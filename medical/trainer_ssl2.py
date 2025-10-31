# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from monai.metrics.utils import do_metric_reduction
from monai.utils.enums import MetricReduction
from tqdm import tqdm
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather, AverageMeter
import torch.utils.data.distributed
import torch.nn.functional as F

def train_epoch(model,
                loader,
                optimizer,
                epoch,
                args,
                loss_func1,
                loss_func2,
                ):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_loss_1 = AverageMeter()
    for idx, batch in enumerate(loader):
        batch = {
            x: batch[x].to(torch.device('cuda', args.rank))
            for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']
        }
        image = batch["image"]
        target = batch["label"]
        loss1 = 0
        loss2 = 0
        for param in model.parameters(): param.grad = None
        if image.shape[0] !=1:
        
            input1_fea, input2_fea, logits = model(image, seg_flag=0)         
            for i in range(len(input1_fea)):
                loss1 +=  loss_func1(F.normalize(input1_fea[i],dim=1),F.normalize(input2_fea[i],dim=1),temperature=0.5)
            target_cons = target[0].unsqueeze(0)
            target_cons = torch.cat((target_cons, target_cons), dim=0)
            loss2 += loss_func2(logits, target_cons)
            
            loss_alpha = 1/len(input1_fea)
            loss = loss_alpha*loss1 + loss2
            loss.backward()
       
            loss2 = 0
            logits = model(image, seg_flag=1)   
            target_cons = target[1].unsqueeze(0)
            target_cons = torch.cat((target_cons, target_cons), dim=0)
            loss2 = loss_func2(logits, target_cons)
            loss2.backward()
        else:
            logits = model(image, seg_flag=1)  
            target_cons = torch.cat((target, target), dim=0)
            loss1 = torch.zeros(1).cuda()
            loss2 = loss_func2(logits, target_cons)
            loss = loss1 + loss2
            loss.backward()
            
        optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_loss_1.update(loss1.item(), n=args.batch_size)
            
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'loss1: {:.4f}'.format(run_loss_1.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

class Trainer:
    def __init__(self, args,
                 train_loader,
                 loss_func1,
                 loss_func2,
                 validator=None,
                 testor=None
                 ):
        pass
        self.args = args
        self.train_loader = train_loader
        self.validator = validator
        self.testor = testor
        self.loss_func1 = loss_func1
        self.loss_func2 = loss_func2

    def train(self, model,
              optimizer,
              scheduler=None,
              start_epoch=0,
              ):
        pass
        args = self.args
        train_loader = self.train_loader
        writer = None

        if args.logdir is not None and args.rank == 0:
            writer = SummaryWriter(log_dir=args.logdir)
            if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)

        val_acc_max_mean = 0.
        val_acc_max = 0.
        test_acc_max_mean = 0.
        test_acc_max = 0.
        for epoch in range(start_epoch, args.max_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            print(args.rank, time.ctime(), 'Epoch:', epoch)
            epoch_time = time.time()
            train_loss = train_epoch(model,
                                     train_loader,
                                     optimizer,
                                     epoch=epoch,
                                     args=args,
                                     loss_func1=self.loss_func1,
                                     loss_func2=self.loss_func2
                                    )
            if args.rank == 0:
                print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                      'time {:.2f}s'.format(time.time() - epoch_time))
            if args.rank==0 and writer is not None:
                writer.add_scalar('train_loss', train_loss, epoch)

            b_new_best = False
            if (epoch+1) % args.val_every == 0 and self.validator is not None:
                if args.distributed:
                    torch.distributed.barrier()
                epoch_time = time.time()
                val_avg_acc = self.validator.run()
                mean_dice = self.validator.metric_dice_avg(val_avg_acc)
                test_avg_acc = self.testor.run()
                test_mean_dice = self.testor.metric_dice_avg(test_avg_acc)
                if args.rank == 0:
                    print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                          'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time),
                          "mean_dice", mean_dice)
                    print('Final test  {}/{}'.format(epoch, args.max_epochs - 1),
                          'acc', test_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time),
                          "test_mean_dice", test_mean_dice)
                    if writer is not None:
                        for name, value in val_avg_acc.items():
                            if "dice" in name.lower():
                                writer.add_scalar(name, value, epoch)
                        writer.add_scalar('mean_dice', mean_dice, epoch)
                        for name, value in test_avg_acc.items():
                            if "dice" in name.lower():
                                writer.add_scalar(name, value, epoch)
                        writer.add_scalar('test_mean_dice', test_mean_dice, epoch)
                    
                    if test_mean_dice > test_acc_max_mean:
                        print('new best ({:.6f} --> {:.6f}). '.format(test_acc_max_mean, test_mean_dice))
                        test_acc_max_mean = test_mean_dice
                        test_acc_max = test_avg_acc
                        b_new_best = True
                        if args.rank == 0 and args.logdir is not None:
                            save_checkpoint(model, epoch, args,
                                            best_acc=test_acc_max_mean,
                                            optimizer=optimizer,
                                            scheduler=scheduler)

                if args.rank == 0 and args.logdir is not None:
                    with open(os.path.join(args.logdir, "log.txt"), "a+") as f:
                        f.write(f"epoch:{epoch+1}, metric:{test_avg_acc}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, avg metric: {test_mean_dice}")
                        f.write("\n")
                        f.write(f"epoch:{epoch+1}, best metric:{test_acc_max}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, best avg metric: {test_acc_max_mean}")
                        f.write("\n")
                        f.write("*" * 20)
                        f.write("\n")

                    save_checkpoint(model,
                                    epoch,
                                    args,
                                    best_acc=test_acc_max,
                                    filename='model_final.pt')
                    if b_new_best:
                        print('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

            if scheduler is not None:
                scheduler.step()

        print('Training Finished !, Best Accuracy: ', test_acc_max_mean)

        return test_acc_max

class Validator:
    def __init__(self,
                 args,
                 model,
                 val_loader,
                 class_list,
                 metric_functions,
                 sliding_window_infer=None,
                 post_label=None,
                 post_pred=None,

                 ) -> None:

        self.val_loader = val_loader
        self.sliding_window_infer = sliding_window_infer
        self.model = model
        self.args = args
        self.post_label = post_label
        self.post_pred = post_pred
        self.metric_functions = metric_functions
        self.class_list = class_list

    def metric_dice_avg(self, metric):
        metric_sum = 0.0
        c_nums = 0
        for m, v in metric.items():
            if "dice" in m.lower():
                metric_sum += v
                c_nums += 1

        return metric_sum / c_nums

    def is_best_metric(self, cur_metric, best_metric):

        best_metric_sum = self.metric_dice_avg(best_metric)
        metric_sum = self.metric_dice_avg(cur_metric)
        if best_metric_sum < metric_sum:
            return True

        return False

    def run(self):
        self.model.eval()
        args = self.args

        assert len(self.metric_functions[0]) == 2

        accs = [None for i in range(len(self.metric_functions))]
        not_nans = [None for i in range(len(self.metric_functions))]
        class_metric = []
        for m in self.metric_functions:
            for clas in self.class_list:
                class_metric.append(f"{clas}_{m[0]}")
        for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):

            batch = {
                x: batch[x].to(torch.device('cuda', args.rank))
                for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']
            }

            label = batch["label"]

            with torch.no_grad():
                if self.sliding_window_infer is not None:
                    logits = self.sliding_window_infer(batch["image"], self.model) # seg_flage此时无意义
                else:
                    logits = self.model(batch["image"],seg_flag=1)

                if self.post_label is not None:
                    label = self.post_label(label)

                if self.post_pred is not None:
                    logits = self.post_pred(logits)

                for i in range(len(self.metric_functions)):
                    acc = self.metric_functions[i][1](y_pred=logits, y=label)
                    acc, not_nan = do_metric_reduction(acc, MetricReduction.MEAN_BATCH)
                    acc = acc.cuda(args.rank)
                    not_nan = not_nan.cuda(args.rank)
                    if accs[i] is None:
                        # accs: [[dice_class_1, dice_class_2], [hd95_class_1, hd95_class_2]]
                        accs[i] = acc
                        not_nans[i] = not_nan
                    else :
                        accs[i] += acc
                        not_nans[i] += not_nan

        if args.distributed:
            # accs: [dice_class_1, dice_class_2, hd95_class_1, hd95_class_2]
            accs = torch.stack(accs).cuda(args.rank).flatten()
            not_nans = torch.stack(not_nans).cuda(args.rank).flatten()
            torch.distributed.barrier()
            gather_list_accs = [torch.zeros_like(accs) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_list_accs, accs)
            gather_list_not_nans = [torch.zeros_like(not_nans) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_list_not_nans, not_nans)

            accs_sum = torch.stack(gather_list_accs).sum(dim=0).flatten()
            not_nans_sum = torch.stack(gather_list_not_nans).sum(dim=0).flatten()

            not_nans_sum[not_nans_sum == 0] = 1
            accs_sum = accs_sum / not_nans_sum
            all_metric_list = {k: v for k, v in zip(class_metric, accs_sum.tolist())}

        else:

            accs = torch.stack(accs, dim=0).flatten()
            not_nans = torch.stack(not_nans, dim=0).flatten()
            not_nans[not_nans == 0] = 1
            accs = accs / not_nans
            all_metric_list = {k: v for k, v in zip(class_metric, accs.tolist())}

        return all_metric_list