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

def create_random_missing_modalities(input_image, num_modalities):
    
    missing_image = input_image.clone()


    num_missing = torch.randint(1, num_modalities, (1,)).item()  
  
    missing_indices = torch.randperm(num_modalities)[:num_missing]
 

    for idx in missing_indices:
        missing_image[:, idx, ...] = 0

    return missing_image
    
def train_epoch(model,
                loader,
                optimizer,
                epoch,
                args,
                loss1_func,
                loss2_func,
                ):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_loss_1 = AverageMeter()
    run_loss_2 = AverageMeter()
    run_loss_3 = AverageMeter()
    for idx, batch in enumerate(loader):
        batch = {
            x: batch[x].to(torch.device('cuda', args.rank))
            for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']
        }

        image = batch["image"]
        target = batch["label"]
        for param in model.parameters(): param.grad = None     
        ##15 types of modal combinations
        all_inputs_list = []
        all_inputs_list.append(image)
        channels_to_zero_1 = [0]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_1, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_2 = [1]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_2, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_3 = [2]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_3, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_4 = [3]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_4, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_5 = [0,1]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_5, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_6 = [0,2]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_6, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_7 = [0,3]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_7, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_8 = [1,2]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_8, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_9 = [1,3]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_9, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_10 = [2,3]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_10, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_11 = [0,1,2]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_11, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_12 = [0,1,3]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_12, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_13 = [0,2,3]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_13, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        channels_to_zero_14 = [1,2,3]
        uncom_image = image.clone()
        uncom_image[:, channels_to_zero_14, :, :, :] = 0
        all_inputs_list.append(uncom_image)
        ####all_input
        grouped_inputs = []
        group_size = 2  
        grouped_inputs.append(all_inputs_list[0])
    
        for i in range(1, len(all_inputs_list)-1, group_size):
            group = all_inputs_list[i:i + group_size]
        
            concatenated_tensor = torch.cat(group, dim=0)         
        
            grouped_inputs.append(concatenated_tensor)
            
        for i in range(len(grouped_inputs)):
        
            ##complete input
            if i ==0:
                with torch.no_grad():
                    e1_complete, e2_complete, e3_complete, e4_complete, e5_complete = model.Encoder_Complete(grouped_inputs[i])    
                complete_logits = model.Decoder(e1_complete, e2_complete, e3_complete, e4_complete, e5_complete) ## complete
                loss1 = loss1_func(complete_logits, target)
                loss1.backward()
                complete_logits = complete_logits.detach()
            else:
                e1, e2, e3, e4, e5 = model.SSL_Adaptor(grouped_inputs[i])    
                logits = model.Decoder(e1, e2, e3, e4, e5) ## complete
                loss2 = loss1_func(logits,torch.sigmoid(torch.cat((complete_logits, complete_logits), dim=0))) 
                loss3_1 = loss2_func(e1,torch.cat((e1_complete, e1_complete), dim=0))
                loss3_2 = loss2_func(e2,torch.cat((e2_complete, e2_complete), dim=0))
                loss3_3 = loss2_func(e3,torch.cat((e3_complete, e3_complete), dim=0))
                loss3_4 = loss2_func(e4,torch.cat((e4_complete, e4_complete), dim=0))
                loss3_5 = loss2_func(e5,torch.cat((e5_complete, e5_complete), dim=0))
                loss3 = 0.2*(loss3_1 + loss3_2 + loss3_3 + loss3_4 + loss3_5)
                loss = loss2 + loss3
                loss.backward() 
                
        optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss_1.update(loss1.item(), n=args.batch_size)
            run_loss_2.update(loss2.item(), n=args.batch_size)
            run_loss_3.update(loss3.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss1: {:.4f}'.format(run_loss_1.avg),
                  'loss2: {:.4f}'.format(run_loss_2.avg),
                  'loss3: {:.4f}'.format(run_loss_3.avg),
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

        test_acc_max_mean_list = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
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
                                     loss1_func=self.loss_func1,
                                     loss2_func=self.loss_func2)
            if args.rank == 0:
                print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                      'time {:.2f}s'.format(time.time() - epoch_time))
            if args.rank==0 and writer is not None:
                writer.add_scalar('train_loss', train_loss, epoch)

            b_new_best = False
            if (epoch+1) % args.val_every == 0 and self.testor is not None:
                if args.distributed:
                    torch.distributed.barrier()
                epoch_time = time.time()
                test_avg_acc = self.testor.run()
                test_mean_dice_list = [] 
                for i in range(len(test_avg_acc)):
                    test_mean_dice = self.testor.metric_dice_avg(test_avg_acc[i])
                    test_mean_dice_list.append(test_mean_dice)                    
                if args.rank == 0:
                    print('Final test  {}/{}'.format(epoch, args.max_epochs - 1),
                          'acc', test_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time),
                          "test_mean_dice", test_mean_dice_list)
                    if writer is not None:
                        for test_avg_acc_idex in range(len(test_avg_acc)): 
                            for name, value in test_avg_acc[test_avg_acc_idex].items():
                                if "dice" in name.lower():
                                    writer.add_scalar(name, value, epoch)
                            writer.add_scalar('test_mean_dice', test_mean_dice_list[test_avg_acc_idex], epoch)

                    test_mean_dice_all_fifth_result = sum(test_mean_dice_list) / len(test_mean_dice_list)
                    test_acc_max_mean_all_fifth_result = sum(test_acc_max_mean_list) / len(test_acc_max_mean_list)
                    if test_mean_dice_all_fifth_result > test_acc_max_mean_all_fifth_result:
                        print('new best ({:.6f} --> {:.6f}). '.format(test_acc_max_mean_all_fifth_result, test_mean_dice_all_fifth_result))
                        test_acc_max_mean_all_fifth_result = test_mean_dice_all_fifth_result
                        test_acc_max_mean_list = test_mean_dice_list
                        b_new_best = True
                        if args.rank == 0 and args.logdir is not None:
                            save_checkpoint(model, epoch, args,
                                            best_acc=test_acc_max_mean_list,
                                            optimizer=optimizer,
                                            scheduler=scheduler)

                if (epoch+1) % args.val_every == 0:
                    if args.distributed:
                        torch.distributed.barrier()
                    if args.rank == 0 and args.logdir is not None:
                        with open(os.path.join(args.logdir, "log.txt"), "a+") as f:
                            f.write(f"epoch:{epoch+1}, train_loss:{train_loss}")
                            f.write("\n")
                        save_checkpoint(model,
                                        epoch,
                                        args,
                                        filename='model_final'+str(epoch)+'.pt')
                
                if args.rank == 0 and args.logdir is not None:
                    with open(os.path.join(args.logdir, "log.txt"), "a+") as f:
                        f.write(f"epoch:{epoch+1}, metric:{test_mean_dice_list}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, avg metric: {sum(test_mean_dice_list) / len(test_mean_dice_list)}")
                        f.write("\n")
                        f.write(f"epoch:{epoch+1}, best metric:{test_acc_max_mean_list}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, best avg metric: {sum(test_acc_max_mean_list) / len(test_acc_max_mean_list)}")
                        f.write("\n")
                        f.write("*" * 20)
                        f.write("\n")

                    save_checkpoint(model,
                                    epoch,
                                    args,
                                    best_acc=test_acc_max_mean_list,
                                    filename='model_final.pt')
                    if b_new_best:
                        print('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

            if scheduler is not None:
                scheduler.step()

        print('Training Finished !, Best Accuracy: ', test_acc_max_mean)

        return test_acc_max_mean_list

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
        accs_list = []
        not_nans_list = []
        for idex_input in range(15):
            accs_list.append(accs)
            not_nans_list.append(not_nans)
        
        for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):

            batch = {
                x: batch[x].to(torch.device('cuda', args.rank))
                for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']
            }

            label = batch["label"]
            image = batch["image"]
            ######15
            all_inputs_list = []
            all_inputs_list.append(image)
            channels_to_zero_1 = [0]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_1, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_2 = [1]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_2, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_3 = [2]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_3, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_4 = [3]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_4, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_5 = [0,1]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_5, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_6 = [0,2]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_6, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_7 = [0,3]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_7, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_8 = [1,2]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_8, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_9 = [1,3]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_9, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_10 = [2,3]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_10, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_11 = [0,1,2]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_11, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_12 = [0,1,3]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_12, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_13 = [0,2,3]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_13, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            channels_to_zero_14 = [1,2,3]
            uncom_image = image.clone()
            uncom_image[:, channels_to_zero_14, :, :, :] = 0
            all_inputs_list.append(uncom_image)
            # print(accs_list)
            for idex_input in range(len(all_inputs_list)):
                with torch.no_grad():
                    if self.sliding_window_infer is not None:
                        if idex_input ==0: 
                            logits = self.sliding_window_infer(all_inputs_list[idex_input], self.model.forward_complete)
                        else:
                            logits = self.sliding_window_infer(all_inputs_list[idex_input], self.model.forward_uncomplete)
                    else:
                        logits = self.model(all_inputs_list[idex_input])
    
                    if self.post_label is not None:
                        label = self.post_label(label)
    
                    if self.post_pred is not None:
                        logits = self.post_pred(logits)
    
                    for i in range(len(self.metric_functions)):
                        acc = self.metric_functions[i][1](y_pred=logits, y=label)
                        acc, not_nan = do_metric_reduction(acc, MetricReduction.MEAN_BATCH)
                        acc = acc.cuda(args.rank)
                        not_nan = not_nan.cuda(args.rank)
                        if accs_list[idex_input][i] is None:
                            # accs: [[dice_class_1, dice_class_2], [hd95_class_1, hd95_class_2]]
                            accs_list[idex_input] = acc
                            not_nans_list[idex_input] = not_nan
                        else :
                            accs_list[idex_input] += acc
                            not_nans_list[idex_input] += not_nan
            

        all_metric_list = []
        for i in range(15):
            accs_list[i] = [mt.as_tensor() for mt in accs_list[i]]
            accs_list[i] = torch.stack(accs_list[i], dim=0).flatten()
            not_nans_list[i] = [mt.as_tensor() for mt in not_nans_list[i]]
            not_nans_list[i] = torch.stack(not_nans_list[i], dim=0).flatten()
            not_nans_list[i][not_nans_list[i] == 0] = 1
            accs_list[i] =  accs_list[i] / not_nans_list[i]
            all_metric = {k: v for k, v in zip(class_metric, accs_list[i].tolist())}
            all_metric_list.append(all_metric)    
        return all_metric_list