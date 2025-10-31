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
    for idx, batch in enumerate(loader):
        batch = {
            x: batch[x].to(torch.device('cuda', args.rank))
            for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']
        }

        image = batch["image"]
        target = batch["image"]
        for param in model.parameters(): param.grad = None
        logits = model(image)
        
      
        logits = torch.relu(logits)

        min_vals = image.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_vals = image.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

    
        target = (image - min_vals) / (max_vals - min_vals).clamp(min=1e-5)
       
        loss1 = loss_func1(logits, target)
        loss2 = loss_func2(logits, target)
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
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
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

            if scheduler is not None:
                scheduler.step()

        print('Training Finished Final loss!', train_loss)

        return train_loss

