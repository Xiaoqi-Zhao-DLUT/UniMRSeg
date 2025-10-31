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
import matplotlib.pyplot as plt
import nibabel as nib

class Trainer:
    def __init__(self, args,
                 train_loader,
                 loss_func,
                 validator=None,
                 testor=None
                 ):
        pass
        self.args = args
        self.train_loader = train_loader
        self.validator = validator
        self.testor = testor
        self.loss_func = loss_func

    def train(self, model,
              optimizer,
              scheduler=None,
              start_epoch=0,
              ):
        pass
        args = self.args
        val_acc_max_mean = 0.
        val_acc_max = 0.
        test_acc_max_mean = 0.
        test_acc_max = 0.
        b_new_best = False
        if args.distributed:
            torch.distributed.barrier()
        epoch_time = time.time()
        val_avg_acc = self.validator.run()
        test_avg_acc = self.testor.run()

        mean_dice_list = []
        test_mean_dice_list = [] 
        for i in range(len(val_avg_acc)):
            mean_dice = self.validator.metric_dice_avg(val_avg_acc[i])
            mean_dice_list.append(mean_dice)
            test_mean_dice = self.testor.metric_dice_avg(test_avg_acc[i])
            test_mean_dice_list.append(test_mean_dice)   
                    
        print('Final validation',
              'acc', val_avg_acc,
              "mean_dice", mean_dice_list)
        print('Final test',
              'acc', test_avg_acc,
              "test_mean_dice", test_mean_dice_list)
        print('Training Finished !, Best Accuracy: ', test_mean_dice_list)
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
                        logits = self.sliding_window_infer(all_inputs_list[idex_input], self.model)
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
            print(accs_list[i])
            not_nans_list[i] = [mt.as_tensor() for mt in not_nans_list[i]]
            not_nans_list[i] = torch.stack(not_nans_list[i], dim=0).flatten()
            not_nans_list[i][not_nans_list[i] == 0] = 1
            print(not_nans_list[i])
            accs_list[i] =  accs_list[i] / not_nans_list[i]
            print(accs_list[i])
            all_metric = {k: v for k, v in zip(class_metric, accs_list[i].tolist())}
            all_metric_list.append(all_metric)    
            print(all_metric)
        print(all_metric_list)
        return all_metric_list