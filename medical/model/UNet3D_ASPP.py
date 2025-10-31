import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ModalityAugmentor:
    def __init__(self, missing_modalities_prob=0.5, mask_prob=0.15):
        self.missing_modalities_prob = missing_modalities_prob
        self.mask_prob = mask_prob

    def random_modalities_drop(self, x):
        """
        Randomly decide whether to drop some modalities for each sample in the batch.
        """
        batch_size, num_modalities, depth, height, width = x.size()

        dropped_x = torch.zeros_like(x)
        keep_modalities = []

        for i in range(batch_size):
            # Decide whether to drop modalities based on missing_modalities_prob
            if random.random() < self.missing_modalities_prob:
                # Randomly select how many modalities to keep (at least 1)
                keep_count = random.randint(1, num_modalities)
                keep_idxs = random.sample(range(num_modalities), keep_count)
            else:
                # Keep all modalities
                keep_idxs = list(range(num_modalities))

            keep_modalities.append(keep_idxs)
            dropped_x[i, keep_idxs] = x[i, keep_idxs]

        return dropped_x, keep_modalities

    def random_modalities_shuffle(self, x, keep_modalities):
        """
        Randomly shuffle the remaining modalities in each batch sample.
        """
        batch_size, num_modalities, depth, height, width = x.size()
        shuffled_x = torch.zeros_like(x)

        for i in range(batch_size):
            keep_idx = keep_modalities[i]
            random.shuffle(keep_idx)  # Shuffle kept modalities
            # print(keep_idx)
            shuffled_x[i, :len(keep_idx)] = x[i, keep_idx]

        return shuffled_x

    def apply_random_mask(self, x):
        """
        Randomly apply a mask to the remaining modalities (similar to MAE).
        """
        batch_size, num_modalities, depth, height, width = x.size()
        mask = torch.rand(batch_size, num_modalities, depth, height, width) < self.mask_prob
        masked_x = x.clone()
        masked_x[mask] = 0  # Apply masking (0 out the masked positions)

        return masked_x

    def __call__(self, x):
        """
        Apply the full augmentation pipeline: random drop, shuffle, and masking.
        """
        # Step 1: Randomly drop some modalities
        x, keep_modalities = self.random_modalities_drop(x)

        # Step 2: Randomly shuffle remaining modalities
        x = self.random_modalities_shuffle(x, keep_modalities)

        # Step 3: Apply spatial masking to the remaining modalities
        x = self.apply_random_mask(x)

        return x


class DoubleConv3D(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW, concatenate along channel axis
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super(ASPP3D, self).__init__()

        self.convs = nn.ModuleList()
        for rate in dilation_rates:
            self.convs.append(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
            )

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.final_conv = nn.Conv3d(len(dilation_rates) * out_channels + out_channels, out_channels, kernel_size=1,
                                    bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res.append(F.interpolate(self.pooling(x), size=x.shape[2:], mode='trilinear', align_corners=False))

        res = torch.cat(res, dim=1)
        res = self.final_conv(res)
        res = self.bn(res)

        return self.relu(res)

class UNet3D_3DASPP(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
        super(UNet3D_3DASPP, self).__init__()
        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        self.aspp = ASPP3D(512, 512)  # Adding 3D-ASPP module here
        self.up1 = Up3D(1024, 256)
        self.up2 = Up3D(512, 128)
        self.up3 = Up3D(256, 64)
        self.up4 = Up3D(128, 64)
        self.outc = OutConv3D(64, 3)



    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape) 
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.aspp(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # print(logits.shape)
        return logits



class UNet3D_3DASPP_ssl1(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
        super(UNet3D_3DASPP_ssl1, self).__init__()
        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        self.aspp = ASPP3D(512, 512)  # Adding 3D-ASPP module here
        self.up1 = Up3D(1024, 256)
        self.up2 = Up3D(512, 128)
        self.up3 = Up3D(256, 64)
        self.up4 = Up3D(128, 64)
        self.outc = OutConv3D(64, 4)
        self.augmentor = ModalityAugmentor(missing_modalities_prob=0.5, mask_prob=0.15)


    def forward(self, x):
    # Apply the augmentation
        x = self.augmentor(x)
        input_drop = x
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.aspp(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # print(logits.shape)
        return input_drop, logits

def create_random_missing_modalities(input_image, num_modalities):
    missing_image = input_image.clone()

 
    num_missing = torch.randint(1, num_modalities, (1,)).item()  # 
    # print(num_missing)
    missing_indices = torch.randperm(num_modalities)[:num_missing]
    # print(missing_indices)
    for idx in missing_indices:
        missing_image[:, idx, ...] = 0

    return missing_image
    


class UNet3D_3DASPP_ssl2_new(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
        super(UNet3D_3DASPP_ssl2_new, self).__init__()
        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        self.aspp = ASPP3D(512, 512)  # Adding 3D-ASPP module here
        self.up1 = Up3D(1024, 256)
        self.up2 = Up3D(512, 128)
        self.up3 = Up3D(256, 64)
        self.up4 = Up3D(128, 64)
        self.outc = OutConv3D(64, 3)

    def Body(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.aspp(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return x1, x2, x3, x4, x5, logits
        
    def Encoder(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.aspp(x4)
        return x1, x2, x3, x4, x5
         
    def forward(self, x, seg_flag=2):
        if self.training:
            num_modalities = x.shape[1] 
            input1 = x[0].unsqueeze(0)  # [1, c, h, w, d]
            if x.shape[0]==1:
                input2 = x[0].unsqueeze(0)  # [1, c, h, w, d]
            else:
                input2 = x[1].unsqueeze(0)
      
            input1_missing = create_random_missing_modalities(input1, num_modalities)
            input2_missing = create_random_missing_modalities(input2, num_modalities)
            
            # [2, c, h, w, d]
            new_input1 = torch.cat((input1, input1_missing), dim=0)
            new_input2 = torch.cat((input2, input2_missing), dim=0)
            new_input = [new_input1, new_input2]
            new_ouput = []
            input1_fea = []
            input2_fea = []
            
            if seg_flag == 0:
                x1, x2, x3, x4, x5, logits = self.Body(new_input[0])
                input1_fea.append(F.adaptive_avg_pool3d(x1, (1, 1, 1)).view(x1.shape[0], -1))
                input1_fea.append(F.adaptive_avg_pool3d(x2, (1, 1, 1)).view(x2.shape[0], -1))
                input1_fea.append(F.adaptive_avg_pool3d(x3, (1, 1, 1)).view(x3.shape[0], -1))
                input1_fea.append(F.adaptive_avg_pool3d(x4, (1, 1, 1)).view(x4.shape[0], -1))
                input1_fea.append(F.adaptive_avg_pool3d(x5, (1, 1, 1)).view(x5.shape[0], -1))
                with torch.no_grad():
                    training_state = self.training
                    self.eval()
                    x1, x2, x3, x4, x5 = self.Encoder(new_input[1])     
                    self.train(training_state)
                    input2_fea.append(F.adaptive_avg_pool3d(x1, (1, 1, 1)).view(x1.shape[0], -1))
                    input2_fea.append(F.adaptive_avg_pool3d(x2, (1, 1, 1)).view(x2.shape[0], -1))
                    input2_fea.append(F.adaptive_avg_pool3d(x3, (1, 1, 1)).view(x3.shape[0], -1))
                    input2_fea.append(F.adaptive_avg_pool3d(x4, (1, 1, 1)).view(x4.shape[0], -1))
                    input2_fea.append(F.adaptive_avg_pool3d(x5, (1, 1, 1)).view(x5.shape[0], -1))
                return input1_fea, input2_fea, logits    
                
            elif seg_flag == 1:
                _, _, _, _, _, logits = self.Body(new_input[1])
                return logits
        else:
            x1 = self.inc(x)
            # print(x1.shape) 
            x2 = self.down1(x1)
            # print(x2.shape)
            x3 = self.down2(x2)
            # print(x3.shape)
            x4 = self.down3(x3)
            # print(x4.shape)
            x5 = self.aspp(x4)
            # print(x5.shape)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            # print(logits.shape)
            return logits

