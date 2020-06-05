'''
https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 200
# CUDA_VISIBLE_DEVICES = 1

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNetLMS(nn.Module):

    def __init__(self, n_lm=2):
        super(UNetLMS).__init__()
        self.n_lm = n_lm
        self.dconv_down1 = double_conv(3, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 96)
        self.dconv_down4 = double_conv(96, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(96 + 128, 96)
        self.dconv_up2 = double_conv(64 + 96, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, self.n_lm, 1)

        self.apply(weights_init)


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        x = x.view(x.shape[0], self.n_lm, IMAGE_SIZE*IMAGE_SIZE)
        x = F.softmax(x, dim=2)
        x = x.view(x.shape[0], self.n_lm, IMAGE_SIZE, IMAGE_SIZE)

        return x


class UNetCLMS(nn.Module):

    def __init__(self, n_lm=2):
        super(UNetCLMS).__init__()
        self.n_lm = n_lm
        self.dconv_down1 = double_conv(6, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 96)
        self.dconv_down4 = double_conv(96, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(96 + 128, 96)
        self.dconv_up2 = double_conv(64 + 96, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, self.n_lm, 1)

        self.apply(weights_init)

    def _cell(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        x = x.view(x.shape[0], self.n_lm, IMAGE_SIZE*IMAGE_SIZE)
        x = F.softmax(x, dim=2)
        x = x.view(x.shape[0], self.n_lm, IMAGE_SIZE, IMAGE_SIZE)

        return x

    def forward(self, x):
        gm = torch.zeros((x.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)).cuda()
        comb_masks = gm
        masks = []

        for i in range(self.n_lm):
            # print(i)
            # print(gm)
            stacked_inputs = torch.cat([x, x*comb_masks], dim=1)
            # print(stacked_inputs.shape)
            # if i == 0:
            #     stacked_inputs = x
            # else:
            #     stacked_inputs = x*(comb_masks-1)*-1

            pm = self._cell(stacked_inputs)

            lms, gm = get_transform_mat(pm)

            # comb_masks = comb_masks + gm
            comb_masks = torch.max(comb_masks, gm)


            if i==0:
                coors = lms
                masks = pm
            else:
                coors = torch.cat([coors, lms], dim=1)
                masks = torch.cat([masks, pm], dim=1)

            # print(coors.shape)

        return comb_masks, masks, coors


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1e-2)



# net = CLMS()
# net.forward(torch.zeros((2,3,200,200)))
