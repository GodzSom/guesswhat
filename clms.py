'''
Conditioal Landmark Selector
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 200
CUDA_VISIBLE_DEVICES = 1

def meshgrid(h):
    xv, yv = torch.meshgrid([torch.arange(0.5, h, 1) / (h / 2) - 1, torch.arange(0.5, h, 1) / (h / 2) - 1])
    return xv.cuda(), yv.cuda()

ranx, rany = meshgrid(IMAGE_SIZE)

def get_gaussian_maps(mu, inv_std):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = torch.linspace(-1.0, 1.0, IMAGE_SIZE).cuda()
    x = torch.linspace(-1.0, 1.0, IMAGE_SIZE).cuda()

    y = torch.reshape(y, (1, 1, IMAGE_SIZE))
    x = torch.reshape(x, (1, 1, IMAGE_SIZE))

    g_y = torch.exp(-(mu_y-y)**2 * inv_std)
    g_x = torch.exp(-(mu_x-x)**2 * inv_std)

    g_y = g_y[:,:,:,None]#tf.expand_dims(g_y, axis=3)
    g_x = g_x[:,:,None,:]#tf.expand_dims(g_x, axis=2)
    g_yx = torch.matmul(g_y, g_x)  # [B, NMAPS, H, W]

    # g_yx = torch.transpose(g_yx, perm=[0, 2, 3, 1])
    g_yx = torch.transpose(g_yx, 1 ,2)
    g_yx = torch.transpose(g_yx, 2, 3)
    return g_yx #* 1.15

def get_transform_mat(probmap, inv_std=20, N_KEYPOINT=1):
    # ranx, rany = meshgrid(IMAGE_SIZE)
    x = torch.sum(probmap * ranx, dim=(2,3))
    y = torch.sum(probmap * rany, dim=(2,3))

    coors = torch.stack([x, y], dim=2)

    masks = get_gaussian_maps(torch.reshape(coors, (-1, N_KEYPOINT, 2)), inv_std)
    masks = torch.transpose(masks, 2, 3)
    masks = torch.transpose(masks, 1, 2)

    return coors, masks

class BasicBlock(nn.Module):

    def __init__(self, n_lm):
        super(BasicBlock, self).__init__()

        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(6, 64, kernel_size=5, stride=4, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = F.leaky_relu(self.bn3(self.conv3(out)))
        out = F.leaky_relu(self.conv4(out))

        d = IMAGE_SIZE
        # print(out.shape)
        out = F.interpolate(out, scale_factor=8, mode='bilinear')

        out = out.view(out.shape[0], 1, d*d)
        out = F.softmax(out, dim=2)
        out = out.view(out.shape[0], 1, d, d)

        # print(out.shape)
        # out = F.pad(out, (10,10,10,10))

        return out

class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockL(nn.Module):

    def __init__(self, n_lm):
        super(BasicBlockL, self).__init__()

        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(6, 96, 5, stride = 2, padding = 0)
        self.norm1 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2 = nn.Conv2d(96, 128, 5, stride = 1, padding = 0)
        self.norm2 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3 = nn.Conv2d(128, 96, 3, stride = 1, padding = 0)
        self.norm3 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv4 = nn.Conv2d(96, 1, 3, stride = 1, padding = 0)
        self.norm4 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.norm1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.norm2(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.norm3(x)

        x = F.leaky_relu(self.conv4(x))
        x = self.norm4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = x.view(-1, 1, 180*180)
        x = F.softmax(x, dim=2)
        x = x.view(-1, 1, 180, 180)

        x = F.pad(x, (10,10,10,10))

        return x


class CLMSwat(nn.Module):
    def __init__(self, block = BasicBlockL, n_lm = 2):
        super(CLMSwat, self).__init__()

        self.n_lm = n_lm
        self.cell = block(self.n_lm)

        self.apply(weights_init)


    def forward(self, x):

        gm = torch.zeros((x.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)).cuda()
        comb_masks = gm
        masks = []

        for i in range(self.n_lm):
            # print(i)
            # print(gm)
            stacked_inputs = torch.cat([x, x*comb_masks], dim=1)
            # print(stacked_inputs.shape)
            pm = self.cell(stacked_inputs)

            lms, gm = get_transform_mat(pm)

            comb_masks = comb_masks + gm

            if i==0:
                coors = lms
                masks = pm
            else:
                coors = torch.cat([coors, lms], dim=1)
                masks = torch.cat([masks, pm], dim=1)

            # print(coors.shape)

        return comb_masks, masks, coors

class CLMS_M(nn.Module):
    def __init__(self, block = BasicBlockL, n_lm = 2):
        super(CLMS_M, self).__init__()
        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2, padding = 3)
        self.norm1 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2 = nn.Conv2d(64, 96, 7, stride = 1, padding = 3)
        self.norm2 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3 = nn.Conv2d(96, 128, 5, stride = 1, padding = 2)
        self.norm3 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv4 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm4 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv5 = nn.Conv2d(128, 96, 5, stride = 1, padding = 1)
        self.norm5 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv6 = nn.Conv2d(96, 64, 3, stride = 1, padding = 1)
        self.norm6 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv7 = nn.Conv2d(64, 1, 3, stride = 1, padding = 1)
        self.norm7 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.apply(weights_init)

    def forward(self, x):

        gm = torch.zeros((x.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)).cuda()
        comb_masks = gm
        masks = []

        for i in range(self.n_lm):
            # print(gm)
            # stacked_inputs = torch.cat([x, x*comb_masks], dim=1)
            # print(stacked_inputs.shape)
            if i == 0:
                stacked_inputs = x
            else:
                stacked_inputs = x*(comb_masks-1)*-1

            l = F.leaky_relu(self.conv1(stacked_inputs))
            l = self.norm1(l)

            l = F.leaky_relu(self.conv2(l))
            l = self.norm2(l)

            l = F.leaky_relu(self.conv3(l))
            l = self.norm3(l)

            l = F.leaky_relu(self.conv4(l))
            l = self.norm4(l)

            l = F.leaky_relu(self.conv5(l))
            l = self.norm5(l)

            l = F.leaky_relu(self.conv6(l))
            l = self.norm6(l)

            l = F.leaky_relu(self.conv7(l))
            l = self.norm7(l)

            l = F.interpolate(l, scale_factor=2, mode='bilinear')

            # print(l.shape)

            l = l.view(l.shape[0], 1, 196*196)
            l = F.softmax(l, dim=2)
            l = l.view(l.shape[0], 1, 196, 196)

            pm = F.pad(l, (2,2,2,2))

            lms, gm = get_transform_mat(pm)

            # comb_masks = comb_masks + gm
            comb_masks = torch.max(comb_masks, gm)


            if i==0:
                coors = lms
                probmaps = pm
                masks = gm
            else:
                coors = torch.cat([coors, lms], dim=1)
                probmaps = torch.cat([probmaps, pm], dim=1)
                masks = torch.cat([masks, gm], dim=1)

            # print(coors.shape)

        return comb_masks, probmaps, coors, masks

class CLMS(nn.Module):
    def __init__(self, block = BasicBlockL, n_lm = 2):
        super(CLMS, self).__init__()
        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(3, 96, 5, stride = 2, padding = 0)
        self.norm1 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2 = nn.Conv2d(96, 128, 5, stride = 1, padding = 0)
        self.norm2 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3 = nn.Conv2d(128, 96, 3, stride = 1, padding = 0)
        self.norm3 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv4 = nn.Conv2d(96, 1, 3, stride = 1, padding = 0)
        self.norm4 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.apply(weights_init)

    def forward(self, x):

        gm = torch.zeros((x.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)).cuda()
        comb_masks = gm
        masks = []

        for i in range(self.n_lm):
            # print(i)
            # print(gm)
            # stacked_inputs = torch.cat([x, x*comb_masks], dim=1)
            # print(stacked_inputs.shape)
            if i == 0:
                stacked_inputs = x
            else:
                stacked_inputs = x*(comb_masks-1)*-1

            l = F.leaky_relu(self.conv1(stacked_inputs))
            l = self.norm1(l)

            l = F.leaky_relu(self.conv2(l))
            l = self.norm2(l)

            l = F.leaky_relu(self.conv3(l))
            l = self.norm3(l)

            l = F.leaky_relu(self.conv4(l))
            l = self.norm4(l)

            l = F.interpolate(l, scale_factor=2, mode='bilinear')

            l = l.view(l.shape[0], 1, 180*180)
            l = F.softmax(l, dim=2)
            l = l.view(l.shape[0], 1, 180, 180)

            pm = F.pad(l, (10,10,10,10))

            lms, gm = get_transform_mat(pm)

            # comb_masks = comb_masks + gm
            comb_masks = torch.max(comb_masks, gm)


            if i==0:
                coors = lms
                probmaps = pm
                masks = gm
            else:
                coors = torch.cat([coors, lms], dim=1)
                probmaps = torch.cat([probmaps, pm], dim=1)
                masks = torch.cat([masks, gm], dim=1)

            # print(coors.shape)

        return comb_masks, probmaps, coors, masks

class CLMS_RR(nn.Module):
    def __init__(self, block = BasicBlockL, n_lm = 2):
        super(CLMS_RR, self).__init__()
        self.n_lm = n_lm

        self.conv1_A = nn.Conv2d(3, 96, 5, stride = 2, padding = 0)
        self.norm1_A = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2_A = nn.Conv2d(96, 128, 5, stride = 1, padding = 0)
        self.norm2_A = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3_A = nn.Conv2d(128, 96, 3, stride = 1, padding = 0)
        self.norm3_A = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv4_A = nn.Conv2d(96, 1, 3, stride = 1, padding = 0)
        self.norm4_A = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv1_B = nn.Conv2d(6, 96, 5, stride = 2, padding = 0)
        self.norm1_B = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2_B = nn.Conv2d(96, 128, 5, stride = 1, padding = 0)
        self.norm2_B = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3_B = nn.Conv2d(128, 96, 3, stride = 1, padding = 0)
        self.norm3_B = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv4_B = nn.Conv2d(96, 1, 3, stride = 1, padding = 0)
        self.norm4_B = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)


        self.apply(weights_init)

    def forward(self, x):

        gm = torch.zeros((x.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)).cuda()
        comb_masks = gm
        masks = []

        l = F.leaky_relu(self.conv1_A(x))
        l = self.norm1_A(l)

        l = F.leaky_relu(self.conv2_A(l))
        l = self.norm2_A(l)

        l = F.leaky_relu(self.conv3_A(l))
        l = self.norm3_A(l)

        l = F.leaky_relu(self.conv4_A(l))
        l = self.norm4_A(l)

        l = F.interpolate(l, scale_factor=2, mode='bilinear')

        l = l.view(l.shape[0], 1, 180*180)
        l = F.softmax(l, dim=2)
        l = l.view(l.shape[0], 1, 180, 180)

        pm = F.pad(l, (10,10,10,10))

        lms, gm = get_transform_mat(pm)

        comb_masks = gm#torch.max(comb_masks, gm)

        coors = lms
        masks = pm

        ##
        stacked_inputs = torch.cat([x, x*comb_masks], dim=1)
        l = F.leaky_relu(self.conv1_B(stacked_inputs))
        l = self.norm1_B(l)

        l = F.leaky_relu(self.conv2_B(l))
        l = self.norm2_B(l)

        l = F.leaky_relu(self.conv3_B(l))
        l = self.norm3_B(l)

        l = F.leaky_relu(self.conv4_B(l))
        l = self.norm4_B(l)

        l = F.interpolate(l, scale_factor=2, mode='bilinear')

        l = l.view(l.shape[0], 1, 180*180)
        l = F.softmax(l, dim=2)
        l = l.view(l.shape[0], 1, 180, 180)

        pm = F.pad(l, (10,10,10,10))

        lms, gm = get_transform_mat(pm)

        comb_masks = torch.max(comb_masks, gm)


        coors = torch.cat([coors, lms], dim=1)
        masks = torch.cat([masks, pm], dim=1)

            # print(coors.shape)

        return comb_masks, masks, coors

class CLMS_L(nn.Module):
    def __init__(self, block = BasicBlockL, n_lm = 2):
        super(CLMS_L, self).__init__()
        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2, padding = 3)
        self.norm1 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2 = nn.Conv2d(64, 96, 7, stride = 1, padding = 3)
        self.norm2 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3 = nn.Conv2d(96, 128, 7, stride = 1, padding = 3)
        self.norm3 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv4 = nn.Conv2d(128, 128, 7, stride = 1, padding = 3)
        self.norm4 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv5 = nn.Conv2d(128, 128, 7, stride = 1, padding = 3)
        self.norm5 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv6 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm6 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv7 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm7 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv8 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm8 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv9 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm9 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv10 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm10 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv11 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.norm11 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv12 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.norm12 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv13 = nn.Conv2d(128, 96, 3, stride = 1, padding = 1)
        self.norm13 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv14 = nn.Conv2d(96, 64, 3, stride = 1, padding = 1)
        self.norm14 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv15 = nn.Conv2d(64, 32, 3, stride = 1, padding = 1)
        self.norm15 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv16 = nn.Conv2d(32, 1, 3, stride = 1, padding = 1)
        self.norm16 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.apply(weights_init)

    def forward(self, x):

        gm = torch.zeros((x.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)).cuda()
        comb_masks = gm
        masks = []

        for i in range(self.n_lm):
            # print(i)
            # print(gm)
            # stacked_inputs = torch.cat([x, x*comb_masks], dim=1)
            # print(stacked_inputs.shape)
            if i == 0:
                stacked_inputs = x
            else:
                stacked_inputs = x*(comb_masks-1)*-1

            l = F.leaky_relu(self.conv1(stacked_inputs))
            l = self.norm1(l)
            l = F.leaky_relu(self.conv2(l))
            l = self.norm2(l)
            l = F.leaky_relu(self.conv3(l))
            l = self.norm3(l)
            l = F.leaky_relu(self.conv4(l))
            l = self.norm4(l)
            l = F.leaky_relu(self.conv5(l))
            l = self.norm5(l)
            l = F.leaky_relu(self.conv6(l))
            l = self.norm6(l)
            l = F.leaky_relu(self.conv7(l))
            l = self.norm7(l)
            l = F.leaky_relu(self.conv8(l))
            l = self.norm8(l)
            l = F.leaky_relu(self.conv9(l))
            l = self.norm9(l)
            l = F.leaky_relu(self.conv10(l))
            l = self.norm10(l)
            l = F.leaky_relu(self.conv11(l))
            l = self.norm11(l)
            l = F.leaky_relu(self.conv12(l))
            l = self.norm12(l)
            l = F.leaky_relu(self.conv13(l))
            l = self.norm13(l)
            l = F.leaky_relu(self.conv14(l))
            l = self.norm14(l)
            l = F.leaky_relu(self.conv15(l))
            l = self.norm15(l)
            l = F.leaky_relu(self.conv16(l))
            l = self.norm16(l)

            l = F.interpolate(l, scale_factor=2, mode='bilinear')

            l = l.view(l.shape[0], 1, 200*200)
            l = F.softmax(l, dim=2)
            pm = l.view(l.shape[0], 1, 200, 200)

            # pm = F.pad(l, (10,10,10,10))

            lms, gm = get_transform_mat(pm)

            # comb_masks = comb_masks + gm
            comb_masks = torch.max(comb_masks, gm)


            if i==0:
                coors = lms
                masks = gm
            else:
                coors = torch.cat([coors, lms], dim=1)
                masks = torch.cat([masks, gm], dim=1)

            # print(coors.shape)

        return comb_masks, masks, coors

class CLMS_L_RR(nn.Module):
    def __init__(self, block = BasicBlockL, n_lm = 2):
        super(CLMS_L_RR, self).__init__()
        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2, padding = 3)
        self.norm1 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv2 = nn.Conv2d(64, 96, 7, stride = 1, padding = 3)
        self.norm2 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv3 = nn.Conv2d(96, 128, 7, stride = 1, padding = 3)
        self.norm3 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv4 = nn.Conv2d(128, 128, 7, stride = 1, padding = 3)
        self.norm4 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv5 = nn.Conv2d(128, 128, 7, stride = 1, padding = 3)
        self.norm5 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv6 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm6 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv7 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm7 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv8 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm8 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv9 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm9 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv10 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.norm10 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv11 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.norm11 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv12 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.norm12 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv13 = nn.Conv2d(128, 96, 3, stride = 1, padding = 1)
        self.norm13 = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv14_A = nn.Conv2d(96, 64, 3, stride = 1, padding = 1)
        self.norm14_A = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv15_A = nn.Conv2d(64, 32, 3, stride = 1, padding = 1)
        self.norm15_A = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv16_A = nn.Conv2d(32, 1, 3, stride = 1, padding = 1)
        self.norm16_A = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv14_B = nn.Conv2d(96, 64, 3, stride = 1, padding = 1)
        self.norm14_B = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv15_B = nn.Conv2d(64, 32, 3, stride = 1, padding = 1)
        self.norm15_B = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.conv16_B = nn.Conv2d(32, 1, 3, stride = 1, padding = 1)
        self.norm16_B = nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)

        self.apply(weights_init)

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

            l = F.leaky_relu(self.conv1(stacked_inputs))
            l = self.norm1(l)
            l = F.leaky_relu(self.conv2(l))
            l = self.norm2(l)
            l = F.leaky_relu(self.conv3(l))
            l = self.norm3(l)
            l = F.leaky_relu(self.conv4(l))
            l = self.norm4(l)
            l = F.leaky_relu(self.conv5(l))
            l = self.norm5(l)
            l = F.leaky_relu(self.conv6(l))
            l = self.norm6(l)
            l = F.leaky_relu(self.conv7(l))
            l = self.norm7(l)
            l = F.leaky_relu(self.conv8(l))
            l = self.norm8(l)
            l = F.leaky_relu(self.conv9(l))
            l = self.norm9(l)
            l = F.leaky_relu(self.conv10(l))
            l = self.norm10(l)
            l = F.leaky_relu(self.conv11(l))
            l = self.norm11(l)
            l = F.leaky_relu(self.conv12(l))
            l = self.norm12(l)
            l = F.leaky_relu(self.conv13(l))
            l = self.norm13(l)
            if i==0:
                l = F.leaky_relu(self.conv14_A(l))
                l = self.norm14_A(l)
                l = F.leaky_relu(self.conv15_A(l))
                l = self.norm15_A(l)
                l = F.leaky_relu(self.conv16_A(l))
                l = self.norm16_A(l)
            else:
                l = F.leaky_relu(self.conv14_B(l))
                l = self.norm14_B(l)
                l = F.leaky_relu(self.conv15_B(l))
                l = self.norm15_B(l)
                l = F.leaky_relu(self.conv16_B(l))
                l = self.norm16_B(l)

            l = F.interpolate(l, scale_factor=2, mode='bilinear')

            l = l.view(l.shape[0], 1, 200*200)
            l = F.softmax(l, dim=2)
            pm = l.view(l.shape[0], 1, 200, 200)

            # pm = F.pad(l, (10,10,10,10))

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

class ResnetCLMS(nn.Module):
    def __init__(self, block = ResnetBlock, num_blocks = [2,2,2,2], n_lm = 2):
        super(ResnetCLMS, self).__init__()
        self.in_planes = 64
        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1,)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # a = block(self.in_planes, planes, stride)
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        gm = torch.zeros((x.shape[0],1,IMAGE_SIZE,IMAGE_SIZE)).cuda()
        comb_masks = gm
        masks = []

        for i in range(self.n_lm):
            stacked_inputs = torch.cat([x, x*comb_masks], dim=1)

            out = F.leaky_relu(self.bn1(self.conv1(stacked_inputs)))
            out = self.layer1(out)
            out = F.leaky_relu(out)
            out = self.layer2(out)
            out = F.leaky_relu(out)
            out = F.leaky_relu(self.bn2(self.conv2(out)))
            out = F.interpolate(out, scale_factor=2, mode='bilinear')
            # print(out.shape)
            out = out.view(-1, 1, IMAGE_SIZE*IMAGE_SIZE)
            out = F.softmax(out, dim=2)
            pm = out.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

            lms, gm = get_transform_mat(pm)

            comb_masks = comb_masks + gm

            if i==0:
                coors = lms
                masks = pm
            else:
                coors = torch.cat([coors, lms], dim=1)
                masks = torch.cat([masks, pm], dim=1)


        return comb_masks, masks, coors

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1e-2)


# net = CLMS()
# net.forward(torch.zeros((2,3,200,200)))
