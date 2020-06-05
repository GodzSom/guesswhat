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


class CLMS(nn.Module):
    def __init__(self, block = BasicBlockL, n_lm = 2):
        super(CLMS, self).__init__()

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
