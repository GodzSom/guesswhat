'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from localpath import getLocalPath
from skimage import io, transform
import torch.optim as optim
import torchvision.transforms as transforms
from random import shuffle
import random

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 200
batch_size = 30
num_epochs = 1000
lr = 0.0001  # initial learning rate

lp = lambda path, clone=True: getLocalPath("/home2/godsom/local_storage", path, clone)
# lp = lambda path, clone=True: path

file_dir =  lp('/home/godsom/Dataset/UTKFace/UTKFace')
PATH_TO_MODELS = lp("/home/godsom/guesswhat/models/resnetC")
transforms_list = [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]

fl = os.listdir(file_dir)
random.seed(555)
random.shuffle(fl)

def get_gaussian_maps(mu, shape_hw=[IMAGE_SIZE,IMAGE_SIZE], inv_std=20):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = torch.linspace(-1.0, 1.0, shape_hw[0]).cuda()
    x = torch.linspace(-1.0, 1.0, shape_hw[1]).cuda()

    y = torch.reshape(y, (1, 1, shape_hw[0]))
    x = torch.reshape(x, (1, 1, shape_hw[1]))

    g_y = torch.exp(-(mu_y-y)**2 * inv_std)
    g_x = torch.exp(-(mu_x-x)**2 * inv_std)

    g_y = g_y[:,:,:,None]#tf.expand_dims(g_y, axis=3)
    g_x = g_x[:,:,None,:]#tf.expand_dims(g_x, axis=2)
    g_yx = torch.matmul(g_y, g_x)  # [B, NMAPS, H, W]

    # g_yx = torch.transpose(g_yx, perm=[0, 2, 3, 1])
    g_yx = torch.transpose(g_yx, 1 ,2)
    g_yx = torch.transpose(g_yx, 2, 3)
    return g_yx #* 1.15

coors = np.zeros((17*17,2))
x, y = 20, 20
for i in range(17*17):
    coors[i,0] = x
    coors[i,1] = y
    x = x + 10
    if x == 180:
        x = 0
        y = y + 10

coors = torch.from_numpy(coors[:,None]).type(torch.FloatTensor).cuda()
coors = (coors*2/IMAGE_SIZE)-1

masks = get_gaussian_maps(torch.reshape(coors, (-1, 1, 2)))
masks = torch.transpose(masks, 2, 3)
masks = torch.transpose(masks, 1, 2)

class FaceDataset(Dataset):
    def __init__(self, root_dir, fl, transform=None, task = 1):
        self.root_dir = root_dir
        self.file_list = fl
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.file_list[idx])
        image = Image.open(img_name)
        label = int(self.file_list[idx].split("_")[self.task])# age gender race

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        # print(sample['image'].shape)
        return sample

def get_dataloader(minibatch_size=batch_size, mode='train'):
    root_dir = file_dir
    task_num = 1
    if mode == 'train':
        transformed_dataset = FaceDataset(root_dir, fl[:20000], transforms.Compose(transforms_list),task_num)
        # transformed_dataset = FaceDataset(root_dir, fl[:200], transforms.Compose(transforms_list),task_num)
        # print(transformed_dataset)
        # assert False
    if mode == 'val':
        transformed_dataset = FaceDataset(root_dir, fl[20000:21500], transforms.Compose(transforms_list),task_num)
        # transformed_dataset = FaceDataset(root_dir, fl[20000:20100], transforms.Compose(transforms_list),task_num)
    if mode == 'test':
        transformed_dataset = FaceDataset(root_dir, fl[21500:], transforms.Compose(transforms_list),task_num)
    dataloader = DataLoader(transformed_dataset, batch_size=minibatch_size, shuffle=True, num_workers=4)
    return dataloader

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        out += x
        out = F.relu(out)
        return out
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetC(nn.Module):
    def __init__(self, block = BasicBlock, num_blocks = [2,2,2,2], num_classes=2):
        super(ResNetC, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(64*IMAGE_SIZE*IMAGE_SIZE, 64)
        self.linear2 = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(-1,64*IMAGE_SIZE*IMAGE_SIZE)
        # print(out.shape)
        out = self.linear1(out)
        out = F.leaky_relu(out)
        out = self.linear2(out)
        # out = F.leaky_relu(out)
        out = F.log_softmax(out, dim=1)

        return out


class ResNetLMS(nn.Module):
    def __init__(self, block = BasicBlock, num_blocks = [2,2,2,2], n_lm = 2):
        super(ResNetLMS, self).__init__()
        self.in_planes = 64
        self.n_lm = n_lm

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.conv2 = nn.Conv2d(64, self.n_lm, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.n_lm,)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # a = block(self.in_planes, planes, stride)
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.relu(self.bn2(self.conv2(out)))

        # print(out.shape)

        out = F.interpolate(out, scale_factor=8, mode='bilinear')
        # print(out.shape)
        out = out.view(-1, self.n_lm, IMAGE_SIZE*IMAGE_SIZE)
        out = F.softmax(out, dim=2)
        out = out.view(-1, self.n_lm, IMAGE_SIZE, IMAGE_SIZE)


        return out

criterion = nn.NLLLoss()
criterion_mse = nn.MSELoss()

# def ResNet18():
#     return ResNet(BasicBlock, [2,2,2,2])
#
# def ResNet34():
#     return ResNet(BasicBlock, [3,4,6,3])
#
# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])
#
# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])
#
# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])
#
#
#
# def test():
#     net = ResNet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())

# net = ResNetC()
# net.forward(torch.zeros((1,3,200,200)))

def train(net, train_dataloader, epochs, filename, checkpoint_frequency=10, val_dataloader=None):
    """
    Args:
        net: An instance of PyTorch's Net class.
        train_dataloader: An instance of PyTorch's Dataloader class.
        epochs: An integer.
        filename: A string. Name of the model saved to drive.
        checkpoint_frequency: An integer. Represents how frequent (in terms
            of number of iterations) the model should be saved to drive.
        val_dataloader: An instance of PyTorch's Dataloader class.

    Returns:
        net: An instance of PyTorch's Net class. The trained network.
        training_loss: A list of numbers that represents the training loss at each checkpoint.
        validation_loss: A list of numbers that represents the validation loss at each checkpoint.
    """
    net.train()
    optimizer = optim.Adam(net.parameters(), lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])

    training_loss, validation_loss, val_accuracy = [], [], []
    checkpoint = 0
    iteration = 0
    running_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            scheduler.step()
            optimizer.zero_grad()
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            # print(masks.shape)
            mm = images*masks[:images.shape[0]]
            outputs = net(mm)
            # print(outputs)
            # print(outputs.shape)

            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels)
            # print(loss)
            running_loss += float(loss.item())
            loss.backward()
            optimizer.step()
            # images = torch.transpose(images, 1 ,2)
            # images = torch.transpose(images, 2, 3)
            # images = images.detach().cpu().numpy()
            # plt.imshow(images[0])
            # plt.show()
            # assert False

        if (epoch+1) % checkpoint_frequency == 0 and val_dataloader is not None:
            training_loss.append(running_loss/checkpoint_frequency)
            val_results = validate(net, val_dataloader)
            validation_loss.append(val_results[0])
            val_accuracy.append(val_results[1])
            print(f'minibatch:{i}, epoch:{epoch}, training_error:{training_loss[-1]}, val_error:{validation_loss[-1]}, val_accu:{val_accuracy[-1]}')

            torch.save({'epoch': epoch,
                        'model_state_dict':net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        f'{PATH_TO_MODELS}/{filename}.pt')

            checkpoint += 1
            running_loss = 0

    return net, training_loss, validation_loss

def validate(net, dataloader):
    net.train()
    total_loss, total, exact_match = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            outputs = net(images*masks[:images.shape[0]])
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).cuda()
            total += len(outputs)
            exact_match += sum(outputs == labels).item()
            # a = (outputs-labels)**2
            # print(outputs.shape)
            # print(labels.shape)


    return total_loss/(i+1), exact_match/total

def test(net, dataloader, c):
    result = {
        'exact_match': 0,
        'total': 0
    }
    if c == 'age':
        result['one_off_match'] = 0

    with torch.no_grad():
        net.eval()
        for i, batch in enumerate(dataloader):
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            outputs = net(images)
            outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).cuda()
            result['total'] += len(outputs)
            result['exact_match'] += sum(outputs == labels).item()
            if c == 'age':
                result['one_off_match'] += (sum(outputs==labels) +
                                            sum(outputs==labels-1) +
                                            sum(outputs==labels+1)).item()

    return result



def train_save(checkpoint_frequency=50):
    """
    Args:
        c: A string. Equals either "age" or "gender".
        fold: An integer. Lies in the range [0, 4] as there are five folds present.
        train_transform_index: An integer. The transforms in the list correesponding
            to this index in the dictionary will be applied on the images.
        checkpoint_frequency: An integer. Represents how frequent (in terms
            of number of iterations) the model should be saved to drive.
    Returns:
        validation_loss: A list of numbers that represents the validation loss at each checkpoint.
    """
    trained_net, training_loss, validation_loss = train(
        ResNetC().cuda(),
        get_dataloader(batch_size, 'train'),
        num_epochs,
        f'gm20_ss',
        checkpoint_frequency,
        get_dataloader(batch_size, 'val')
    )

    plt.plot(list(map(lambda x: checkpoint_frequency * x, (list(range(1, len(validation_loss)+1))))), validation_loss, label='validation_loss')
    plt.plot(list(map(lambda x: checkpoint_frequency * x, (list(range(1, len(training_loss)+1))))), training_loss, label='training_loss')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

    return validation_loss



if __name__ == "__main__":
    train_save()
