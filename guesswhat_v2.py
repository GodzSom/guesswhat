import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from localpath import getLocalPath
from classifier import Classifier, FaceDataset, LmSelector2
from skimage import io, transform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import random
import argparse
from scipy.stats import multivariate_normal
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_di_n', type=str, default='1', help='model_directory')
args = parser.parse_args()

IMAGE_SIZE = 200
batch_size = 500
num_epochs = 1000
N_KEYPOINT = 1
lr = 0.0001  # initial learning rate
SCALE = 0.5
checkpoint_frequency = 10

lp = lambda path, clone=True: getLocalPath("/home2/godsom/local_storage", path, clone)
# lp = lambda path, clone=True: path

file_dir =  lp('/home/godsom/Dataset/UTKFace/UTKFace')
PATH_TO_MODELS = lp("/home/godsom/guesswhat/models/v2",clone=False)
PATH_TO_OUTPUTS = lp("/home/godsom/guesswhat/outputs/v2",clone=False)
transforms_list = [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]

if not os.path.exists(PATH_TO_OUTPUTS+'/'+args.model_di_n):
    os.makedirs(PATH_TO_OUTPUTS+'/'+args.model_di_n)


fl = os.listdir(file_dir)
random.seed(555)
random.shuffle(fl)

def meshgrid(h):
    xv, yv = torch.meshgrid([torch.arange(0.5, h, 1) / (h / 2) - 1, torch.arange(0.5, h, 1) / (h / 2) - 1])
    return xv.to(device), yv.to(device)

def get_gaussian_maps(mu, shape_hw=[IMAGE_SIZE,IMAGE_SIZE], inv_std=2):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = torch.linspace(-1.0, 1.0, shape_hw[0]).to(device)
    x = torch.linspace(-1.0, 1.0, shape_hw[1]).to(device)

    y = torch.reshape(y, (1, 1, shape_hw[0]))
    x = torch.reshape(x, (1, 1, shape_hw[1]))

    g_y = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_y - y) * inv_std)))
    g_x = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_x - x) * inv_std)))

    g_y = g_y[:,:,:,None]#tf.expand_dims(g_y, axis=3)
    g_x = g_x[:,:,None,:]#tf.expand_dims(g_x, axis=2)
    g_yx = torch.matmul(g_y, g_x)  # [B, NMAPS, H, W]

    # g_yx = torch.transpose(g_yx, perm=[0, 2, 3, 1])
    g_yx = torch.transpose(g_yx, 1 ,2)
    g_yx = torch.transpose(g_yx, 2, 3)
    return g_yx * 1.15

def get_transform_mat(probmap):
    ranx, rany = meshgrid(IMAGE_SIZE)
    x = torch.sum(probmap * ranx, dim=(2,3))
    y = torch.sum(probmap * rany, dim=(2,3))
    # s = torch.ones(x.shape) * SCALE
    # zeros = torch.zeros(x.shape)
    # print('***')
    # print(probmap.shape)
    # print(x.shape)

    coors = torch.stack([x, y], dim=2)

    masks = get_gaussian_maps(torch.reshape(coors, (-1, N_KEYPOINT, 2)))
    masks = torch.transpose(masks, 2, 3)
    masks = torch.transpose(masks, 1, 2)
    # plt.imshow(masks[0,:,:,0].cpu().detach().numpy(), cmap='gray')
    # plt.show()
    # print(torch.max(masks[0,:,:,0]))


    # transform_mat = torch.stack([s, zeros, x, zeros, s, y], dim=1)
    # mask = torch.zeros(batch_size,1,IMAGE_SIZE,IMAGE_SIZE)
    # mask[:,:,0,0] = 1
    # print(torch.stack([x, y], dim=2))
    # print(torch.stack([x, y], dim=2).shape)


    # print(ranx)
    # print((probmap * ranx).shape)
    # assert False

    return coors, masks

def get_dataloader(minibatch_size=batch_size, mode='train'):
    root_dir = file_dir
    if mode == 'train':
        transformed_dataset = FaceDataset(root_dir, fl[:20000], transforms.Compose(transforms_list))
        # transformed_dataset = FaceDataset(root_dir, fl[:1000], transforms.Compose(transforms_list))
    if mode == 'val':
        # transformed_dataset = FaceDataset(root_dir, fl[20000:21500], transforms.Compose(transforms_list))
        transformed_dataset = FaceDataset(root_dir, fl[20000:21000], transforms.Compose(transforms_list))
    if mode == 'test':
        transformed_dataset = FaceDataset(root_dir, fl[21500:], transforms.Compose(transforms_list))
    dataloader = DataLoader(transformed_dataset, batch_size=minibatch_size, shuffle=True, num_workers=4)
    return dataloader

criterion = nn.NLLLoss()

def train(cl_net, lm_net, train_dataloader, epochs, filename, checkpoint_frequency=checkpoint_frequency, val_dataloader=None):
    cl_optimizer = optim.Adam(cl_net.parameters(), lr)
    cl_scheduler = optim.lr_scheduler.MultiStepLR(cl_optimizer, milestones=[10000])
    lm_optimizer = optim.Adam(lm_net.parameters(), lr)
    lm_scheduler = optim.lr_scheduler.MultiStepLR(lm_optimizer, milestones=[10000])

    if os.path.exists(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt'):
    #     os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
    # if os.path.exists(lp("/home/godsom/guesswhat/logs/v2"+'/'+args.model_di_n,clone=False)):
        checkpoint = torch.load(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt')
        cl_net.load_state_dict(checkpoint['cl_state_dict'])
        lm_net.load_state_dict(checkpoint['lm_state_dict'])
        cl_optimizer.load_state_dict(checkpoint['cl_optimizer_state_dict'])
        lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        os.makedirs(lp("/home/godsom/guesswhat/logs/v2"+'/'+args.model_di_n,clone=False))
        os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
        start_epoch = 0

    writer = SummaryWriter(lp("/home/godsom/guesswhat/logs/v2/"+args.model_di_n,clone=False))

    cl_net.train()
    lm_net.train()

    training_loss, validation_loss, val_accuracy = [], [], []
    checkpoint = 0
    iteration = 0
    running_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            # print(i)
            cl_scheduler.step()
            cl_optimizer.zero_grad()
            lm_scheduler.step()
            lm_optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)
            lms = lm_net(images)

            # lms = get_transform_mat(probmap)[0]
            lms = (lms.cpu().detach().numpy() + 1)*IMAGE_SIZE/2
            pos = np.dstack(np.mgrid[0:IMAGE_SIZE:1, 0:IMAGE_SIZE:1])
            masks = np.zeros((images.shape[0], IMAGE_SIZE,IMAGE_SIZE))
            for a in range(images.shape[0]):
                var = multivariate_normal(mean=lms[a], cov=300)
                map = var.pdf(pos)
                map = map/np.max(map)
                masks[a] = map

            masks = torch.from_numpy(masks).float().to(device)[:,None]

            masked_images = images * masks.repeat(1,3,1,1)

            outputs = cl_net(masked_images)

            loss = criterion(outputs, labels)
            running_loss += float(loss.item())
            loss.backward()
            cl_optimizer.step()
            lm_optimizer.step()

            outputs = torch.argmax(outputs, 1)
            accu = sum(outputs == labels).item()/outputs.shape[0]

        if (epoch+1) % checkpoint_frequency == 0 and val_dataloader is not None:
            training_loss.append(running_loss/i)
            val_results = validate(cl_net, lm_net, val_dataloader, epoch)
            validation_loss.append(val_results[0])
            val_accuracy.append(val_results[1])
            print(f'minibatch:{i}, epoch:{epoch+start_epoch}, training_error:{training_loss[-1]}, val_error:{validation_loss[-1]}, val_accu:{val_accuracy[-1]}')

            torch.save({'epoch': epoch+start_epoch,
                        'cl_state_dict':cl_net.state_dict(),
                        'cl_optimizer_state_dict': cl_optimizer.state_dict(),
                        'lm_state_dict':lm_net.state_dict(),
                        'lm_optimizer_state_dict': lm_optimizer.state_dict()},
                        f'{PATH_TO_MODELS}/{args.model_di_n}/{filename}.pt')

            writer.add_scalar('training loss', training_loss[-1], epoch+start_epoch)
            writer.add_scalar('validation loss', validation_loss[-1], epoch+start_epoch)
            writer.add_scalar('val accuracy', val_accuracy[-1], epoch+start_epoch)
            writer.add_scalar('train accuracy',accu, epoch+start_epoch)

        running_loss = 0

    return training_loss, validation_loss

def validate(cl_net, lm_net, dataloader,epoch):
    # net.train()
    total_loss, total, exact_match = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            lms = lm_net(images)

            # lms = get_transform_mat(probmap)[0]
            lms = (lms.cpu().detach().numpy() + 1)*IMAGE_SIZE/2
            pos = np.dstack(np.mgrid[0:IMAGE_SIZE:1, 0:IMAGE_SIZE:1])
            masks = np.zeros((images.shape[0], IMAGE_SIZE,IMAGE_SIZE))
            for a in range(images.shape[0]):
                var = multivariate_normal(mean=lms[a], cov=300)
                map = var.pdf(pos)
                map = map/np.max(map)
                masks[i] = map

            masks = torch.from_numpy(masks).float().to(device)[:,None]

            masked_images = images * masks.repeat(1,3,1,1)

            outputs = cl_net(masked_images)

            loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            # print(outputs)
            # op = map(lambda x: torch.max(x, 0)[1], outputs)
            # outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).to(device)
            outputs = torch.argmax(outputs, 1)
            total += outputs.shape[0]
            exact_match += sum(outputs == labels).item()
            if i==0:
                save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/maskedimg_{:04d}.png'.format(epoch))

            # print(torch.stack([outputs, labels], dim=1))
            # print('-------')
            # print(lms)
    return total_loss/i, exact_match/total

# def get_validation_error(c, fold, train_transform_index):
#     filename = get_model_filename(c, fold, train_transform_index)
#     net = Classifier().to(device)
#     net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{filename}'))
#     return validate(net, get_dataloader(batch_size))

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
            images, labels = batch['image'].to(device), batch['label'].to(device)
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap)

            masked_images = images * masks.repeat(1,3,1,1)

            outputs = cl_net(masked_images)

            loss = criterion(outputs, labels)
            outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).to(device)
            result['total'] += len(outputs)
            result['exact_match'] += sum(outputs == labels).item()
            if c == 'age':
                result['one_off_match'] += (sum(outputs==labels) +
                                            sum(outputs==labels-1) +
                                            sum(outputs==labels+1)).item()

    return result



def train_save(checkpoint_frequency=checkpoint_frequency):
    training_loss, validation_loss = train(
        Classifier().to(device),
        LmSelector2(N_KEYPOINT).to(device),
        get_dataloader(batch_size, 'train'),
        num_epochs,
        args.model_di_n,
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

train_save()
