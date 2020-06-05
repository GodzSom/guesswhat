import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from localpath import getLocalPath
from classifier import Classifier, FaceDataset, LmSelector
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
parser.add_argument('-mode', type=str, default='train', help='mode')
args = parser.parse_args()

IMAGE_SIZE = 200
batch_size = 50
num_epochs = 1000
N_KEYPOINT = 1
lr = 0.0001  # initial learning rate
SCALE = 0.5
checkpoint_frequency = 10
inv_std = 20

lp = lambda path, clone=True: getLocalPath("/home2/godsom/local_storage", path, clone)
# lp = lambda path, clone=True: path

file_dir =  lp('/home/godsom/Dataset/UTKFace/UTKFace')
PATH_TO_MODELS = lp("/home/godsom/guesswhat/models/v1",clone=False)
PATH_TO_OUTPUTS = lp("/home/godsom/guesswhat/outputs/v1",clone=False)
transforms_list = [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]


if not os.path.exists(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train'):
    os.makedirs(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train')


fl = os.listdir(file_dir)
random.seed(555)
random.shuffle(fl)

def meshgrid(h):
    xv, yv = torch.meshgrid([torch.arange(0.5, h, 1) / (h / 2) - 1, torch.arange(0.5, h, 1) / (h / 2) - 1])
    return xv.to(device), yv.to(device)

ranx, rany = meshgrid(IMAGE_SIZE)

def variance_loss(probmap, uv):
  """Computes the variance loss as part of Sillhouette consistency.
  Args:
    probmap: [batch, num_kp, h, w] The distribution map of keypoint locations.
    ranx: X-axis meshgrid.
    rany: Y-axis meshgrid.
    uv: [batch, num_kp, 2] Keypoint locations (in NDC).
  Returns:
    The variance loss.
  """

  ran = torch.stack([ranx, rany], axis=2)

  sh = ran.shape
  # [batch, num_kp, vh, vw, 2]
  ran = torch.reshape(ran, (1, 1, sh[0], sh[1], 2))

  sh = uv.shape
  uv = torch.reshape(uv, (sh[0], sh[1], 1, 1, 2))

  diff = torch.sum((uv - ran)**2, axis=4)
  diff *= probmap

  return torch.mean(torch.sum(diff, axis=[2, 3]))

def get_gaussian_maps(mu, inv_std):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    # print(inv_std)

    y = torch.linspace(-1.0, 1.0, IMAGE_SIZE).to(device)
    x = torch.linspace(-1.0, 1.0, IMAGE_SIZE).to(device)

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

def get_transform_mat(probmap, inv_std=inv_std):
    # ranx, rany = meshgrid(IMAGE_SIZE)
    x = torch.sum(probmap * ranx, dim=(2,3))
    y = torch.sum(probmap * rany, dim=(2,3))
    # s = torch.ones(x.shape) * SCALE
    # zeros = torch.zeros(x.shape)
    # print('***')
    # print(probmap.shape)
    # print(x.shape)

    coors = torch.stack([x, y], dim=2)

    masks = get_gaussian_maps(torch.reshape(coors, (-1, N_KEYPOINT, 2)), inv_std)
    masks = torch.transpose(masks, 2, 3)
    masks = torch.transpose(masks, 1, 2)

    return coors, masks

def get_dataloader(minibatch_size=batch_size, mode='train'):
    root_dir = file_dir
    if mode == 'train':
        transformed_dataset = FaceDataset(root_dir, fl[:20000], transforms.Compose(transforms_list))
        # transformed_dataset = FaceDataset(root_dir, fl[:100], transforms.Compose(transforms_list))
    if mode == 'val':
        transformed_dataset = FaceDataset(root_dir, fl[20000:21500], transforms.Compose(transforms_list))
        # transformed_dataset = FaceDataset(root_dir, fl[20000:20100], transforms.Compose(transforms_list))
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

    inv_std_loop = inv_std
    anneal_rate = 1.0

    if os.path.exists(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt'):
    #     os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
    # if os.path.exists(lp("/home/godsom/guesswhat/logs/v1"+'/'+args.model_di_n,clone=False)):
        checkpoint = torch.load(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt')
        cl_net.load_state_dict(checkpoint['cl_state_dict'])
        lm_net.load_state_dict(checkpoint['lm_state_dict'])
        cl_optimizer.load_state_dict(checkpoint['cl_optimizer_state_dict'])
        lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        if not os.path.exists("/home/godsom/guesswhat/logs/v1"+'/'+args.model_di_n):
            os.makedirs(lp("/home/godsom/guesswhat/logs/v1"+'/'+args.model_di_n,clone=False))
            os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
        start_epoch = 0

    writer = SummaryWriter(lp("/home/godsom/guesswhat/logs/v1/"+args.model_di_n,clone=False))


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
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap, inv_std=inv_std_loop)

            masked_images = images * masks.repeat(1,3,1,1)

            outputs = cl_net(masked_images)

            loss = criterion(outputs, labels) + variance_loss(probmap, lms)
            running_loss += float(loss.item())
            loss.backward()
            cl_optimizer.step()
            lm_optimizer.step()

            outputs = torch.argmax(outputs, 1)
            accu = sum(outputs == labels).item()/outputs.shape[0]

        if (epoch+1) % checkpoint_frequency == 0 and val_dataloader is not None:
            training_loss.append(running_loss/i)
            val_results = validate(cl_net, lm_net, val_dataloader, epoch+start_epoch, inv_std_loop)
            validation_loss.append(val_results[0])
            val_accuracy.append(val_results[1])
            print(f'epoch:{epoch+start_epoch}, training_error:{training_loss[-1]}, val_error:{validation_loss[-1]}, val_accu:{val_accuracy[-1]}')

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

            save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/masked_{:04d}.png'.format(epoch+start_epoch))
            save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/hm_{:04d}.png'.format(epoch+start_epoch))
            if inv_std_loop < 20:
                inv_std_loop = inv_std_loop * anneal_rate

        running_loss = 0

    return training_loss, validation_loss

def validate(cl_net, lm_net, dataloader,epoch,inv_std_loop):
    # net.train()
    total_loss, total, exact_match = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap, inv_std=inv_std_loop)

            masked_images = images * masks.repeat(1,3,1,1)

            outputs = cl_net(masked_images)

            loss = criterion(outputs, labels)  + variance_loss(probmap, lms)
            total_loss += float(loss.item())
            # print(outputs)
            # op = map(lambda x: torch.max(x, 0)[1], outputs)
            # outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).to(device)
            outputs = torch.argmax(outputs, 1)
            total += outputs.shape[0]
            exact_match += sum(outputs == labels).item()
            if i==0:
                save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/maskedimg_{:04d}.png'.format(epoch))
                save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/hm_{:04d}.png'.format(epoch))

            # print(torch.stack([outputs, labels], dim=1))
            # print('-------')
            # print(torch.max(masks))
    return total_loss/i, exact_match/total

# def get_validation_error(c, fold, train_transform_index):
#     filename = get_model_filename(c, fold, train_transform_index)
#     net = Classifier().to(device)
#     net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{filename}'))
#     return validate(net, get_dataloader(batch_size))

def test():
    if not os.path.exists(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test'):
        os.makedirs(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test')

    cl_net = Classifier().to(device)
    lm_net = LmSelector(N_KEYPOINT).to(device)

    checkpoint = torch.load(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt')
    cl_net.load_state_dict(checkpoint['cl_state_dict'])
    lm_net.load_state_dict(checkpoint['lm_state_dict'])

    result = {
        'exact_match': 0,
        'total': 0
    }

    total_loss, total, exact_match = 0, 0, 0

    with torch.no_grad():
        cl_net.eval()
        lm_net.eval()
        for i, batch in enumerate(get_dataloader(batch_size, 'test')):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap)

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
            if i<50:
                save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/maskedimg_{:04d}.png'.format(i))
                save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/hm_{:04d}.png'.format(i))
            else:
                break

    return result



def train_save(checkpoint_frequency=checkpoint_frequency):
    training_loss, validation_loss = train(
        Classifier().to(device),
        LmSelector(N_KEYPOINT).to(device),
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

if args.mode == 'train':
    train_save()
else:
    results = test()
