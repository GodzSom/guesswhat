#dot
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
from math import pi
from PIL import Image
from shutil import copyfile
from localpath import getLocalPath
from classifier import FaceDataset, Classifier#, Classifier, ResNetLMS_L_v8
from resnet import ResNetLMS_L_v7
# from clms import ResNetLMS_L_v8
from skimage import io, transform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import random
import argparse
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


# CUDA_VISIBLE_DEVICES = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_di_n', type=str, default='1', help='model_directory')
parser.add_argument('-mode', type=str, default='train', help='mode')
args = parser.parse_args()

mk = ['+r','+m','+b','+g','+c','|r','|m','|b','|g','|c']
IMAGE_SIZE = 200
batch_size = 30
num_epochs = 1000
N_KEYPOINT = 1
lr = 0.0001  # initial learning rate
SCALE = 0.5
checkpoint_frequency = 10
inv_std = 20
MAX_D_PENALTY = 25.0

lp = lambda path, clone=True: getLocalPath("/home2/godsom/local_storage", path, clone)
# lp = lambda path, clone=True: path

file_dir =  lp('/home/godsom/Dataset/UTKFace/UTKFace')
PATH_TO_MODELS = lp("/home/godsom/guesswhat/models/v8",clone=False)
PATH_TO_OUTPUTS = lp("/home/godsom/guesswhat/outputs/v8",clone=False)
transforms_list = [transforms.Resize((IMAGE_SIZE)), transforms.ToTensor()]


if not os.path.exists(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train'):
    os.makedirs(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train')


fl = os.listdir(file_dir)
fl.sort()
random.seed(555)
random.shuffle(fl)
# print(fl)

def meshgrid(h):
    xv, yv = torch.meshgrid([torch.arange(0.5, h, 1) / (h / 2) - 1, torch.arange(0.5, h, 1) / (h / 2) - 1])
    return xv.cuda(), yv.cuda()

ranx, rany = meshgrid(IMAGE_SIZE)

def dis_pel_loss(dis):
    return torch.mean(torch.max(MAX_D_PENALTY-dis,torch.zeros_like(dis)))/MAX_D_PENALTY

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

def create_blurred_circular_mask(mask_shape, radius, center=None, sigma=10):
    # assert (len(mask_shape) == 2)
    if center is None:
        x_center = int(mask_shape[1] / float(2))
        y_center = int(mask_shape[0] / float(2))
        center = (x_center, y_center)
    else:
        # print(center)
        x_center = int(0.5*IMAGE_SIZE*(center[0,1]+1))
        y_center = int(0.5*IMAGE_SIZE*(center[0,0]+1))
        center = (x_center, y_center)

    y, x = np.ogrid[-y_center:mask_shape[0] - y_center, -x_center:mask_shape[1] - x_center]
    mask = x * x + y * y <= radius * radius
    grid = np.zeros(mask_shape)
    grid[mask] = 1

    if sigma is not None:
        grid = scipy.ndimage.filters.gaussian_filter(grid, sigma)
    return torch.from_numpy(grid)

def get_gaussian_maps(mu, inv_std=inv_std):
    """
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    # mu_y[0] = 0.0
    # mu_x[0] = 0.0

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

    # zero_tensor = torch.zeros(g_yx.shape).cuda()
    # g_yx = torch.where(g_yx > 0.3, g_yx, zero_tensor)

    # print(g_yx[0, 100, 100:125])

    return g_yx #* 1.15

def get_transform_mat(probmap, inv_std=inv_std):
    ranx, rany = meshgrid(IMAGE_SIZE)
    x = torch.sum(probmap * ranx, dim=(2,3))
    y = torch.sum(probmap * rany, dim=(2,3))

    coors = torch.stack([x, y], dim=2)

    # masks = get_gaussian_maps(torch.reshape(coors, (-1, N_KEYPOINT, 2)), inv_std)
    masks = torch.zeros((probmap.shape[0],IMAGE_SIZE,IMAGE_SIZE,1))#.cuda()

    masks = np.zeros((probmap.shape[0],IMAGE_SIZE,IMAGE_SIZE,1))
    for i in range(masks.shape[0]):
        masks[i,:,:,0] = create_blurred_circular_mask([IMAGE_SIZE,IMAGE_SIZE], 30, coors[i], sigma=10)
        # masks[i,:,:,0] = create_blurred_circular_mask([IMAGE_SIZE,IMAGE_SIZE], 30, coors[i], sigma=2)

    # print(masks.shape)
    # plt.imshow(1-masks)
    # plt.show()
    # masks = get_gaussian_maps(torch.reshape(coors, (-1, N_KEYPOINT, 2)), inv_std)
    # print(masks[0].shape)
    # plt.imshow(1-masks[0,:,:,0].detach().cpu().numpy())
    # plt.show()
    # assert False
    masks = torch.transpose(torch.from_numpy(masks).float(), 2, 3)
    # masks = torch.transpose(masks, 2, 3)
    masks = torch.transpose(masks, 1, 2)

    return coors, (1-masks).cuda()


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
    dataloader = DataLoader(transformed_dataset, batch_size=minibatch_size, shuffle=False, num_workers=4)
    return dataloader

criterion = nn.NLLLoss(reduce=True)

def train(cl_net, lm_net, train_dataloader, epochs, filename, checkpoint_frequency=checkpoint_frequency, val_dataloader=None):
    cl_optimizer = optim.Adam(cl_net.parameters(), lr)
    cl_scheduler = optim.lr_scheduler.MultiStepLR(cl_optimizer, milestones=[10000])
    lm_optimizer = optim.Adam(lm_net.parameters(), lr)
    lm_scheduler = optim.lr_scheduler.MultiStepLR(lm_optimizer, milestones=[200,400,600,800,1000],gamma=0.1)

    inv_std_loop = inv_std
    anneal_rate = 1.0,

    if os.path.exists(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt'):
    #     os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
    # if os.path.exists(lp("/home/godsom/guesswhat/logs/v8"+'/'+args.model_di_n,clone=False)):
        checkpoint = torch.load(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt')
        cl_net.load_state_dict(checkpoint['cl_state_dict'])
        lm_net.load_state_dict(checkpoint['lm_state_dict'])
        cl_optimizer.load_state_dict(checkpoint['cl_optimizer_state_dict'])
        lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        if not os.path.exists(lp("/home/godsom/guesswhat/logs/v8"+'/'+args.model_di_n,clone=False)):
            os.makedirs(lp("/home/godsom/guesswhat/logs/v8"+'/'+args.model_di_n,clone=False))
            os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
        start_epoch = 0

    writer = SummaryWriter(lp("/home/godsom/guesswhat/logs/v8/"+args.model_di_n,clone=False))

    # checkpoint_cl = torch.load(lp("/home/godsom/guesswhat/models/classifier/gm20_n"+str(N_KEYPOINT)+".pt", clone=False))
    # cl_net.load_state_dict(checkpoint_cl['model_state_dict'])

    cl_net.train()
    lm_net.train()


    training_loss, validation_loss, val_accuracy = [], [], []
    checkpoint = 0
    iteration = 0
    running_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            # print(i)
            # print(lm_net.parameters)
            lm_optimizer.zero_grad()
            images, blurred_image, labels, fn = batch['image'].cuda(), batch['blurred_image'].cuda(), batch['label'].cuda(), batch['fn']
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap, inv_std=inv_std_loop)

            if N_KEYPOINT==1:
                masks = masks[:,0,:,:][:,None].repeat(1,3,1,1)
                masked_images = images*masks + blurred_image*(1-masks)

            outputs = cl_net(masked_images)

            loss = torch.sum(criterion(outputs, labels)) + variance_loss(probmap, lms)# + dis_pel_loss(dist)
            neg_loss = -loss
            running_loss += float(loss.item())
            neg_loss.backward(retain_graph=True)
            # neg_loss.backward()
            lm_optimizer.step()

            ##---------------------------------------
            cl_optimizer.zero_grad()
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap, inv_std=inv_std_loop)

            if N_KEYPOINT==1:
                masks = masks[:,0,:,:][:,None].repeat(1,3,1,1)
                masked_images = images*masks + blurred_image*(1-masks)

            outputs = cl_net(masked_images)
            loss = criterion(outputs, labels) + variance_loss(probmap, lms)
            running_loss += float(loss.item())
            loss.backward()
            cl_optimizer.step()

            outputs = torch.argmax(outputs, 1)
            # print(outputs)
            accu = sum(outputs == labels).item()/outputs.shape[0]

        if (epoch) % 10 == 0 and val_dataloader is not None:
            # if inv_std_loop < 20:
            #     inv_std_loop = inv_std_loop * anneal_rate
            training_loss.append(running_loss/i)
            val_results = validate(cl_net, lm_net, val_dataloader, epoch+start_epoch, inv_std_loop,start_epoch)
            validation_loss.append(val_results[0])
            val_accuracy.append(val_results[1])
            print(f'epoch:{epoch+start_epoch}, training_error:{training_loss[-1]}, training_accu:{accu}, val_error:{validation_loss[-1]}, val_accu:{val_accuracy[-1]}')

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

            if N_KEYPOINT ==2:
                probmap = probmap.detach().cpu().numpy()
                masked_images = torch.transpose(masked_images, 1 ,2)
                masked_images = torch.transpose(masked_images, 2, 3)
                masked_images = masked_images.detach().cpu().numpy()

                plt.figure(figsize=(12,12))
                plt.subplot(2, 2, 1)
                plt.imshow(probmap[0][0]/np.max(probmap[0]))
                plt.subplot(2, 2, 2)
                plt.imshow(probmap[0][1]/np.max(probmap[1]))
                plt.subplot(2, 2, 3)
                plt.imshow(masked_images[0])
                # plt.subplot(2, 2, 4)
                # plt.imshow(masked_images[1])
                plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/t_{:04d}.png'.format(epoch+start_epoch))
                plt.close()
            elif N_KEYPOINT == 4:
                probmap = probmap.detach().cpu().numpy()
                masked_images = torch.transpose(masked_images, 1 ,2)
                masked_images = torch.transpose(masked_images, 2, 3)
                masked_images = masked_images.detach().cpu().numpy()

                plt.figure(figsize=(12,12))
                plt.subplot(2, 2, 1)
                plt.imshow(probmap[0][0]/np.max(probmap[0]))
                plt.subplot(2, 2, 2)
                plt.imshow(probmap[0][1]/np.max(probmap[1]))
                plt.subplot(2, 2, 3)
                plt.imshow(probmap[0][2]/np.max(probmap[2]))
                plt.subplot(2, 2, 4)
                plt.imshow(masked_images[0])
                plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/t_{:04d}.png'.format(epoch+start_epoch))
                plt.close()
            else:
                save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/masked_{:04d}_A.png'.format(epoch+start_epoch))
                # save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/hm_{:04d}_A.png'.format(epoch+start_epoch))
                save_image(masked_images[1], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/masked_{:04d}_B.png'.format(epoch+start_epoch))
                # save_image(probmap[1]/torch.max(probmap[1]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/hm_{:04d}_B.png'.format(epoch+start_epoch))

                # diff_pm = probmap[0]-probmap[1]
                # save_image(diff_pm/torch.max(diff_pm), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/diff_hm_{:04d}.png'.format(epoch+start_epoch))
                # diff_img = masked_images[0]-masked_images[1]
                # save_image(diff_img, PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/diff_img_{:04d}.png'.format(epoch+start_epoch))
                # print(torch.max(diff_pm))
                # print(torch.max(diff_img))


        running_loss = 0

    return training_loss, validation_loss

def validate(cl_net, lm_net, dataloader,epoch,inv_std_loop,start_epoch):
    # net.train()
    total_loss, total, exact_match = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, blurred_image, labels, fn = batch['image'].cuda(), batch['blurred_image'].cuda(), batch['label'].cuda(), batch['fn']
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap, inv_std=inv_std_loop)

            if N_KEYPOINT==1:
                masks = masks[:,0,:,:][:,None].repeat(1,3,1,1)
                masked_images = images*masks + blurred_image*(1-masks)


            outputs = cl_net(masked_images)

            # print(labels)
            loss = torch.sum(criterion(outputs, labels))
            total_loss += float(loss.item())
            # print(outputs)
            # op = map(lambda x: torch.max(x, 0)[1], outputs)
            # outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).cuda()
            outputs = torch.argmax(outputs, 1)
            total += outputs.shape[0]
            exact_match += sum(outputs == labels).item()
            if i==0:
                if N_KEYPOINT ==2:
                    probmap = probmap.cpu()
                    masked_images = torch.transpose(masked_images, 1 ,2)
                    masked_images = torch.transpose(masked_images, 2, 3)
                    masked_images = masked_images.cpu()

                    plt.figure(figsize=(12,12))
                    plt.subplot(2, 2, 1)
                    plt.imshow(probmap[0][0]/torch.max(probmap[0][0]))
                    plt.subplot(2, 2, 2)
                    plt.imshow(probmap[0][1]/torch.max(probmap[1][1]))
                    plt.subplot(2, 2, 3)
                    plt.imshow(masked_images[0])
                    # plt.subplot(2, 2, 4)
                    # plt.imshow(masked_images[1])
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/val_{:04d}.png'.format(epoch+start_epoch))
                    plt.close()
                elif N_KEYPOINT ==4:
                    probmap = probmap.cpu()
                    masked_images = torch.transpose(masked_images, 1 ,2)
                    masked_images = torch.transpose(masked_images, 2, 3)
                    masked_images = masked_images.cpu()

                    plt.figure(figsize=(12,12))
                    plt.subplot(2, 2, 1)
                    plt.imshow(probmap[0][0]/torch.max(probmap[0][0]))
                    plt.subplot(2, 2, 2)
                    plt.imshow(probmap[0][1]/torch.max(probmap[0][1]))
                    plt.subplot(2, 2, 3)
                    plt.imshow(probmap[0][2]/torch.max(probmap[0][2]))
                    plt.subplot(2, 2, 4)
                    plt.imshow(masked_images[0])
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/val_{:04d}.png'.format(epoch+start_epoch))
                    plt.close()
                else:
                    save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/maskedimg_{:04d}.png'.format(epoch+start_epoch))
                    save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/hm_{:04d}.png'.format(epoch+start_epoch))

            # print(torch.stack([outputs, labels], dim=1))
            # print('-------')
            # print(torch.max(masks))
    return total_loss/i, exact_match/total

# def get_validation_error(c, fold, train_transform_index):
#     filename = get_model_filename(c, fold, train_transform_index)
#     net = Classifier().cuda()
#     net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{filename}'))
#     return validate(net, get_dataloader(batch_size))

def test():
    if not os.path.exists(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test'):
        os.makedirs(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test')

    cl_net = Classifier().cuda()
    lm_net = ResNetLMS_L_v8(n_lm=N_KEYPOINT).cuda()

    checkpoint = torch.load(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt')
    # cl_net.load_state_dict(checkpoint['model_state_dict'])
    lm_net.load_state_dict(checkpoint['lm_state_dict'])

    checkpoint_cl = torch.load(lp("/home/godsom/guesswhat/models/classifier/gm20_n"+str(N_KEYPOINT)+".pt", clone=False))
    cl_net.load_state_dict(checkpoint_cl['model_state_dict'])

    result = {
        'exact_match': 0,
        'total': 0
    }

    total_loss, total, exact_match = 0, 0, 0

    with torch.no_grad():
        cl_net.eval()
        lm_net.eval()
        for i, batch in enumerate(get_dataloader(batch_size, 'test')):
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            probmap = lm_net(images)

            lms, masks = get_transform_mat(probmap, inv_std=inv_std)

            if N_KEYPOINT==1:
                masked_images = images * masks[:,0,:,:][:,None].repeat(1,3,1,1)
            elif N_KEYPOINT==2:
                masked_images1 = images * masks[:,0,:,:][:,None].repeat(1,3,1,1)

                comb_masks2 = torch.max(masks[:,0,:,:], masks[:,1,:,:])
                masked_images2 = images * comb_masks2[:,None].repeat(1,3,1,1)

                masked_images = torch.cat([masked_images2, masked_images1], dim=0)
                labels = torch.cat([labels, labels], dim=0)

            elif N_KEYPOINT==4:
                masked_images1 = images * masks[:,0,:,:][:,None].repeat(1,3,1,1)

                comb_masks2 = torch.max(masks[:,0,:,:], masks[:,1,:,:])
                masked_images2 = images * comb_masks2[:,None].repeat(1,3,1,1)

                comb_masks3 = torch.max(comb_masks2, masks[:,2,:,:])
                masked_images3 = images * comb_masks3[:,None].repeat(1,3,1,1)

                comb_masks4 = torch.max(comb_masks3, masks[:,3,:,:])
                masked_images4 = images * comb_masks4[:,None].repeat(1,3,1,1)

                masked_images = torch.cat([masked_images4, masked_images3, masked_images2, masked_images1], dim=0)
                labels = torch.cat([labels, labels, labels, labels], dim=0)

            outputs = cl_net(masked_images)

            loss = criterion(outputs, labels) + 1*variance_loss(probmap, lms)# + dis_pel_loss(dist)
            total_loss += float(loss.item())
            # print(outputs)
            # op = map(lambda x: torch.max(x, 0)[1], outputs)
            # outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).cuda()
            outputs = torch.argmax(outputs, 1)
            total += outputs.shape[0]
            exact_match += sum(outputs == labels).item()
            if i<50:
                if N_KEYPOINT ==2:
                    probmap = probmap.cpu()
                    masked_images = torch.transpose(masked_images, 1 ,2)
                    masked_images = torch.transpose(masked_images, 2, 3)
                    masked_images = masked_images.cpu()
                    images = torch.transpose(images, 1 ,2)
                    images = torch.transpose(images, 2, 3)
                    images = images.cpu()
                    lms = lms.cpu()
                    lms = (lms+1)*IMAGE_SIZE/2

                    plt.figure(figsize=(24,12))
                    plt.subplot(1, 2, 1)
                    plt.imshow(images[0])
                    for q in range(N_KEYPOINT):
                        # print(lms.shape)
                        plt.plot([lms[0][q][1]], [lms[0][q][0]], mk[q], mew=15.0)
                    plt.subplot(1, 2, 2)
                    plt.imshow(masked_images[0])
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/out_{:04d}.png'.format(i))
                    plt.close()

                    plt.figure(figsize=(12,12))
                    plt.subplot(2, 2, 1)
                    plt.imshow(probmap[0][0]/torch.max(probmap[0]))
                    plt.subplot(2, 2, 2)
                    plt.imshow(probmap[0][1]/torch.max(probmap[1]))
                    plt.subplot(2, 2, 3)
                    plt.imshow(masked_images[0])
                    plt.subplot(2, 2, 4)
                    plt.imshow(images[0])
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/test_{:04d}.png'.format(i))
                    plt.close()
                elif N_KEYPOINT ==4:
                    probmap = probmap.cpu()
                    masked_images = torch.transpose(masked_images, 1 ,2)
                    masked_images = torch.transpose(masked_images, 2, 3)
                    masked_images = masked_images.cpu()
                    images = torch.transpose(images, 1 ,2)
                    images = torch.transpose(images, 2, 3)
                    images = images.cpu()
                    lms = lms.cpu()
                    lms = (lms+1)*IMAGE_SIZE/2

                    plt.figure(figsize=(24,12))
                    plt.subplot(1, 2, 1)
                    plt.imshow(images[0])
                    for q in range(N_KEYPOINT):
                        # print(lms.shape)
                        plt.plot([lms[0][q][1]], [lms[0][q][0]], mk[q], mew=15.0)
                    plt.subplot(1, 2, 2)
                    plt.imshow(masked_images[0])
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/out_{:04d}.png'.format(i))
                    plt.close()

                    plt.figure(figsize=(12,12))
                    plt.subplot(2, 2, 1)
                    plt.imshow(probmap[0][0]/torch.max(probmap[0]))
                    plt.subplot(2, 2, 2)
                    plt.imshow(probmap[0][1]/torch.max(probmap[1]))
                    plt.subplot(2, 2, 3)
                    plt.imshow(probmap[0][2]/torch.max(probmap[2]))
                    plt.subplot(2, 2, 4)
                    plt.imshow(probmap[0][3]/torch.max(probmap[3]))
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/hm_{:04d}.png'.format(i))
                    plt.close()
                else:
                    save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/maskedimg_{:04d}.png'.format(i))
                    save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test/hm_{:04d}.png'.format(i))
                print(i)
                print(labels[0], outputs[0])
            else:
                break

    return result



def train_save(checkpoint_frequency=checkpoint_frequency):
    training_loss, validation_loss = train(
        Classifier().cuda(),
        ResNetLMS_L_v7(n_lm=N_KEYPOINT).cuda(),
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