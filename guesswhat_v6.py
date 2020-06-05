#ranked lms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from localpath import getLocalPath
from classifier import FaceDataset, Classifier_v5, LmSelector_v5#, Classifier_v5, LmSelector_v5
# from resnet import ResNetLMS_L
# from clms import ResNetLMS_L
from skimage import io, transform
import torch.nn as nn
import matplotlib.patches as P
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import random
import argparse
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


# CUDA_VISIBLE_DEVICES = 1
# device = torch.device('cuda:1' if torch.cuda().is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_di_n', type=str, default='1', help='model_directory')
parser.add_argument('-mode', type=str, default='train', help='mode')
args = parser.parse_args()

mk = ['+r','+m','+b','+g','+c','|r','|m','|b','|g','|c']
IMAGE_SIZE = 200
batch_size = 8#40
num_epochs = 1000
N_KEYPOINT = 1
lr = 0.0001  # initial learning rate
SCALE = 0.5
checkpoint_frequency = 10
nll_w = 2
inv_std = 20
MAX_D_PENALTY = 25.0

lp = lambda path, clone=True: getLocalPath("/home2/godsom/local_storage", path, clone)
# lp = lambda path, clone=True: path

file_dir =  lp('/home/godsom/Dataset/UTKFace/UTKFace')
PATH_TO_MODELS = lp("/home/godsom/guesswhat/models/v6",clone=False)
PATH_TO_OUTPUTS = lp("/home/godsom/guesswhat/outputs/v6",clone=False)
transforms_list = [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]

cx, cy = torch.meshgrid([torch.arange(25, IMAGE_SIZE-24, 10), torch.arange(25, IMAGE_SIZE-24, 10)])
COORS = torch.stack([cx, cy], axis=2)
COORS = torch.reshape(COORS, (1,-1,2)).cuda()
# print(COORS)
# assert False

if not os.path.exists(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train'):
    os.makedirs(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train')


fl = os.listdir(file_dir)
fl.sort()
random.seed(555)
random.shuffle(fl)

def meshgrid(h):
    xv, yv = torch.meshgrid([torch.arange(0.5, h, 1) / (h / 2) - 1, torch.arange(0.5, h, 1) / (h / 2) - 1])
    return xv.cuda(), yv.cuda()

ranx, rany = meshgrid(7)

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.cuda())


def stn(lms, img):
    lm_coor = torch.reshape(lms, (-1,2))
    s = torch.ones_like(lm_coor[:,0]) * 0.25
    zeros = torch.zeros_like(lm_coor[:,0])
    transform_mat = torch.stack([s, zeros, lm_coor[:,0]*2/IMAGE_SIZE-1, zeros, s, lm_coor[:,1]*2/IMAGE_SIZE-1], 1)
    transform_mat = torch.reshape(transform_mat, (-1,2,3))
    # print('!!!!!')
    # print(transform_mat[0])
    input_img= tile(img,0,N_KEYPOINT)#img.repeat(N_KEYPOINT,1,1,1)
    # input_img = torch.transpose(input_img_tiled, 1 ,2)
    # input_img = torch.transpose(input_img, 2 ,3)
    # print(input_img.size())
    grid = F.affine_grid(transform_mat, input_img.size())
    out_fmap = F.grid_sample(input_img, grid)

    return out_fmap


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

  vl = torch.mean(torch.sum(diff, axis=[2, 3]))

  return vl

def get_gaussian_maps(mu, inv_std=inv_std):
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

def get_transform_mat(probmap, inv_std=inv_std):
    # ranx, rany = meshgrid(IMAGE_SIZE)
    x = torch.sum(probmap * ranx, dim=(2,3))
    y = torch.sum(probmap * rany, dim=(2,3))

    # x = torch.max(probmap)[1]
    # # y = torch.sum(probmap * rany, dim=(2,3))
    #
    # print(x)
    # assert False

    coors = torch.stack([x, y], dim=2)

    # masks = get_gaussian_maps(torch.reshape(coors, (-1, N_KEYPOINT, 2)), inv_std)
    # masks = torch.transpose(masks, 2, 3)
    # masks = torch.transpose(masks, 1, 2)

    return coors#, masks

def get_dataloader(minibatch_size=batch_size, mode='train'):
    root_dir = file_dir
    if mode == 'train':
        transformed_dataset = FaceDataset(root_dir, fl[:20000], transforms.Compose(transforms_list))
        # transformed_dataset = FaceDataset(root_dir, fl[:16], transforms.Compose(transforms_list))
    if mode == 'val':
        transformed_dataset = FaceDataset(root_dir, fl[20000:21496], transforms.Compose(transforms_list))
        # transformed_dataset = FaceDataset(root_dir, fl[20000:20016], transforms.Compose(transforms_list))
    if mode == 'test':
        transformed_dataset = FaceDataset(root_dir, fl[21500:], transforms.Compose(transforms_list))
    dataloader = DataLoader(transformed_dataset, batch_size=minibatch_size, shuffle=True, num_workers=4)
    return dataloader

criterion = nn.NLLLoss(reduce=False)

def train(cl_net, lm_net, train_dataloader, epochs, filename, checkpoint_frequency=checkpoint_frequency, val_dataloader=None):
    cl_optimizer = optim.Adam(cl_net.parameters(), lr)
    cl_scheduler = optim.lr_scheduler.MultiStepLR(cl_optimizer, milestones=[10000])
    lm_optimizer = optim.Adam(lm_net.parameters(), lr)
    lm_scheduler = optim.lr_scheduler.MultiStepLR(lm_optimizer, milestones=[200,400,600,800,1000],gamma=0.1)

    inv_std_loop = inv_std
    anneal_rate = 1.0,

    if os.path.exists(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt'):
    #     os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
    # if os.path.exists(lp("/home/godsom/guesswhat/logs/v6"+'/'+args.model_di_n,clone=False)):
        checkpoint = torch.load(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt')
        cl_net.load_state_dict(checkpoint['cl_state_dict'])
        lm_net.load_state_dict(checkpoint['lm_state_dict'])
        cl_optimizer.load_state_dict(checkpoint['cl_optimizer_state_dict'])
        lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        if not os.path.exists(lp("/home/godsom/guesswhat/logs/v6"+'/'+args.model_di_n,clone=False)):
            os.makedirs(lp("/home/godsom/guesswhat/logs/v6"+'/'+args.model_di_n,clone=False))
            os.makedirs(PATH_TO_MODELS+'/'+args.model_di_n)
        start_epoch = 0

    writer = SummaryWriter(lp("/home/godsom/guesswhat/logs/v6/"+args.model_di_n,clone=False))

    # checkpoint_cl = torch.load(lp("/home/godsom/guesswhat/models/Classifier_v5/gm20_n"+str(N_KEYPOINT)+".pt", clone=False))
    # cl_net.load_state_dict(checkpoint_cl['model_state_dict'])

    cl_net.train()
    lm_net.train()


    training_loss, validation_loss, val_accuracy = [], [], []
    checkpoint = 0
    iteration = 0
    running_loss = 0


    for epoch in range(epochs):
        ta = 0
        for i, batch in enumerate(train_dataloader):
            # print(i)
            # print(lm_net.parameters)
            cl_optimizer.zero_grad()
            lm_optimizer.zero_grad()
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            bs = images.shape[0]
            windows = F.unfold(images, kernel_size=50, stride = 25)
            windows = torch.transpose(windows, 1, 2)
            n_win_sqrt = int(windows.shape[1]**0.5)
            windows = torch.reshape(windows, (windows.shape[0]*windows.shape[1],3,50,50))

            patch_score = lm_net(windows)
            patch_score = torch.reshape(patch_score, (bs, n_win_sqrt*n_win_sqrt, N_KEYPOINT))
            patch_score = torch.transpose(patch_score, 1, 2)
            patch_score = F.softmax(patch_score, dim=2)
            probmap = torch.reshape(patch_score, (bs, N_KEYPOINT, n_win_sqrt, n_win_sqrt))
            patch_score = torch.reshape(patch_score, (bs, N_KEYPOINT, n_win_sqrt*n_win_sqrt))
            patch_score = torch.transpose(patch_score, 0, 1)
            patch_score = torch.reshape(patch_score, (N_KEYPOINT, bs*n_win_sqrt*n_win_sqrt))

            lms = get_transform_mat(probmap, inv_std=20)

            outputs = cl_net(windows)

            # print(labels)
            # print(tile(labels, 0, n_win_sqrt*n_win_sqrt))
            # print(patch_score.shape)
            tiled_labels = tile(labels, 0, n_win_sqrt*n_win_sqrt)
            vl = variance_loss(probmap, lms)
            loss = torch.sum(criterion(outputs, tiled_labels) * patch_score)*nll_w + vl
            running_loss += float(loss.item())
            loss.backward()
            cl_optimizer.step()
            lm_optimizer.step()

            outputs = torch.argmax(outputs, 1)
            train_accu = (outputs == tiled_labels)
            accu = sum(train_accu).item()/outputs.shape[0]

            patch_score_all = torch.reshape(patch_score, (N_KEYPOINT, bs, n_win_sqrt*n_win_sqrt))
            top_score, ts_idx = torch.topk(patch_score_all, 1)

            ta = ta + sum(train_accu[ts_idx[0,:]]).item()/bs

        if (epoch+1) % checkpoint_frequency == 0 and val_dataloader is not None:
            # if inv_std_loop < 20:
            #     inv_std_loop = inv_std_loop * anneal_rate

            # patch_score_all = torch.reshape(patch_score, (N_KEYPOINT, bs, n_win_sqrt*n_win_sqrt))
            # top_score, ts_idx = torch.topk(patch_score_all, 1)

            training_loss.append(running_loss/i)
            val_results = validate(cl_net, lm_net, val_dataloader, epoch+start_epoch, inv_std_loop,start_epoch)
            validation_loss.append(val_results[0])
            val_accuracy.append(val_results[1])
            print(f'epoch:{epoch+start_epoch}, training_error:{training_loss[-1]}, train_var_loss:{vl}, val_error:{validation_loss[-1]}, val_accu:{val_accuracy[-1]}')
            print(f'top accuracy, training:{ta/(i+1)}, val:{val_results[2]}')

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
                patch_score_plt = patch_score_all[:,0]
                top_score, ts_idx = torch.topk(patch_score_plt, 1)
                patches_plot = torch.transpose(windows, 1, 2)
                patches_plot = torch.transpose(patches_plot, 2, 3).detach().cpu().numpy()

                print('train')
                print(top_score)

                plt.figure(figsize=(12,12))
                plt.subplot(1, 2, 1)
                plt.imshow(patches_plot[ts_idx[0][0]])
                plt.subplot(1, 2, 2)
                plt.imshow(patches_plot[ts_idx[1][0]])
                plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/p_{:04d}.png'.format(epoch+start_epoch))
                plt.close()
            elif N_KEYPOINT ==1:
                patch_score_plt = patch_score_all[:,0]
                top_score, ts_idx = torch.topk(patch_score_plt, 1)
                patches_plot = torch.transpose(windows, 1, 2)
                patches_plot = torch.transpose(patches_plot, 2, 3).detach().cpu().numpy()


                plt.figure(figsize=(12,12))
                # plt.subplot(1, 2, 1)
                plt.imshow(patches_plot[ts_idx[0][0]])
                # plt.subplot(1, 2, 2)
                # plt.imshow(patches_plot[ts_idx[1][0]])
                plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/p_{:04d}_r.png'.format(epoch+start_epoch))
                plt.close()

                print('train')
                print(top_score)
            else:
                save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/masked_{:04d}.png'.format(epoch+start_epoch))
                save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/train/hm_{:04d}.png'.format(epoch+start_epoch))


        running_loss = 0

    return training_loss, validation_loss

def validate(cl_net, lm_net, dataloader,epoch,inv_std_loop,start_epoch):
    # net.train()
    total_loss, total, exact_match = 0, 0, 0
    va = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            bs = images.shape[0]
            windows = F.unfold(images, kernel_size=50, stride = 25)
            windows = torch.transpose(windows, 1, 2)
            n_win_sqrt = int(windows.shape[1]**0.5)
            windows = torch.reshape(windows, (windows.shape[0]*windows.shape[1],3,50,50))

            patch_score = lm_net(windows)
            patch_score = torch.reshape(patch_score, (bs, n_win_sqrt*n_win_sqrt, N_KEYPOINT))
            patch_score = torch.transpose(patch_score, 1, 2)
            patch_score = F.softmax(patch_score, dim=2)
            probmap = torch.reshape(patch_score, (bs, N_KEYPOINT, n_win_sqrt, n_win_sqrt))
            patch_score = torch.reshape(patch_score, (bs, N_KEYPOINT, n_win_sqrt*n_win_sqrt))
            patch_score = torch.transpose(patch_score, 0, 1)
            patch_score = torch.reshape(patch_score, (N_KEYPOINT, bs*n_win_sqrt*n_win_sqrt))

            lms = get_transform_mat(probmap, inv_std=20)

            outputs = cl_net(windows)

            # print(labels)
            # print(tile(labels, 0, n_win_sqrt*n_win_sqrt))
            # print(patch_score.shape)
            vl = variance_loss(probmap, lms)
            tiled_labels = tile(labels, 0, n_win_sqrt*n_win_sqrt)
            loss = torch.sum(criterion(outputs, tiled_labels) * patch_score)*nll_w + vl
            # print(variance_loss(probmap, lms))
            total_loss += float(loss.item())
            # print(outputs)
            # op = map(lambda x: torch.max(x, 0)[1], outputs)
            # outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).cuda()
            outputs = torch.argmax(outputs, 1)
            total += outputs.shape[0]
            val_accu = outputs == tiled_labels
            exact_match += sum(val_accu).item()

            patch_score_all = torch.reshape(patch_score, (N_KEYPOINT, bs, n_win_sqrt*n_win_sqrt))
            top_score_all, ts_idx_all = torch.topk(patch_score_all, 1)
            if i==0:
                if N_KEYPOINT ==2:
                    patch_score_plt = patch_score_all[:,0]
                    top_score, ts_idx = torch.topk(patch_score_plt, 1)
                    patches_plot = torch.transpose(windows, 1, 2)
                    patches_plot = torch.transpose(patches_plot, 2, 3).detach().cpu().numpy()

                    print('val')
                    print(top_score)

                    plt.figure(figsize=(12,12))
                    plt.subplot(1, 2, 1)
                    plt.imshow(patches_plot[ts_idx[0][0]])
                    plt.subplot(1, 2, 2)
                    plt.imshow(patches_plot[ts_idx[1][0]])
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/p_{:04d}.png'.format(epoch+start_epoch))
                    plt.close()
                elif N_KEYPOINT ==1:
                    patch_score_plt = patch_score_all[:,0]
                    top_score, ts_idx = torch.topk(patch_score_plt, 1)
                    patches_plot = torch.transpose(windows, 1, 2)
                    patches_plot = torch.transpose(patches_plot, 2, 3).detach().cpu().numpy()


                    plt.figure(figsize=(12,12))
                    # plt.subplot(1, 2, 1)
                    plt.imshow(patches_plot[ts_idx[0][0]])
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(patches_plot[ts_idx[1][0]])
                    plt.savefig(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/p_{:04d}_r.png'.format(epoch+start_epoch))
                    plt.close()

                    print('val')
                    print(top_score)
                    # print(val_accu[ts_idx_all[0,:]])
                    # print(bs)
                    print(sum(val_accu[ts_idx_all[0,:]]).item()/bs)
                else:
                    save_image(masked_images[0], PATH_TO_OUTPUTS+'/'+args.model_di_n+'/maskedimg_{:04d}.png'.format(epoch+start_epoch))
                    save_image(probmap[0]/torch.max(probmap[0]), PATH_TO_OUTPUTS+'/'+args.model_di_n+'/hm_{:04d}.png'.format(epoch+start_epoch))

            # print('--val')
            # # print(top_score)
            # print(val_accu[ts_idx_all[0,:]])
            # print(bs)
            # print(sum(val_accu[ts_idx_all[0,:]]).item()/bs)
            va = va + sum(val_accu[ts_idx_all[0,:]]).item()/bs
    return total_loss/(i+1), exact_match/total, va/(i+1)

# def get_validation_error(c, fold, train_transform_index):
#     filename = get_model_filename(c, fold, train_transform_index)
#     net = Classifier_v5().cuda()
#     net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{filename}'))
#     return validate(net, get_dataloader(batch_size))

def test():
    if not os.path.exists(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test'):
        os.makedirs(PATH_TO_OUTPUTS+'/'+args.model_di_n+'/test')

    cl_net = Classifier_v5().cuda()
    lm_net = LmSelector_v5(n_lm=N_KEYPOINT).cuda()

    checkpoint = torch.load(PATH_TO_MODELS+'/'+args.model_di_n+f'/{args.model_di_n}.pt')
    # cl_net.load_state_dict(checkpoint['model_state_dict'])
    lm_net.load_state_dict(checkpoint['lm_state_dict'])

    # checkpoint_cl = torch.load(lp("/home/godsom/guesswhat/models/Classifier_v5/gm20_n"+str(N_KEYPOINT)+".pt", clone=False))
    # cl_net.load_state_dict(checkpoint_cl['model_state_dict'])

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
        Classifier_v5().cuda(),
        LmSelector_v5(n_lm=N_KEYPOINT).cuda(),
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
