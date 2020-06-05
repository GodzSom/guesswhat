from __future__ import division
from __future__ import print_function
import math
from utils.sfm_utils import SfMData
from utils.utils import *
from utils.mpi_utils import outputMPI
from skimage import io, transform
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from utils.unet_part import *
import torch as pt
import numpy as np
import os
import sys
import time
from itertools import repeat
import cv2
import socket
import torch.autograd as autograd
from timeit import default_timer as timer
from torch.autograd import Variable
from appearance import Network2 
pt.backends.cudnn.enabled = True

os.environ["CUDA_VISIBLE_DEVICES"]="5"
parser = argparse.ArgumentParser()
parser.add_argument('-layers', type=int, default=40)
parser.add_argument('-epochs', type=int, default=100000)

parser.add_argument('-offset', type=int, default=0)
parser.add_argument('-l2', type=float, default=100000)
parser.add_argument('-l1', type=float, default=0)
parser.add_argument('-tva', type=float, default=0.5)
parser.add_argument('-tvc', type=float, default=0.005)
parser.add_argument('-tv01', type=float, default=1)
parser.add_argument('-dmin', type=float, default=-1)
parser.add_argument('-dmax', type=float, default=-1)
parser.add_argument('-scale', type=float, default=0.4)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-decaylr', type=float, default=0.5)
parser.add_argument('-lra', type=float, default=4)
parser.add_argument('-lrc', type=float, default=1)
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-model_dir', type=str, default="e1")
parser.add_argument('-restart', action='store_true')
parser.add_argument('-invz', action='store_true')
parser.add_argument('-predict', action='store_true')

parser.add_argument('-dataset', type=str, default="trevi_appearance")#img657
parser.add_argument('-ref_img', type=str, default="9_UltimateRome_Trevi-Fountain.jpg")#9_UltimateRome_Trevi-Fountain.jpg
parser.add_argument('-img_wildcard', type=str, default="")

args = parser.parse_args()

class OrbiterDataset(Dataset):
  def __init__(self, dataset, ref_img, scale, img_wildcard="", transform=None):
    self.scale = scale
    self.dataset = dataset
    self.transform = transform
    self.sfm = SfMData(dataset,
                       ref_img=ref_img,
                       dmin=args.dmin,
                       dmax=args.dmax,
                       scale=scale)

    self.sfm.ref_rT = pt.from_numpy(self.sfm.ref_img['r']).t()
    self.sfm.ref_t = pt.from_numpy(self.sfm.ref_img['t'])

    t = self.sfm.ref_cam
    self.sfm.ref_k = pt.tensor(
      [[t['fx'], 0, t['px']],
       [0, t['fy'], t['py']],
       [0, 0, 1]])

    self.imgs = []
    
    self.ref_id = -1
    for i, ind in enumerate(self.sfm.imgs):
      img = self.sfm.imgs[ind]
      if img_wildcard == "" or img_wildcard in img['path']:
        self.imgs.append(img)
        if ref_img in img['path']:
          self.ref_id = len(self.imgs) - 1
    self.imgs = self.imgs
    
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    if pt.is_tensor(idx):
        idx = idx.tolist()



    img = io.imread(self.dataset + "/" + self.imgs[idx]['path'])

    if self.scale != 1:
      h, w = img.shape[:2]
      newh = round(h * self.scale)
      neww = round(w * self.scale)
      if len(img.shape) <3:
          img = np.tile(img[:,:,np.newaxis], [1, 1, 3])
      '''
      notredame 
      h_threshold = 665
      w_threshold = 490
      trevi 
      h_threshold = 500
      w_threshold = 750
      '''
      h_threshold = 400
      w_threshold = 600
      if newh >= h_threshold or neww >= w_threshold:
        x_scale = w_threshold/neww
        y_scale = h_threshold/newh

        img = cv2.resize(img, (w_threshold, h_threshold),interpolation=cv2.INTER_AREA)

      else:
        img = cv2.resize(img, (neww, newh),interpolation=cv2.INTER_AREA)
        x_scale = 1.0
        y_scale = 1.0
    
    img = np.transpose(img, [2, 0, 1]).astype(np.float32)
    
    if np.max(img) > 1:
      img /= 255.0
    
    im = self.imgs[idx]
    cam = self.sfm.cams[im['camera_id']]
    feature = {
      'image': img[:3,:,:],
      'r': im['r'],
      't': im['t'],
      'fx': cam['fx'] * x_scale,
      'fy': cam['fy'] * y_scale,
      'px': cam['px'] * x_scale,
      'py': cam['py'] * y_scale,
      'path': im['path']
    }

    return feature

def computeHomography(sfm, feature, d):
  r = feature['r'][0]
  t = feature['t'][0]
  fx = feature['fx'][0]
  fy = feature['fy'][0]
  px = feature['px'][0]
  py = feature['py'][0]

  new_r = pt.matmul(r, sfm.ref_rT)
  new_t = pt.matmul(pt.matmul(-r, sfm.ref_rT), sfm.ref_t) + t

  n = pt.tensor([[0.0, 0.0, 1.0]])
  Ha = new_r.t()
  Hb = pt.matmul(pt.matmul(pt.matmul(Ha, new_t), n), Ha)
  Hc = pt.matmul(pt.matmul(n, Ha), new_t)[0]

  ki = pt.tensor([[fx, 0, px],
                 [0, fy, py],
                 [0, 0, 1]], dtype=pt.float).inverse()

  return pt.matmul(pt.matmul(sfm.ref_k, Ha + Hb/(-d-Hc)), ki)
def computePSV(sfm, planes, mpi_sig, feature, train_img):
  warp, mask = computeHomoWarp(sfm,
                               mpi_sig.shape[-2:],
                               args.offset,
                               [feature['height'][0], feature['width'][0]],
                               0,
                               feature, planes)
  samples = F.grid_sample(mpi_sig[:, 3:], warp, align_corners=True)
  weight = pt.cumprod(1 - pt.cat([pt.zeros_like(samples[:1]), samples[:-1]], 0), 0)
  # netTrans = weight * samples
  netTrans = weight

  tiled = train_img.repeat(args.layers, 1, 1, 1)
  tiled = pt.cat([tiled, netTrans], 1)

  iwarp, imask = computeHomoWarp(sfm,
                               train_img.shape[-2:],
                               0,
                               mpi_sig.shape[-2:],
                               -args.offset,
                               feature, planes, True)

  samples = F.grid_sample(tiled, iwarp, align_corners=True)
  return samples
def computeHomoWarp(sfm,input_shape, input_offset,output_shape, output_offset, feature, planes, inv=False):
  y, x = pt.meshgrid([
    pt.arange(0, output_shape[0], dtype=pt.float) + output_offset,
    pt.arange(0, output_shape[1], dtype=pt.float) + output_offset])

  coords = pt.stack([x, y, pt.ones_like(x)], 2).cuda()

  cxys = []
  for i, v in enumerate(planes):
    H = computeHomography(sfm, feature, v)
    if inv:
      H = H.inverse()

    newCoords = pt.matmul(coords, H.t().cuda())

    cxys.append(
        (newCoords[None, :, :, :2] / newCoords[:, :, 2:] + input_offset) /
        (pt.tensor([input_shape[1]-1, input_shape[0]-1]).cuda()) * 2 - 1)

  warp = pt.cat(cxys, 0)

  warp2 = pt.cat([cxys[0], cxys[-1]], 0)
  tmp = (warp2 >=-1) & (warp2 <=1)
  mask = (tmp[0, :, :, 0] & tmp[0, :, :, 1] & tmp[1, :, :, 0] & tmp[1, :, :, 1]).float()
  return warp, mask


def getPlanes(sfm):
  if args.invz:
    return 1/np.linspace(1, 0.0001, args.layers) * sfm.dmin
  else:
    return np.linspace(sfm.dmin, sfm.dmax, args.layers)
def totalVariation(images):
  pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
  pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
  sum_axis = [1, 2, 3]

  tot_var = (
      pt.sum(pt.abs(pixel_dif1), dim=sum_axis) +
      pt.sum(pt.abs(pixel_dif2), dim=sum_axis))

  return tot_var / (images.shape[2]-1) / (images.shape[3]-1) * 306081




def weights_init(m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
      



class generator(nn.Module):
  def __init__(self, shape):
    super(generator, self).__init__()

    #img = cv2.cvtColor(cv2.resize(cv2.imread('/data/orbiter/datasets/' + args.dataset + '/dense/images/9_UltimateRome_Trevi-Fountain.jpg' ),
    # (shape[2], shape[1])), cv2.COLOR_BGR2RGB)[np.newaxis, :, :, :]

    #img = cv2.cvtColor(cv2.resize(cv2.imread('/data/orbiter/datasets/' + args.dataset + '/dense/images/image657.jpg' ),
    # (shape[2], shape[1])), cv2.COLOR_BGR2RGB)[np.newaxis, :, :, :]

    #img = np.tile(img, [args.layers, 1, 1, 1])/255.0
    
    #color_init = np.load('/data/orbiter_pytorch/notredame_after_c.npy')
    #alpha_init = np.load('/data/orbiter_pytorch/notredame_after_a.npy')
    
    color_init = np.load('/data/orbiter_pytorch/trevi_after_c.npy')
    alpha_init = np.load('/data/orbiter_pytorch/trevi_after_a.npy')
    
    with pt.no_grad():
        #mpic = pt.tensor(color_init).permute(0, 3, 1, 2).cuda()
        #mpia = pt.tensor(alpha_init).permute(0, 3, 1, 2).cuda()
      mpic = pt.tensor(color_init).cuda()
      mpia = pt.tensor(alpha_init).cuda()

    #mpic.requires_grad = False
    #mpia.requires_grad = False
    
    #self.mpia = nn.Parameter(mpia)
    #self.mpic = nn.Parameter(mpic * mpia)
    self.mpia = mpia
    self.mpic = mpic * mpia
    
    
    #self.laplacian = Lap(3)
    self.in_channels = 3
    filter = [self.in_channels, 16, 32, 64, 128, 256, 512]
    self.laplacian = Lap()
    self.down = nn.ModuleList([Down(filter[i], filter[i + 1], 'res') for i in range(len(filter) - 1)])
    self.intermediate = Up(filter[-1], filter[-2], 'style2d')
    self.style_inter1 = style(filter[-1], filter[-2])
    self.style_inter2 = style(filter[-2], filter[-2])
    #self.style_inter3 = style(filter[-1], filter[-2])
    self.encoder = AutoEncoder()
    
    self.up = nn.ModuleList([Up(filter[i], filter[i - 2], 'style2d') for i in range(len(filter) - 1, 1, -1)])
    self.style1 = nn.ModuleList([style(filter[i], filter[i - 2]) for i in range(len(filter) - 1, 1, -1)])
    self.style2 = nn.ModuleList([style(filter[i], filter[i]) for i in range(len(filter) - 3, -1, -1)])
    #self.style3 = nn.ModuleList([style(filter[i], filter[i - 2]) for i in range(len(filter) - 1, 1, -1)])
    
    
    self.last = nn.Sequential(
                    nn.Conv2d(2*self.in_channels, 3, 3, 1, padding = 1),
                    nn.LeakyReLU(0.2),
                    #state size h/2 x w/2 x 8
                    nn.Conv2d(3, 3, 3, 1, padding = 1),
                    nn.Sigmoid()
    )
  

  def forward(self, sfm, feature, is_training):
    #print(pt.max(self.mpic))
    #print(pt.max(self.mpia))
    #exit()
    self.mpi_sig = pt.cat([self.mpic, self.mpia], 1)
    
    conv = [pt.cat([self.mpic], 1)]
    
    for i in range(len(self.down)):
      conv.append(self.down[i](conv[i]))
    
    
    latent_code, recon = self.encoder(F.interpolate(feature['image'].cuda(), [128*2, 128*2]))
    
    scale1, bias1 = self.style_inter1(latent_code)
    scale2, bias2 = self.style_inter2(latent_code)
    #scale3, bias3 = self.style_inter3(latent_code)
    deconv = [self.intermediate(conv[-1], conv[-2], [scale1, bias1, scale2, bias2])]
    
    for i in range(len(self.up)):
      scale1, bias1 = self.style1[i](latent_code)
      scale2, bias2 = self.style2[i](latent_code)
      #scale3, bias3 = self.style3[i](latent_code)
      mod = [scale1, bias1, scale2, bias2]
      
      deconv.append(self.up[i](deconv[i], conv[-(i+3)], mod))
    
    
    #self.new_sig =  pt.squeeze(pt.cat([self.last(deconv[-1]), self.mpia], 1), dim = 0).permute(1, 0, 2, 3)
    self.new_sig = pt.cat([self.last(deconv[-1]), self.mpia], 1)
    #self.new_sig = self.last(deconv[-1])
    if not is_training:
      return self.new_sig
    else:
      planes = getPlanes(sfm)

      warp, mask = computeHomoWarp(sfm,
                                  self.new_sig.shape[-2:],
                                  args.offset,
                                  feature['image'].shape[-2:],
                                  0,
                                  feature, planes)

      # For a good visualization of align_coners parameter:
      # see: https://discuss.pytorch.org/uploads/default/original/2X/6/6a242715685b8192f07c93a57a1d053b8add97bf.png
      #samples = F.grid_sample(pt.cat([self.new_sig, pt.sigmoid(self.mpia)], 1), warp, align_corners=True)
      samples = F.grid_sample(self.new_sig, warp, align_corners=True)
      weight = pt.cumprod(1 - pt.cat([pt.zeros_like(samples[:1, :1]), samples[:-1, 3:]], 0), 0)
      output = pt.sum(weight * samples[:, :3] * samples[:, 3:], dim=0, keepdim=True)
      #pyr_out = self.laplacian(output * mask)[0]
      #pyr_real = self.laplacian(feature['image'].cuda() * mask)[0]


      # values = tf.reshape(tf.convert_to_tensor(np.linspace(0, 1, FLAGS.layers), tf.float32), [FLAGS.layers, 1, 1, 1])
      # depthmap = tf.reduce_sum(weight * samples[:, :, :, 3:] * values, axis=0, keepdims=True)
      return output, mask
class discriminator(nn.Module):
  def __init__(self, shape):
    super(discriminator, self).__init__()
    self.shape = shape
    filter = [3, 16, 32, 64, 128, 256]
    self.down = nn.ModuleList([])
    for i in range(len(filter) - 1):
      self.down.append(Down_dis(filter[i], filter[i + 1]))
    
    weight = pt.empty((1, filter[-1], 3, 3)).cuda()
    self.last = nn.Parameter(weight)
  def forward(self, x):
    x = F.interpolate(x, self.shape)
    
    for f in self.down:
      x = f(x)
    
    demod = pt.rsqrt(pt.sum(self.last ** 2, dim = [1, 2, 3]) + 1e-8)[:,np.newaxis, np.newaxis, np.newaxis]
    out = F.conv2d(x, self.last * demod, stride =1, padding = 1)
    return out
class discriminator2(nn.Module):
  def __init__(self, shape):
    super(discriminator, self).__init__()
    self.shape = shape
    filter = [3, 16, 32, 64, 128, 256]
    self.down = nn.ModuleList([Down(filter[i], filter[i + 1], 'res') for i in range(len(filter) - 1)])
    
    self.last = nn.Sequential(nn.Conv2d(256, 1, kernel_size = 3, stride = 1, padding = 0, bias = False),
                  )
    
  def forward(self, x):
    x = F.interpolate(x, self.shape)
    for f in self.down:
      x = f(x)
    for f in self.last:
      x = f(x)
    return x

def predict():
    is_training = False
    #dpath = getOrbiterDataset(args.dataset, '/media/dome/hdd/dome/colmap_file/')
    dpath = '/data/orbiter/datasets/' + args.dataset
    dataset = OrbiterDataset(dpath, ref_img=args.ref_img, scale=args.scale, img_wildcard=args.img_wildcard)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Network1_back((args.layers,
                   dataset.sfm.ref_cam['height'] + args.offset * 2,
                   dataset.sfm.ref_cam['width'] + args.offset * 2,
                   )).cuda()
    for i, features in enumerate(dataloader):
        if i ==0:
            break
    app_img = io.imread("/data/orbiter/datasets/" +args.dataset+'/dense/images//trevi-fountain-rome-night-cr-getty.jpg')
    #app_img = cv2.resize(app_img, (128*2, 128*2),interpolation=cv2.INTER_AREA)
    app_img = np.transpose(app_img, [2, 0, 1]).astype(np.float32)/255
    
    #features['image'] = pt.tensor(app_img).cuda()[np.newaxis,:,:,:]
    runpath = "runs/"
    ckpt = runpath + args.model_dir + "/ckpt.pt"
    checkpoint = pt.load(ckpt)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.forward(dataset.sfm, features, False)
    
    
    app = features['image'][0].permute(1, 2, 0).cpu().detach().numpy()
    
    outputMPI((model.new_sig).permute([0, 2, 3, 1]).cpu().detach().numpy(),
              dataset.sfm,
              getPlanes(dataset.sfm),
              runpath + args.model_dir,
              args.layers,
              1,
              args.offset,
              args.invz)
    plt.imsave('/data/orbiter_pytorch/html/' + args.model_dir + '/app_image.png', app)
    print('Finished Training')
def train():
  #dpath = getOrbiterDataset(args.dataset, '/media/dome/hdd/dome/colmap_file/')
  dpath = '/data/orbiter/datasets/' + args.dataset
  #dpath = '/media/dome/hdd/dome/colmap_file/' + args.dataset

  dataset = OrbiterDataset(dpath, ref_img=args.ref_img, scale=args.scale, img_wildcard=args.img_wildcard)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = 5)

  netG = generator((args.layers,
                 dataset.sfm.ref_cam['height'] + args.offset * 2,
                 dataset.sfm.ref_cam['width'] + args.offset * 2,
                 )).cuda()

  netApp = Network2().cuda()
  if args.restart:
    os.system("rm -rf " + "runs/" + args.model_dir)

  # torch prefers [batch, channel, h, w]





  # writer = SummaryWriter("runs/" + args.model_dir)
  # runpath = "/home2/supasorn/local_storage/home2/research/orbiter_pytorch/runs/"
  

  #var = [name for name, param in model1.named_parameters() if 'encoder' in name]
  #print(var)
  #exit()
  #seed = 0
  seed = 4
  #seed = 6
  pt.cuda.manual_seed_all(seed)
  netG.apply(weights_init)
  
  
  lr = args.lr
  optimizer = pt.optim.Adam(netG.parameters(), lr= lr)

  #new_lr = nn.optim.lr_scheduler.MultiStepLR(optimizer, milestones = np.arange(1, args.epoch),
    #                                    gamma=args.decsylr, last_epoch=-1)
  '''
  optimizer = pt.optim.Adam([
    {'params': model.mpic, 'lr': lr * args.lrc},
    {'params': model.mpia, 'lr': lr * args.lra}])
  
  for name, param in model1.named_parameters():
    print(name, param.shape)
  print('\n')
  exit()
  for name, param in model2.named_parameters():
    print(name, param.shape)
  '''
  
  
  
  def load_model2():
    params1 = netG.named_parameters()
    dict_params1 = dict(params1)

    runpath = "runs/"
    pretrain_ckpt = runpath + 'trevi_fc_pretrain' + "/ckpt.pt"   
    checkpoint = pt.load(pretrain_ckpt)
    netApp.load_state_dict(checkpoint['model_state_dict'])
    for name1, param in netApp.named_parameters():
      for name2, value in dict_params1.items():
        if name1 == name2:
          
          dict_params1[name2].data.copy_(param.data)
          dict_params1[name2].requires_grad = False
    return dict_params1
  pretrain_ae = load_model2()
  netG.load_state_dict(pretrain_ae)
  
  start_epoch = 0
  runpath = "runs/"
  writer = SummaryWriter(runpath + args.model_dir)

  ckpt = runpath + args.model_dir + "/ckpt.pt"
  checkpoint = None
  if os.path.exists(ckpt):
    checkpoint = pt.load(ckpt)
    netG.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print("Loading model at epoch %d" % start_epoch)

  step = start_epoch * len(dataset)
  ts = TrainingStatus(num_steps=args.epochs * len(dataset))
  '''
  for name, param in model1.named_parameters():
    if 'encoder'  not in name:
      param.requires_grad = False
  k = 0
  '''
  for epoch in range(start_epoch, args.epochs):
    epoch_loss_total = 0
    epoch_mse = 0
    '''
    if epoch == start_epoch + 30:
      k = 1
      for name, param in model1.named_parameters():
        param.requires_grad = True
    '''
    
    for i, feature in enumerate(dataloader):
      ts.tic()

      optimizer.zero_grad()

      gt = feature['image'].cuda() # "ground-truth"
      
      output, mask= netG(dataset.sfm, feature, True)
      
      #mse =  pt.mean((gt - recon)**2)
      mse = pt.mean((mask * (output- gt ))**2)

      if args.l1 > 0:

        loss_recon = args.l1 * pt.mean(pt.abs(mask * (output - gt)))
      else:
        ##resize_gt = F.interpolate(gt, [128*2, 128*2])
        ##ae_loss = pt.mean((recon - resize_gt)**2)
        loss_recon = args.l2 * ( mse)

      #tva = args.tva * pt.mean(totalVariation(model1.new_sig[:, 3:]))
      #tvc = args.tvc * pt.mean(totalVariation(model1.new_sig[:, :3]))
      # tv01 = pt.mean(0.25-(0.5 - mpi_sig[:, 3:])**2)

      loss_total = (loss_recon) #(tva +  tvc)) # + args.tv01 * tv01
      #loss_total = loss_recon
      epoch_loss_total += loss_total
      epoch_mse += mse

      loss_total.backward()
      
      optimizer.step()

      print(ts.toc(step, loss_total.item()))

      if step % 100 == 0:
        writer.add_scalar('loss/total', loss_total, step)
        #writer.add_scalar('aeloss/total', args.l2 * ae_loss, step)
        writer.add_scalar('lr', lr, step)
        #writer.add_image('images/0_recon', pt.cat([resize_gt[0], recon[0]], 1), step)

        writer.add_image('images/0_gt', pt.cat([gt[0], output[0]], 1), step)
        writer.add_image('images/2_mpic', make_grid(netG.new_sig[:, :3] * netG.new_sig[:, 3:], 1), step)
        writer.add_image('images/3_mpia', make_grid( netG.new_sig[:, 3:], 1), step)

      step += 1

      if step % 1000000 == 0:
        lr *= args.decaylr
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
        print("Learning rate = %f" % lr)

    epoch_loss_total /= len(dataset)
    epoch_mse /= len(dataset)
    writer.add_scalar('loss/total', epoch_loss_total, step)
    writer.add_scalar('loss/mse', epoch_mse, step)

    if (epoch+1) % 10 == 0 or epoch == args.epochs-1:
      print("checkpointing model...")
      pt.save({
        'epoch': epoch,
        'model_state_dict': netG.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, runpath + args.model_dir + "/ckpt.pt")

  # save_image(mpi[0], "mpi0.png")
  # save_image(output[0], "output.png")
  predict()
  
  print('Finished Training')


def trainGAN():
  #dpath = getOrbiterDataset(args.dataset, '/media/dome/hdd/dome/colmap_file/')
  dpath = '/data/orbiter/datasets/' + args.dataset
  #dpath = '/media/dome/hdd/dome/colmap_file/' + args.dataset

  dataset = OrbiterDataset(dpath, ref_img=args.ref_img, scale=args.scale, img_wildcard=args.img_wildcard)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = 10)

  netG = generator((args.layers,
                 dataset.sfm.ref_cam['height'] + args.offset * 2,
                 dataset.sfm.ref_cam['width'] + args.offset * 2,
                 )).cuda()
  netD = discriminator((dataset.sfm.ref_cam['height'] + args.offset * 2,
                 dataset.sfm.ref_cam['width'] + args.offset * 2,
                 )).cuda()
  netApp = Network2().cuda()
  if args.restart:
    os.system("rm -rf " + "runs/" + args.model_dir)

  '''
  seed = 4
  pt.cuda.manual_seed_all(seed)
  netG.apply(weights_init)
  netD.apply(weights_init)
  '''
  lr = args.lr

  Goptimizer = pt.optim.Adam(netG.parameters(), lr= lr)
  Doptimizer = pt.optim.Adam(netD.parameters(), lr= lr)
  
  
  
  def pretrain_ae():
    params1 = netG.named_parameters()
    dict_params1 = dict(params1)

    runpath = "runs/"
    pretrain_ckpt = runpath + 'trevi_fc_pretrain' + "/ckpt.pt"   
    checkpoint = pt.load(pretrain_ckpt)
    netApp.load_state_dict(checkpoint['model_state_dict'])
    for name1, param in netApp.named_parameters():
      for name2, value in dict_params1.items():
        if name1 == name2:
          
          dict_params1[name2].data.copy_(param.data)
          dict_params1[name2].requires_grad = False
    return dict_params1
  
  netG.load_state_dict(pretrain_ae())
  
  start_epoch = 0
  runpath = "runs/"
  writer = SummaryWriter(runpath + args.model_dir)

  ckpt1 = runpath + args.model_dir + "/gen_ckpt.pt"
  checkpoint = None
  if os.path.exists(ckpt1):
    checkpoint = pt.load(ckpt1)
    netG.load_state_dict(checkpoint['model_state_dict'])
    Goptimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print("Loading generator model at epoch %d" % start_epoch)
    ckpt2 = runpath + args.model_dir + "/dis_ckpt.pt"
    checkpoint = pt.load(ckpt2)
    netD.load_state_dict(checkpoint['model_state_dict'])
    Doptimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print("Loading discriminator model at epoch %d" % start_epoch)

  step = start_epoch * len(dataset)
  ts = TrainingStatus(num_steps=args.epochs * len(dataset))
  '''
  for name, param in model1.named_parameters():
    if 'encoder'  not in name:
      param.requires_grad = False
  k = 0
  '''
  one = pt.tensor(1, dtype=pt.float).cuda()
  mone = (one * -1).cuda()
  for name, param in netG.named_parameters():
    if 'encoder'  not in name:
      param.requires_grad = True
  for epoch in range(start_epoch, args.epochs):
    epoch_loss_total = 0
    epoch_mse = 0
    '''
    if epoch == start_epoch + 30:
      k = 1
      for name, param in model1.named_parameters():
        param.requires_grad = True
    '''
    
    for jk, feature in enumerate(dataloader):
      d_loss_real = 0
      d_loss_fake = 0
      gt = feature['image'].cuda()
      
      num_critics = 5
      
      lamb_cs_fake = 10
      lamb_cs_real = 10
      start = timer()
      '''
      clip_value = 10
      for p in netD.parameters():
        p.data.clamp_(-clip_value, clip_value)
      '''
      for i in range(num_critics):
        netD.zero_grad()
        
        fake_img, mask = netG(dataset.sfm, feature, True)
        transform_fake_img = transform(fake_img.detach())
        fake_out = netD(fake_img.detach() * mask.detach())
        transform_fake_out = netD(transform_fake_img * mask.detach())
        cs_fake = lamb_cs_fake * pt.mean((fake_out - transform_fake_out) ** 2)
        cs_fake.backward(one)

        transform_gt = transform(gt.detach())
        transform_gt_out = netD(transform_gt * mask.detach())
        gt_out = netD(gt.detach() * mask.detach())
        cs_real = lamb_cs_real * pt.mean((gt_out - transform_gt_out) ** 2)
        cs_real.backward(one)

       
        fake_img, mask = netG(dataset.sfm, feature, True)
        fake_out = netD(fake_img.detach() * mask.detach())
        d_loss_fake = fake_out.mean()
        d_loss_fake.backward(one)
        
         # "ground-truth"
        
        real_out = netD(gt * mask.detach())
        d_loss_real = real_out.mean()
        d_loss_real.backward(mone)
        
        d_loss = - d_loss_real + d_loss_fake + cs_fake + cs_real
        
        Wasserstein_d = - d_loss_real + d_loss_fake
        pt.nn.utils.clip_grad_value_(netD.parameters(), 10)
        Doptimizer.step()
      end = timer()
      print('STEP {}'.format(1418 * epoch + jk))
      print('D loss: {}'.format(d_loss))
      #print(f'---train D elapsed time: {end - start}')
      

      start = timer()
      num_gen = 1 
      for i in range(num_gen):
        netG.zero_grad()
        
        fake_img, mask = netG(dataset.sfm, feature, True)
        g_loss = netD(fake_img * mask)
        #g_loss = args.l2 * (fake_img - gt).mean()
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        #g_loss.backward()
        fake_img, mask = netG(dataset.sfm, feature, True)
        l2_loss = 100 * pt.mean((mask * (fake_img - gt ))**2)
        l2_loss.backward()
        
        Goptimizer.step()
        g_cost = - g_loss +l2_loss
      end = timer()
     
      print('G loss: {}'.format(g_cost))
      #print(f'---train G elapsed time: {end - start}')
      print('\n')
      '''
      '''
      
      if step % 100 == 0:
        writer.add_scalar('gloss/total', g_cost, step)
        writer.add_scalar('dloss/total', d_loss, step)
        writer.add_scalar('Wasserstein distance', Wasserstein_d, step)
        #writer.add_scalar('aeloss/total', args.l2 * ae_loss, step)
        writer.add_scalar('lr', lr, step)
        #writer.add_image('images/0_recon', pt.cat([resize_gt[0], recon[0]], 1), step)

        writer.add_image('images/0_gt', pt.cat([gt[0] * mask, fake_img[0] * mask], 1), step)
        writer.add_image('images/2_mpic', make_grid(netG.new_sig[:, :3] * netG.new_sig[:, 3:], 1), step)
        writer.add_image('images/3_mpia', make_grid( netG.new_sig[:, 3:], 1), step)

      step += 1

      if step % 1000000 == 0:
        lr *= args.decaylr
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
        print("Learning rate = %f" % lr)

    if (epoch+1) % 10 == 0 or epoch == args.epochs-1:
      print("checkpointing model...")
      pt.save({
        'epoch': epoch,
        'model_state_dict': netG.state_dict(),
        'optimizer_state_dict': Goptimizer.state_dict(),
        }, runpath + args.model_dir + "/gen_ckpt.pt")
      pt.save({
        'epoch': epoch,
        'model_state_dict': netD.state_dict(),
        'optimizer_state_dict': Doptimizer.state_dict(),
        }, runpath + args.model_dir + "/dis_ckpt.pt")
  predict()

def transform(img, flags = 2):
  if flags == 1:
    shift = pt.randint(0, 10, (2, )).cuda()
    pad_img = nn.ZeroPad2d((shift[0], shift[0], shift[1], shift[1]))(img)
    crop_img = pad_img[:, :, :img.shape[2], :img.shape[3]]
    return crop_img
  elif flags == 2:
   return pt.flip(img, [3])
    
def main():
  trainGAN()
  

if __name__ == "__main__":
  sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
  main()
