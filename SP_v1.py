import torch
from torch.utils.data import Dataset, DataLoader
from classifier import Classifier
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from localpath import getLocalPath
from skimage import io, transform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from random import shuffle
from torchvision.utils import save_image
import random

IMAGE_SIZE = 200

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu' if torch.cuda.is_available() else '??')

lp = lambda path, clone=True: path

file_dir =  lp('/home/godsom/Dataset/UTKFace/UTKFace')
PATH_TO_MODELS = lp("/home/godsom/guesswhat/models/classifier/v2_m2.pt")
transforms_list = [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
transform_img = transforms.Compose(transforms_list)

fl = os.listdir(file_dir)
random.seed(555)
random.shuffle(fl)

ir = 20021
img_name = os.path.join(file_dir, fl[ir])
image = Image.open(img_name)
image = transform_img(image)

net = Classifier().to(device)
checkpoint = torch.load(PATH_TO_MODELS)
net.load_state_dict(checkpoint['model_state_dict'])

print(net(image[None].to(device)))

# image =  torch.transpose(image, 0, 1)
# image =  torch.transpose(image, 1, 2)
# plt.imshow(image)
# plt.show()

masks = np.zeros((31*31,IMAGE_SIZE,IMAGE_SIZE))
x, y = 0, 0
for i in range(31*31):
    masks[i,y:y+50,x:x+50] = 1
    x = x + 5
    if x == 155:
        x = 0
        y = y + 5


# plt.imshow(masks[-1])
# plt.show()
# assert False

masks = masks[:,None]

masked_img = image[None].repeat(31*31,1,1,1) * masks

print(masked_img.shape)
# masked_img =  torch.transpose(masked_img, 1, 2)
# masked_img =  torch.transpose(masked_img, 2, 3)
# print(masked_img.shape)

# prob = net(masked_img.to(device).type(torch.FloatTensor).cuda())
prob = net(masked_img.to(device).type(torch.FloatTensor))
prob_am = torch.argmax(prob, 1)
print('mean: ', torch.mean(prob_am.type(torch.FloatTensor)))

prob_male = prob[:,0]
prob_female = prob[:,1]

prob_male = (prob_male - torch.mean(prob_male)) / torch.std(prob_male)
prob_female = (prob_female - torch.mean(prob_female)) / torch.std(prob_female)

prob_female = torch.reshape(prob_female, (31,31))
prob_male = torch.reshape(prob_male, (31,31))

# print(net(image[None].to(device).type(torch.FloatTensor).cuda()))
image =  torch.transpose(image, 0, 1)
image =  torch.transpose(image, 1, 2)
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.subplot(2, 2, 3)
plt.imshow(prob_male.detach().numpy() )
plt.subplot(2, 2, 4)
plt.imshow(prob_female.detach().numpy() )
plt.show()
