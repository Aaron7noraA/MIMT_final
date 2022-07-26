from matplotlib.colors import to_rgba_array
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
from math import log10
from torchvision import transforms
from util.ssim import MS_SSIM

def psnr(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2)
    psnr = 20 * log10(data_range) - 10 * log10(mse.item())
    return psnr

def msssim(imgs1, imgs2, data_range=1.):
    criterion = MS_SSIM(data_range=data_range)
    return criterion(imgs1, imgs2)

def mse(image0, image1):
    return np.sum(np.square(image1 - image0)) / np.prod(image1.shape)

def mse2psnr(mse, data_range=1.):
    """PSNR for numpy mse"""
    return 20 * log10(data_range) - 10 * log10(mse)

def read_frame_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    if input_image.shape[-2:] == (1080, 1920):
        transform = transforms.CenterCrop((1024, 1920))
        input_image = transform(input_image)
        
    elif input_image.shape[-2:] == (480, 832):
        transform = transforms.CenterCrop((448, 832))
        input_image = transform(input_image)
    
    elif input_image.shape[-2:] == (240, 416):
        transform = transforms.CenterCrop((192, 384))
        input_image = transform(input_image)
    
    elif input_image.shape[-2:] == (720, 1280):
        transform = transforms.CenterCrop((704, 1280))
        input_image = transform(input_image)
    elif input_image.shape[-2:] == (1080, 2048):
        transform = transforms.CenterCrop((1024, 2048))
        input_image = transform(input_image)
        
    input_image = input_image.unsqueeze(0)/255
    return input_image

video_root = "/work/u1481110/models/torchDVC/CANFVC_Plus/CLIC_Test_CANFVC-Lite-GOP12-SSIM-debug-1024/23e266612abe7b8767587d6e77a5eb3c6b8a71c6bf4c4ff2b1c11cc478cc7244/gt_frame"
save_root = "/work/u1481110/models/torchDVC/CANFVC_Plus/CLIC_Test_CANFVC-Lite-GOP12-SSIM-debug-1024/23e266612abe7b8767587d6e77a5eb3c6b8a71c6bf4c4ff2b1c11cc478cc7244/mc_frame"

if os.path.exists('./compare.txt'):
    os.remove('./compare.txt')
    
for j in range(1, 301):
    if j % 12 == 1 :
        continue
    ori_pth = f'{video_root}/frame_{j}.png'
    rec_pth =  f'{save_root}/frame_{j}.png'
    with open('compare.txt', 'a') as fp:
        ori_frame = np.transpose(np.asarray(Image.open(ori_pth).convert('RGB'), dtype=np.float32), (2, 0, 1))
        rec_frame = np.transpose(np.asarray(Image.open(rec_pth).convert('RGB'), dtype=np.float32), (2, 0, 1))
        fp.write(f'frame {j}: mse={mse(ori_frame, rec_frame)}\n')
        fp.write(f'frame {j}: msssim={msssim(torch.from_numpy(ori_frame).unsqueeze(0), torch.from_numpy(rec_frame).unsqueeze(0)).mean().item()}\n')
