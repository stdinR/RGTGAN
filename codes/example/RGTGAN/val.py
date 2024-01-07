import argparse
import glob
import logging
import math
import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.archs.RGTNet_arch as RGTNet_arch
import options.options as option
import scipy.io as sio
from data import create_dataloader, create_dataset
from utils import util

from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--model_RGTNet_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')

    return parser

###############################################################################
# args = get_parser().parse_args()
# model_RGTNet_path = args.model_RGTNet_path
# exp_name = args.exp_name
# dataset = args.dataset
# save_path = args.save_path
scale = 4

exp_name = 'RGTGAN_KaggleSRD_val_1'
save_path = './results'
model_RGTNet_path = '/Path/to/model/for/example/300000_G.pth'
dataset = 'val_KaggleSRD_1'

print(dataset, exp_name)

model = RGTNet_arch.RGTNet(ngf=64, n_blocks=16)
model.load_state_dict(torch.load(model_RGTNet_path), strict=True)
model.eval()
model = model.cuda()

if dataset == 'RRSSRD_1':
    img_list = glob.glob('../dataset/val/val_1st/LR_UX4/*')
    Ref_path = '../dataset/val/val_1st/google_Ref'  # Ref
    Ref_DUX4_path = '../dataset/val/val_1st/google_Ref_DUX4'  #
    LR_path = '../dataset/val/val_1st/LR'
    HR_path = '../dataset/val/val_1st/HR'
elif dataset == 'RRSSRD_2':
    img_list = glob.glob('../dataset/val/val_2nd/LR_UX4/*')
    Ref_path = '../dataset/val/val_2nd/google_Ref'  # Ref
    Ref_DUX4_path = '../dataset/val/val_2nd/google_Ref_DUX4'  #
    LR_path = '../dataset/val/val_2nd/LR'
    HR_path = '../dataset/val/val_2nd/HR'
elif dataset == 'RRSSRD_3':
    img_list = glob.glob('../dataset/val/val_3rd/LR_UX4/*')
    Ref_path = '../dataset/val/val_3rd/google_Ref'  # Ref
    Ref_DUX4_path = '../dataset/val/val_3rd/google_Ref_DUX4'  #
    LR_path = '../dataset/val/val_3rd/LR'
    HR_path = '../dataset/val/val_3rd/HR'
elif dataset == 'RRSSRD_4':
    img_list = glob.glob('../dataset/val/val_4th/LR_UX4/*')
    Ref_path = '../dataset/val/val_4th/google_Ref'  # Ref
    Ref_DUX4_path = '../dataset/val/val_4th/google_Ref_DUX4'  #
    LR_path = '../dataset/val/val_4th/LR'
    HR_path = '../dataset/val/val_4th/HR'

elif dataset == 'KaggleSRD_1':
    img_list = glob.glob('../dataset/val/KaggleSRD_test/val_1/LR_UX4/*')
    Ref_path = '../dataset/val/KaggleSRD_test/val_1/Ref'
    Ref_DUX4_path = '../dataset/val/KaggleSRD_test/val_1/Ref_DUX4'
    LR_path = '../dataset/val/KaggleSRD_test/val_1/LR'
    HR_path = '../dataset/val/KaggleSRD_test/val_1/HR'
elif dataset == 'KaggleSRD_2':
    img_list = glob.glob('../dataset/val/KaggleSRD_test/val_2/LR_UX4/*')
    Ref_path = '../dataset/val/KaggleSRD_test/val_2/Ref'
    Ref_DUX4_path = '../dataset/val/KaggleSRD_test/val_2/Ref_DUX4'
    LR_path = '../dataset/val/KaggleSRD_test/val_2/LR'
    HR_path = '../dataset/val/KaggleSRD_test/val_2/HR'
elif dataset == 'KaggleSRD_3':
    img_list = glob.glob('../dataset/val/KaggleSRD_test/val_3/LR_UX4/*')
    Ref_path = '../dataset/val/KaggleSRD_test/val_3/Ref'
    Ref_DUX4_path = '../dataset/val/KaggleSRD_test/val_3/Ref_DUX4'
    LR_path = '../dataset/val/KaggleSRD_test/val_3/LR'
    HR_path = '../dataset/val/KaggleSRD_test/val_3/HR'
elif dataset == 'KaggleSRD_4':
    img_list = glob.glob('../dataset/val/KaggleSRD_test/val_4/LR_UX4/*')
    Ref_path = '../dataset/val/KaggleSRD_test/val_4/Ref'
    Ref_DUX4_path = '../dataset/val/KaggleSRD_test/val_4/Ref_DUX4'
    LR_path = '../dataset/val/KaggleSRD_test/val_4/LR'
    HR_path = '../dataset/val/KaggleSRD_test/val_4/HR'


util.mkdir_and_rename(osp.join(save_path,'{}_exp{}'.format(dataset, exp_name)))

PSNR_avg = 0
SSIM_avg = 0

stat_time = 0

for ii, img_path in enumerate(sorted(img_list)):
    # if '160' in img_path:
    #print('0')
    base_name = osp.splitext(osp.basename(img_path))[0]
    use_name = base_name + '.jpg'
    #print('1')
    # print(use_name)

    img_GT = cv2.imread(osp.join(HR_path,use_name))

    img_LR = cv2.imread(osp.join(LR_path,use_name)) / 255.
    img_LR = img_LR[:, :, [2, 1, 0]]

    img_LR = torch.from_numpy(
        np.ascontiguousarray(np.transpose(
            img_LR, (2, 0, 1)))).float().unsqueeze(0).cuda()

    img_LR_UX4 = cv2.imread(img_path) / 255.
    img_LR_UX4 = img_LR_UX4[:, :, [2, 1, 0]]

    img_LR_UX4 = torch.from_numpy(
        np.ascontiguousarray(np.transpose(
            img_LR_UX4, (2, 0, 1)))).float().unsqueeze(0).cuda()

    img_Ref = cv2.imread(osp.join(Ref_path,use_name)) / 255.
    img_Ref = img_Ref[:, :, [2, 1, 0]]
    img_Ref = torch.from_numpy(
        np.ascontiguousarray(np.transpose(
            img_Ref, (2, 0, 1)))).float().unsqueeze(0).cuda()

    img_Ref_DUX4 = cv2.imread(osp.join(Ref_DUX4_path,use_name)) / 255.
    img_Ref_DUX4 = img_Ref_DUX4[:, :, [2, 1, 0]]
    img_Ref_DUX4 = torch.from_numpy(
        np.ascontiguousarray(np.transpose(
            img_Ref_DUX4, (2, 0, 1)))).float().unsqueeze(0).cuda()

    with torch.no_grad():
        begin_time = time.time()
        output = model(img_LR, img_LR_UX4, img_Ref, img_Ref_DUX4)
        end_time = time.time()
        stat_time += (end_time-begin_time)


    output = util.tensor2img(output.squeeze(0))

    # save images
    save_path_name = osp.join(save_path,'{}_exp{}/{}.jpg'.format(
        dataset, exp_name, base_name))#png
    util.save_img(output, save_path_name)
  
    # calculate PSNR
    sr_img, gt_img = util.crop_border([output, img_GT], scale)
    PSNR_avg += util.calculate_psnr(sr_img, gt_img)
    SSIM_avg += util.calculate_ssim(sr_img, gt_img)

print('average PSNR: ', PSNR_avg / len(img_list))
print('average SSIM: ', SSIM_avg / len(img_list))
print('time: ', stat_time / len(img_list))

