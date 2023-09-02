#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 12:22:11 2023

@author: nadja
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
from torchvision.transforms import RandomCrop, RandomRotation, RandomVerticalFlip, RandomHorizontalFlip
import numpy as np
#import cv2
import scipy.io as sio
import random


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 
    
def generate_mask(image,ratio = 0.9,size_window=(5, 5)):
    'returns input without information of pixels at positions where mask is 1'
    size_data = image.shape
    num_sample = int(size_data[1] * size_data[2] * (1 - ratio))
    mask = torch.ones(size_data)
    output = torch.clone(image)

    idy_msk = torch.randint(0, size_data[1], (size_data[0], num_sample, size_data[3]))
    idx_msk = torch.randint(0, size_data[2], (size_data[0], num_sample, size_data[3]))

    idy_neigh = torch.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, (size_data[0],num_sample,size_data[3]))
    idx_neigh = torch.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, (size_data[0],num_sample,size_data[3]))

    idy_msk_neigh = idy_msk + idy_neigh
    idx_msk_neigh = idx_msk + idx_neigh
# if we are at the boundary and idy_msk_neigh<0, these errors are corrected in the following lines####
    idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[1] - (idy_msk_neigh >= size_data[1]) * size_data[1]
    idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[2] - (idx_msk_neigh >= size_data[2]) * size_data[2]

    id_msk = (idy_msk, idx_msk)
    id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)
    O = torch.zeros(size_data[0], num_sample,2,size_data[3])
    O[:,:,0]=idx_msk
    O[:,:,1]= idy_msk
    for i in range(size_data[0]):
        output[i,idx_msk,idy_msk[i],:]= image[i,idx_msk_neigh[i],idy_msk_neigh[i]]
        mask[i,idx_msk[i],idy_msk[i],:] = 0.0

    masked_input = output
    mask = 1-mask

    return masked_input, mask


'''---fidelity term that we add for denoising loss---'''
def fidelity_term(im, pred):
    fid = torch.mean((torch.dot(pred.flatten(), im.flatten())/torch.sum(pred) - im)**2 * pred)
    + torch.mean((torch.dot(1-pred.flatten(), im.flatten())/torch.sum(1-pred) - im)**2 * (1-pred))
    return fid

def augment_images(img):
    #crop = RandomCrop(size = (64,64))
   # rot = RandomRotation(180,expand = True)
    flipv = RandomVerticalFlip(0.5)
    fliph = RandomHorizontalFlip(0.5)
    #rot_ims = rot(img)
   # cropped_ims = crop(img)
    flip_ims = flipv(img)
    flip_ims = fliph(flip_ims)
    return flip_ims

def augment_images2(img,img1,img2):
    img = torch.cat((img,img1,img2), dim=0)
    #crop = RandomCrop(size = (64,64))
    #rot = RandomRotation(180,expand = True)
    flipv = RandomVerticalFlip(0.5)
    fliph = RandomHorizontalFlip(0.5)
    #rot_ims = rot(img)
   # cropped_ims = crop(img)
    flip_ims = flipv(img)
    flip_ims = fliph(flip_ims)

    return flip_ims[0:1],flip_ims[1:2], flip_ims[2:3]

def extract_random_patches(image, patch_size, N):
    #get image dimensions
    _, height, width ,_ = image.shape
    
    #calculate the number of patches that can be extracted from the image
    num_patches_x = width - patch_size + 1
    num_patches_y = height - patch_size + 1
    num_total_patches = num_patches_x*num_patches_y
    
    #generate random indices for patch locations
    patch_indices = torch.randint(low = 0, high = num_total_patches, size = (N,))
    
    #Calculate the row and column indices for each patch
    row_indices = (patch_indices // num_patches_x).long()
    col_indices = (patch_indices % num_patches_x).long()
    
    patches = torch.empty((N,patch_size,patch_size,1))
    
    #extract patches using indexing
    for i in range(N):
        patches[i] = image[0, row_indices[i]:row_indices[i]+patch_size, col_indices[i]:col_indices[i]+patch_size,:]
        patches[i] = (patches[i]-torch.mean(patches[i]))/torch.std(patches[i])

    
    return patches
    
    
def extract_random_patches2(image, image1, image2, patch_size, N):
    #get image dimensions
    _, height, width ,_ = image.shape
    
    #calculate the number of patches that can be extracted from the image
    num_patches_x = width - patch_size + 1
    num_patches_y = height - patch_size + 1
    num_total_patches = num_patches_x*num_patches_y
    
    #generate random indices for patch locations
    patch_indices = torch.randint(low = 0, high = num_total_patches, size = (N,))
    
    #Calculate the row and column indices for each patch
    row_indices = (patch_indices // num_patches_x).long()
    col_indices = (patch_indices % num_patches_x).long()
    
    patches = torch.empty((N,patch_size,patch_size,1))
    patches1 = torch.empty((N,patch_size,patch_size,1))
    patches2 = torch.empty((N,patch_size,patch_size,1))

    #extract patches using indexing
    for i in range(N):
        patches[i] = image[0, row_indices[i]:row_indices[i]+patch_size, col_indices[i]:col_indices[i]+patch_size,:]
        #patches[i] = (patches[i]-torch.mean(patches[i]))/torch.std(patches[i])
        patches1[i] = image1[0, row_indices[i]:row_indices[i]+patch_size, col_indices[i]:col_indices[i]+patch_size,:]
        patches2[i] = image2[0, row_indices[i]:row_indices[i]+patch_size, col_indices[i]:col_indices[i]+patch_size,:]    
    return patches, patches1, patches2






    
