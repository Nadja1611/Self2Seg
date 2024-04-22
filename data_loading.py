# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:09:34 2024

@author: johan
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import torch


#Work in progress, does not work yet

def load_dataset(path, resize_to = 256, noise_level = 0., channel = 'gray'):
    # function to load a dataset:
    # Can ether be a directory with npz or png files or npz or png file
    if path.endswith('.npz'):
        data = np.load(path)
        data = torch.tensor(data)
        data = data +noise_level* torch.randn_like(data)*torch.max(data)
    
    elif path.endswith('png'):
        data = plt.imread(path)
        data = torch.tensor(data)
        data = data +noise_level* torch.randn_like(data)*torch.max(data)
        
    else:
        files = os.listdidir(path)
        data_list = []
        for file in files:
            if file.endswith('.npz'):
                data = np.load(path+file)
            
            elif path.endswith('png'):
                data = plt.imread(path+file)
                
            shape = data.shape
            if shape[2] == 3:
                if channel == 0:
                    data = data[:,:,0]
                if channel == 1:
                    data = data[:,:,1]
                if channel == 2:
                    data = data[:,:,2]
                if channel == 'gray':
                    data = rgb2gray(data)
            if shape[1] != resize_to or shape[0] != resize_to:
                data = resize(data,(256,256))
            data = torch.tensor(data)
            data = data +noise_level* torch.randn_like(data)*torch.max(data)
            data_list.append(data)
        data = torch.stack(data_list)
            
            
            
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray