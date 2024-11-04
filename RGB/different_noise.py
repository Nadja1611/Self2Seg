# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:23:24 2024

@author: JohannesSchwab
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.util import random_noise
import torch
from noise2seg import noise2seg
import argparse

VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description="Arguments for segmentation network.")
parser.add_argument(
    "--output_directory",
    type=str,
    help="directory for outputs",
    default="/home/nadja/Self2Seg/Self2Seg/results_test/",
)
parser.add_argument(
    "--input_directory",
    type=str,
    help="directory for input files",
    default="/home/nadja/Self2Seg/Self2Seg/data/",
)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=2e-4)
parser.add_argument(
    "--method", type=str, help="joint/sequential or only Chan Vese cv", default="joint"
)
parser.add_argument(
    "--lam", type=list, help="regularization parameter of CV", default=[0.01, 0.02]
)
parser.add_argument("--dataset", type=str, help="Which dataset", default="DSB2018_n30")
parser.add_argument(
    "--number_of_denoisers",
    type=int,
    help="1 (constant background), 2 (denoising expert for foreground and background)",
    default=2,
)
parser.add_argument(
    "--use_filter",
    type=str,
    help="if yes, then filters are applied to (\phi - f), we need this for the cell examples",
    default="yes",
)
parser.add_argument("--mu", type=float, help="parameter mu acceleration", default=0)
parser.add_argument(
    "--initialization",
    type=str,
    help="How should the segmentation be initialized? (options are threshold, denoise+threshold, box",
    default="threshold",
)
parser.add_argument(
    "--denoised_provided", type=bool, help="do we have denoised folder?", default=False
)

device = "cuda:0"
torch.manual_seed(0)
""" set necessary parameters, which initialization should be used """
""" should the differences be filtered"""
""" constant background or two denoising experts """
""" joint method """
""" which dataset should be used for the experiment """
""" which values of regularization parameter lambda  """
args = parser.parse_args()
lambdas = np.linspace(0.04, 0.08, 9)
# args.initialization = "denoise+threshold"
# args.number_of_denoisers=1
# args.method = "joint"
# args.dataset = "data_n30.npz"
# args.lam = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
# args.denoised_provided = True
# args.box_fg = None
# args.box_bg = None

args.initialization = "threshold"
args.number_of_denoisers = 2
args.method = "joint"
args.dataset = "left.jpg"
args.lam = [0.01, 0.02]
args.denoised_provided = False
args.box_fg = torch.zeros((1, 256, 256))
args.box_fg[::, -140:-110, 100:130] = 1
args.box_bg = torch.zeros((1, 256, 256))
args.box_bg[::, 10:50, 40:120] = 1

#### to do filter: kernel size 1 default
####### read in the dataset we want to denoise and segment #############

# For cell data:
if args.dataset.endswith(".npz"):
    data = np.load(args.input_directory + args.dataset)
    data = data["X_train"]
# For an arbitrary image
if args.dataset.endswith(".png") or args.dataset.endswith(".jpg"):
    ##### load brodatz image (example in paper with means of left and right half are the same but denoiser learns structure)
    data = plt.imread(args.input_directory + args.dataset)
    print(data.shape)    
    img = data[:, :300, :]
    img = resize(img, (128, 256, 3), preserve_range=True)
    minner = np.min(img)
    maxer = np.max(img)
    img_clean = np.copy(img).astype(np.uint8)
    f = torch.tensor(img/np.max(img))
    ##### add dimension
    f = f.unsqueeze(0).to(device)
    ##### create halfs with same mean
    # f[:,:,:140]=f[:,:,:140]+torch.abs(torch.mean(f[:,:,:140])-torch.mean(f[:,:,140:]))
    ## add gaussian noise
    print(f.shape)
    img_left = f 
    img_right = torch.fliplr(f) 
    img = torch.zeros(1,256,256,3).to(device)
    img_left = img_left-torch.min(img_left)
    img_clean = torch.zeros(256,256,3)
    img_clean[:128,:,:] = f
    img_clean[128:,:,:] = torch.fliplr(f)
    img_right = img_right-torch.min(img_right)
    img_left = img_left/torch.max(img_left)
    img_right = img_right/torch.max(img_right)
    img[:,:128,:,:]= torch.tensor(random_noise(img_left.cpu().numpy(),mode='gaussian',var = 0.03)).to(device)
    img[:,128:,:,:] = img_right+0.2*torch.max(img_right)*(torch.rand_like(img_right)-0.5).to(device)
    for lam in lambdas:
        seg, den = noise2seg(
            img,
            initialization=args.initialization,
            lam=lam,
            learning_rate=args.learning_rate,
            use_filter=False,
            box_foreground=args.box_fg,
            box_background=args.box_bg,
            number_of_denoisers=2,
            device=device,
        )
        den = (den-torch.min(den))
        den = den / torch.max(den)
        den = np.clip(den.cpu().numpy()*maxer + minner, 0, 255).astype(np.uint8)
        img_noisy = np.clip(img[0].cpu().numpy()*maxer+minner, 0, 255).astype(np.uint8)
        image_name = args.dataset.split(".")[0]
        print(np.max(img_noisy),np.min(img_noisy), np.max(den), np.min(den), maxer, minner)
        print(img_clean.shape)

        plt.figure()
        plt.subplot(221)
        plt.imshow(den)
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(seg[0].cpu())
        plt.axis('off')
        plt.subplot(223)
        plt.imshow(img_noisy)
        plt.axis('off')
        plt.subplot(224)
        plt.imshow(img_clean)
        plt.axis('off')
        plt.show()
        plt.savefig(
            args.output_directory
            + image_name
            + "_final_diff_lambda_"
            + str(lam)
            + ".svg"
        )

        np.savez_compressed(
            args.output_directory
            + image_name
            + "_final_diff_results_lambda_"
            + str(lam)
            + ".npz",
            seg=seg.cpu(),
            den=den,
            img=img_noisy,
            clean=f.cpu(),
        )
