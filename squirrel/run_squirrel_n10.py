



import os
import numpy as np
from math import sqrt
from Functions_pytorch import *
import matplotlib.pyplot as plt
from PIL import Image                                                                 
import imageio
from skimage.color import rgb2gray
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray
import torch
#import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model_N2F_every4 import *
from utils import *
from torch.optim import Adam
import time
import argparse
VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--output_directory', type=str, 
                    help='directory for outputs', default="C:/Users/nadja/Documents/Cambridge/images")
parser.add_argument('--input_directory', type=str, 
                    help='directory for input files', default = "C:/Users/nadja/Documents/Cambridge/images")
parser.add_argument('--learning_rate', type=float, 
                    help='learning rate', default=0.0005)
parser.add_argument('--method', type = str, help="joint/sequential or only Chan Vese cv", default = "joint")
parser.add_argument('--lam', type = float, help = "regularization parameter of CV", default = 0.0000001)
parser.add_argument('--ratio', type = int, help = "What is the ratio of masked pixels in N2v", default = 0.3)
parser.add_argument('--experiment', type = str, help = "What hyperparameter are we looking at here? Assigns the folder we want to produce with Lambda, if we make lambda tests for example", default = "/Lambda")
parser.add_argument('--patient', type = int, help = "Which patient index do we use", default = 0)
parser.add_argument('--dataset', type = str, help = "Which dataset", default = "DSB2018_n20")
parser.add_argument('--fid', type = float, help = "do we add fidelity term?", default = 0.0)


device = 'cuda:0'

torch.manual_seed(0)

args = parser.parse_args()
os.chdir("C:/Users/nadja/Documents/Cambridge/images")



args.lam=0.005
determine_stop=[]
img = plt.imread("squrirel.jpg")
img = img[:,:,1]
img = resize(img,(256,256))
f = torch.tensor(img)
f = f 
f = f.unsqueeze(0).to(device)
f = f +0.1* torch.randn_like(f)*torch.max(f)
gt = torch.tensor(f).to(device)



mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)
mynet.mu = 0.0001
n_it =8
args.method = "joint"
cols = []
first_loss = []
curr_loss = []
median = torch.median(f)
lastepoch = False
if args.method == "joint" or args.method == "cv":
    for i in range(n_it):
      if args.method == "joint":
        mynet.iteration_index = i
        if i == 0:
            plt.imshow(mynet.f[0].cpu())
            plt.show()
            mynet.x = torch.zeros_like(mynet.f)
            mynet.x[:,120:200,80:140]=1
         #   mynet.x[:,30:60,10:30]=1

          #  mynet.x = torch.tensor(gt_mask[:,0]).to(device)
            #mynet.x[:,100:400,100:400:]=1
            mynet.first_mask = torch.clone(mynet.x).to(device)
           # background_mask = 1-torch.tensor(gt_mask[:,0,:,:]).to(device)
            background_mask = torch.zeros_like(mynet.f)
            background_mask[:,20:230,-70:-20]=1

            plt.subplot(1,2,1)
            plt.imshow(mynet.x.cpu()[0])
            plt.subplot(1,2,2)
            plt.imshow(background_mask.cpu()[0])
            plt.show()
            mynet.N2Fstep(mynet.x, "foreground")
            mynet.notdone = True
            mynet.N2Fstep(background_mask, "background")
            mynet.notdone = True
            determine_stop.append(mynet.en[-100:])
            
        else:
            if i == 1:
                mynet.first_mask = torch.clone(mynet.x)
            previous_mask = torch.round(torch.clone(mynet.x))
            mynet.reference = mynet.x
            mynet.segmentation_step2denoisers_acc_bg_constant(mynet.f,8000, mynet.f)
            mynet.reinitialize_network()
            plt.imshow(mynet.x[0].cpu())
            plt.show()
            mynet.N2Fstep(((mynet.x>0.5).float()), "foreground")
            mynet.mu = mynet.mu*2
            mynet.notdone = True
            mynet.N2Fstep(1-(mynet.x>0.5).float(), "background")
            determine_stop.append(mynet.en[-100:])

            mynet.notdone = True

            plt.plot(mynet.fidelity_fg, label = "foreground_loss")
            plt.plot(mynet.fidelity_bg[:], label = "background_loss")
            plt.plot(mynet.tv, label = "TV")
            plt.plot(mynet.fidelity_fg_d_bg, label = "fg denoiser on bg")
            plt.plot(mynet.fidelity_bg_d_fg, label = "bg denoiser on fg")
            plt.plot(mynet.difference, label = "fg loss-bg loss on whole image")

            plt.legend()
            plt.show()
            
            plt.plot(mynet.en,label = 'total energy')
            plt.legend()
            plt.show()

  
        if i>-1:

            first_loss.append(mynet.previous_loss)
            curr_loss.append(mynet.current_loss)
            try:
                if curr_loss[-1] > curr_loss[-2]  and i>1:
                    lastepoch = True

            except: #
                print('Ich bin Jake')
            
        if i>-1:
            #check if energy is still movinga a lot
            if len(determine_stop)>12 and np.mean(determine_stop[-1])/np.mean(determine_stop[-2])>0.9:
                print("finished")
                break
            

        if i%1 == 0:
            plt.subplot(3,2,1)
            plt.imshow(mynet.x[0].cpu())
            plt.colorbar()
            plt.subplot(3,2,2)
            plt.imshow(f[0].cpu())
            plt.colorbar()
            plt.subplot(3,2,3)
            plt.imshow(mynet.f1[0].cpu(),cmap ='inferno')
            plt.colorbar()
            plt.subplot(3,2,4)
            plt.imshow(mynet.mu_r2[0].cpu(),cmap ='inferno')
            plt.colorbar()    
            plt.subplot(3,2,5)
            plt.imshow(torch.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
            plt.colorbar()   
            plt.subplot(3,2,6)
            plt.imshow(torch.abs(mynet.mu_r2[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
            plt.colorbar()   
            plt.show()


            plt.plot(mynet.val_loss_list_N2F, label = "mean denoising performance in current mask")
            plt.title("Denoiser on current mask")
            plt.legend()
            plt.show()


            plt.figure(figsize=(20,10))
            plt.subplot(2,1,1)
            plt.imshow(mynet.net.conv1.conv1.weight[0][0].detach().cpu())
            plt.title("net1")
            plt.subplot(2,1,2)
            plt.imshow(mynet.net1.conv1.conv1.weight[0][0].detach().cpu())
            plt.title("net2")
           

            plt.show()
            
summe = torch.round(mynet.x.cpu()[0])*mynet.f1[0].cpu()+ (1-torch.round(mynet.x[0].cpu()))*mynet.mu_r2[0].cpu()           
############# here, we compute psnr and ssim
def compute_psnr_normalized(f,denoised):
    a = np.linspace(-0.4,0.4,100)
    b = np.linspace(0.1,4,100)
    A, B = np.meshgrid(a,b)
    err = torch.zeros(100,100)

    for i in range(100):
        for j in range(100):
            err[i,j] = torch.mean((f[0].cpu()-(denoised.cpu()-A[i,j])/B[i,j])**2)

    indi = torch.argmin(err)
    in1 = indi.div(100, rounding_mode = 'floor')
    in2 = indi % 100
    
    new = (denoised.cpu()-A[in1,in2])/B[in1,in2]
    psnr = skimage.metrics.peak_signal_noise_ratio(f[0].cpu().numpy(),new.cpu().numpy(), data_range=None)
    return psnr




#### denoise on whole image
os.chdir("C:/Users/nadja/Documents/Cambridge/images")



args.lam=0.06
determine_stop=[]



args.patient= 44




img = plt.imread("squrirel.jpg")
img = img[:,:,1]
img = resize(img,(256,256))
f = torch.tensor(img)
f = f 
f = f.unsqueeze(0).to(device)
f[:,:,:140]=f[:,:,:140]+torch.abs(torch.mean(f[:,:,:140])-torch.mean(f[:,:,140:]))
f = f +0.1* torch.randn_like(f)*torch.max(f)
gt = torch.tensor(f).to(device)

mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)

mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)
mynet.mu = 0.000
n_it =3
args.method = "joint"
cols = []
first_loss = []
curr_loss = []
median = torch.median(f)
lastepoch = False
if args.method == "joint" or args.method == "cv":
    for i in range(n_it):
      if args.method == "joint":
        mynet.iteration_index = i
        if i == 0:
            plt.imshow(mynet.f[0].cpu())
            plt.show()
            mynet.x = torch.ones_like(mynet.f)
         #   mynet.x[:,30:60,10:30]=1

          #  mynet.x = torch.tensor(gt_mask[:,0]).to(device)
            #mynet.x[:,100:400,100:400:]=1
            mynet.first_mask = torch.clone(mynet.x).to(device)
           # background_mask = 1-torch.tensor(gt_mask[:,0,:,:]).to(device)


            mynet.N2Fstep(mynet.x, "foreground")
            mynet.notdone = True

summe = torch.round(mynet.x.cpu()[0])*mynet.f1[0].cpu()+ (1-torch.round(mynet.x[0].cpu()))*mynet.mu_r2[0].cpu()           

def compute_psnr_normalized(f,denoised):
    a = np.linspace(-0.4,0.4,100)
    b = np.linspace(0.1,4,100)
    A, B = np.meshgrid(a,b)
    err = torch.zeros(100,100)

    for i in range(100):
        for j in range(100):
            err[i,j] = torch.mean((f[0].cpu()-(denoised.cpu()-A[i,j])/B[i,j])**2)

    indi = torch.argmin(err)
    in1 = indi.div(100, rounding_mode = 'floor')
    in2 = indi % 100
    
    new = (denoised.cpu()-A[in1,in2])/B[in1,in2]
    psnr = skimage.metrics.peak_signal_noise_ratio(f[0].cpu().numpy(),new.cpu().numpy(), data_range=None)
    return psnr




#### denoise on whole image
os.chdir("C:/Users/nadja/Documents/Cambridge/images")



args.lam=0.06
determine_stop=[]



args.patient= 44




img = plt.imread("brodi.png")
img = img[:,:,1]
img = resize(img,(256,256))
f = torch.tensor(img)
f = f 
f = f.unsqueeze(0).to(device)
f[:,:,:140]=f[:,:,:140]+torch.abs(torch.mean(f[:,:,:140])-torch.mean(f[:,:,140:]))
f = f +0.1* torch.randn_like(f)*torch.max(f)
gt = torch.tensor(f).to(device)

mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)

mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)
mynet.mu = 0.000
n_it =1
args.method = "joint"
cols = []
first_loss = []
curr_loss = []
median = torch.median(f)
lastepoch = False
if args.method == "joint" or args.method == "cv":
    for i in range(n_it):
      if args.method == "joint":
        mynet.iteration_index = i
        if i == 0:
            plt.imshow(mynet.f[0].cpu())
            plt.show()
            mynet.x = torch.ones_like(mynet.f)
         #   mynet.x[:,30:60,10:30]=1

          #  mynet.x = torch.tensor(gt_mask[:,0]).to(device)
            #mynet.x[:,100:400,100:400:]=1
            mynet.first_mask = torch.clone(mynet.x).to(device)
           # background_mask = 1-torch.tensor(gt_mask[:,0,:,:]).to(device)


            mynet.N2Fstep(mynet.x, "foreground")
            mynet.notdone = True

img = plt.imread("brodi.png")
img = img[:,:,1]
img = resize(img,(256,256))
f = torch.tensor(img)
f = f 
f = f.unsqueeze(0).to(device)
f[:,:,:140]=f[:,:,:140]+torch.abs(torch.mean(f[:,:,:140])-torch.mean(f[:,:,140:]))

psnr_prop = compute_psnr_normalized(f,summe)
psnr_only = compute_psnr_normalized(f, mynet.f1[0].cpu())

print('PSNR proposed method is: ', psnr_prop)
print('PSNR denoising only is: ', psnr_only)


f_norm = f[0]-torch.min(f[0])
f_norm = f_norm/torch.max(f_norm)

summe_norm = summe-torch.min(summe)
summe_norm = summe_norm/torch.max(summe_norm)

glob_norm = mynet.f1[0]-torch.min(mynet.f1[0])
glob_norm = glob_norm/torch.max(glob_norm)

mssim, S = skimage.metrics.structural_similarity(f_norm.cpu().numpy(), summe_norm.cpu().numpy(),win_size = 7, data_range = 1., full = True)
mssim1, S1 = skimage.metrics.structural_similarity(f_norm.cpu().numpy(), glob_norm.cpu().numpy(), win_size = 7, data_range = 1., full = True)


print('SSIM proposed is: ', mssim)
print('SSIM denoising only is: ', mssim1)