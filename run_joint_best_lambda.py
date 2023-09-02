# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:44:33 2023

@author: nadja
"""



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
from model_N2F_faster_padding import *
from utils import *
from torch.optim import Adam
import time
import argparse
VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--output_directory', type=str, 
                    help='directory for outputs', default="C:/Users/nadja/Documents/Cambridge/data")
parser.add_argument('--input_directory', type=str, 
                    help='directory for input files', default = "C:/Users/nadja/Documents/Cambridge/data")
parser.add_argument('--learning_rate', type=float, 
                    help='learning rate', default=0.001)
parser.add_argument('--method', type = str, help="joint/sequential or only Chan Vese cv", default = "joint")
parser.add_argument('--lam', type = float, help = "regularization parameter of CV", default = 0.0000001)
parser.add_argument('--ratio', type = int, help = "What is the ratio of masked pixels in N2v", default = 0.3)
parser.add_argument('--experiment', type = str, help = "What hyperparameter are we looking at here? Assigns the folder we want to produce with Lambda, if we make lambda tests for example", default = "/Lambda")
parser.add_argument('--patient', type = int, help = "Which patient index do we use", default = 30)
parser.add_argument('--dataset', type = str, help = "Which dataset", default = "DSB2018_n30")
parser.add_argument('--fid', type = float, help = "do we add fidelity term?", default = 0.0)


device = 'cuda:0'

torch.manual_seed(0)

args = parser.parse_args()
os.chdir("C:/Users/nadja/Documents/Cambridge/data")

args.dataset = "data_n50.npz"
noise = "50"
determine_stop=[]






data = np.load("C:/Users/nadja/Documents/Cambridge/data/"+args.dataset)

f = data["X_train"]

mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)

mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
mynet.initialize(f)
f=mynet.normalize(f)
n_it =100
args.method = "joint"
cols = []
first_loss = []
curr_loss = []
median = torch.median(f)
lastepoch = False
if args.method == "joint" or args.method == "cv":
    for patient in range(32,50):
        
        data = np.load("C:/Users/nadja/Documents/Cambridge/data/"+args.dataset)
        
        f = torch.tensor(data["X_train"][patient:patient+1]).to(device)
        
        try:
            os.mkdir("C:/Users/nadja/Documents/Cambridge/Results/results_joint_best_lambda_"+args.dataset[5:-4]+"/P"+str(patient))
        except:
            pass
        for lam in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]:
        #for lam in [0.0075, 0.015]:
            
            print(lam)
            mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
            mynet.initialize(f)
            mynet.learning_rate=0.001
            f=mynet.normalize(f)
            
            args.method = "joint"
            cols = []
            first_loss = []
            curr_loss = []
            median = torch.median(f)
            lastepoch = False
            mynet.lam = lam
            for i in range(n_it):
              if args.method == "joint":
                mynet.iteration_index = i
                if i == 0:
                    
                   
                    mynet.x = torch.ones_like(mynet.f)
                    mynet.N2Fstep2(mynet.x, "foreground")
                    mynet.notdone = True
    
                    denoised = torch.clone(mynet.f2)
                    denoised = denoised-torch.min(denoised)
                    denoised = denoised/torch.max(denoised)
                    mynet.x = (denoised>0.5).float() 
                   # mynet.x = (mynet.f>0.6).float()
                    mynet.first_mask = torch.clone(mynet.x)
                    background_mask = 1-mynet.x
                    
                    #cell denoising step
                    mynet.reinitialize_network()
                    mynet.N2Fstep(mynet.x, "foreground")
                    mynet.notdone = True
                    #bg denoising step
                    mynet.denoising_step_r2()
                    mynet.notdone = True
                    
                    determine_stop.append(mynet.en[-100:])
    
    
                    plt.imshow(mynet.x.cpu()[0])
                    plt.title("init mask")
                    plt.show()
                    
      
    
    
                    
        
                else:
                    if i == 1:
                        mynet.first_mask = torch.clone(mynet.x)
                    previous_mask = torch.round(torch.clone(mynet.x))
                    mynet.reference = mynet.x
                    mynet.segmentation_step2denoisers_acc_bg_constant(mynet.f,8000, mynet.f)
                    plt.subplot(1,2,1)
                    plt.imshow(f[0].cpu())
                    plt.subplot(1,2,2)
                    plt.imshow(mynet.x[0].cpu())
                    plt.show()
                    #here, we re-init the nw and the nw parameters
                    mynet.reinitialize_network()
                    mynet.N2Fstep(((mynet.x>0.5).float()), "foreground")
    
                    mynet.notdone = True
                    mynet.denoising_step_r2()
                    determine_stop.append(mynet.en[-100:])
    
                    plt.imshow((mynet.f[0].cpu()-mynet.f1[0].cpu())**2)
                    plt.show()
                    mynet.notdone = True
        
                    # plt.plot(mynet.fidelity_fg, label = "foreground_loss")
                    # plt.plot(mynet.fidelity_bg[:], label = "background_loss")
                    # plt.plot(mynet.tv, label = "TV")
                    # plt.plot(mynet.fidelity_fg_d_bg, label = "fg denoiser on bg")
                    # plt.plot(mynet.fidelity_bg_d_fg, label = "bg denoiser on fg")
                    # plt.plot(mynet.difference, label = "fg loss-bg loss on whole image")
        
                    # plt.legend()
                    # plt.show()
                    
                    plt.plot(mynet.en,label = 'total energy')
                    plt.legend()
                    plt.show()
        
          
        
                    first_loss.append(mynet.previous_loss)
                    curr_loss.append(mynet.current_loss)
                    try:
                        if curr_loss[-1] > curr_loss[-2]  and i>1:
                            lastepoch = True
        
                    except: #
                        print('Ich bin Jake')
                    
                    if i>-1:
                        if len(determine_stop) > 3:
                            print((np.mean(determine_stop[-2])/np.mean(determine_stop[-3]))/(np.mean(determine_stop[-1])/np.mean(determine_stop[-2])))
    
                        #check if energy is still movinga a lot
                        if len(determine_stop)>3 and (np.mean(determine_stop[-2])/np.mean(determine_stop[-3]))/(np.mean(determine_stop[-1])/np.mean(determine_stop[-2]))>0.85:
                            print("finished")
                            #we have to delete all saved energy values for the current lambda
                            determine_stop=[]
                            #plt.figure()
                            plt.imshow(torch.round(mynet.x.cpu()[0]))
                            plt.savefig("C:/Users/nadja/Documents/Cambridge/Results/results_joint_best_lambda_"+args.dataset[5:-4]+"/img_"+str(patient)+"_"+str(lam)+".png")
                            np.savez_compressed("C:/Users/nadja/Documents/Cambridge/Results/results_joint_best_lambda_"+args.dataset[5:-4]+"/P"+str(patient)+"/results_"+str(lam)+".npz", f1 = mynet.f1.cpu(), seg = mynet.x.cpu())
                            break
                        
                        
    
