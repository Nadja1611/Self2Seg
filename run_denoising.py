# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:37:33 2023

@author: nadja
"""




import os
import numpy as np
from Functions_pytorch import *
import matplotlib.pyplot as plt
import scipy
import torch
#import matplotlib.pyplot as plt
import numpy as np
from model_N2F_faster_padding import *
from utils import *
import argparse
VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--output_directory', type=str, 
                    help='directory for outputs', default="C:/Users/nadja/Documents/Cambridge/data")
parser.add_argument('--input_directory', type=str, 
                    help='directory for input files', default = "C:/Users/nadja/Documents/Cambridge/data")
parser.add_argument('--learning_rate', type=float, 
                    help='learning rate', default=0.0001)
parser.add_argument('--method', type = str, help="joint/sequential or only Chan Vese cv", default = "joint")
parser.add_argument('--lam', type = float, help = "regularization parameter of CV", default = 0.0000001)
parser.add_argument('--patient', type = int, help = "Which patient index do we use", default = 0)
parser.add_argument('--dataset', type = str, help = "Which dataset", default = "DSB2018_n30")


device = 'cuda:0'

torch.manual_seed(0)

args = parser.parse_args()
################## which dataset do we want to segment########
args.dataset = "data_n50.npz"
try:
    os.mkdir(args.input_directory[:-5]+"/results_seq_pad_"+args.dataset[5:-4])
except:
    pass
determine_stop=[]
data = np.load(args.input_directory+"/"+args.dataset)
n_it =1

for patient in range(50):
    
    data = np.load(args.input_directory+"/"+args.dataset)
    
    f = torch.tensor(data["X_train"][patient:patient+1]).to(device)
    
    try:
        os.mkdir(args.input_directory[:-5] + "/results_seq_padnew_"+args.dataset[5:-4]+"/P"+str(patient))
    except:
        pass
    
    for lam in [0.001]:
        mynet = Denseg_N2F(learning_rate = args.learning_rate, lam = args.lam)
        mynet.method = "seq"
        mynet.initialize(f)
        mynet.learning_rate=0.001
        f=mynet.normalize(f)
        for i in range(n_it):
            mynet.iteration_index = i
            if i == 0:
                mynet.x = torch.ones_like(f)
                mynet.first_mask = torch.clone(mynet.x)

                mynet.N2Fstep(mynet.x, "foreground")


                plt.imshow(mynet.f1[0].cpu())
                plt.show()
                mynet.f=torch.clone(mynet.f1)
                #### normalize i.e. [0,1] normalization the given image
                mynet.f1 = mynet.f1-torch.min(mynet.f1)
                mynet.f1 = mynet.f1/torch.max(mynet.f1)
                ### show initialization for sequential and so on
                mynet.x = (mynet.f1>0.5).float()
                plt.imshow(mynet.x.cpu()[0])
                plt.title("start")
                plt.show()
                mynet.f = torch.clone(mynet.f1)
                

                np.savez_compressed(args.input_directory[:-5]+"/results_seq_pad_"+args.dataset[5:-4]+"/P"+str(patient)+"/results_"+str(lam)+".npz", f1 = mynet.f1.cpu(), seg = mynet.x.cpu())

      
    