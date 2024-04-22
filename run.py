import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
from noise2seg import noise2seg
import argparse
VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--output_directory', type=str, 
                    help='directory for outputs', default="C:/Users/nadja/Documents/Cambridge/results")
parser.add_argument('--input_directory', type=str, 
                    help='directory for input files', default = "C:/Users/nadja/Documents/Cambridge/data/")
parser.add_argument('--learning_rate', type=float, 
                    help='learning rate', default=0.001)
parser.add_argument('--method', type = str, help="joint/sequential or only Chan Vese cv", default = "joint")
parser.add_argument('--lam', type = list, help = "regularization parameter of CV", default =[0.01,0.02])
parser.add_argument('--dataset', type = str, help = "Which dataset", default = "DSB2018_n30")
parser.add_argument('--number_of_denoisers', type = int, help="1 (constant background), 2 (denoising expert for foreground and background)", default = 2)
parser.add_argument('--use_filter', type = str, help="if yes, then filters are applied to (\phi - f), we need this for the cell examples", default = "yes")
parser.add_argument('--mu', type = float, help="parameter mu acceleration", default = 0)
parser.add_argument('--initialization', type = str, help="How should the segmentation be initialized? (options are threshold, denoise+threshold, box", default = "threshold")
parser.add_argument('--denoised_provided', type = bool, help="do we have denoised folder?", default = False)

device = 'cuda:0'
torch.manual_seed(0)
''' set necessary parameters, which initialization should be used '''
''' should the differences be filtered'''
''' constant background or two denoising experts '''
''' joint method '''
''' which dataset should be used for the experiment '''
''' which values of regularization parameter lambda  '''
args = parser.parse_args()
# args.initialization = "denoise+threshold"
# args.number_of_denoisers=1
# args.method = "joint"
# args.dataset = "data_n30.npz"
# args.lam = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
# args.denoised_provided = True
# args.box_fg = None
# args.box_bg = None

args.initialization = "boxes"
args.number_of_denoisers=2
args.method = "joint"
args.dataset = "brodi.png"
args.lam = [0.01,0.02]
args.denoised_provided = False
args.box_fg = torch.zeros((1,256,256))
args.box_fg[::,10:60,10:70] = 1
args.box_bg = torch.zeros((1,256,256))
args.box_bg[::,10:60,140:200] = 1

#### to do filter: kernel size 1 default
####### read in the dataset we want to denoise and segment #############

# For cell data:
if args.dataset.endswith(".npz"):
    data = np.load(args.input_directory+args.dataset)
    data = data['X_train']
# For an arbitrary image    
if args.dataset == "brodi.png":
    ##### load brodatz image (example in paper with means of left and right half are the same but denoiser learns structure)
    data = plt.imread(args.input_directory + "brodi.png")
    img = data[:,:,1]
    img = resize(img,(256,256))
    f = torch.tensor(img)
    ##### add dimension
    f = f.unsqueeze(0).to(device)
    ##### create halfs with same mean
    f[:,:,:140]=f[:,:,:140]+torch.abs(torch.mean(f[:,:,:140])-torch.mean(f[:,:,140:]))
    ## add gaussian noise
    img = f +0.1* torch.randn_like(f)*torch.max(f)
    seg, den = noise2seg(img, initialization=args.initialization, lam = 0.02, box_foreground = args.box_fg, box_background = args.box_bg, number_of_denoisers=2)


for j in range(data.shape[0]):
    j=3
    img = data[j:j+1]
    seg, den = noise2seg(img, lam = 0.04)
                            
                        
    
        
   