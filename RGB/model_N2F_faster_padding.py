

import time

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



from layer import *
from utils import *
from Functions_pytorch import *




torch.manual_seed(0)

class TwoCon(nn.Module):

    def __init__(self, in_channels, out_channels, pad=False):
        super().__init__()
        if pad == True:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect")
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode = "reflect")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode = "reflect")

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoCon(3, 128,pad = True)
        self.conv2 = TwoCon(128, 128)
        self.conv3 = TwoCon(128, 128)
        self.conv4 = TwoCon(128, 128)  
        self.conv6 = nn.Conv2d(128,3,1)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        x = self.conv6(x)
        return x




def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    loader = T.Compose([T.ToPILImage(),T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2))),T.ToTensor()])
    image = torch.tensor(image).float()
    image= loader(image)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)


class Denseg_N2F:
    def __init__(
        self,
        learning_rate: float = 1e-1,
        lam: float = 0.01,
        device: str = 'cuda:0',
        verbose = False,
    ):
        self.learning_rate = learning_rate
        self.lam = lam
        self.sigma_tv = 1/2
        self.mu = 0
        self.tau = 1/4
        self.theta = 1.0
        self.method = "joint"
        self.p = []
        self.q=[]
        self.r = []
        self.x_tilde = []
        self.device = device
        self.fid=[]
        self.tv=[]
        self.fidelity_fg = []
        self.fidelity_bg = []
        self.en = []
        self.iteration = 0
        self.f1 = None
        self.f2 = []
        self.verbose = True
        self.notdone = True
        self.net = Net().to(self.device)
        self.net1 = Net().to(self.device)
        self.mu_r2 = None
        self.optimizer1 = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.optimizer2= optim.Adam(self.net1.parameters(), lr=self.learning_rate)
        self.val_loss_list_N2F = []
        self.bg_loss_list = []
        self.number_its_N2F=500
        self.fidelity_fg_d_bg =[]
        self.fidelity_bg_d_fg = []
        self.val_loss_list_N2F_firstmask = []
        self.old_x = 0
        self.p1 = 0
        self.first_mask = 0
        self.val_loss_list_N2F_currentmask = []
        self.current_mask = 0
        self.first_loss = 4
        self.current_loss = torch.tensor(1)
        self.previous_loss = torch.tensor(2)
        self.iteration_index = 0
        self.variance = []
        self.rotation_list=[]
        self.use_filter = False
        self.ps_other = -0.1
        
    def normalize(self,f): 
        '''normalize image to range [0,1]'''
        f = f.float()
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
        return f

        
    def initialize(self,f):
        #prepare input for segmentation
        f = self.normalize(f)
        self.p = gradient(torch.clone(torch.sum(f,-1)))
        self.q = torch.clone(f)
        self.r = torch.clone(f)
        self.x_tilde = torch.clone(torch.sum(f,-1))
        self.x = torch.clone(f)
        self.f = torch.clone(f)


 ######################## Classical Chan Vese step############################################       
    def segmentation_step(self,f):
        f_orig = torch.clone(f)
        # for segmentaion process, the input should be normalized and the values should lie in [0,1]
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)

        self.p = p1.clone()
        self.q = q1.clone()
        self.r = r1.clone()
        # Update primal variables
        x_old = torch.clone(self.x)  
        self.x = proj_unitintervall(x_old + self.tau*div(p1) - self.tau*adjoint_der_Fid1(x_old, f, self.q) - self.tau *
                               adjoint_der_Fid2(x_old, f, self.r))  # proximity operator of indicator function on [0,1]
        self.x_tilde = self.x + self.theta*(self.x-x_old)
        if self.verbose == True:
            fidelity = norm1(Fid1(torch.clone(self.x), f)) + norm1(Fid2(torch.clone(self.x),f))
            total = norm1(gradient(torch.clone(self.x)))
            self.fid.append(fidelity.cpu())
            tv_p = norm1(gradient(torch.clone(self.x)))
            self.tv.append(total.cpu())
            energy = fidelity +self.lam* total
            self.en.append(energy.cpu())
          #  plt.plot(np.array(self.tv), label = "TV")
            if self.iteration %999 == 0:  
                plt.plot(np.array(self.en[:]), label = "energy")
                plt.plot(np.array(self.fid[:]), label = "fidelity")
              #  plt.plot(self.lam*np.array(self.tv[498:]))
                plt.legend()
                plt.show()
        self.iteration += 1


    def compute_energy(self,x=None):
        if x == None:
            x = torch.clone(self.x)

        diff1 = torch.sum((torch.clone(self.f-self.f1)**2).float(),-1)
        diff2 = torch.sum(((self.f - self.mu_r2)**2).float(),-1)
        energy = torch.sum(diff1*x)+ torch.sum(diff2*(1-x)) + self.lam*norm1(gradient(torch.clone(x)))
        return energy.cpu()
    
    def compute_fidelity(self,x = None):
        if x == None:
            x = torch.clone(self.x)
        diff1 = torch.sum((torch.clone(self.f-self.f1)**2).float())
        diff2 = torch.sum((torch.clone(self.f - self.mu_r2)**2).float())
        fidelity = torch.sum(diff1*x)+ torch.sum(diff2*(1-x))
        return fidelity.cpu()

##################### accelerated segmentation algorithm bg constant############################
        ''' this is the version of the accelerated segmentation step, where the reference mask'''
        ''' is added to ensure that mask does not grow to fast, default for mu = 0, then it is '''
        ''' the standard method without acceleration! '''
    def segmentation_step2denoisers_acc_bg_constant(self,f, iterations, gt):
        energy_beginning = self.compute_energy()
        f_orig = torch.clone(f).to(self.device)
        f1 = torch.clone(self.f1)

        # compute difference between noisy input and denoised image
        #compute difference between constant of background and originial noisy image
        kernel = torch.ones(1,3,5,5)/25
        kernel = kernel.to(self.device)
        kernel = kernel.to(self.device)
        # f = torch.clone(self.f)       
        diff1 = torch.clone((f_orig-f1)).float()
        n_d1 = torch.sqrt(torch.sum(diff1**2))
        diff1 = diff1
        if self.use_filter == True :
            print("filter", self.method)
            diff1 = torch.nn.functional.conv2d(diff1.movedim(3,1), kernel,padding = 2)
            diff1 = diff1.movedim(1,3)
        print('diff 1',torch.sum(diff1**2))
            
        diff2 = ((torch.clone(f_orig - self.mu_r2))).float()
        n_d2 = torch.sqrt(torch.sum(diff2**2))
        diff2 = diff2*n_d1/n_d2
        if self.use_filter == True :
            print("filter", self.method)
            diff2 = torch.nn.functional.conv2d(diff2.movedim(3,1), kernel,padding = 2)
            diff2 = diff2.movedim(1,3)
        print('diff 2', torch.sum(diff2**2))
            
        

       # diff1 = torch.nn.functional.conv2d(diff2, kernel,padding = 1)
        plt.subplot(131)
        plt.imshow(torch.sum(diff2**2,-1).cpu()[0])
        plt.colorbar()
        plt.title("diff2")
        plt.subplot(132)
        plt.imshow(torch.sum(diff1**2,-1).cpu()[0])
        plt.colorbar()
        plt.title("diff1")
        plt.subplot(133)
        plt.imshow(torch.sum(diff1**2,-1).cpu()[0]<torch.sum(diff2**2,-1).cpu()[0])
        plt.title('diff1>diff2')
        plt.show()


        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)
        self.q = q1.clone()
        self.r = r1.clone()
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        for i in range(iterations):
            self.en.append(self.compute_energy())
            p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
            # Fidelity term without norm (change this in funciton.py)
            #self.p = p1.clone()
            self.p = p1.clone()
            # Update primal variables
            self.x_old = torch.clone(self.x)  
            # constant difference term
            #filteing for smoother differences between denoised images and noisy input images
            self.x = proj_unitintervall((self.x_old + self.tau*div(p1) - self.tau*(torch.sum((diff1)**2,-1)) +  self.tau*(torch.sum((diff2)**2,-1) + self.mu*self.reference))/(1+self.tau*self.mu)) # proximity operator of indicator function on [0,1]
            ######acceleration variables
            self.theta=1/np.sqrt(1+2*self.tau*self.mu)
            self.tau=self.theta*self.tau
            self.sigma_tv = self.sigma_tv/self.theta
            ###### 
            self.x_tilde = self.x + self.theta*(self.x-self.x_old)
           # self.x = torch.round(self.x)
            if self.verbose == True:
                fidelity = self.compute_fidelity()
                fid_den = torch.sum(torch.sum((diff1)**2,-1)*self.x)
                fid_fg_denoiser_bg = (torch.sum(torch.sum((diff1)**2,-1)*(1-self.x))).cpu()
                fid_bg_denoiser_fg = (torch.sum(torch.sum((diff2)**2,-1)*(self.x))).cpu()
                self.fidelity_bg_d_fg.append(fid_bg_denoiser_fg)
                self.fidelity_fg_d_bg.append(fid_fg_denoiser_bg)
                self.fidelity_fg.append(fid_den.cpu())
                fid_const =( torch.sum(torch.sum(diff2**2,-1)*(1-self.x))).cpu()
                self.fidelity_bg.append(fid_const)
                total = norm1(gradient(self.x))
                self.fid.append(fidelity.cpu())
                tv_p = norm1(gradient(self.x))

                self.tv.append(total.cpu())
                energy = fidelity + self.lam*tv_p
                #self.en.append(energy.cpu())
                self.en.append(self.compute_energy())

                
                gt_bin = torch.clone(gt)
                gt_bin[gt_bin > 1] = 1
                seg = torch.round(torch.clone(self.x))

              #  plt.plot(np.array(self.tv), label = "TV")
            if i == 9999:  
                plt.plot(np.array(self.fidelity_fg), label = "forground_loss")
                plt.plot(np.array(self.fidelity_bg[:]), label = "background_loss")
                plt.plot(np.array(self.en[:]), label = "energy")
                plt.plot(np.array(self.fid[:]), label = "fidelity")
                plt.plot(np.array(self.tv), label = "TV")

              #  plt.plot(self.lam*np.array(self.tv[498:]))
                plt.legend()
                plt.show()


  



    def denoising_step_r1(self):
        ''' fidelity foreground is assumed to be constant '''
        f = torch.clone(self.f)
        self.f1 = torch.sum(f*torch.stack(3*[self.x],-1),dim = [0,1])/(torch.sum(self.x)+1e-5)
        print(self.f1.shape)

    def denoising_step_r2(self):
        ''' fidelity background is assumed to be constant '''
        f = torch.clone(self.f)
        self.mu_r2 = torch.sum(f*torch.stack(3*[(1-self.x)],-1), dim = [0,1])/(torch.sum(1-self.x)+1e-5)

 
      
    def preprocessing_N2F(self,f, loss_mask):
            img = f[0].cpu().numpy()#*loss_mask[0].cpu().numpy()
            img = np.expand_dims(img,axis=0)
            img = np.moveaxis(img,3,1)
            
            img_test = f[0].cpu().numpy()
            img_test = np.expand_dims(img_test,axis=0)
            img_test  = np.moveaxis(img_test,3,1)
            
            minner = np.min(img)
            img = img -  minner
            maxer = np.max(img)
            img = img/ maxer
            img = img.astype(np.float32)
            img = img[0]
            
            minner_test = np.min(img_test)
            img_test = img_test -  minner_test
            maxer_test = np.max(img_test)
            img_test = img_test/ maxer
            img_test = img_test.astype(np.float32)
            img_test = img_test[0]
    
            shape = img.shape
    
             
            listimgH_mask = []
            listimgH = []
            Zshape = [shape[1],shape[2]]
            if shape[1] % 2 == 1:
                Zshape[0] -= 1
            if shape[2] % 2 == 1:
                Zshape[1] -=1  
            imgZ = img[:,:Zshape[0],:Zshape[1]]
            imgM = loss_mask[0,:Zshape[0],:Zshape[1]].cpu().numpy()

            imgin = np.zeros((3,Zshape[0]//2,Zshape[1]),dtype=np.float32)
            imgin2 = np.zeros((3,Zshape[0]//2,Zshape[1]),dtype=np.float32)
                     
            imgin_mask = np.zeros((3,Zshape[0]//2,Zshape[1]),dtype=np.float32)
            imgin2_mask = np.zeros((3,Zshape[0]//2,Zshape[1]),dtype=np.float32)
            for i in range(imgin.shape[1]):
                for j in range(imgin.shape[2]):
                    if j % 2 == 0:
                        imgin[:,i,j] = imgZ[:,2*i+1,j]
                        imgin2[:,i,j] = imgZ[:,2*i,j]
                        imgin_mask[:,i,j] = imgM[2*i+1,j]
                        imgin2_mask[:,i,j] = imgM[2*i,j]

                    if j % 2 == 1:
                        imgin[:,i,j] = imgZ[:,2*i,j]
                        imgin2[:,i,j] = imgZ[:,2*i+1,j]
                        imgin_mask[:,i,j] = imgM[2*i,j]
                        imgin2_mask[:,i,j] = imgM[2*i+1,j]

            imgin = torch.from_numpy(imgin)
            imgin = torch.unsqueeze(imgin,0)
            imgin = imgin.to(self.device)
            imgin2 = torch.from_numpy(imgin2)
            imgin2 = torch.unsqueeze(imgin2,0)
            imgin2 = imgin2.to(self.device)
            listimgH.append(imgin)
            listimgH.append(imgin2)
            
            
            imgin_mask = torch.from_numpy(imgin_mask)
            imgin_mask = torch.unsqueeze(imgin_mask,0)
            imgin_mask = imgin_mask.to(self.device)
            imgin2_mask = torch.from_numpy(imgin2_mask)
            imgin2_mask = torch.unsqueeze(imgin2_mask,0)
            imgin2_mask = imgin2_mask.to(self.device)
            listimgH_mask.append(imgin_mask)
            listimgH_mask.append(imgin2_mask)        

            listimgV = []
            listimgV_mask=[]
            Zshape = [shape[1],shape[2]]
            if shape[1] % 2 == 1:
                Zshape[0] -= 1
            if shape[2] % 2 == 1:
                 Zshape[1] -=1  
            imgZ = img[:,:Zshape[0],:Zshape[1]]
            imgM = loss_mask[0,:Zshape[0],:Zshape[1]].cpu()
    
             
            imgin3 = np.zeros((3,Zshape[0],Zshape[1]//2),dtype=np.float32)
            imgin4 = np.zeros((3,Zshape[0],Zshape[1]//2),dtype=np.float32)
            imgin3_mask = np.zeros((3,Zshape[0],Zshape[1]//2),dtype=np.float32)
            imgin4_mask = np.zeros((3,Zshape[0],Zshape[1]//2),dtype=np.float32)
            for i in range(imgin3.shape[1]):
                for j in range(imgin3.shape[2]):
                    if i % 2 == 0:
                        imgin3[:,i,j] = imgZ[:,i,2*j+1]
                        imgin4[:,i,j] = imgZ[:,i, 2*j]
                        imgin3_mask[:,i,j] = imgM[i,2*j+1]
                        imgin4_mask[:,i,j] = imgM[i, 2*j]
                    if i % 2 == 1:
                        imgin3[:,i,j] = imgZ[:,i,2*j]
                        imgin4[:,i,j] = imgZ[:,i,2*j+1]
                        imgin3_mask[:,i,j] = imgM[i,2*j]
                        imgin4_mask[:,i,j] = imgM[i,2*j+1]
            imgin3 = torch.from_numpy(imgin3)
            imgin3 = torch.unsqueeze(imgin3,0)
            imgin3 = imgin3.to(self.device)
            imgin4 = torch.from_numpy(imgin4)
            imgin4 = torch.unsqueeze(imgin4,0)
            imgin4 = imgin4.to(self.device)
            listimgV.append(imgin3)
            listimgV.append(imgin4)
            
            
            imgin3_mask = torch.from_numpy(imgin3_mask)
            imgin3_mask = torch.unsqueeze(imgin3_mask,0)
            imgin3_mask = imgin3_mask.to(self.device)
            imgin4_mask = torch.from_numpy(imgin4_mask)
            imgin4_mask = torch.unsqueeze(imgin4_mask,0)
            imgin4_mask = imgin4_mask.to(self.device)
            listimgV_mask.append(imgin3_mask)
            listimgV_mask.append(imgin4_mask)        

    
            img = torch.from_numpy(img)
         
            img = torch.unsqueeze(img,0)
            img = img.to(self.device)
             
            listimgV1 = [[listimgV[0],listimgV[1]]]
            listimgV2 = [[listimgV[1],listimgV[0]]]
            listimgH1 = [[listimgH[1],listimgH[0]]]
            listimgH2 = [[listimgH[0],listimgH[1]]]
            listimg = listimgH1+listimgH2+listimgV1+listimgV2# + self.rotation_list
             
            listimgV1_mask = [[listimgV_mask[0],listimgV_mask[1]]]
            listimgV2_mask = [[listimgV_mask[1],listimgV_mask[0]]]
            listimgH1_mask = [[listimgH_mask[1],listimgH_mask[0]]]
            listimgH2_mask = [[listimgH_mask[0],listimgH_mask[1]]]
            listimg_mask = listimgH1_mask+listimgH2_mask+listimgV1_mask+listimgV2_mask #+ self.rotation_list_mask
            
            img_test = torch.from_numpy(img_test)
            img_test = torch.unsqueeze(img_test,0)
            img_test = img_test.to(self.device)
            return listimg, listimg_mask, img_test, maxer, minner
    
    def reinitialize_network(self):
        #self.net = Net()
        #self.net.to(self.device)
        self.optimizer1 = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        #self.net1 = Net().to(self.device)
        self.optimizer2 = optim.Adam(self.net1.parameters(), lr=self.learning_rate)
        
        
    def N2Fstep(self, mask,b_mask, region="foreground"):
        start_time = time.time()
        while self.notdone: 
            self.previous_loss = torch.clone(self.current_loss)
    
            if self.f1 == None:
                self.f1 = torch.clone(self.f)
            if self.mu_r2 == None:
                self.mu_r2 = torch.clone(self.f)
    
            f = torch.clone(self.f)
            loss_mask=torch.round(torch.clone(mask)).detach()
    
            listimg, listimg_mask, img_test, maxer, minner = self.preprocessing_N2F(f, loss_mask)
            
            running_loss1=0.0
            running_loss2=0.0
            maxpsnr = -np.inf
            timesince = 0
            last10 = []
            last10psnr = [-0.00001]*105
            last10psnr_other = [0]*105
            stop = False
            better_than_last = False
            min_epochs = 500
            epoch = 0
            #cleaned = np.zeros_like(inputs)
            
            while (stop == False and epoch<min_epochs):#((timesince < 50 or np.mean(last10psnr)<np.mean(last10psnr_other) or better_than_last == False)) and (stop == False and epoch<min_epochs):
                #print(timesince<20, np.mean(last10psnr)<np.mean(last10psnr_other),stop, better_than_last)
                #print(timesince<20 or np.mean(last10psnr)<np.mean(last10psnr_other)or better_than_last and stop == False)
                indx = np.random.randint(0,len(listimg))
                data = listimg[indx]
                data_mask = listimg_mask[indx]
                inputs = data[0]
                labello = data[1]
                loss_mask = data_mask[1]
                epoch+=1
                
                if region == "foreground":
                    self.optimizer1.zero_grad()
                    inputs = torch.nn.functional.pad(inputs,(1,1,1,1),mode = 'reflect')
                    [s0,s1,s2,s3] = inputs.shape
                    randind1 = torch.randint(s1,(1,))
                    randind2 = torch.randint(s2,(1,))
                    randind3 = torch.randint(s3,(1,))
                    const = inputs[:,randind1,randind2,randind3]
                    #inputs = inputs-const
                    outputs = self.net(inputs)
                    proportion = torch.sum(loss_mask)/torch.sum(torch.ones_like(loss_mask))
                    #labello = labello -const
                    
                    sumshape = s0+s1+s2+s3
                    loss1 = (1-proportion)*torch.nn.functional.mse_loss(outputs*loss_mask,labello*loss_mask)#+0.0001*proportion*(torch.nn.functional.mse_loss(outputs*(1-loss_mask)-labello*(1-loss_mask),(1-loss_mask)))#torch.nn.functional.binary_cross_entropy(outputs,labello,weight = loss_mask) #+ torch.sum(torch.min(self.f1)-torch.clip(outputs,max=torch.min(self.f1)))#+ 0.1*torch.sum((outputs -  torch.mean(outputs))**2)#/torch.sum(loss_mask)
                    loss1.backward()
                    self.optimizer1.step()
                    #print(loss1.item())
                    running_loss1+=loss1.item()
                    
                elif region == "background":
                    self.optimizer2.zero_grad()
                    inputs = torch.nn.functional.pad(inputs,(1,1,1,1),mode = 'reflect')
                    [s0,s1,s2,s3] = inputs.shape
                    randind1 = torch.randint(s1,(1,))
                    randind2 = torch.randint(s2,(1,))
                    randind3 = torch.randint(s3,(1,))
                    const = inputs[:,randind1,randind2,randind3]
                    #inputs = inputs-const
                    outputs = self.net1(inputs)
                    #labello = labello - 
                    proportion = torch.sum(loss_mask)/torch.sum(torch.ones_like(loss_mask))
                    
                    
                    loss1 = (1-proportion)*torch.nn.functional.mse_loss(outputs*loss_mask,labello*loss_mask)#+0.0001*proportion*(torch.nn.functional.mse_loss(outputs*(1-loss_mask)-labello*(1-loss_mask),(1-loss_mask)))#torch.nn.functional.binary_cross_entropy(outputs,labello,weight = loss_mask) #+ torch.sum(torch.min(self.f1)-torch.clip(outputs,max=torch.min(self.f1)))#+ 0.1*torch.sum((outputs -  torch.mean(outputs))**2)#/torch.sum(loss_mask)
                    loss1.backward()
                    self.optimizer2.step()
                    running_loss1+=loss1.item()
                    
                with torch.no_grad():
                    if region == "foreground":
                        inputs = torch.nn.functional.pad(img_test.detach(),(1,1,1,1),mode = 'reflect')
                        outputstest = self.net(inputs).detach()
                    elif region == "background":
                        inputs = torch.nn.functional.pad(img_test.detach(),(1,1,1,1),mode = 'reflect')
                        outputstest = self.net1(inputs).detach()

    
                    #self.current_loss = (torch.sum((outputstest[0]-img_test[0])**2*self.x)/torch.sum(self.x)).cpu()
                    # compute the loss of the denoising in the current mask
                    self.val_loss_list_N2F.append((torch.sum((outputstest[0]-img_test[0])**2*self.x)/torch.sum(self.x)).cpu())
    
                    cleaned = outputstest[0,:,:,:].cpu().detach().numpy() 
                    noisy = img_test.cpu().detach().numpy()
                    last10.append(cleaned)
    
                    ps = -np.sum((noisy-cleaned)**2*np.asarray(torch.round(mask).cpu()))/np.sum(np.asarray(torch.round(mask).cpu()))
                    ps_other = -np.sum((noisy-cleaned)**2*np.asarray(torch.round(b_mask).cpu()))/np.sum(np.asarray(torch.round(b_mask).cpu()))

                    last10psnr.pop(0)
                    last10psnr_other.pop(0)
                    last10psnr.append(ps)
                    last10psnr_other.append(ps_other)

                    
                    if ps > maxpsnr:
                        maxpsnr = ps
                        outclean = cleaned*maxer+minner
                        timesince = 0
                    else:
                        timesince+=1.0

                
                
                H = np.mean(np.array(last10), axis=0)
                mmask = np.stack(3*[mask[0].cpu()],0)
                H1 = np.asarray(1*H[:,1:-1,1:-1])
                H1 = H1[mmask[:,1:-1,1:-1]>0.5]
                


                

    
               
                if region == "foreground":
                    for g in self.optimizer1.param_groups:
                        learning_rate = g['lr'] 
                        
                elif region == "background":
                    for g in self.optimizer2.param_groups:
                        learning_rate = g['lr']  
                
                if self.ps_other == 0:
                    better_than_last = True
                elif np.mean(last10psnr) > self.ps_other:
                    better_than_last = True
                    #print(np.mean(last10psnr),self.ps_other)
                elif np.mean(last10psnr) < self.ps_other:
                    better_than_last = False
                    #print(np.mean(last10psnr),self.ps_other)  
                    if timesince>50:
                        g["lr"] /= 2
                        print("Reducing learning rate", g["lr"])
                        timesince = 0
                if epoch%100 == 0:
                    plt.plot(last10psnr, label = 'trained region')
                    plt.plot(self.ps_other*np.ones_like(last10psnr), label = 'other region last')
                    plt.plot(last10psnr_other, label = 'other region')
                    plt.legend()
                    plt.show()
               # print(np.mean(last10psnr)<np.mean(last10psnr_other), g['lr']>1e-6, timesince)
                if timesince > 50 and g["lr"]>1e-6: #and np.mean(last10psnr)>np.mean(last10psnr_other) 
                       g["lr"] /= 2
                       print("Reducing learning rate", g["lr"])
                       timesince = 0
                # if epoch%500 == 0:
                #      #g["lr"] /= 2
                #      #print("Reducing learning rate", g["lr"])
                #      timesince = 0
                if g["lr"]<1e-6:
                    stop = True
                    
                

                       
                #print(np.sum((H1-np.mean(H1))>0))
                if np.sum((H1-np.mean(H1))>0) <= 9.5 and learning_rate != 0.000005:
                    learning_rate = 0.000005
                    g["lr"]=learning_rate
                    print("Reducing learning rate", learning_rate)
                    #self.notdone = False
                else:
                    self.notdone = True
                    #print("--- %s seconds ---" % (time.time() - start_time))
                    start_time = time.time()
                #print(timesince<20 or np.mean(last10psnr)<np.mean(last10psnr_other) or better_than_last ==False)
                #print(stop==False)
                #print('fuck',(timesince<20 or np.mean(last10psnr)<np.mean(last10psnr_other) or better_than_last==False) and (stop == False))


               
            self.ps_other = -np.sum((noisy-cleaned)**2*np.asarray(torch.round(b_mask).cpu()))/np.sum(np.asarray(torch.round(b_mask).cpu()))

            self.notdone = False
            if region == "foreground":
                #if we are in foreground Region, we want to have f1 returned
                self.f1 = torch.from_numpy(H).to(self.device)
                self.f1 = self.f1.movedim(0,2)
                self.f1 = self.f1.unsqueeze(0)
                
            elif region == "background":
                #if we are in the bg region, our result is called mu_r2
                self.mu_r2 = torch.from_numpy(H).to(self.device)
                self.mu_r2 = self.mu_r2.movedim(0,2)
                self.mu_r2 = self.mu_r2.unsqueeze(0) 
            print(self.f1.shape,self.f.shape)
            self.en.append(self.compute_energy())
            print('setting back learning rate')
            #self.learning_rate /= 2
            g["lr"] = self.learning_rate
            print('I did ', epoch, ' denoising iterations')
            

