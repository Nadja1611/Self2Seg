# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:33:46 2024

@author: johan
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from model_N2F_faster_padding import Denseg_N2F




def noise2seg(image,
              n_iter=100,
              lam = 0.01, 
              learning_rate = 0.001,
              method = 'joint',
              initialization = 'denoise+threshold',
              device = 'cuda:0',
              use_filter = True,
              box_foreground = None,
              box_background = None,
              initialization_image = None,
              number_of_denoisers = 1,
              ):
    
    if torch.is_tensor(image) == False:
        image = torch.tensor(image).to(device)
    
    if initialization_image != None:
        if torch.is_tensor(initialization_image) == False:
            initialization_image = torch.tensor(initialization_image).to(device)
        
    mynet = Denseg_N2F(learning_rate = learning_rate, lam = lam)
    mynet.initialize(image)
    mynet.learning_rate = learning_rate
    #####  normalize such that values of f are in [0,1]
    image = mynet.normalize(image)
    ##### use filter for background, in case bg is assumed to be constant but very noisy
    if use_filter == True:
        mynet.use_filter = "yes"           

    curr_loss = []
    determine_stop = []
    lastepoch = False
    
    for i in range(n_iter):
      if method == "joint":
        mynet.iteration_index = i
        if i == 0:
            ''' two denoisers and initialization with boxes '''
            if initialization == "boxes":
                ## this method is used for the squirrel, zebra and brodatz example where thresholding is not very well working
                if box_foreground == None or box_background == None:
                    print('You need to define boxes')

                else:  
                    print("boxes are provided")
                    mynet.x = torch.tensor(box_foreground).to(device)
                    mynet.background_mask = torch.tensor(box_background).to(device)
                    plt.subplot(121)
                    plt.imshow(mynet.f[0].cpu())
                    plt.title('Initialization image')
                    plt.subplot(122)
                    plt.imshow(mynet.x.cpu()[0])
                    plt.title("Foreground box")  
                    plt.show()
                    mynet.f1 = image.to(device)
                    mynet.f1 = mynet.f1-torch.min(mynet.f1)
                    mynet.f1 = mynet.f1/torch.max(mynet.f1)


            
            if initialization == "threshold":
                '''Only threshold the input image or another given image (initialization_image)'''
                if initialization_image == None:
                    mynet.f1 = image

                else:
                    mynet.f1 = initialization_image

                mynet.f1 = mynet.f1-torch.min(mynet.f1)
                mynet.f1 = mynet.f1/torch.max(mynet.f1)

              
                ### initialization using thresholding
                mynet.x = (mynet.f1>0.5).float()
                plt.subplot(121)
                plt.imshow(mynet.f1.cpu()[0])
                plt.title('Initialization image')
                plt.subplot(122)
                plt.imshow(mynet.x.cpu()[0])
                plt.title("Segmentation with threshold 0.5")   
                plt.show()

                mynet.first_mask = torch.clone(mynet.x)
                background_mask = 1-mynet.x
                
            if initialization == "denoise+threshold":
                ''' constant background assumption and initialization with denoised images being thresholded '''
                
                '''specify the path to the denoised dataset'''
                #### explanation: We in our experiments did one iteration of denoising of the given image, and saved the denoised image. We here start with the denoised versions, do thresholding
                ###               and obtain the init. Then we go on and work with the provided noisy image
                
                mynet.x = torch.ones_like(mynet.f)
                mynet.N2Fstep(mynet.x)

                mynet.f1 = mynet.f1-torch.min(mynet.f1)
                mynet.f1 = mynet.f1/torch.max(mynet.f1)

              
                ### initialization using thresholding
                mynet.x = (mynet.f1>0.5).float()
                plt.subplot(121)
                plt.imshow(mynet.f1.cpu()[0])
                plt.title('Initialization image')
                plt.subplot(122)
                plt.imshow(mynet.x.cpu()[0])
                plt.title("Segmentation with threshold 0.5")
                plt.show()

                mynet.first_mask = torch.clone(mynet.x)
                background_mask = 1-mynet.x
                
            ''' Denoising step '''
            ### We reinit the network parameters such that it starts from scratch on each modified mask again
            mynet.reinitialize_network()
            #### denoising using noise2fast on foreground masked image
            mynet.N2Fstep(mynet.x, "foreground")
            mynet.notdone = True
            #bg denoising step for constant bg assumption
            if number_of_denoisers ==1:
                mynet.denoising_step_r2()
                mynet.notdone = True
            if number_of_denoisers == 2:
                mynet.reinitialize_network()
                mynet.N2Fstep(mynet.background_mask, "background")
                mynet.notdone = True           
            determine_stop.append(mynet.en[-100:])

        else: #meaning if i >= 1
            if i == 1:
                mynet.first_mask = torch.clone(mynet.x)
            ### we threshold the mask which is ok thresholding theorem ;)
            previous_mask = torch.round(torch.clone(mynet.x))
            mynet.reference = mynet.x
            #### first segmentation step  8000 cv iterations
            mynet.segmentation_step2denoisers_acc_bg_constant(mynet.f,8000, mynet.f)
            plt.subplot(1,2,1)
            plt.imshow(mynet.f[0].cpu())
            plt.subplot(1,2,2)
            plt.imshow(mynet.x[0].cpu())
            plt.title("Input and obtained mask at epoch "+str(i))
            plt.show()
            #here, we re-init the nw and the nw parameters and do another denoising step on update masks
            mynet.reinitialize_network()
            mynet.N2Fstep(((mynet.x>0.5).float()), "foreground")

            mynet.notdone = True
            if number_of_denoisers == 2:
                print("two denoisers are running")
                mynet.reinitialize_network()
                mynet.N2Fstep((((1-mynet.x)>0.5).float()), "background")

            if number_of_denoisers == 1:
                mynet.denoising_step_r2()
            determine_stop.append(mynet.en[-100:])
            plt.subplot(121)
            plt.imshow(mynet.f1[0].cpu())
            plt.title('Denoised image')
            plt.subplot(122)
            plt.imshow((mynet.f[0].cpu()-mynet.f1[0].cpu())**2)
            plt.title("Difference between input and denoised output")
            plt.show()
            mynet.notdone = True


            
            plt.plot(mynet.en,label = 'total energy')
            plt.legend()
            plt.show()

  
            curr_loss.append(mynet.current_loss)
            try:
                if mynet.previous_loss > mynet.current_loss  and i>1:
                    lastepoch = True

            except: #
                print('We are not done yet')
            
            if i>=0:
                if len(determine_stop) > 3:
                    print((np.mean(determine_stop[-2])/np.mean(determine_stop[-3]))/(np.mean(determine_stop[-1])/np.mean(determine_stop[-2])))

                #check if energy is still movinga a lot
                if len(determine_stop)>3 and (np.mean(determine_stop[-2])/np.mean(determine_stop[-3]))/(np.mean(determine_stop[-1])/np.mean(determine_stop[-2]))>0.85:
                    print("finished")
                    #we have to delete all saved energy values for the current lambda
                    plt.imshow(torch.round(mynet.x.cpu()[0]))
                    plt.title('Final Segmentation')
                    plt.show()
                    
                    #plt.figure()
                    return mynet.x, mynet.f1
                    
