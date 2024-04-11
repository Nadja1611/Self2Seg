

import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.transform import resize
import os
import torch
import numpy as np
os.chdir("C:/Users/nadja/Documents/Cambridge/images")

img = plt.imread("squrirel.jpg")
img = img[:,:,1]
img = resize(img,(256,256))
img = torch.tensor(img)
f = img +0.5* torch.randn_like(img)*torch.max(img)
x = torch.zeros_like(f)
x[120:200,80:140]=1
x=np.asarray(x)
image = img_as_float(f)
# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(image, mu=0.6, lambda1=1.1, lambda2=1, tol=3e-4,
               max_num_iter=20000000, dt=0.5, init_level_set=x,
               extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv[2])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()