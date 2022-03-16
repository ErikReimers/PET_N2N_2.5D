import numpy as np
import cv2
import matplotlib.pyplot as plt
from generator import TrainingImageGenerator, ValGenerator
import time

gt_dir="not_specified"
#gt_dir = "GT/fr10/"
#gt_dir = "GT/Summed_label/"
#sum_label="True"
sum_label="False"
#image_dir='PET_images/Summed_label_mixed/'
image_dir='PET_images/fr10/'
#image_dir = 'PETMR_contrast_phantom/u05/'
#image_dir = 'ConnorImages/'
nb_val_images = 5
rotations="False"
batch_size = 5
image_size=128
volume_shape = (207,256,256)
#volume_shape = (89,276,276)
slices=1



generator = TrainingImageGenerator(image_dir, batch_size=batch_size, image_size=image_size,rotations=rotations,sum_label=sum_label,volume_shape=volume_shape,slices=slices)


val_generator = ValGenerator(image_dir,gt_dir=gt_dir,nb_val_images=nb_val_images,rotations=rotations,sum_label=sum_label,volume_shape=volume_shape,slices=slices)


xg,yg=generator[0]

print(len(generator))


for ii in range(3):
    
    #print('xg Min: %.3f, Max: %.3f' % (xg.min(), xg.max()))
    #print('yg Min: %.3f, Max: %.3f' % (yg.min(), yg.max()))


    xv,yv=val_generator[ii]

    #print('xv Min: %.3f, Max: %.3f' % (xv.min(), xv.max()))
    #print('yv Min: %.3f, Max: %.3f' % (yv.min(), yv.max()))


    f, axarr = plt.subplots(2,2)
    #axarr[0,0].imshow(np.squeeze(xg[ii,:,:,:]), vmin=0, vmax=np.max(xg[ii,:,:,:]) if sum_label == "True" else np.max(xg[ii,:,:,:])*0.1)
    #axarr[0,1].imshow(np.squeeze(yg[ii,:,:,:]), vmin=0, vmax=np.max(yg[ii,:,:,:]) if sum_label == "True" else np.max(yg[ii,:,:,:])*0.1)
    #axarr[1,0].imshow(np.squeeze(xv), vmin=0, vmax=np.max(xv) if sum_label == "True" else np.max(xv)*0.1)
    #axarr[1,1].imshow(np.squeeze(yv), vmin=0, vmax=np.max(yv) if sum_label == "True" or gt_dir != "not_specified" else np.max(yv)*0.1)
    
    axarr[0,0].imshow(np.squeeze(xg[ii,:,:,0]), vmin=0, vmax=np.max(xg[ii,:,:,:]))
    axarr[0,1].imshow(np.squeeze(yg[ii,:,:,0]), vmin=0, vmax=np.max(yg[ii,:,:,:]))
    axarr[1,0].imshow(np.squeeze(xv[0,:,:,0]), vmin=0, vmax=np.max(xv))
    axarr[1,1].imshow(np.squeeze(yv[0,:,:,0]), vmin=0, vmax=np.max(yv))

    #print(xv.shape)
    #print(yv.shape)
    #print(np.max(xg[ii,:,:,:]))
    #print(np.max(yg[ii,:,:,:]))

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show()
    

