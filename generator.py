from __future__ import division
from pathlib import Path
import random
import numpy as np
import cv2
import itertools
from scipy.ndimage.interpolation import rotate
import imutils
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import time
from natsort import natsorted
import matplotlib.pyplot as plt

#Grabs a random batch of noisy pairs
class TrainingImageGenerator(Sequence):
    def __init__(self, volume_dir, batch_size=8, image_size=69, rotations="True", sum_label="False",volume_shape=(207,256,256),slices=1):
        
        volume_suffixes = (".i")
        self.volume_paths = natsorted([p for p in Path(volume_dir).glob("**/*") if p.suffix.lower() in volume_suffixes], key=str)
        self.volume_nb = len(self.volume_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.volume_shape = volume_shape
        self.volume_dir = volume_dir
        self.rotations = rotations
        self.sum_label = sum_label
        self.slices = slices

        #Check if the specified folder containing the PET volumes is empty
        if self.volume_nb == 0:
            raise ValueError("volume dir '{}' does not include any volumes".format(volume_dir))

        #Figure out if cropping needs to be done. UNET requires image shape to be multiply of 16
        crop_x = volume_shape[1]//16*16
        crop_y = volume_shape[2]//16*16
        self.new_volume_shape = [self.volume_shape[0],crop_x,crop_y]

        #Save the volumes into RAM
        all_volume_data = np.zeros((np.prod(self.new_volume_shape),len(self.volume_paths)))
        for ii in range(len(self.volume_paths)):
            fid = open(self.volume_paths[ii], "r")
            #Reshape the volumes so their size is a multiple of 16
            noise_volume = np.fromfile(fid, dtype=np.float32)
            noise_volume = np.reshape(noise_volume, volume_shape)
            noise_volume = noise_volume[:,0:crop_x,0:crop_y]
            all_volume_data[:,ii] = noise_volume.flatten()
            fid.close()

        #normalize to 0->1
        all_volume_data = all_volume_data/np.max(all_volume_data)
        self.volume_tensor = all_volume_data

        #if sum_label is true, sum up all the volumes
        if self.sum_label == "True":
            self.volume_two = np.sum(self.volume_tensor,1)

        #Come up with a list of possible combinations for the data
        self.combinations = list(itertools.permutations(range(len(self.volume_paths)),2))

    #len shows how many batches fit within the number of training image pairs
    def __len__(self):
        #if sum_label is true, then the options are directly connected to the number of volumes in the folder
        if self.sum_label == "True":
            return self.volume_nb*self.new_volume_shape[0] // self.batch_size
        #But if not, then we can use all the different combinational options
        else:
            return len(list(itertools.permutations(range(len(self.volume_paths)), 2)))*self.new_volume_shape[0] // self.batch_size
    
    #getitem returns a batch worth of noisy image patch pairs
    def __getitem__(self, idx):
        
        #make the variables to hold the patches
        x = np.zeros((self.batch_size, self.image_size, self.image_size, self.slices), dtype=np.float64)
        y = np.zeros((self.batch_size, self.image_size, self.image_size, self.slices), dtype=np.float64)
        sample_id = 0
        

         
        #Keep creating patch pairs until the counter equals the batch size
        while True:
            #Choose a random combination of data
            combination = random.choice(self.combinations)

            #The first volume of the pair
            volume_one = np.reshape(self.volume_tensor[:,combination[0]], self.new_volume_shape)

            #If you want the target to be a sum, than do that (target will always be the same volume)
            if self.sum_label == "True":

                volume_two = np.reshape(self.volume_two, self.new_volume_shape)

            #otherwise the target will be the second randomly selected volume
            else:
                volume_two = np.reshape(self.volume_tensor[:,combination[1]], self.new_volume_shape)

            #Choose a random slice
            axial_slice_nb = random.choice(range(self.slices//2,self.new_volume_shape[0]-self.slices//2))
            volume_one = volume_one.transpose((1, 2, 0))
            volume_two = volume_two.transpose((1, 2, 0))
            image_one = volume_one[:,:,axial_slice_nb-self.slices//2:axial_slice_nb+1+self.slices//2]
            image_two = volume_two[:,:,axial_slice_nb-self.slices//2:axial_slice_nb+1+self.slices//2]

            #Check if both images are just all zeros, if so skip and try again
            if image_one.any() or image_two.any():
                

                #If you want to randomly rotate the dataset, do that
                if self.rotations == "True":
                    degrees = random.random()*360

                    image_one = imutils.rotate(image_one,angle=degrees)
                    image_two = imutils.rotate(image_two,angle=degrees)

                w, h, _ = image_one.shape

                #Check that the patch "image_size" is smaller than the original image itself, and create the smaller patches
                if h >= self.image_size and w >= self.image_size:

                    i = np.random.randint(h - self.image_size + 1)
                    j = np.random.randint(w - self.image_size + 1)
                    patch_one = image_one[i:i + self.image_size, j:j + self.image_size,:]
                    patch_two = image_two[i:i + self.image_size, j:j + self.image_size,:]

                    #check that both patches aren't all zeros
                    if patch_one.any() or patch_two.any():
                        x[sample_id,:,:,:] = patch_one
                        y[sample_id,:,:,:] = patch_two

                        sample_id += 1
                        
                        #Once the counter reaches the batch_size return the set of noisy pairs
                        if sample_id == self.batch_size:
                            return x,y

#Grabs all the volumes and returns them as matching validation image pairs
class ValGenerator(Sequence):
    #Initalize the class
    def __init__(self, volume_dir, gt_dir="not_specified", nb_val_images=32, rotations="True",sum_label="True",volume_shape=(207,256,256),slices=1):     
        volume_suffixes = (".i")
        self.volume_paths = natsorted([p for p in Path(volume_dir).glob("**/*") if p.suffix.lower() in volume_suffixes])
        self.gt_paths = natsorted([p for p in Path(gt_dir).glob("**/*") if p.suffix.lower() in volume_suffixes])
        self.volume_shape = volume_shape
        self.volume_nb = len(self.volume_paths)
        self.data = []
        self.rotations = rotations
        self.sum_label = sum_label
        self.nb_val_images = nb_val_images
        self.volume_dir = volume_dir
        self.slices = slices

        #Check if the specified folder is empty
        if self.volume_nb == 0:
            raise ValueError("image dir '{}' does not include any volumes".format(volume_dir))

        #if a ground truth directory is specified, check if that's empty
        if gt_dir != "not_specified":
            gt_volume_nb = len(self.gt_paths)
            if gt_volume_nb == 0:
                raise ValueError("gt image dir '{}' does not include any volumes".format(gt_dir))

        #Figure out if cropping needs to be done. UNET requires image shape to be multiply of 16
        crop_x = volume_shape[1]//16*16
        crop_y = volume_shape[2]//16*16
        self.new_volume_shape = [self.volume_shape[0],crop_x,crop_y]

        #Save the volumes in ram
        all_volume_data = np.zeros((np.prod(self.new_volume_shape),len(self.volume_paths)))
        for ii in range(len(self.volume_paths)):
            
            fid = open(self.volume_paths[ii], "r")
            #Shape the volumes to make their shape a multiple of 16
            noise_volume = np.fromfile(fid, dtype=np.float32)
            noise_volume = np.reshape(noise_volume, volume_shape)
            noise_volume = noise_volume[:,0:crop_x,0:crop_y]
            all_volume_data[:,ii] = noise_volume.flatten()
            fid.close()

        #Normalize to 0->1
        all_volume_data = all_volume_data/np.max(all_volume_data)
        self.volume_tensor = all_volume_data

        #if sum_label is true, sum up all the volumes in folder
        if self.sum_label == "True":
            self.volume_two = np.sum(self.volume_tensor,1)

        #if ground_truth is given add that into ram
        if gt_dir != "not_specified":
            all_gt_data = np.zeros((np.prod(self.new_volume_shape),len(self.gt_paths)))
            for ii in range(len(self.gt_paths)):
                fid = open(self.gt_paths[ii], "r")
                #Crop to be multiple of 16
                gt_volume = np.fromfile(fid, dtype=np.float32)
                gt_volume = np.reshape(gt_volume, volume_shape)
                gt_volume = gt_volume[:,0:crop_x,0:crop_y]
                all_gt_data[:,ii] = gt_volume.flatten()
                fid.close()
            #Normalize to 0->1
            all_gt_data = all_gt_data/np.max(all_gt_data)
            self.gt_tensor = all_gt_data

        #Check if you want to label to always just be the sum of all volumes
        if self.sum_label == "True" and gt_dir != "not_specified":
            self.volume_two = np.sum(self.gt_tensor,1)
 
        #list the possible combinations
        combinations = list(itertools.permutations(range(len(self.volume_paths)),2))

        #Run through each validation pair and add them to the self.data
        for ii in range(nb_val_images):
            
            #Pick a random combination
            combination = random.choice(combinations)

            #Open the first volume
            volume_one = np.reshape(self.volume_tensor[:,combination[0]], self.new_volume_shape)

            #If sum_label is true, then the second label volume will be a summation of the volumes
            if self.sum_label == "True":
                    volume_two = np.reshape(self.volume_two, self.new_volume_shape)

            #If not doing a summation, then just use the second randomly selected volume
            else:
                if gt_dir != "not_specified":
                    volume_two = np.reshape(self.gt_tensor, self.new_volume_shape)
                else:
                    volume_two = np.reshape(self.volume_tensor[:,combination[1]], self.new_volume_shape)

            volume_one = volume_one.transpose((1, 2, 0))
            volume_two = volume_two.transpose((1, 2, 0))

            image_one = np.zeros((self.new_volume_shape[1],self.new_volume_shape[2],slices),dtype=np.float64)
            image_two = np.zeros((self.new_volume_shape[1],self.new_volume_shape[2],slices),dtype=np.float64)

            #Keep grabing random slices until it finds a non-zero slice
            while not image_one.any() or not image_two.any():
                axial_slice_nb = random.choice(range(self.slices//2,self.volume_shape[0]-self.slices//2))
                image_one = volume_one[:,:,axial_slice_nb-self.slices//2:axial_slice_nb+1+self.slices//2]
                image_two = volume_two[:,:,axial_slice_nb-self.slices//2:axial_slice_nb+1+self.slices//2]

            #Add the rotations
            if self.rotations == "True":
                degrees = random.random()*360
                image_one = imutils.rotate(image_one,angle=degrees)
                image_two = imutils.rotate(image_two,angle=degrees)

            x = np.expand_dims(image_one.astype(np.float64),0)
            y = np.expand_dims(image_two.astype(np.float64),0)

            self.data.append([x, y])     
            #self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)]) 

    #len will be the number of validation pairs
    def __len__(self):
        return self.nb_val_images

    #getitem will return the self.data of the index specified
    def __getitem__(self, idx):
        return self.data[idx]
        

