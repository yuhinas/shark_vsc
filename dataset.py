import torch
import torch.utils.data as data

#import random
import numpy as np
import torchvision.transforms as transforms

import skimage.io as io
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
most_of_the_time = lambda aug: iaa.Sometimes(0.9, aug)
usually = lambda aug: iaa.Sometimes(0.75, aug)
always = lambda aug: iaa.Sometimes(1, aug)
charm = lambda aug: iaa.Sometimes(0.33, aug)
seldom = lambda aug: iaa.Sometimes(0.2, aug)

augseq_all = iaa.Sequential([
    #iaa.Fliplr(0.5),
    #iaa.Flipud(0.5),
    usually(iaa.Affine(
            scale={"x": (.9, 1.2), "y": (.9, 1.2)}, # scale images to 90-100% of their size, individually per axis
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -10 to +10 percent (per axis)
            rotate=(-5,5), # rotate by -10 to +10 degrees
            mode='wrap',
            #cval=127,
        )),
    #sometimes(iaa.Affine(
    #        rotate=(90,90), # rotate by -10 to +10 degrees
    #        mode='wrap',
    #        #cval=127,
    #    )),
    iaa.Multiply((0.9, 1.1)),
    iaa.ContrastNormalization((0.9, 1.1)),
])
      
class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_path,
                input_height=128, input_width=128, output_height=128, output_width=128,
                aug=True):
        
        super(ImageDatasetFromFile, self).__init__()
                
        self.image_filenames = image_list 
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.aug = aug
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
          
        img = io.imread(self.image_filenames[index])
        if self.aug:
            if len(img.shape) < 3:
                img = img.reshape(img.shape[0], img.shape[1], 1)
                img = np.repeat(img, 3, axis=2)
            img = augseq_all.augment_images([img])
        else:
            if len(img.shape) < 3:
                img = img.reshape(img.shape[0], img.shape[1], 1)
                img = np.repeat(img, 3, axis=2)
            img = [img]

        img = self.input_transform(img[0])
        
        return img, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

