# python main_vsc.py 
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
#from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import VSC
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid#, save_image
import pandas as pd
from touch_dir import touch_dir
from average_meter import AverageMeter
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    
str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in [".jpg", ".png", ".jpeg",".bmp"])

def encoding_vsc():

    cudnn.benchmark = True

    #--------------build VSC models -------------------------
    #print(opt.hdim, str_to_list(opt.channels), opt.output_height)
    #vsc_model = VSC(cdim=3, hdim=512, channels=str_to_list("32, 64, 128, 256, 512, 512"), image_size=256).cuda()
    vsc_model = VSC(cdim=3, hdim=512, channels=str_to_list('64, 128, 256, 512, 512'), image_size=128, parallel=True).cuda()
    vsc_model.eval()

    pretrained_default = './model/vsc/model_local_epoch_23310_iter_0.pth'
    load_model(vsc_model, pretrained_default)

    #-----------------load dataset--------------------------
    #image_list = [x for x in glob.iglob('./tesri/*', recursive=True) if is_image_file(x)]
    dataroot = "/home/jovyan/sharky2/images"
    image_list = [x for x in glob.iglob(dataroot + '/*/**', recursive=True) if is_image_file(x)]
    valid_list = image_list[:]
    assert len(valid_list) > 0
    
    #image_cache = np.load('./wolrdwide_lepidoptera_yolov4_cropped_and_padded.npy', allow_pickle=True)
    
    valid_set = ImageDatasetFromFile(valid_list, '', aug=False)
    valid_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=False, num_workers=10)
            
    mu_s = np.empty((len(valid_list), 512))
    logvar_s = np.empty((len(valid_list), 512))
    logspike_s = np.empty((len(valid_list), 512))

    all_filenames = np.array([])
    with torch.no_grad():
        for iteration, (batch, filenames) in enumerate(valid_data_loader, 0):
            
            print(iteration, end='\r')
            #print(filenames)
            
            real= Variable(batch).cuda()

            mu, logvar, logspike = vsc_model.encode(real)
            #z = model.reparameterize(mu, logvar, logspike)

            all_filenames = np.append(all_filenames, filenames)

            from_ = iteration * 50 # batch size
            to_ = from_ + batch.size(0)

            mu_s[from_:to_,] = mu.detach().data.cpu().numpy()
            logvar_s[from_:to_,] = logvar.detach().data.cpu().numpy()
            logspike_s[from_:to_,] = logspike.detach().data.cpu().numpy()

            del real
            del mu
            del logvar
            del logspike
        print()

    with torch.no_grad():
        repeatN = 1000
        codes_ = vsc_model.reparameterize(torch.from_numpy(mu_s).cuda(), torch.from_numpy(logvar_s).cuda(), torch.from_numpy(logspike_s).cuda())
        for rep_ in range(1, repeatN):
            print('Repeating %d' % rep_, end='\r')
            codes_ = codes_ + vsc_model.reparameterize(torch.from_numpy(mu_s).cuda(), torch.from_numpy(logvar_s).cuda(), torch.from_numpy(logspike_s).cuda())
        print()
        codes = (codes_ / repeatN).detach().data.cpu().numpy()

    df = pd.DataFrame(data=codes, columns=range(codes.shape[1]))
    mu_df = pd.DataFrame(data=mu_s, columns=range(mu_s.shape[1]))
    logvar_df = pd.DataFrame(data=logvar_s, columns=range(logvar_s.shape[1]))
    logspike_df = pd.DataFrame(data=logspike_s, columns=range(logspike_s.shape[1]))
    
    df['filename'] = all_filenames
    mu_df['filename'] = all_filenames
    logvar_df['filename'] = all_filenames
    logspike_df['filename'] = all_filenames
    #print(df)

    to_save = "./save/repeated_codes"
    touch_dir(to_save)

    #np.save("%s/codes.npy" % to_save, codes)
    df.to_csv("%s/codes.csv" % to_save, sep="\t")
    mu_df.to_csv("%s/mu_s.csv" % to_save, sep="\t")
    logvar_df.to_csv("%s/logvar_s.csv" % to_save, sep="\t")
    logspike_df.to_csv("%s/logspike_s.csv" % to_save, sep="\t")

if __name__ == '__main__':
    encoding_vsc()
