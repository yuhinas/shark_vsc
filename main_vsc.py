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
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import *
from math import log10
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
# from tensorboardX import SummaryWriter

from average_meter import AverageMeter
import glob
#import visdom
#viz = visdom.Visdom()

parser = argparse.ArgumentParser()
#parser.add_argument('--channels', default="64, 128, 256, 512, 512, 512", type=str, help='the list of channel numbers')
#parser.add_argument('--channels', default="32, 64, 128, 256, 512, 512", type=str, help='the list of channel numbers')
parser.add_argument('--channels', default="64, 128, 256, 512, 512", type=str, help='the list of channel numbers')
parser.add_argument("--hdim", type=int, default=512, help="dim of the latent code, Default=512")
parser.add_argument("--save_iter", type=int, default=2, help="Default=1")
parser.add_argument("--test_iter", type=int, default=1372, help="Default=1000")
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=8)
parser.add_argument('--dataroot', default="/home/jovyan/sharky2/images", type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data', default=-1)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--input_height', type=int, default=128, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=128, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=128, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=128, help='the width  of the output image to network')
parser.add_argument("--nEpochs", type=int, default=60000, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--parallel', action='store_true', help='for multiple GPUs')
parser.add_argument('--outf', default='results/vsc/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', action='store_true', help='enables tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg",".bmp"])
    
def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=opt.nrow), cur_iter)
    

def main():
    
    global opt, model
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    #if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    #if torch.cuda.is_available() and not opt.cuda:
    #    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
    is_scale_back = False
    
    #--------------build models -------------------------
    if opt.parallel:
        model = VSC(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height, parallel=True).cuda()
    else:
        model = VSC(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height).cuda()
    #model = VSC(cdim=3, hdim=512, channels=str_to_list('32, 64, 128, 256, 512, 512'), image_size=256).cuda()

    pretrained_default = 'model/vsc/model_local_epoch_%d_iter_0.pth' % opt.start_epoch
    #pretrained_default = 'model/vsc/model_local_epoch_%d_iter_0.pth' % 2000

    if opt.pretrained:
        load_model(model, opt.pretrained)
    elif os.path.isfile(pretrained_default):
        print ("Loading default pretrained %d..." % opt.start_epoch)
        load_model(model, pretrained_default)
    
    #print(model)
            
    optimizerE = optim.Adam(model.encoder.parameters(), lr=opt.lr)
    optimizerG = optim.Adam(model.decoder.parameters(), lr=opt.lr)
    
    #-----------------load dataset--------------------------
    image_list = [x for x in glob.iglob(opt.dataroot + '/*/**', recursive=True) if is_image_file(x)]
    #train_list = image_list[:opt.trainsize]
    train_list = image_list[:]
    assert len(train_list) > 0
    
    train_set = ImageDatasetFromFile(train_list, opt.dataroot, aug=True)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)
    num_of_batches = len(train_data_loader)
    
    start_time = time.time()

    #cur_iter = 0        
    #cur_iter = int(np.ceil(float(opt.trainsize) / float(opt.batchSize)) * opt.start_epoch)
    cur_iter = len(train_data_loader) * (opt.start_epoch - 1)
    
    def train_vsc(epoch, iteration, batch, cur_iter):
        
        #if len(batch.size()) == 3:
        #    batch = batch.unsqueeze(0)
        #print(batch.size())
            
        batch_size = batch.size(0)
                       
        real= Variable(batch).cuda() 

        noise = Variable(torch.zeros(batch_size, opt.hdim).normal_(0, 1)).cuda()
        fake = model.sample(noise)

        time_cost = time.time()-start_time
        info = "====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time_cost)
        
        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================                  
        #real_mu, real_logvar, z, rec = model(real)
        real_mu, real_logvar, real_logspike, z, rec = model(real)
        
        loss_rec =  model.reconstruction_loss(rec, real, True)
        
        #loss_kl = model.kl_loss(real_mu, real_logvar).mean()
        loss_prior = model.prior_loss(real_mu, real_logvar, real_logspike)
                    
        #loss = loss_rec + loss_kl
        
        # CHANGE SINCE EPOCH 2870
        # we also change the lr to 0.0001
        #loss = loss_rec + loss_prior
        loss = 5 * loss_rec + loss_prior
        
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        loss.backward()                   
        optimizerE.step()
        optimizerG.step()
     
        #info += 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec.data[0], loss_kl.data[0])

        am_rec.update(loss_rec.item())
        am_prior.update(loss_prior.item())

        info += 'Rec: {:.4f}({:.4f}), Prior: {:.4f}({:.4f}), '.format(am_rec.val, am_rec.avg, am_prior.val, am_prior.avg)
        print(info, end='\r')

        # viz_idx = cur_iter
        # if cur_iter % 10 is 0:
        #     viz.line([[am_rec.avgN, am_rec.avg]], [viz_idx], win='loss_rec', update='append', opts=dict(title='Rec'))
        #     viz.line([[am_prior.avgN, am_prior.avg]], [viz_idx], win='loss_prior', update='append', opts=dict(title='Prior'))

        if (iteration+1) % opt.test_iter is 0:  
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec, fake], cur_iter)   
            else:
                vutils.save_image(torch.cat([real[:16], rec[:16], fake[:8]], dim=0).data.cpu(), '{}/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)
                #not_enough = rec - denoised
                #too_much = -not_enough
                #vutils.save_image(torch.cat([not_enough, too_much], dim=0).data.cpu(), '{}/overlay/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)
                with open('./model/vsc/losses.log','a') as loss_log:
                    loss_log.write(
                        "\t".join([
                            str(cur_iter),
                            str(epoch), 
                            '%.2f' % time_cost,
                            '%.4f' % am_rec.avg,
                            '%.4f\n' % am_prior.avg
                        ])
                    )
        elif cur_iter % 100 is 0:
            vutils.save_image(torch.cat([real[:24], rec[:24], fake[:24]], dim=0).data.cpu(), '{}/image_up_to_date.jpg'.format(opt.outf),nrow=opt.nrow)
    #----------------Train by epochs--------------------------
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        #save models
        save_epoch = (epoch//opt.save_iter)*opt.save_iter

        if epoch == save_epoch:
            save_checkpoint(model, save_epoch, 0, '')
        
        model.train()
        model.c = 50 + epoch * model.c_delta
    
        am_rec = AverageMeter()
        am_prior = AverageMeter()

        for iteration, (batch, filenames) in enumerate(train_data_loader, 0):
            #--------------train------------
            train_vsc(epoch, iteration, batch, cur_iter)            
            cur_iter += 1

        print()
            
def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
            
def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "model/vsc/" + prefix +"model_local_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/vsc/"):
        os.makedirs("model/vsc/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()    