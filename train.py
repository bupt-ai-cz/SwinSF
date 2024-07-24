import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
#import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from dataloader import *
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
import os
from torch.utils.tensorboard import SummaryWriter   
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

def set_parser():
    parser = argparse.ArgumentParser(description='SwinSpikeFormer')
    parser.add_argument('--data_mode', type=str, choices=["250","1000"], help='the resolution of the datasets')
    parser.add_argument('--dataset_path', default='./datasets/spike_x4k', help='the dataset for training and testing')
    parser.add_argument('--device', default='cuda:0', help='the gpu device useed in training')
    parser.add_argument('--device_ids', type = str,default='', help='the gpu device useed in training')
    parser.add_argument('--load_model',type=str,default='',help='the model saved in training and loaded in testing')
    parser.add_argument('--epochs', type=int, default=900, metavar='N', help='number of epochs to train(default:200)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate(default:0.0005)')
    parser.add_argument('--momentum', default=0.9, type=float, help='the momentum for SGD optimizer 0.9')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='the weight decay for optimizer 1e-4')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of trainset')
    parser.add_argument('--resume_path',type = str,default='',help='Path for resume model.')
    args = parser.parse_args()
    return args


def train_val(net,epoch,lr,train_loader,val_loader,device,args):
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    save_path = os.path.join('checkpoint',time_string)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    writer = SummaryWriter(save_path)
    torch.cuda.empty_cache()
    loss = nn.SmoothL1Loss(reduction = 'mean' , beta = 1.0)
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters(),lr = lr , betas = (0.9,0.999))
    scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
    best = {'epoch':0 , 'ssim':0 , 'psnr':0 }
    for i in range(epoch):
        torch.cuda.empty_cache()
        loss_epoch_train = [] 
        ssim_epoch_train = []
        psnr_epoch_train = []
        loss_epoch_test = [] 
        ssim_epoch_test = []
        psnr_epoch_test = []
        for [data_iter,ik_7,ik_14,ik_21,ik_28,ik_35,_] in tqdm(iter(train_loader)):
            #torch.cuda.empty_cache()
            net.train()
            optimizer.zero_grad()
            optimizer.step()
            data_iter = data_iter.to(device)
            #print(data_iter.shape)
            ik_iter = ik_21.to(device)
            ik_l = ik_14.to(device)
            ik_r = ik_28.to(device)
            #print(label_iter)
            img_pred,im_l,im_r = net(data_iter)
            img_pred = torch.clamp(img_pred , 0 ,255)
            im_l = torch.clamp(im_l , 0 ,255)
            im_r = torch.clamp(im_r , 0 ,255)
            l = loss(ik_iter,img_pred) + 0.1*(loss(ik_l , im_l) + loss(ik_r , im_r))
            img_pred = img_pred.cpu().detach().numpy().astype(np.uint8)
            ik_iter = ik_iter.cpu().detach().numpy().astype(np.uint8)
            p = psnr(img_pred,ik_iter)
            s = ssim(img_pred[0,0,:,:],ik_iter[0,0,:,:])
            loss_epoch_train.append(l.item())
            psnr_epoch_train.append(p)
            ssim_epoch_train.append(s)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        for [data_iter,ik_iter,_] in tqdm(iter(val_loader)):
            net.eval()
            with torch.no_grad():
                data_iter = data_iter.to(device)
                ik_iter = ik_iter.to(device)
                ik_l = ik_14.to(device)
                ik_r = ik_28.to(device)
                img_pred,im_l,im_r = net(data_iter)
                img_pred = torch.clamp(img_pred , 0 ,255)
                im_l = torch.clamp(im_l , 0 ,255)
                im_r = torch.clamp(im_r , 0 ,255)
                l = loss(ik_iter,img_pred) + 0.1*(loss(ik_l , im_l) + loss(ik_r , im_r))
                img_pred = img_pred.cpu().detach().numpy().astype(np.uint8)
                ik_iter = ik_iter.cpu().detach().numpy().astype(np.uint8)
                p = psnr(img_pred,ik_iter)
                s = ssim(img_pred[0,0,:,:],ik_iter[0,0,:,:])
                loss_epoch_test.append(l.item())
                psnr_epoch_test.append(p)
                ssim_epoch_test.append(s)
        scheduler.step()
        writer.add_scalar('train/loss',l.item(),epoch)
        writer.add_scalar('train/psnr',p,epoch)
        writer.add_scalar('train/ssim',s,epoch)
        writer.add_scalar('test/loss',l.item(),epoch)
        writer.add_scalar('test/psnr',p,epoch)
        writer.add_scalar('test/ssim',s,epoch)
        print("epoch : {}  \ntrain_loss : {}  train_psnr : {}  train_ssim : {} \ntest_loss : {}  test_psnr : {}  test_ssim : {}".format(
              str(i) , np.mean(loss_epoch_train) , np.mean(psnr_epoch_train) , np.mean(ssim_epoch_train) , np.mean(loss_epoch_test) , np.mean(psnr_epoch_test) , np.mean(ssim_epoch_test)))
        
        if np.mean(psnr_epoch_test) > best['psnr']:
            best['epoch'] = i
            best['psnr'] = np.mean(psnr_epoch_test)
            best['ssim'] = np.mean(ssim_epoch_test)
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i}
            torch.save(state , os.path.join(save_path,'best_model_ssim:{}_psnr:{}_epoch:{}_.pth'.format(best['ssim'],best['psnr'],best['epoch'])))
            print('save the best model!')
        print("now the best psnr is {} , best ssim is {} , appear in epoch {}".format(best['psnr'],best['ssim'],best['epoch']))


def main():
    args = set_parser()
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('cuda is avaliable,training on gpu!')
    else:
        print('Training on cpu!')
    device = torch.device(args.device if train_on_gpu else "cpu")
    if args.device_ids:
        device_ids = list(args.device_ids)
    else:
        device_ids = list(args.device.split(':')[-1])
    device_ids = [int(i) for i in device_ids]
    print(device_ids)
    epoch = args.epochs
    lr = args.lr
    batch_size = args.batch_size

    if args.data_mode == "250":
        from models.SwinSF_250 import SwinSpikeFormer
        net = SwinSpikeFormer(img_size=(250,400), patch_size=2, in_chans=41,ref_ch = 28,out_chans = 1,
                    embed_dim=96, depths=[6,6], num_heads=[2, 2],
                    window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv')
        spike_train = Load_DataSet_Reds(dataset_path = args.dataset_path, mode = 'train')
        spike_test = Load_DataSet_Reds(dataset_path = args.dataset_path, mode = 'test')

    elif args.data_mode == "1000":
        from models.SwinSF_1000 import SwinSpikeFormer
        net = SwinSpikeFormer(img_size=(1000,1000), patch_size=4, in_chans=41, ref_ch = 28,out_chans = 1,
                 embed_dim=64, depths=[6, 6], num_heads=[2, 2],
                 window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv')
        spike_train = Load_DataSet_X4K(dataset_path = args.dataset_path, mode = 'train')
        spike_test = Load_DataSet_X4K(dataset_path = args.dataset_path, mode = 'test')

    net = torch.nn.DataParallel(net,device_ids = device_ids)
    print("the network is {}\n ------------------------------------------------------------------------------------------------------------------------------------------------------------".format(net))
    if args.load_model:
        if os.path.isfile(args.load_model):
            print("=> loading checkpoint '{}'".format(args.load_model))
            checkpoint = torch.load(args.load_model)
            net.load_state_dict(checkpoint['net'])
            print("=> loaded checkpoint '{}' ".format(args.load_model))
    net.to(device)
    train_loader = data.DataLoader(dataset=spike_train,batch_size=batch_size,shuffle=True)
    test_loader = data.DataLoader(dataset=spike_test,batch_size=1,shuffle=False)
    train_val(net,epoch,lr,train_loader,test_loader,device,args)


if __name__ =="__main__": 
    main()

