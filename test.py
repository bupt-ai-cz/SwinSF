import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
#import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from dataloader import *
from metrics.niqe import *
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
import os
from torch.utils.tensorboard import SummaryWriter   
from tqdm import tqdm
import cv2


def set_parser():
    parser = argparse.ArgumentParser(description='SwinSpikeFormer')
    parser.add_argument('--data_mode', type=str, choices=["250","1000","no_gt"], help='the resolution of the datasets')
    parser.add_argument('--dataset_path', default='./datasets/spike_x4k', help='the dataset for training and testing')
    parser.add_argument('--device', default='cuda:1', help='the gpu device useed in testing')
    parser.add_argument('--load_model',type=str,default='',help='the model saved in training and loaded in testing')
    parser.add_argument('--save_image', type=bool, default=False, metavar='N', help='whether saving images or not when testing')
    parser.add_argument('--save_path', default='save_images_reds', help='path to save reconstructed image')
    args = parser.parse_args()
    return args

def main():
    args = set_parser()
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('cuda is avaliable, testing on gpu!')
    else:
        print('Testing on cpu!')
    device = torch.device(args.device if train_on_gpu else "cpu")
    device_ids = list(args.device.split(':')[-1])
    device_ids = [int(i) for i in device_ids]

    if args.data_mode == "250":
        from models.SwinSF_250 import SwinSpikeFormer
        net = SwinSpikeFormer(img_size=(250,400), patch_size=2, in_chans=41,ref_ch = 28,out_chans = 1,
                    embed_dim=96, depths=[6,6], num_heads=[2, 2],
                    window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv')
        spike_test = Load_DataSet_Reds(dataset_path = args.dataset_path, mode = 'test')

    elif args.data_mode == "1000":
        from models.SwinSF_1000 import SwinSpikeFormer
        net = SwinSpikeFormer(img_size=(1000,1000), patch_size=4, in_chans=41, ref_ch = 28,out_chans = 1,
                 embed_dim=64, depths=[6, 6], num_heads=[2, 2],
                 window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv')
        spike_test = Load_DataSet_X4K(dataset_path = args.dataset_path, mode = 'test')
    elif args.data_mode == "no_gt":
        from models.SwinSF_250 import SwinSpikeFormer
        net = SwinSpikeFormer(img_size=(250,400), patch_size=2, in_chans=41,ref_ch = 28,out_chans = 1,
                    embed_dim=96, depths=[6,6], num_heads=[2, 2],
                    window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv')
        spike_test = Load_DataSet_classA(dataset_path = args.dataset_path, mode = 'test')
    net = torch.nn.DataParallel(net,device_ids = device_ids)
    print("the network is {}\n ------------------------------------------------------------------------------------------------------------------------------------------------------------".format(net))
    if args.load_model:
        if os.path.isfile(args.load_model):
            print("=> loading checkpoint '{}'".format(args.load_model))
            checkpoint = torch.load(args.load_model, map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            print("=> loaded checkpoint '{}' ".format(args.load_model))
    net.to(device)
    test_loader = data.DataLoader(dataset=spike_test,batch_size=1,shuffle=False)
    torch.cuda.empty_cache()

    if args.data_mode != "no_gt":
        loss = nn.SmoothL1Loss(reduction = 'mean' , beta = 1.0)
        loss_epoch_test = [] 
        ssim_epoch_test = []
        psnr_epoch_test = []
        i = 0
        for [data_iter,gt_iter,name] in tqdm(iter(test_loader)):
            name = name[0]
            net.eval()
            with torch.no_grad():
                data_rec_by_tfitfp = np.array(data_iter).squeeze(0)
                data_iter = data_iter.to(device)
                gt_iter = gt_iter.to(device)
                img_pred,_,_ = net(data_iter)
                img_pred = torch.clamp(img_pred , 0 ,255)
                l = loss(gt_iter,img_pred)
                img_pred = img_pred.cpu().detach().numpy().astype(np.uint8)
                gt_iter = gt_iter.cpu().detach().numpy().astype(np.uint8)
                if args.save_image:
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    save_path = args.save_path
                    cv2.imwrite(os.path.join(save_path,name+'_spike.png'),data_rec_by_tfitfp[21,:,:]*255)
                    cv2.imwrite(os.path.join(save_path,name+'_ik_gt.png'),gt_iter[0,0,:,:])
                    cv2.imwrite(os.path.join(save_path,name+'_ik_Ours.png'),img_pred[0,0,:,:])
                p = psnr(img_pred,gt_iter)
                s = ssim(img_pred[0,0,:,:],gt_iter[0,0,:,:])
                loss_epoch_test.append(l.item())
                psnr_epoch_test.append(p)
                ssim_epoch_test.append(s)
                print("{} psnr :{:.4f} ssim:{:.4f}".format(name, p, s))
                i += 1

        print("total loss : {}  psnr : {}  ssim : {}  ".format(np.mean(loss_epoch_test) , np.mean(psnr_epoch_test) , np.mean(ssim_epoch_test)))
                                                                                                            
    else:
        niqe_list = []
        i = 0
        for [data_iter,name] in tqdm(iter(test_loader)):
            name = name[0]
            net.eval()
            with torch.no_grad():
                data_rec_by_tfitfp = np.array(data_iter).squeeze(0)
                data_iter = data_iter.to(device)
                data_iter = data_iter[:,300:341,:,:]
                img_pred,_,_ = net(data_iter)
                img_pred = torch.clamp(img_pred , 0 ,255)
                img_pred = img_pred.cpu().detach().numpy().astype(np.uint8)
                if args.save_image:
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    save_path = args.save_path
                    cv2.imwrite(os.path.join(save_path,name+'_spike.png'),data_rec_by_tfitfp[21,:,:]*255)
                    cv2.imwrite(os.path.join(save_path,name+'_ik_Ours.png'),img_pred[0,0,:,:])
                niqe_ik = niqe(img_pred[0,0,:,:])
                niqe_list.append(niqe_ik)
                i += 1
                print("{} niqe :{:.4f}".format(name, niqe_ik))
                i += 1

        print("total niqe: {}  ".format(np.mean(niqe_list)))

    
if __name__ == "__main__":
    main()

