import numpy as np
from torchvision import transforms as T
import torch
import os
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

class Load_DataSet_Reds(Dataset):
    def __init__(self , dataset_path , mode=None):
        self.dataset_path = dataset_path
        self.mode = mode
        self.filename_list = os.listdir(os.path.join(self.dataset_path,self.mode,'input'))

    def __getitem__(self, index):
        if self.mode == 'train':
            data_index = self.filename_list[index]
            dat_path = os.path.join(self.dataset_path,self.mode,'input',data_index)
            data  = self.get_np_from_dat(dat_path)
            data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
            data_name = data_index.split('.')[0].split('id')[0]
            ik_7 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(7) + '.png'),False)
            ik_14 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(14) + '.png'),False)
            ik_21 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(21) + '.png'),False)
            ik_28 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(28) + '.png'),False)
            ik_35 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(35) + '.png'),False)
            
            return data,ik_7,ik_14,ik_21,ik_28,ik_35,data_name
        elif self.mode == 'test':
            data_index = self.filename_list[index]
            dat_path = os.path.join(self.dataset_path,self.mode,'input',data_index)
            data_name = data_index.split('.')[0]
            gt_path = os.path.join(self.dataset_path,self.mode,'gt',data_name+'.png')
            #print(dat_path)
            data  = self.get_np_from_dat(dat_path)
            data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
            data = data[151 - 21 : 151 + 20, :, :]
            gt = self.get_gt(gt_path,False)
            return data,gt,data_name
    
    def __len__(self):
        return len(self.filename_list)

    def get_np_from_dat(self,file_path):
        f =  open(file_path,'rb')
        video_seq = f.read()
        video_seq = np.frombuffer(video_seq, 'b')
        spike_matrix = self.RawToSpike(video_seq,250,400,True)
        return spike_matrix

    def RawToSpike(self,video_seq, h, w,flipud):
        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = h*w
        img_num = len(video_seq)//(img_size//8)
        SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
        pix_id = np.arange(0,h*w)
        pix_id = np.reshape(pix_id, (h, w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(img_num):
            id_start = img_id*img_size//8
            id_end = id_start + img_size//8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)
            if flipud:
                SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id, :, :] = (result == comparator)

        return SpikeMatrix

    def get_gt(self,png_path,is_resize):
        if is_resize:
            w,h = 400,250
            img = np.array(Image.open(png_path).convert('L').resize((w, h),Image.ANTIALIAS))
        else:
            img = np.array(Image.open(png_path).convert('L'))
        return torch.from_numpy(img.copy()).type(torch.FloatTensor).unsqueeze(0)



class Load_DataSet_classA(Dataset):
    def __init__(self , dataset_path , mode=None):
        self.dataset_path = dataset_path
        self.mode = mode
        self.filename_list = os.listdir(os.path.join(self.dataset_path,'test'))

    def __getitem__(self, index):
        data_index = self.filename_list[index]
        dat_path = os.path.join(self.dataset_path,'test',data_index)
        #print(dat_path)
        data  = self.get_np_from_dat(dat_path)
        data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
        data_name = data_index.split('.')[0]
        return data,data_name
    
    def __len__(self):
        return len(self.filename_list)

    def get_np_from_dat(self,file_path):
        f =  open(file_path,'rb')
        video_seq = f.read()
        video_seq = np.frombuffer(video_seq, 'b')
        spike_matrix = self.RawToSpike(video_seq,250,400,True)
        return spike_matrix

    def RawToSpike(self,video_seq, h, w,flipud):
        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = h*w
        img_num = len(video_seq)//(img_size//8)
        SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
        pix_id = np.arange(0,h*w)
        pix_id = np.reshape(pix_id, (h, w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(img_num):
            id_start = img_id*img_size//8
            id_end = id_start + img_size//8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)
            if flipud:
                SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id, :, :] = (result == comparator)

        return SpikeMatrix

class Load_DataSet_X4K(Dataset):
    def __init__(self , dataset_path , mode=None):
        self.dataset_path = dataset_path
        self.mode = mode
        self.filename_list = os.listdir(os.path.join(self.dataset_path,self.mode,'input'))

    def __getitem__(self, index):
        if self.mode == 'train':
            data_index = self.filename_list[index]
            dat_path = os.path.join(self.dataset_path,self.mode,'input',data_index)
            data  = self.get_np_from_dat(dat_path)
            data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
            data_name = data_index.split('.d')[0].split('id')[0]
            ik_7 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + '_key_id' + str(7) + '.png'),False)
            ik_14 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + '_key_id' + str(14) + '.png'),False)
            ik_21 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + '_key_id' + str(21) + '.png'),False)
            ik_28 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + '_key_id' + str(28) + '.png'),False)
            ik_35 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + '_key_id' + str(35) + '.png'),False)
            '''
            ids = [7,14,21,28,35]
            for each_id in ids:
                each_path = os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(each_id) + '.png')
                code = """ik_{} = self.get_gt('{}',False)""".format(str(each_id),each_path)
                print(code)
                exec(code)
            '''
            return data,ik_7,ik_14,ik_21,ik_28,ik_35,data_index
        elif self.mode == 'test':
            data_index = self.filename_list[index]
            dat_path = os.path.join(self.dataset_path,self.mode,'input',data_index)
            data_name = data_index.split('.')[0].split('id')[0]
            gt_path = os.path.join(self.dataset_path,self.mode,'gt',data_name + '_key_id' + str(14) + '.png')
            #print(dat_path)
            data  = self.get_np_from_dat(dat_path)
            data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
            gt = self.get_gt(gt_path,False)
            return data,gt,data_index
    
    def __len__(self):
        return len(self.filename_list)

    def get_np_from_dat(self,file_path):
        f =  open(file_path,'rb')
        video_seq = f.read()
        video_seq = np.frombuffer(video_seq, 'b')
        spike_matrix = self.RawToSpike(video_seq,1000,1000,False)
        return spike_matrix

    def RawToSpike(self,video_seq, h, w,flipud):
        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = h*w
        img_num = len(video_seq)//(img_size//8)
        SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
        pix_id = np.arange(0,h*w)
        pix_id = np.reshape(pix_id, (h, w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(img_num):
            id_start = img_id*img_size//8
            id_end = id_start + img_size//8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)
            if flipud:
                SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id, :, :] = (result == comparator)

        return SpikeMatrix

    def get_gt(self,png_path,is_resize):
        img = np.array(Image.open(png_path).convert('L'))
        return torch.from_numpy(img.copy()).type(torch.FloatTensor).unsqueeze(0)
