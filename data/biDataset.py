import os
import numpy as np
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageOps
import torch.nn as nn
import random
random.seed(0)


def one_hot(seg,num_class):
    seg = seg.squeeze(0)
    seg_one_hot = nn.functional.one_hot(seg.to(torch.int64),num_class)
    return seg_one_hot.permute(2,0,1)

class biDatasetCTStable(Dataset):

    def __init__(self,index_left=0,index_right=5000,res=1024,tr = 10,
                 num_class=3,listdir=None,path='./USegv5',
                 gt=False,return_seg=False):  # crop_size,

        if listdir is not None:
            self.imlist = np.loadtxt(listdir,dtype='str')
            self.path = path
        else:
            self.imlist = os.listdir(path)
            self.path = path

        self.imlist = self.imlist[index_left:index_right]
        self.train_transform_lung = transforms.Compose([

            transforms.Resize([res, res]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.tr = tr
        self.train_transform_seg = nn.Upsample((res, res))
        self.num_class = num_class
        self.return_seg = return_seg
        self.gt = gt

    def __getitem__(self, idx):
        if idx < len(self.imlist):

            filename = self.imlist[idx]
            seg_name = '%s/%s'%(self.path.replace('USeg','Segmentation'),filename)
            seg = Image.open(seg_name)
            seg = torch.Tensor(np.array([[np.array(seg)]]))
            seg[seg >= self.num_class - 1] = self.num_class - 1
            # useg = assign_label(img)
            useg_name = '%s/%s'%(self.path,filename)
            useg = Image.open(useg_name)
            useg = np.array(np.array(useg) / self.tr, dtype='int') * self.tr
            useg = torch.Tensor(np.array([[(useg/255 - 0.5)/0.5]]))
            useg = self.train_transform_seg(useg).squeeze(1)
            seg = self.train_transform_seg(seg).squeeze(1)
            seg = one_hot(seg,self.num_class)
            img_name = '%s/%s'%(self.path.replace('USeg','CTMontage'),filename)

            img_origin = Image.open(img_name)
            img_origin = ImageOps.grayscale(img_origin)
            img_origin = self.train_transform_lung(img_origin)
            if self.gt:
                img_origin = torch.cat([img_origin,seg],0)
            else:
                img_origin = torch.cat([img_origin, seg], 0)
            if self.return_seg:
                return {'label': seg, 'path': filename, 'instance': seg, 'image': img_origin}
            else:
                return {'label': seg, 'path': filename, 'instance': useg, 'image': img_origin}


    def __len__(self):
        return len(self.imlist)


