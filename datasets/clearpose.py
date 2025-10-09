import os
import cv2
import torch
import json
import random

import numpy as np
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ClearPose(Dataset):

    def __init__(self, config, spilt):
        super().__init__()
        self.datadir = config["path"]
        with open(self.datadir + 'config.json', 'r') as f:
            self.grouping = json.load(f)
        self.spilt = spilt
        self.datalist = self.make_dataset()
        self.datalen = len(self.datalist)
        print(f"clearpose {spilt} dataset size:  {self.datalen}")
        self.im_size = (config["transforms"]["im_resize"], config["transforms"]["im_resize"])
        self.transform_image, self.transform_depth, self.transform_seg = self.get_transforms()
        self.p_flip = config['transforms']['p_flip'] if spilt=='train' else 0
        self.p_crop = config['transforms']['p_crop'] if spilt=='train' else 0
        self.p_rot = config['transforms']['p_rot'] if spilt=='train' else 0
        self.im_resize = config['transforms']['im_resize']
        self.depth_min = config["depth_min"]
        self.depth_max = config["depth_max"]


    def __getitem__(self, id):
        rgb_path, depth_path, seg_path = self.datalist[id]
        rgb = np.array(Image.open(rgb_path))
        depth_gt = np.array(Image.open(depth_path))
        seg_gt = np.array(Image.open(seg_path))

        rgb = rgb / 255.0

        depth_gt = depth_gt / 1000.0
        depth_gt[depth_gt < self.depth_min] = self.depth_min
        depth_gt[depth_gt > self.depth_max] = self.depth_min
        depth_gt = (depth_gt - self.depth_min) / (self.depth_max - self.depth_min)

        seg_gt = torch.from_numpy(np.ascontiguousarray(seg_gt)).long().unsqueeze(0)
        seg_gt[seg_gt != 0] = 1
        
        rgb = self.transform_image(rgb.astype(np.float32))
        depth_gt = self.transform_depth(depth_gt.astype(np.float32))
        seg_gt = self.transform_seg(seg_gt)

        # rgb, depth_gt, seg_gt = self.augmentation(rgb, depth_gt, seg_gt)
        
        depth_gt = depth_gt.squeeze(0)
        seg_gt = seg_gt.squeeze(0)

        zero_mask = (depth_gt > 0.0)
        loss_mask = (seg_gt == 1)

        data_dict = {
            'rgb': rgb,
            'depth_gt': depth_gt,
            'seg_gt': seg_gt,
            'depth_min': torch.tensor(self.depth_min),
            'depth_max': torch.tensor(self.depth_max),
            'zero_mask': torch.BoolTensor(zero_mask),
            'loss_mask': torch.BoolTensor(loss_mask)
        }

        return data_dict
        
    
    def __len__(self):
        return self.datalen
    
    
    def make_dataset(self):
        datalist = []
        config = self.grouping[self.spilt]
        assert os.path.isdir(self.datadir)
        for set, value in config.items():
            for scene, number in value.items():
                base_path = os.path.join(self.datadir, set, scene)
                meta_path = os.path.join(base_path, 'metadata.mat')
                for index in range(number):
                    rgb_path = os.path.join(base_path, str(index).zfill(6) + '-color.png')
                    depth_path = os.path.join(base_path, str(index).zfill(6) + '-depth_true.png')
                    mask_path = os.path.join(base_path, str(index).zfill(6) + '-label.png')
                    if os.path.exists(rgb_path):
                        datalist.append((rgb_path, depth_path, mask_path))
                    else:
                        continue

        return datalist
    
    def get_transforms(self):
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.im_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_depth = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.im_size)
        ])
        transform_seg = transforms.Compose([
            transforms.Resize(self.im_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])
        return transform_image, transform_depth, transform_seg
    
    def augmentation(self, rgb, depth_gt, seg_gt):
        if random.random() < self.p_flip:
            rgb = TF.hflip(rgb)
            depth_gt = TF.hflip(depth_gt)
            seg_gt = TF.hflip(seg_gt) 

        if random.random() < self.p_crop:
            random_size = random.randint(256, self.im_resize-1)
            max_size = self.im_resize - random_size
            left = int(random.random() * max_size)
            top = int(random.random() * max_size)
            rgb = TF.crop(rgb, top, left, random_size, random_size)
            depth_gt = TF.crop(depth_gt, top, left, random_size, random_size)
            seg_gt = TF.crop(seg_gt, top, left, random_size, random_size)
            

            rgb = transforms.Resize((self.im_resize, self.im_resize))(rgb)
            depth_gt = transforms.Resize((self.im_resize, self.im_resize))(depth_gt)
            seg_gt = transforms.Resize((self.im_resize, self.im_resize), interpolation=transforms.InterpolationMode.NEAREST)(seg_gt)

        if random.random() < self.p_rot:
            ## rotate
            random_angle = random.random()*20 - 10
            mask = torch.ones((1,self.im_resize,self.im_resize))
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            rgb = TF.rotate(rgb, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            depth_gt = TF.rotate(depth_gt, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            seg_gt = TF.rotate(seg_gt, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
        
            left = torch.argmax(mask[:,0,:]).item()
            top = torch.argmax(mask[:,:,0]).item()
            coin = min(left, top)
            size = self.im_resize - 2 * coin

            rgb = TF.crop(rgb, coin, coin, size, size)
            depth_gt = TF.crop(depth_gt, coin, coin, size, size)
            seg_gt = TF.crop(seg_gt, coin, coin, size, size)
            ## resize
            rgb = transforms.Resize((self.im_resize, self.im_resize))(rgb)
            depth_gt = transforms.Resize((self.im_resize, self.im_resize))(depth_gt)
            seg_gt = transforms.Resize((self.im_resize, self.im_resize), interpolation=transforms.InterpolationMode.NEAREST)(seg_gt)

        return rgb, depth_gt, seg_gt