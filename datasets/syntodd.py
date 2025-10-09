import os
import pathlib
import json
import pickle
import cv2
import torch
import random

import zstandard as zstd
import numpy as np
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from torchvision import transforms


MAX_DEPTH = 3.0
MIN_DEPTH = 0.0


def decompress_datapoint(cbuf, disable_final_decompression=False):
    cctx = zstd.ZstdDecompressor()
    buf = cctx.decompress(cbuf)
    x = pickle.loads(buf)
    if not disable_final_decompression:
        x.decompress()
    return x

class LocalReadHandle:

    def __init__(self, dataset_path, uid):
        self.dataset_path = dataset_path
        self.uid = uid

    def read(self, disable_final_decompression=False):
        path = os.path.join(self.dataset_path, f'{self.uid}.pickle.zstd')
        with open(path, 'rb') as fh:
            dp = decompress_datapoint(fh.read(), disable_final_decompression=disable_final_decompression)
        if not hasattr(dp, 'uid'):
            dp.uid = self.uid
        assert dp.uid == self.uid, f'dp uid is {dp.uid}, self.uid is {self.uid}'
        return dp


class SynTodd(Dataset):

    def __init__(self, config, mode):
        super().__init__()
        if mode not in ['train', 'val', 'test']:
            raise AttributeError('Invalid mode.')
        self.dataset_path = pathlib.Path(config["paths"][f"{mode}_path"])
        self.num_views = config["num_multiview"]
        self.num_samples = config[f"{mode}_num_samples"]
        self.prepare_dataset()
        print(f"syntodd {mode} dataset size: ", len(self.handles))
        self.config = config
        self.mode = mode

        self.transform_image, self.transform_depth, self.transform_seg = self.get_transforms()
        self.p_flip = config['transforms']['p_flip'] if mode=='train' else 0
        self.p_crop = config['transforms']['p_crop'] if mode=='train' else 0
        self.p_rot = config['transforms']['p_rot'] if mode=='train' else 0
        self.im_resize = config['transforms']['im_resize']

    def __len__(self):
        return len(self.handles)

    def getMultiviewSample(self, idx):
        
        dp_list = [dp.read() for dp in self.handles[idx]]
        stereo_list = [dp.stereo for dp in dp_list]
        anaglyph = self.CombineMultiview(stereo_list)
        image = anaglyph[0]

        seg_target = dp_list[0].segmentation
        seg_target = torch.from_numpy(np.ascontiguousarray(seg_target)).long()

        dp_list[0].depth = np.nan_to_num(dp_list[0].depth, nan=0.0, posinf=0.0)
        dp_list[0].depth[dp_list[0].depth > MAX_DEPTH] = 0.0
        dp_list[0].depth[dp_list[0].depth < MIN_DEPTH] = 0.0
        assert not np.isnan(dp_list[0].depth).any(), 'Depth should not have nan!!'
        depth_target = torch.from_numpy(np.ascontiguousarray(dp_list[0].depth)).float()

        depth_target = depth_target.unsqueeze(0)
        seg_target = seg_target.unsqueeze(0)
        
        image = self.transform_image(image)
        depth_target = self.transform_depth(depth_target)
        seg_target = self.transform_seg(seg_target)

        # image, depth_target, seg_target = self.augmentation(image, depth_target, seg_target)

        depth_target = depth_target.squeeze(0)
        seg_target = seg_target.squeeze(0)
        zero_mask = (depth_target != 0.0)
        loss_mask = (seg_target == 2)
        
        data_dict = {
            'rgb': image,
            'depth_gt': depth_target,
            'seg_gt': seg_target,
            'depth_min': MIN_DEPTH,
            'depth_max': MAX_DEPTH,
            'zero_mask': zero_mask,
            'loss_mask': loss_mask
            # 'camera_intrinsic': camera_intrinsic,
            # 'camera_poses': camera_poses,
            # 'pose_gt': pose_target,
            # 'box_gt': box_target,
            # 'keypoint_gt': kp_target,
            # 'scene_name': scene_name
        }

        return data_dict
    
    def __getitem__(self, idx):
        return self.getMultiviewSample(idx)
    
    def CombineMultiview(self, stereo_dps):
        images_combined = []
        for stereo_dp in stereo_dps:
            cv2.normalize(stereo_dp.left_color, stereo_dp.left_color, 0, 255, cv2.NORM_MINMAX)
            image = stereo_dp.left_color.transpose((2,0,1))
            image = image * 1. / 255.0
            images_combined.append(image)
        
        return torch.from_numpy(np.ascontiguousarray(images_combined)).float()
    
    def get_transforms(self):
        im_size = self.config['transforms']['im_resize']
        transform_image = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_depth = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.Lambda(lambda x: (x - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH))
        ])
        transform_seg = transforms.Compose([
            transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        return transform_image, transform_depth, transform_seg
    
    def prepare_dataset(self):
        self.handles = []
        config_file = open(os.path.join(self.dataset_path, f"{self.num_views}_{self.num_samples}.json"))
        data = json.load(config_file)
        groupings = data['data']
        config_file.close()

        for group in groupings:
            group_handle = []
            for element in group:
                uid = os.path.basename(element).split('.')[0]
                group_handle.append(LocalReadHandle(self.dataset_path, int(uid)))
            self.handles.append(group_handle)
        self.handles = sorted(self.handles, key= lambda x: x[0].uid)

    def augmentation(self, image, depth_target, seg_target):
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            depth_target = TF.hflip(depth_target)
            seg_target = TF.hflip(seg_target)
            
        if random.random() < self.p_crop:
            random_size = random.randint(256, self.im_resize-1)
            max_size = self.im_resize - random_size
            left = int(random.random() * max_size)
            top = int(random.random() * max_size)
            image = TF.crop(image, top, left, random_size, random_size)
            depth_target = TF.crop(depth_target, top, left, random_size, random_size)
            seg_target = TF.crop(seg_target, top, left, random_size, random_size)
            

            image = transforms.Resize((self.im_resize, self.im_resize))(image)
            depth_target = transforms.Resize((self.im_resize, self.im_resize))(depth_target)
            seg_target = transforms.Resize((self.im_resize, self.im_resize), interpolation=transforms.InterpolationMode.NEAREST)(seg_target)

        if random.random() < self.p_rot:
            ## rotate
            random_angle = random.random()*20 - 10
            mask = torch.ones((1,self.im_resize,self.im_resize))
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            depth_target = TF.rotate(depth_target, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            seg_target = TF.rotate(seg_target, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
        
            left = torch.argmax(mask[:,0,:]).item()
            top = torch.argmax(mask[:,:,0]).item()
            coin = min(left, top)
            size = self.im_resize - 2 * coin

            image = TF.crop(image, coin, coin, size, size)
            depth_target = TF.crop(depth_target, coin, coin, size, size)
            seg_target = TF.crop(seg_target, coin, coin, size, size)
            ## resize
            image = transforms.Resize((self.im_resize, self.im_resize))(image)
            depth_target = transforms.Resize((self.im_resize, self.im_resize))(depth_target)
            seg_target = transforms.Resize((self.im_resize, self.im_resize), interpolation=transforms.InterpolationMode.NEAREST)(seg_target)

        return image, depth_target, seg_target

    
def isnan(x):
  return x != x