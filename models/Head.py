import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x


############# multi-scale supervision #################
class MultiscaleHead(nn.Module):
    def __init__(self, channels, nclasses):
        super(MultiscaleHead, self).__init__()
        self.head_depth = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(channels // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU()
            nn.Sigmoid()
        )
        self.head_seg = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(channels // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, depth_feature, seg_feature):
        depth = self.head_depth(depth_feature)
        seg = self.head_seg(seg_feature)
        
        return depth, seg


############# seg head #################
class HeadSeg(nn.Module):
    def __init__(self, features, nclasses=3):
        super(HeadSeg, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0)
        )
        

    def forward(self, x):
        seg = self.head(x)
        return seg


############# depth head #################
class HeadDepth(nn.Module):
    def __init__(self, features):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU()
            nn.Sigmoid()
        )

    def forward(self, x):
        depth = self.head(x)
        return depth  