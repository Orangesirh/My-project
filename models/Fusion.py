import numpy as np
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class GateConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        out = self.conv(x)
        out = self.relu(out)
        out = nn.functional.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=True)
        return out


class GGA(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(features, features, 1, bias=False)

    def forward(self, x, gate):
        attention_map = self.gate_conv(torch.cat([x, gate], dim=1))
        out = x * attention_map
        out = self.out_conv(out)
        return out


################ multi-scale fusion #######################
class Fusion(nn.Module):
    def __init__(self, resample_dim, nclasses):
        super(Fusion, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2_depth = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.res_conv2_seg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.res_conv2_depth_skip = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(resample_dim+1, resample_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.res_conv2_seg_skip = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(resample_dim+3, resample_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.gate_conv_depth = GateConvUnit(resample_dim)
        self.gate_conv_seg = GateConvUnit(resample_dim)
        self.gate_depth = GGA(resample_dim)
        self.gate_seg = GGA(resample_dim)

        self.ca_depth = ChannelAttention(resample_dim)
        self.ca_seg = ChannelAttention(resample_dim)
        self.sa_depth = SpatialAttention()
        self.sa_seg = SpatialAttention()
        
        
    def forward(self, reassemble, index, previous_depth=None, previous_seg=None, out_depths=None, out_segs=None):
        ## reassemble: [256, 12, 12], [256, 24, 24], [256, 48, 48], [256, 96, 96]
        ## previous: None, [256, 24, 24], [256, 48, 48], [256, 96, 96]
        ## depth: None, [1, 48, 48], [1, 96, 96], [1, 192, 192]
        if previous_depth == None and previous_seg == None:
            previous_depth = torch.zeros_like(reassemble)
            previous_seg = torch.zeros_like(reassemble)
        output_feature = self.res_conv1(reassemble)
        output_depth = output_feature + previous_depth
        output_seg = output_feature + previous_seg
        if len(out_depths) != 0 and len(out_segs) != 0:
            depth = out_depths[-1][3-index]
            seg = out_segs[-1][3-index]
            depth = self.gate_conv_depth(depth)
            output_depth = self.gate_depth(output_depth, depth)
            seg = self.gate_conv_seg(seg)
            output_seg = self.gate_seg(output_seg, seg)

        depth_attention_channel = self.ca_depth(output_depth)
        seg_attention_channel = self.ca_seg(output_seg)
        output_seg = output_seg * depth_attention_channel
        output_depth = output_depth * seg_attention_channel
        depth_attention_spatial = self.sa_depth(output_depth)
        seg_attention_spatial = self.sa_seg(output_seg)
        output_seg = output_seg * depth_attention_spatial
        output_depth = output_depth * seg_attention_spatial
        output_seg = output_seg
        output_depth = output_depth
        output_seg = nn.functional.interpolate(output_seg, scale_factor=2, mode="bilinear", align_corners=True)
        output_depth = nn.functional.interpolate(output_depth, scale_factor=2, mode="bilinear", align_corners=True)
        return output_depth, output_seg
    