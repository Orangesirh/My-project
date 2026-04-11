"""
models/Fusion.py — 消融版本：CAM + EDS-Fusion

改动说明：
  Stage3 将 DCCA 替换为 Channel Attention Module（CAM）
    - CAM.forward 返回纯通道注意力权重 (B, C, 1, 1)，不包含特征值本身
    - 交叉调制：output_depth = output_depth * seg_cam_attn
               output_seg   = output_seg   * depth_cam_attn
    - 与参考实现（document #25）完全一致
  Stage4 使用 EDS-Fusion（由构造参数 use_eds_at_finest 控制）
  去掉 from models.dcca import DCCA
  其余模块完全不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== CAM（Channel Attention Module）====================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    forward 返回纯注意力权重 (B, C, 1, 1)，不含特征值。
    """
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
        return self.sigmoid(out)          # (B, C, 1, 1) 纯权重图


# ==================== SpatialAttention ====================

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1   = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


# ==================== EdgeExtractor ====================

class EdgeExtractor(nn.Module):
    def __init__(self, in_channels=256, reduce_channels=8):
        super(EdgeExtractor, self).__init__()

        self.channel_reduce = nn.Conv2d(
            in_channels, reduce_channels, kernel_size=1, bias=False)

        self.sobel_conv = nn.Conv2d(
            reduce_channels, 2, kernel_size=3,
            stride=1, padding=1, bias=False)

        sobel_kx = torch.tensor([[1.,  0., -1.],
                                  [2.,  0., -2.],
                                  [1.,  0., -1.]])
        sobel_ky = torch.tensor([[1.,  2.,  1.],
                                  [0.,  0.,  0.],
                                  [-1., -2., -1.]])
        with torch.no_grad():
            for i in range(reduce_channels):
                self.sobel_conv.weight.data[0, i] = sobel_kx
                self.sobel_conv.weight.data[1, i] = sobel_ky

        self.sobel_conv.weight.requires_grad = False

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_reduced = self.channel_reduce(x)
        edges     = self.sobel_conv(x_reduced)
        edge_map  = self.edge_enhance(edges)
        return edge_map


# ==================== EDSFusion ====================

class EDSFusion(nn.Module):
    def __init__(self, channels=256):
        super(EDSFusion, self).__init__()

        self.global_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
        )

        self.edge_extractor = EdgeExtractor(channels)

        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.refine = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        global_small = self.global_branch(x)
        global_map   = torch.sigmoid(
            F.interpolate(global_small, size=(H, W),
                          mode='bilinear', align_corners=True)
        )

        edge_map = self.edge_extractor(x)

        weights = F.softmax(self.fusion_weights, dim=0)
        fused   = weights[0] * global_map + weights[1] * edge_map

        stacked = torch.cat([global_map, edge_map], dim=1)
        refined = self.refine(stacked)

        attention = 0.6 * refined + 0.4 * fused

        return attention


# ==================== 辅助模块 ====================

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.relu  = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class GateConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv2d(features, features, kernel_size=1,
                              stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = nn.functional.interpolate(out, scale_factor=0.5,
                                        mode="bilinear", align_corners=True)
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
        attn = self.gate_conv(torch.cat([x, gate], dim=1))
        return self.out_conv(x * attn)


# ==================== Fusion 主模块 ====================

class Fusion(nn.Module):
    """
    消融版本：Stage3 使用 CAM，Stage4 使用 EDS-Fusion

    Stage 1 : ResidualConvUnit
    Stage 2 : GGA 门控
    Stage 3 : CAM 跨任务通道注意力
                ca_depth(output_depth) → (B,C,1,1) 权重，调制 output_seg
                ca_seg(output_seg)     → (B,C,1,1) 权重，调制 output_depth
    Stage 4 : EDS-Fusion / SpatialAttention / identity（由构造参数控制）
    Stage 5 : 双线性上采样 ×2
    """
    def __init__(self,
                 resample_dim,
                 coord_reduction=16,
                 use_eds_at_finest=False,
                 use_identity=False):
        super(Fusion, self).__init__()

        self.use_eds_at_finest = use_eds_at_finest
        self.use_identity      = use_identity

        # ===== Stage 1 =====
        self.res_conv1 = ResidualConvUnit(resample_dim)

        # ===== Stage 2: 门控 =====
        self.gate_conv_depth = GateConvUnit(resample_dim)
        self.gate_conv_seg   = GateConvUnit(resample_dim)
        self.gate_depth      = GGA(resample_dim)
        self.gate_seg        = GGA(resample_dim)

        # ===== Stage 3: CAM 跨任务通道注意力 =====
        # reduction 使用参考实现默认值 16，coord_reduction 参数保留但此处固定
        self.ca_depth = ChannelAttention(resample_dim, reduction=coord_reduction)
        self.ca_seg   = ChannelAttention(resample_dim, reduction=coord_reduction)

        # ===== Stage 4: 空间注意力 =====
        if use_identity:
            pass
        elif use_eds_at_finest:
            self.eds_fusion_depth = EDSFusion(channels=resample_dim)
            self.eds_fusion_seg   = EDSFusion(channels=resample_dim)
        else:
            self.sa_depth = SpatialAttention()
            self.sa_seg   = SpatialAttention()

    def forward(self, reassemble, index, previous_depth=None, previous_seg=None,
                out_depths=None, out_segs=None):

        # ---- Stage 1 ----
        if previous_depth is None and previous_seg is None:
            previous_depth = torch.zeros_like(reassemble)
            previous_seg   = torch.zeros_like(reassemble)

        output_feature = self.res_conv1(reassemble)
        output_depth   = output_feature + previous_depth
        output_seg     = output_feature + previous_seg

        # ---- Stage 2: 门控引导 ----
        if out_depths is not None and out_segs is not None:
            if len(out_depths) != 0 and len(out_segs) != 0:
                depth = out_depths[-1][3 - index]
                seg   = out_segs[-1][3 - index]
                depth = self.gate_conv_depth(depth)
                output_depth = self.gate_depth(output_depth, depth)
                seg   = self.gate_conv_seg(seg)
                output_seg   = self.gate_seg(output_seg, seg)

        # ---- Stage 3: CAM 跨任务通道注意力 ----
        # ca_depth/ca_seg 返回纯权重 (B, C, 1, 1)，交叉调制
        depth_cam_attn = self.ca_depth(output_depth)   # (B, C, 1, 1)
        seg_cam_attn   = self.ca_seg(output_seg)       # (B, C, 1, 1)
        output_depth_ca = output_depth * seg_cam_attn  # 深度被分割通道权重调制
        output_seg_ca   = output_seg   * depth_cam_attn  # 分割被深度通道权重调制

        # ---- Stage 4: 空间注意力 ----
        if self.use_identity:
            output_depth = output_depth_ca
            output_seg   = output_seg_ca
        elif self.use_eds_at_finest:
            depth_spatial_attn = self.eds_fusion_depth(output_depth_ca)
            seg_spatial_attn   = self.eds_fusion_seg(output_seg_ca)
            output_depth = output_depth_ca * seg_spatial_attn
            output_seg   = output_seg_ca   * depth_spatial_attn
        else:
            depth_spatial_attn = self.sa_depth(output_depth_ca)
            seg_spatial_attn   = self.sa_seg(output_seg_ca)
            output_depth = output_depth_ca * seg_spatial_attn
            output_seg   = output_seg_ca   * depth_spatial_attn

        # ---- Stage 5: 上采样 ----
        output_depth = nn.functional.interpolate(
            output_depth, scale_factor=2, mode="bilinear", align_corners=True)
        output_seg   = nn.functional.interpolate(
            output_seg,   scale_factor=2, mode="bilinear", align_corners=True)

        return output_depth, output_seg