"""
models/Fusion.py — 消融版本：CoordinateAttention + 混合 Stage4

改动：
  Stage3 将 DCCA 替换为原版 CoordinateAttention（Hou et al., CVPR 2021）
    - CoordinateAttention 已在本文件内定义，无需额外导入
    - 去掉 from models.dcca import DCCA
    - forward 返回调制后特征 identity * a_h * a_w，与 DCCA 返回值形状语义一致
    - 交叉调制方式与原版 Fusion.py 保持一致
  Stage4 由构造参数控制（EDS-Fusion 或 SpatialAttention）
  其余模块完全不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== CoordinateAttention（原版，用于消融 Stage3）====================

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    原版 CoordinateAttention（Hou et al., CVPR 2021）
    forward 返回调制后特征 identity * a_h * a_w，形状 (B, C, H, W)
    用于消融实验，替换 DCCA，验证坐标平滑改进的有效性。
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y   = torch.cat([x_h, x_w], dim=2)
        y   = self.conv1(y)
        y   = self.bn1(y)
        y   = self.act(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w   # (B, C, H, W) 调制后特征


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
    消融版本：CoordinateAttention + 混合 Stage4

    Stage 1 : ResidualConvUnit
    Stage 2 : GGA 门控
    Stage 3 : CoordinateAttention 跨任务交互（消融，替换 DCCA）
                coord_att_depth(output_depth) → 调制后特征，交叉作用于 output_seg
                coord_att_seg(output_seg)     → 调制后特征，交叉作用于 output_depth
    Stage 4 : EDS-Fusion / SpatialAttention / identity（由构造参数控制）
    Stage 5 : 双线性上采样 ×2
    """
    def __init__(self,
                 resample_dim,
                 coord_reduction=32,
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

        # ===== Stage 3: 原版 CoordinateAttention（消融，替换 DCCA）=====
        self.coord_att_depth = CoordinateAttention(
            inp=resample_dim, oup=resample_dim, reduction=coord_reduction)
        self.coord_att_seg   = CoordinateAttention(
            inp=resample_dim, oup=resample_dim, reduction=coord_reduction)

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

        # ---- Stage 3: CoordinateAttention 跨任务交互 ----
        # 返回调制后特征 (B, C, H, W)，交叉作用于对方分支
        depth_coord_attn = self.coord_att_depth(output_depth)
        seg_coord_attn   = self.coord_att_seg(output_seg)

        output_depth_ca = output_depth * seg_coord_attn
        output_seg_ca   = output_seg   * depth_coord_attn

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