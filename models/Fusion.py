import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dcca import DCCA


# ==================== CoordinateAttention（保留用于对比）====================

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
    """原始 CoordinateAttention（保留用于对比实验）"""
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
        return identity * a_w * a_h


# ==================== SpatialAttention（粗尺度 fallback）====================

class SpatialAttention(nn.Module):
    """简单空间注意力，用于粗尺度（index=2,3: 24×24, 12×12）。"""
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
    """
    边缘提取器。

    流程：
      1. channel_reduce: 1×1 卷积将 256 通道压缩到 8 通道（可学习）。
         目的：避免 Conv2d(256→2) 时 256 通道叠加求和导致
               高激活通道主导梯度响应，Sobel 语义失效。
      2. sobel_conv: 对 8 通道特征做 Sobel 边缘检测（权重冻结，不可学习）。
         固定 Sobel 算子整个训练过程始终保持真正的梯度检测语义，
         与 SE-MDE (Zuo et al., CVIU 2025) 的固定 Sobel 设计一致。
      3. edge_enhance: 将 2 通道梯度图（x/y 方向）融合为 1 通道边缘图（可学习）。

    参数量：
      channel_reduce: 256×8 = 2,048
      sobel_conv:     8×2×9 = 144（冻结，不参与优化）
      edge_enhance:   2×8×9 + 16(BN) + 9 = 177
      合计可训练参数: 2,048 + 177 = 2,225
    """
    def __init__(self, in_channels=256, reduce_channels=8):
        super(EdgeExtractor, self).__init__()

        # Step 1: 通道压缩
        self.channel_reduce = nn.Conv2d(
            in_channels, reduce_channels, kernel_size=1, bias=False)

        # Step 2: Sobel 卷积（低维空间，语义干净）
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

        # 冻结 Sobel 算子：整个训练过程保持真正的边缘检测语义
        self.sobel_conv.weight.requires_grad = False

        # Step 3: 梯度图融合为单通道边缘响应
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),  # 后接BN，不需要bias
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_reduced = self.channel_reduce(x)      # (B, 256, H, W) → (B, 8, H, W)
        edges     = self.sobel_conv(x_reduced)  # (B, 8, H, W)  → (B, 2, H, W)
        edge_map  = self.edge_enhance(edges)    # (B, 2, H, W)  → (B, 1, H, W)
        return edge_map


# ==================== EDSFusion（双路版）====================

class EDSFusion(nn.Module):
    """
    EDS-Fusion: Enhanced Dual-Scale Spatial Attention with Edge Guidance

    双路并行注意力：
      1. Global-SAM : 压缩到 4×4 再双线性上采样，保留全局空间变化能力。
                      修复原版 AdaptiveAvgPool2d(1) 导致单一标量广播的失效问题。
      2. Edge-SAM   : 固定 Sobel 算子 + 可学习融合，针对透明物体边界。

    仅用于最细尺度（idx=0: 96×96）。
    深度分支与分割分支在 Fusion 类中各自实例化，权重不共享。
    """
    def __init__(self, channels=256):
        super(EDSFusion, self).__init__()

        # ===== Global-SAM =====
        self.global_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),           # 压缩到 4×4 保留空间结构
            nn.Conv2d(channels // 4, 1, kernel_size=1),
        )

        # ===== Edge-SAM =====
        self.edge_extractor = EdgeExtractor(channels)

        # ===== 可学习融合权重 =====
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        # ===== 双路细化网络 =====
        self.refine = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        输入:  x (B, C, H, W)
        输出:  attention_map (B, 1, H, W)，值域 [0, 1]
        """
        B, C, H, W = x.size()

        # ---- Global-SAM ----
        global_small = self.global_branch(x)               # (B, 1, 4, 4)
        global_map   = torch.sigmoid(
            F.interpolate(global_small, size=(H, W),
                          mode='bilinear', align_corners=True)
        )                                                   # (B, 1, H, W)

        # ---- Edge-SAM ----
        edge_map = self.edge_extractor(x)                  # (B, 1, H, W)

        # ---- 可学习加权融合 ----
        weights = F.softmax(self.fusion_weights, dim=0)
        fused   = weights[0] * global_map + weights[1] * edge_map

        # ---- 双路细化网络 ----
        stacked = torch.cat([global_map, edge_map], dim=1)  # (B, 2, H, W)
        refined = self.refine(stacked)                       # (B, 1, H, W)

        # ---- 残差混合 ----
        # global_map 和 edge_map 均已过 Sigmoid，fused 天然在 [0,1]，
        # 无需再做 sigmoid（否则变化范围从 [0,1] 压缩到 [0.5,0.73]）
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
    语义与几何融合模块

    四阶段架构：
      Stage 1 : ResidualConvUnit — 基础特征提取
      Stage 2 : GGA 门控 — 跨迭代历史引导
      Stage 3 : DCCA — 深度↔分割跨任务坐标注意力
      Stage 4 : 空间注意力
                  idx=0（96×96）→ EDS-Fusion（Global-SAM + Edge-SAM）
                    eds_fusion_depth 与 eds_fusion_seg 为独立实例
                  idx=1（48×48）→ identity（恒等透传，跳过 Stage 4）
                  idx=2（24×24）→ SpatialAttention
                  idx=3（12×12）→ SpatialAttention
      Stage 5 : 双线性上采样 ×2

    参数：
        use_eds_at_finest : bool
            由 Model.py 按 idx==0 在构造时设置。
    """
    def __init__(self,
                 resample_dim,
                 coord_reduction=32,
                 use_eds_at_finest=False,
                 use_identity=False):
        super(Fusion, self).__init__()

        self.use_eds_at_finest = use_eds_at_finest
        self.use_identity      = use_identity   # True → Stage 4 恒等透传

        # ===== Stage 1 =====
        self.res_conv1 = ResidualConvUnit(resample_dim)

        # ===== Stage 2: 门控 =====
        self.gate_conv_depth = GateConvUnit(resample_dim)
        self.gate_conv_seg   = GateConvUnit(resample_dim)
        self.gate_depth      = GGA(resample_dim)
        self.gate_seg        = GGA(resample_dim)

        # ===== Stage 3: DCCA 跨任务坐标注意力 =====
        self.coord_att_depth = DCCA(inp=resample_dim, oup=resample_dim,
                                    reduction=coord_reduction)
        self.coord_att_seg   = DCCA(inp=resample_dim, oup=resample_dim,
                                    reduction=coord_reduction)

        # ===== Stage 4: 空间注意力 =====
        if use_identity:
            pass   # 不初始化任何模块，forward 直接透传 DCCA 输出
        elif use_eds_at_finest:
            self.eds_fusion_depth = EDSFusion(channels=resample_dim)
            self.eds_fusion_seg   = EDSFusion(channels=resample_dim)
        else:
            self.sa_depth = SpatialAttention()
            self.sa_seg   = SpatialAttention()

    def forward(self, reassemble, index, previous_depth=None, previous_seg=None,
                out_depths=None, out_segs=None):
        """
        参数：
            reassemble     : 当前尺度编码器特征 (B, C, H, W)
            index          : 当前尺度索引，0=最细(96×96) … 3=最粗(12×12)
            previous_depth : 上一尺度深度特征
            previous_seg   : 上一尺度分割特征
        返回：
            output_depth, output_seg (B, C, 2H, 2W)
        """

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

        # ---- Stage 3: DCCA 跨任务交互 ----
        depth_coord_attn = self.coord_att_depth(output_depth)
        seg_coord_attn   = self.coord_att_seg(output_seg)

        output_depth_ca = output_depth * seg_coord_attn
        output_seg_ca   = output_seg   * depth_coord_attn

        # ---- Stage 4: 空间注意力 ----
        if self.use_identity:
            # 恒等透传：跳过空间注意力，直接使用 DCCA 输出
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