"""
Fusion Module with EDS-Fusion Integration
完整版 - 可直接替换项目中的 models/Fusion.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== CoordinateAttention（保留）====================

class h_sigmoid(nn.Module):
    """Hard Sigmoid激活函数"""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish激活函数"""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    坐标注意力模块（保持不变）
    论文: Coordinate Attention for Efficient Mobile Network Design (CVPR2021)
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# ==================== SpatialAttention（用于粗尺度）====================

class SpatialAttention(nn.Module):
    """简单的空间注意力（用于粗尺度）"""
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


# ==================== EdgeExtractor ====================

class EdgeExtractor(nn.Module):
    """
    边缘提取器（轻量化版本）
    
    特点：
    1. Sobel算子初始化（可学习）
    2. 轻量增强网络
    3. 适配不同通道数
    """
    def __init__(self, in_channels=256):
        super(EdgeExtractor, self).__init__()
        
        # Sobel卷积（可学习）
        self.sobel_conv = nn.Conv2d(in_channels, 2, kernel_size=3, 
                                    stride=1, padding=1, bias=False)
        
        # 初始化为Sobel算子
        sobel_kx = torch.tensor([[1., 0., -1.], 
                                 [2., 0., -2.], 
                                 [1., 0., -1.]])
        sobel_ky = torch.tensor([[1., 2., 1.], 
                                 [0., 0., 0.], 
                                 [-1., -2., -1.]])
        
        # 扩展到所有输入通道（简化版：取均值）
        sobel_kx = sobel_kx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        sobel_ky = sobel_ky.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        with torch.no_grad():
            self.sobel_conv.weight[0] = sobel_kx.mean(dim=0, keepdim=True)
            self.sobel_conv.weight[1] = sobel_ky.mean(dim=0, keepdim=True)
        
        # 边缘增强网络（轻量）
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: edge_map (B, 1, H, W)
        """
        edges = self.sobel_conv(x)          # (B, 2, H, W)
        edge_map = self.edge_enhance(edges) # (B, 1, H, W)
        return edge_map


# ==================== EDS-Fusion ====================

class EDSFusion(nn.Module):
    """
    EDS-Fusion: Enhanced Dual-Scale Spatial Attention with Edge Guidance
    
    三路并行注意力：
    1. Local-SAM: 高分辨率局部注意力（保留细节）
    2. Global-SAM: 全局上下文（理解整体形状）
    3. Edge-SAM: 边缘线索（针对透明物体边界）
    
    只在最细尺度使用，其他尺度用简单SA
    """
    def __init__(self, channels=256):
        super(EDSFusion, self).__init__()
        
        # ===== Local-SAM =====
        self.local_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        
        # ===== Global-SAM（简化：只用1×1池化）=====
        self.global_conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.BatchNorm2d(1)
        )
        
        # ===== Edge Path =====
        self.edge_extractor = EdgeExtractor(channels)
        
        # ===== 融合门控（自适应加权）=====
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # ===== 最终细化 =====
        self.refine = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        输入: x (B, C, H, W)
        输出: attention_map (B, 1, H, W)
        """
        B, C, H, W = x.size()
        
        # ========== Local-SAM: 局部细节 ==========
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        local_map = self.local_conv(torch.cat([max_out, avg_out], dim=1))
        
        # ========== Global-SAM: 全局上下文 ==========
        global_pool = F.adaptive_avg_pool2d(x, (1, 1)).expand(B, C, H, W)
        global_max = torch.max(global_pool, dim=1, keepdim=True)[0]
        global_avg = torch.mean(global_pool, dim=1, keepdim=True)
        global_map = self.global_conv(torch.cat([global_max, global_avg], dim=1))
        
        # ========== Edge-SAM: 边缘线索 ==========
        edge_map = self.edge_extractor(x)
        
        # ========== 可学习融合 ==========
        weights = self.fusion_gate(x)  # (B, 3, 1, 1)
        fused_map = (weights[:, 0:1, :, :] * local_map + 
                     weights[:, 1:2, :, :] * global_map + 
                     weights[:, 2:3, :, :] * edge_map)
        
        # ========== 最终细化 ==========
        stacked = torch.cat([local_map, global_map, edge_map], dim=1)
        attention = self.refine(stacked)
        
        # 残差连接
        attention = 0.6 * attention + 0.4 * torch.sigmoid(fused_map)
        
        return attention


# ==================== 其他保留模块 ====================

class ResidualConvUnit(nn.Module):
    """残差卷积单元"""
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
    """门控卷积单元"""
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
    """门控全局注意力"""
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


# ==================== 主Fusion模块 ====================

class Fusion(nn.Module):
    """
    语义和几何融合模块（EDS-Fusion版本）
    
    改进架构：
    Stage 1: ResidualConv + GateUnit (基础特征处理)
    Stage 2: CoordinateAttention (语义级：通道+空间位置) 
    Stage 3: EDS-Fusion (空间级：多尺度+边缘，只在最细尺度)
    Stage 4: 残差连接和上采样
    
    参数：
        resample_dim: 特征维度 (default: 256)
        nclasses: 分割类别数
        coord_reduction: CoordAttention降维比例 (default: 32)
        use_eds_at_finest: 是否只在最细尺度使用EDS-Fusion (default: True)
    """
    def __init__(self, resample_dim, nclasses, coord_reduction=32, 
                 use_eds_at_finest=True):
        super(Fusion, self).__init__()
        
        self.use_eds_at_finest = use_eds_at_finest
        
        # ===== Stage 1: 残差卷积单元 =====
        self.res_conv1 = ResidualConvUnit(resample_dim)
        
        # 深度分支的卷积层
        self.res_conv2_depth = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )
        
        # 分割分支的卷积层
        self.res_conv2_seg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # ===== Stage 2: 门控单元 =====
        self.gate_conv_depth = GateConvUnit(resample_dim)
        self.gate_conv_seg = GateConvUnit(resample_dim)
        self.gate_depth = GGA(resample_dim)
        self.gate_seg = GGA(resample_dim)

        # ===== Stage 3: CoordinateAttention（语义级）=====
        self.coord_att_depth = CoordinateAttention(
            inp=resample_dim, 
            oup=resample_dim, 
            reduction=coord_reduction
        )
        self.coord_att_seg = CoordinateAttention(
            inp=resample_dim, 
            oup=resample_dim, 
            reduction=coord_reduction
        )
        
        # ===== Stage 4: EDS-Fusion（空间级，只在最细尺度）=====
        if use_eds_at_finest:
            self.eds_fusion = EDSFusion(channels=resample_dim)
        
        # 简单SA（用于粗尺度）
        self.sa_depth = SpatialAttention()
        self.sa_seg = SpatialAttention()
        
    def forward(self, reassemble, index, previous_depth=None, previous_seg=None, 
                out_depths=None, out_segs=None):
        """
        前向传播
        
        参数：
            reassemble: 当前尺度的编码器特征 [B, C, H, W]
            index: 当前尺度索引 (3→粗, 0→细)
            previous_depth/seg: 上一尺度的特征
            out_depths/segs: 上一次迭代的预测结果
            
        返回：
            output_depth, output_seg: 融合后的深度和分割特征
        """
        ## ========== Stage 1: 初始化 + 残差卷积 ==========
        if previous_depth is None and previous_seg is None:
            previous_depth = torch.zeros_like(reassemble)
            previous_seg = torch.zeros_like(reassemble)
            
        output_feature = self.res_conv1(reassemble)
        output_depth = output_feature + previous_depth
        output_seg = output_feature + previous_seg
        
        ## ========== Stage 2: 门控机制 ==========
        if out_depths is not None and out_segs is not None:
            if len(out_depths) != 0 and len(out_segs) != 0:
                depth = out_depths[-1][3-index]
                seg = out_segs[-1][3-index]
                depth = self.gate_conv_depth(depth)
                output_depth = self.gate_depth(output_depth, depth)
                seg = self.gate_conv_seg(seg)
                output_seg = self.gate_seg(output_seg, seg)

        ## ========== Stage 3: CoordinateAttention跨任务交互 ==========
        # 提取位置感知的语义特征
        depth_coord_attn = self.coord_att_depth(output_depth)
        seg_coord_attn = self.coord_att_seg(output_seg)
        
        # 交叉调制：语义级融合
        output_seg_ca = output_seg * depth_coord_attn
        output_depth_ca = output_depth * seg_coord_attn
        
        # ========== 特征范围检查和归一化 ==========
        # 防止数值爆炸
        if output_seg_ca.abs().max() > 50:
            output_seg_ca = output_seg_ca / (output_seg_ca.abs().max() + 1e-6) * 10
    
        if output_depth_ca.abs().max() > 50:
            output_depth_ca = output_depth_ca / (output_depth_ca.abs().max() + 1e-6) * 10

        ## ========== Stage 4: 空间注意力 ==========
        if self.use_eds_at_finest and index == 0:
            # 最细尺度：使用EDS-Fusion
            depth_spatial_attn = self.eds_fusion(output_depth_ca)
            seg_spatial_attn = self.eds_fusion(output_seg_ca)
        else:
            # 粗尺度：使用简单SA（保持效率）
            depth_spatial_attn = self.sa_depth(output_depth_ca)
            seg_spatial_attn = self.sa_seg(output_seg_ca)
        
        # 应用空间注意力
        output_seg = output_seg_ca * depth_spatial_attn
        output_depth = output_depth_ca * seg_spatial_attn
        
        ## ========== Stage 5: 上采样到下一尺度 ==========
        output_seg = nn.functional.interpolate(
            output_seg, scale_factor=2, mode="bilinear", align_corners=True)
        output_depth = nn.functional.interpolate(
            output_depth, scale_factor=2, mode="bilinear", align_corners=True)
        
        return output_depth, output_seg