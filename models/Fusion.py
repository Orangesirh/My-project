"""
Fusion Module with Coordinate Attention for MODEST Framework
改进：用CoordAttention替换原始的ChannelAttention，保留更多空间位置信息
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 新增：CoordAttention相关组件 ====================

class h_sigmoid(nn.Module):
    """Hard Sigmoid激活函数（用于CoordAttention）"""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish激活函数（用于CoordAttention）"""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    坐标注意力模块
    论文: Coordinate Attention for Efficient Mobile Network Design (CVPR2021)
    
    优势：
    1. 保留空间位置信息（H和W方向分别编码）
    2. 对透明物体边界敏感
    3. 适合深度估计和分割任务
    
    参数：
        inp: 输入通道数
        oup: 输出通道数（通常与inp相同）
        reduction: 降维比例，控制中间层通道数 (default: 32)
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        
        # H方向和W方向的自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 保留H维度
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 保留W维度

        # 计算中间层通道数（最少8个通道）
        mip = max(8, inp // reduction)

        # 共享的1×1卷积用于降维和特征编码
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        # 分别为H和W方向生成注意力权重
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播
        
        输入: x (B, C, H, W)
        输出: out (B, C, H, W) - 经过坐标注意力加权的特征
        
        流程：
        1. 沿H和W方向分别池化，保留位置信息
        2. 拼接后通过共享卷积编码
        3. 分割并生成H和W的注意力权重
        4. 将权重应用到原始特征上
        """
        identity = x
        B, C, H, W = x.size()
        
        # Step 1: 位置信息编码
        # 压缩水平方向: (B, C, H, W) → (B, C, H, 1)
        x_h = self.pool_h(x)
        # 压缩垂直方向: (B, C, H, W) → (B, C, 1, W) → (B, C, W, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Step 2: 坐标注意力生成
        # 拼接H和W方向的特征: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        # 通过Conv降维并编码: (B, C, H+W, 1) → (B, mip, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Step 3: 分割回H和W方向
        # (B, mip, H+W, 1) → x_h:(B, mip, H, 1), x_w:(B, mip, W, 1)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, mip, W, 1) → (B, mip, 1, W)

        # Step 4: 生成注意力权重
        # 恢复通道数并通过sigmoid生成权重
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)

        # Step 5: 应用注意力
        # 使用广播机制同时应用H和W的权重
        out = identity * a_w * a_h  # (B,C,H,W) * (B,C,1,W) * (B,C,H,1)

        return out


# ==================== 保留原有的其他模块 ====================

class SpatialAttention(nn.Module):
    """空间注意力模块（保持不变）"""
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
    """残差卷积单元（保持不变）"""
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
    """门控卷积单元（保持不变）"""
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
    """门控全局注意力（保持不变）"""
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


# ==================== 改进的Fusion模块 ====================

class Fusion(nn.Module):
    """
    语义和几何融合模块（改进版）
    
    主要改进：
    1. 用CoordinateAttention替换原始的ChannelAttention
    2. 保留空间位置信息，提升透明物体边界感知能力
    3. 更适合深度估计任务
    
    参数：
        resample_dim: 特征维度 (default: 256)
        nclasses: 分割类别数
        coord_reduction: CoordAttention的降维比例 (default: 32)
    """
    def __init__(self, resample_dim, nclasses, coord_reduction=32):
        super(Fusion, self).__init__()
        
        # 残差卷积单元
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

        # 门控单元
        self.gate_conv_depth = GateConvUnit(resample_dim)
        self.gate_conv_seg = GateConvUnit(resample_dim)
        self.gate_depth = GGA(resample_dim)
        self.gate_seg = GGA(resample_dim)

        # ========== 核心改进：CoordinateAttention ==========
        # 替换原来的ChannelAttention，保留位置信息
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
        # ==================================================
        
        # 空间注意力（保持不变）
        self.sa_depth = SpatialAttention()
        self.sa_seg = SpatialAttention()
        
    def forward(self, reassemble, index, previous_depth=None, previous_seg=None, 
                out_depths=None, out_segs=None):
        """
        前向传播
        
        参数：
            reassemble: 当前尺度的编码器特征
            index: 当前尺度索引 (3,2,1,0 从粗到细)
            previous_depth/seg: 上一尺度的特征
            out_depths/segs: 上一次迭代的预测结果
            
        返回：
            output_depth, output_seg: 融合后的深度和分割特征
        """
        ## 初始化
        if previous_depth is None and previous_seg is None:
            previous_depth = torch.zeros_like(reassemble)
            previous_seg = torch.zeros_like(reassemble)
            
        ## 残差卷积
        output_feature = self.res_conv1(reassemble)
        output_depth = output_feature + previous_depth
        output_seg = output_feature + previous_seg
        
        ## 门控机制（如果有历史预测）
        if out_depths is not None and out_segs is not None:
            if len(out_depths) != 0 and len(out_segs) != 0:
                depth = out_depths[-1][3-index]
                seg = out_segs[-1][3-index]
                depth = self.gate_conv_depth(depth)
                output_depth = self.gate_depth(output_depth, depth)
                seg = self.gate_conv_seg(seg)
                output_seg = self.gate_seg(output_seg, seg)

        ## ========== 核心改进：坐标注意力交叉融合 ==========
        # 通过CoordinateAttention提取位置感知的特征
        # 相比原来的CAM，这里同时编码了通道和空间位置信息
        depth_coord_attn = self.coord_att_depth(output_depth)
        seg_coord_attn = self.coord_att_seg(output_seg)
        
        # 交叉调制：用分割的位置信息增强深度，反之亦然
        output_seg = output_seg * depth_coord_attn
        output_depth = output_depth * seg_coord_attn
        ## ===================================================
        
        ## 空间注意力进一步细化
        depth_attention_spatial = self.sa_depth(output_depth)
        seg_attention_spatial = self.sa_seg(output_seg)
        output_seg = output_seg * depth_attention_spatial
        output_depth = output_depth * seg_attention_spatial
        
        ## 上采样到下一尺度
        output_seg = nn.functional.interpolate(
            output_seg, scale_factor=2, mode="bilinear", align_corners=True)
        output_depth = nn.functional.interpolate(
            output_depth, scale_factor=2, mode="bilinear", align_corners=True)
        
        return output_depth, output_seg


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("="*60)
    print("Testing CoordinateAttention Module")
    print("="*60)
    
    # 测试CoordinateAttention
    coord_att = CoordinateAttention(inp=256, oup=256, reduction=32)
    test_input = torch.randn(2, 256, 48, 48)
    test_output = coord_att(test_input)
    
    print(f"Input shape:   {test_input.shape}")
    print(f"Output shape:  {test_output.shape}")
    
    # 统计参数量
    coord_params = sum(p.numel() for p in coord_att.parameters())
    print(f"Parameters:    {coord_params:,}")
    
    print("\n" + "="*60)
    print("Testing Fusion Module with CoordAttention")
    print("="*60)
    
    # 测试完整的Fusion模块
    fusion = Fusion(resample_dim=256, nclasses=3, coord_reduction=32)
    reassemble = torch.randn(2, 256, 24, 24)
    
    depth_out, seg_out = fusion(
        reassemble, 
        index=2, 
        previous_depth=None, 
        previous_seg=None, 
        out_depths=[], 
        out_segs=[]
    )
    
    print(f"Input shape:       {reassemble.shape}")
    print(f"Depth output:      {depth_out.shape}")
    print(f"Seg output:        {seg_out.shape}")
    
    # 统计总参数量
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"Total parameters:  {total_params:,}")
    
    # 对比原始CAM的参数量（理论值）
    # 原始CAM: C×(C/r) + (C/r)×C = 2×C²/r
    original_cam_params = 2 * 256 * 256 // 16  # r=16
    print(f"\n参数量对比：")
    print(f"  原始CAM (理论):    ~{original_cam_params:,}")
    print(f"  CoordAttention:     {coord_params:,}")
    print(f"  增加比例:           {(coord_params/original_cam_params - 1)*100:.1f}%")
    
    print("\n" + "="*60)
    print("✅ All tests passed! You can now use this module.")
    print("="*60)