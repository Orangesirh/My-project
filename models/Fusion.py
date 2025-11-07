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

# ==================== 新增：TripletAttention简化版（几何级）====================

class ZPool(nn.Module):
    """Z-Pool：同时进行最大池化和平均池化"""
    def forward(self, x):
        return torch.cat([
            torch.max(x, 1)[0].unsqueeze(1),
            torch.mean(x, 1).unsqueeze(1)
        ], dim=1)


class AttentionGate(nn.Module):
    """
    基础注意力门
    用于TripletAttention的各个分支
    """
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, 
                     padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(1)
        )
    
    def forward(self, x):
        """
        输入: x - 任意形状的特征 (B, dim1, dim2, dim3)
        输出: 经过注意力加权的特征
        """
        x_compress = self.compress(x)  # (B, 2, dim2, dim3)
        x_out = self.conv(x_compress)   # (B, 1, dim2, dim3)
        scale = torch.sigmoid(x_out)    # 生成权重
        return x * scale                # 加权


class TripletAttentionSimplified(nn.Module):
    """
    简化版TripletAttention（方案C专用）
    
    只保留C-W和H-C两个分支：
    - C-W分支：捕获垂直方向的几何模式（透明物体轮廓线）
    - H-C分支：捕获水平方向的几何模式（液体界面、反射边界）
    
    省略H-W分支，因为CoordinateAttention已经处理了空间关系
    """
    def __init__(self):
        super(TripletAttentionSimplified, self).__init__()
        self.cw = AttentionGate()  # C-W交互
        self.hc = AttentionGate()  # H-C交互
    
    def forward(self, x):
        """
        几何维度的跨维度交互
        
        Args:
            x: 输入特征 (B, C, H, W)
        
        Returns:
            融合后的特征 (B, C, H, W)
        """
        # === C-W分支：垂直方向几何模式 ===
        # 旋转：将C和W作为空间维度，在H上压缩
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()  # (B,C,H,W) → (B,H,C,W)
        x_out1 = self.cw(x_perm1)                      # 在H维度上压缩并生成注意力
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()  # 旋转回来
        
        # === H-C分支：水平方向几何模式 ===
        # 旋转：将H和C作为空间维度，在W上压缩
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()  # (B,C,H,W) → (B,W,H,C)
        x_out2 = self.hc(x_perm2)                      # 在W维度上压缩并生成注意力
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()  # 旋转回来
        
        # === 双分支融合 ===
        # 使用加权平均而非简单平均，以保持梯度稳定性
        out = 0.5 * (x_out1 + x_out2)
        
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


# ==================== 方案C：增强版Fusion模块 ====================

class Fusion(nn.Module):
    """
    语义和几何融合模块（方案C完整实现）
    
    改进链路：
    1. ResidualConv + GateUnit: 基础特征处理
    2. CoordinateAttention: 语义级跨任务交互（通道+空间位置）
    3. TripletAttention: 几何级跨维度交互（C-W + H-C）
    4. 残差连接: alpha*Triplet + CoordAtt（保持锐度）
    5. SpatialAttention: 最终空间细化（可选）
    
    参数：
        resample_dim: 特征维度 (default: 256)
        nclasses: 分割类别数
        coord_reduction: CoordAttention降维比例 (default: 32)
        use_triplet: 是否启用TripletAttention (default: True)
        use_final_sa: 是否使用最终的SpatialAttention (default: False)
    """
    def __init__(self, resample_dim, nclasses, coord_reduction=32, 
                 use_triplet=True, use_final_sa=False):
        super(Fusion, self).__init__()
        
        self.use_triplet = use_triplet
        self.use_final_sa = use_final_sa
        
        # === Stage 1: 残差卷积单元 ===
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

        # === Stage 2: 门控单元 ===
        self.gate_conv_depth = GateConvUnit(resample_dim)
        self.gate_conv_seg = GateConvUnit(resample_dim)
        self.gate_depth = GGA(resample_dim)
        self.gate_seg = GGA(resample_dim)

        # === Stage 3: CoordinateAttention（语义级）===
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
        
        # === Stage 4: TripletAttention（几何级，新增）===
        if self.use_triplet:
            self.triplet_att = TripletAttentionSimplified()
            
            # 可学习的融合权重（初始化为0.5）
            self.alpha_depth = nn.Parameter(torch.tensor(0.5))
            self.beta_seg = nn.Parameter(torch.tensor(0.5))
        
        # === Stage 5: 空间注意力（最终细化，可选）===
        if self.use_final_sa:
            self.sa_depth = SpatialAttention()
            self.sa_seg = SpatialAttention()
        
    def forward(self, reassemble, index, previous_depth=None, previous_seg=None, 
                out_depths=None, out_segs=None):
        """
        前向传播
        
        参数：
            reassemble: 当前尺度的编码器特征 [B, C, H, W]
            index: 当前尺度索引 (3,2,1,0 从粗到细)
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
        
        # ========== 添加：特征范围检查和归一化 ==========
        # 检测异常值
        if output_seg.abs().max() > 50:
            # 归一化到合理范围
            output_seg_ca = output_seg_ca / (output_seg_ca.abs().max() + 1e-6) * 10
    
        if output_depth.abs().max() > 50:
            output_depth_ca = output_depth_ca / (output_depth_ca.abs().max() + 1e-6) * 10
        # ==============================================


        ## ========== Stage 4: TripletAttention + 残差连接（新增）==========
        if self.use_triplet:
            # 几何级跨维度交互
            output_depth_geo = self.triplet_att(output_depth_ca)
            output_seg_geo = self.triplet_att(output_seg_ca)
            
            # 残差融合（关键！防止过度平滑）
            # alpha/beta可学习，初始值0.5表示平衡
            output_depth = self.alpha_depth * output_depth_geo + output_depth_ca
            output_seg = self.beta_seg * output_seg_geo + output_seg_ca
        else:
            # 如果不使用Triplet，直接使用CoordAtt的结果
            output_depth = output_depth_ca
            output_seg = output_seg_ca
        
        ## ========== Stage 5: 最终空间注意力细化（可选）==========
        if self.use_final_sa:
            depth_attention_spatial = self.sa_depth(output_depth)
            seg_attention_spatial = self.sa_seg(output_seg)
            output_seg = output_seg * depth_attention_spatial
            output_depth = output_depth * seg_attention_spatial
        
        ## ========== Stage 6: 上采样到下一尺度 ==========
        output_seg = nn.functional.interpolate(
            output_seg, scale_factor=2, mode="bilinear", align_corners=True)
        output_depth = nn.functional.interpolate(
            output_depth, scale_factor=2, mode="bilinear", align_corners=True)
        
        return output_depth, output_seg

