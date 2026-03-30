"""
models/dcca.py

DCCA: Depthwise-Smoothed Coordinate Attention
基于原版 CoordinateAttention (Hou et al., CVPR 2021) 的改进版本

改进点：
  在原版 CA 生成的 1D 注意力权重序列上，增加逐通道 Conv1d 平滑 + 残差连接，
  强制相邻空间位置的权重保持连续过渡，减少透明物体边界区域的注意力权重阶跃噪声。

相比项目原始 dcca.py 修正的三个问题：
  1. ReLU → GELU
       原版使用 ReLU，会将 Conv1d 输出的负值截断为 0，
       截断操作破坏了权重序列的连续性，与"平滑边界"的改进动机相矛盾。
       GELU 为软激活，对负值区域保留平滑梯度，不产生硬截断。

  2. Conv1d(groups=1) → Conv1d(groups=oup)，即普通卷积改为 Depthwise 卷积
       原版 Conv1d(oup, oup, k=3) 是全通道卷积，在平滑 H 方向第 i 个通道的权重时
       同时混入了其余所有通道的信息，破坏了各通道独立的坐标注意力语义。
       改为 Depthwise 后每个通道独立做 1D 平滑，符合设计意图。
       额外收益：参数量从 oup×oup×3=196,608 降至 oup×3=768（per direction），
       减少约 99.6%，不引入额外计算负担。

  3. 增加 assert inp == oup
       forward 最后一步 `identity * a_h * a_w` 要求 identity 的通道数等于 oup，
       即 inp 必须等于 oup。原代码对此无约束，可能产生静默的广播错误。
       本项目中 inp=oup=resample_dim=256，assert 在运行时显式保证此假设。
"""

import torch
import torch.nn as nn


# ==================== 激活函数 ====================

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


# ==================== 原版 CoordinateAttention（保留用于对比实验）====================

class CoordinateAttention(nn.Module):
    """
    原版 CoordinateAttention，保留用于 ablation 对比。
    来源：Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021.
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

        x_h = self.pool_h(x)                       # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)           # (B, C, H+W, 1)
        y = self.conv1(y)                           # (B, mip, H+W, 1)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)   # (B,mip,H,1), (B,mip,W,1)
        x_w = x_w.permute(0, 1, 3, 2)              # (B, mip, 1, W)

        a_h = self.conv_h(x_h).sigmoid()            # (B, oup, H, 1)
        a_w = self.conv_w(x_w).sigmoid()            # (B, oup, 1, W)

        return identity * a_h * a_w


# ==================== DCCA（改进版，用于实际训练）====================

class DCCA(nn.Module):
    """
    DCCA: Depthwise-Smoothed Coordinate Attention

    在原版 CA 的基础上，对生成的 1D 注意力权重序列
    施加 Depthwise Conv1d 局部平滑 + 残差连接，
    增强透明物体边界区域的注意力连续性。

    Shape 流程（以 inp=oup=256, H=W=96 为例）：
        x          : (B, 256, 96, 96)
        pool_h(x)  : (B, 256, 96,  1)
        pool_w(x)  : (B, 256,  1, 96) → permute → (B, 256, 96,  1)
        cat + conv1: (B,   8, 192, 1)   # mip = max(8, 256//32) = 8
        split      : (B,   8, 96,  1), (B, 8, 96, 1) → permute → (B,8,1,96)
        conv_h/w   : e_h(B,256,96,1),  e_w(B,256,1,96)
        squeeze    : e_h_1d(B,256,96), e_w_1d(B,256,96)
        dw-conv1d  : smooth 后 residual 叠加 → 同形状
        sigmoid+   : a_h(B,256,96,1),  a_w(B,256,1,96)
        unsqueeze  : broadcast 至 (B,256,96,96)
        out        : x * a_h * a_w → (B,256,96,96)

    注意：本模块要求 inp == oup（由 assert 保证），
    项目中实际调用均为 inp=oup=resample_dim=256。
    """
    def __init__(self, inp, oup, reduction=32):
        super(DCCA, self).__init__()

        # inp 必须等于 oup，否则最后的 identity * a_h * a_w 会广播错误
        assert inp == oup, (
            f"DCCA requires inp == oup, got inp={inp}, oup={oup}. "
            "If different channel dimensions are needed, add a projection layer."
        )

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        # ===== Stage 1: 与原版 CA 相同的坐标编码 =====
        self.conv1  = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # ===== Stage 2: DCCA 改进——Depthwise Conv1d 平滑 =====
        #
        # 修正1（原代码问题）：groups=1 → groups=oup（Depthwise）
        #   普通 Conv1d 在平滑第 i 通道的 H 方向权重时会混入其余通道信息，
        #   破坏各通道坐标注意力的独立性。
        #   Depthwise 卷积每通道独立平滑，语义正确，参数量也从 oup²×3 降至 oup×3。
        #
        # 修正2（原代码问题）：ReLU → GELU
        #   ReLU 将负值硬截断为 0，导致权重曲线在零值处产生折点，
        #   引入新的不连续性，与"平滑边界"目标相矛盾。
        #   GELU 为软激活（x·Φ(x)），对负值区域保留平滑梯度，无硬截断。
        self.conv1d_h = nn.Sequential(
            nn.Conv1d(oup, oup, kernel_size=3, padding=1,
                      bias=False, groups=oup),   # Depthwise: 每通道独立平滑
            nn.BatchNorm1d(oup),
            nn.GELU()                            # 软激活，无硬截断
        )

        self.conv1d_w = nn.Sequential(
            nn.Conv1d(oup, oup, kernel_size=3, padding=1,
                      bias=False, groups=oup),   # Depthwise
            nn.BatchNorm1d(oup),
            nn.GELU()
        )

    def forward(self, x):
        """
        输入/输出: (B, C, H, W)，C = inp = oup
        """
        identity = x
        B, C, H, W = x.size()

        # ---- 坐标信息编码（原版 CA 步骤）----
        x_h = self.pool_h(x)                       # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)           # (B, C, H+W, 1)
        y = self.conv1(y)                           # (B, mip, H+W, 1)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)   # (B,mip,H,1), (B,mip,W,1)
        x_w = x_w.permute(0, 1, 3, 2)              # (B, mip, 1, W)

        # ---- 生成原始 1D 权重向量 ----
        e_h = self.conv_h(x_h)   # (B, oup, H, 1)
        e_w = self.conv_w(x_w)   # (B, oup, 1, W)

        # ---- DCCA 改进：Depthwise Conv1d 平滑 + 残差连接 ----
        e_h_1d = e_h.squeeze(-1)                          # (B, oup, H)
        e_h_smooth = self.conv1d_h(e_h_1d) + e_h_1d      # 残差：保留原始信息
        #                                                  # shape: (B, oup, H)

        e_w_1d = e_w.squeeze(-2)                          # (B, oup, W)
        e_w_smooth = self.conv1d_w(e_w_1d) + e_w_1d      # 残差
        #                                                  # shape: (B, oup, W)

        # ---- 激活为注意力权重并广播 ----
        a_h = torch.sigmoid(e_h_smooth.unsqueeze(-1))     # (B, oup, H, 1) → 广播至 W
        a_w = torch.sigmoid(e_w_smooth.unsqueeze(-2))     # (B, oup, 1, W) → 广播至 H

        return identity * a_h * a_w                        # (B, C, H, W)