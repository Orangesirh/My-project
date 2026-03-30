"""
Multi-Scale Dilated Attention (MSDA) for MODEST
Adapted for transparent object depth estimation and segmentation

Key modifications from original DilateFormer:
1. Optimized for MODEST's feature dimensions
2. Efficient implementation for different scales
3. Compatible with (B,C,H,W) format used in Fusion.py

Bug fix:
  原代码用 outputs[i] = ... 对 x 的 view 做 inplace 写入，
  导致 autograd 报错：
    "modified by an inplace operation ... CopySlices"
  修复：改用 list 收集每个 scale 的结果，最后 torch.stack + reshape，
  完全不依赖预分配 tensor，消除 inplace 操作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilateAttention(nn.Module):
    """
    Single-scale dilated attention

    Args:
        head_dim: Dimension per attention head
        qk_scale: Scale factor for Q·K (default: head_dim^-0.5)
        attn_drop: Dropout rate for attention weights
        kernel_size: Size of local window
        dilation: Dilation rate for sparse sampling
    """
    def __init__(self, head_dim, qk_scale=None, attn_drop=0., kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            stride=1
        )
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        """
        Args:
            q, k, v: [B, d, H, W] where d = head_dim * num_heads_per_scale
        Returns:
            x: [B, H, W, d]
        """
        B, d, H, W = q.shape
        num_heads = d // self.head_dim

        # Query: [B, d, H, W] → [B, h, H*W, 1, head_dim]
        q = q.reshape(B, num_heads, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2)

        # Key: [B, d, H, W] → [B, h, H*W, head_dim, k*k]
        k = self.unfold(k)
        k = k.reshape(B, num_heads, self.head_dim,
                      self.kernel_size ** 2, H * W).permute(0, 1, 4, 2, 3)

        # attn: [B, h, H*W, 1, k*k]  显存占用极小（局部attention）
        attn = (q @ k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Value: [B, d, H, W] → [B, h, H*W, k*k, head_dim]
        v = self.unfold(v)
        v = v.reshape(B, num_heads, self.head_dim,
                      self.kernel_size ** 2, H * W).permute(0, 1, 4, 3, 2)

        # Output: [B, h, H*W, 1, k*k] @ [B, h, H*W, k*k, head_dim]
        #       → [B, h, H*W, 1, head_dim] → [B, H, W, d]
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)

        return x


class MultiScaleDilatedAttention(nn.Module):
    """
    Multi-Scale Dilated Attention for MODEST

    Bug fix说明：
      原代码：
        outputs = x.reshape(...).permute(...)   # outputs 是 x 的 view，共享存储
        for i in range(N):
            outputs[i] = self.dilate_attention[i](...)  # inplace 写入 x 的 view
        → autograd 报 CopySlices inplace 错误

      修复后：
        out_list = []
        for i in range(N):
            out_list.append(self.dilate_attention[i](...))  # 新tensor，无inplace
        x = torch.stack(out_list, dim=1).reshape(B, H, W, C)
        → 消除所有 inplace 操作
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 kernel_size=3,
                 dilation=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        if dilation is None:
            dilation = [1, 2]

        self.dim = dim
        self.num_heads = num_heads
        self.dilation = dilation
        self.num_dilation = len(dilation)
        self.kernel_size = kernel_size

        assert num_heads % self.num_dilation == 0, \
            f"num_heads ({num_heads}) must be divisible by num_dilation ({self.num_dilation})"

        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)

        self.dilate_attention = nn.ModuleList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
            for i in range(self.num_dilation)
        ])

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # QKV projection: [B, C, H, W] → [B, 3*C, H, W]
        qkv = self.qkv(x)

        # Split into num_dilation groups
        # [B, 3*C, H, W] → [B, 3, num_dilation, C//num_dilation, H, W]
        # → permute → [num_dilation, 3, B, C//num_dilation, H, W]
        qkv = qkv.reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W)
        qkv = qkv.permute(2, 1, 0, 3, 4, 5)

        # ---- 修复：用 list 收集，不做 inplace 赋值 ----
        out_list = []
        for i in range(self.num_dilation):
            # qkv[i][0/1/2]: [B, C//num_dilation, H, W]
            # dilate_attention返回: [B, H, W, C//num_dilation]
            out_i = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
            out_list.append(out_i)

        # stack: [num_dilation, B, H, W, C//num_dilation]
        # → permute → [B, H, W, num_dilation, C//num_dilation]
        # → reshape → [B, H, W, C]
        x = torch.stack(out_list, dim=0).permute(1, 2, 3, 0, 4).reshape(B, H, W, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        # [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


# ===== Test =====
if __name__ == '__main__':
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_cases = [
        {'name': 'fusions[1] 48x48', 'H': 48, 'W': 48, 'C': 256, 'dilation': [1, 2]},
        {'name': 'fusions[2] 24x24', 'H': 24, 'W': 24, 'C': 256, 'dilation': [2, 4]},
        {'name': 'fusions[3] 12x12', 'H': 12, 'W': 12, 'C': 256, 'dilation': [4, 8]},
    ]

    print("="*60)
    print("MSDA inplace修复验证（梯度反传测试）")
    print("="*60)

    for tc in test_cases:
        x = torch.randn(2, tc['C'], tc['H'], tc['W'],
                        requires_grad=True).to(device)
        msda = MultiScaleDilatedAttention(
            dim=tc['C'], num_heads=8, kernel_size=3,
            dilation=tc['dilation']).to(device)

        try:
            y = msda(x)
            loss = y.mean()
            loss.backward()
            grad_ok = x.grad is not None
            print(f"  [{tc['name']}] output={list(y.shape)} "
                  f"grad={'✓' if grad_ok else '✗'}")
        except Exception as e:
            print(f"  [{tc['name']}] ✗ {e}")