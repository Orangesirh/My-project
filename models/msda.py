"""
Multi-Scale Dilated Attention (MSDA) for MODEST
Adapted for transparent object depth estimation and segmentation

Key modifications from original DilateFormer:
1. Optimized for MODEST's feature dimensions
2. Efficient implementation for different scales
3. Compatible with (B,C,H,W) format used in Fusion.py
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
        
        # Unfold: extracts local patches with dilation
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
        
        # Reshape query: [B, d, H, W] → [B, h, head_dim, 1, H*W] → [B, h, H*W, 1, head_dim]
        q = q.reshape(B, num_heads, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2)
        
        # Extract local keys via unfold: [B, d, H, W] → [B, d*k*k, H*W]
        k = self.unfold(k)
        k = k.reshape(B, num_heads, self.head_dim, 
                      self.kernel_size ** 2, H * W).permute(0, 1, 4, 2, 3)
        # → [B, h, H*W, head_dim, k*k]
        
        # Compute attention: [B, h, H*W, 1, head_dim] @ [B, h, H*W, head_dim, k*k]
        # → [B, h, H*W, 1, k*k]
        attn = (q @ k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Extract and apply to values
        v = self.unfold(v)
        v = v.reshape(B, num_heads, self.head_dim, 
                      self.kernel_size ** 2, H * W).permute(0, 1, 4, 3, 2)
        # → [B, h, H*W, k*k, head_dim]
        
        # Weighted sum: [B, h, H*W, 1, k*k] @ [B, h, H*W, k*k, head_dim]
        # → [B, h, H*W, 1, head_dim] → [B, H, W, d]
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        
        return x


class MultiScaleDilatedAttention(nn.Module):
    """
    Multi-Scale Dilated Attention for MODEST
    
    Key design:
    - Splits channels into N groups (N = num dilation rates)
    - Each group processes with different dilation rate
    - Parallel computation then concatenate
    
    For transparent objects:
    - Small dilation captures local refraction details
    - Large dilation captures global light propagation
    
    Args:
        dim: Input feature dimension (must match Fusion layer)
        num_heads: Total number of attention heads (must be divisible by len(dilation))
        kernel_size: Local window size (default: 3)
        dilation: List of dilation rates for multi-scale
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Dropout rate for attention
        proj_drop: Dropout rate for output projection
    """
    def __init__(self, 
                 dim, 
                 num_heads=8,
                 kernel_size=3,
                 dilation=[1, 2, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.dilation = dilation
        self.num_dilation = len(dilation)
        self.kernel_size = kernel_size
        
        # Ensure heads can be evenly divided among scales
        assert num_heads % self.num_dilation == 0, \
            f"num_heads ({num_heads}) must be divisible by num_dilation ({self.num_dilation})"
        
        head_dim = dim // num_heads
        self.head_dim = head_dim
        
        # QKV projection (Conv2d for spatial preservation)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        
        # Create attention module for each dilation rate
        self.dilate_attention = nn.ModuleList([
            DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
            for i in range(self.num_dilation)
        ])
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - Input features from Fusion (MODEST format)
        Returns:
            x: [B, C, H, W] - Multi-scale enhanced features
        """
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        # [B, C, H, W] → [B, 3*C, H, W]
        qkv = self.qkv(x)
        
        # Reshape for multi-scale processing
        # [B, 3*C, H, W] → [B, 3, num_scales, C//num_scales, H, W]
        # → [num_scales, 3, B, C//num_scales, H, W]
        qkv = qkv.reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W)
        qkv = qkv.permute(2, 1, 0, 3, 4, 5)
        
        # Prepare output container
        # [B, C, H, W] → [B, num_scales, C//num_scales, H, W]
        # → [num_scales, B, H, W, C//num_scales]
        outputs = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W)
        outputs = outputs.permute(1, 0, 3, 4, 2)
        
        # Process each scale independently
        for i in range(self.num_dilation):
            # qkv[i][0]: Q, qkv[i][1]: K, qkv[i][2]: V
            # Each is [B, C//num_scales, H, W]
            outputs[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
            # → [B, H, W, C//num_scales]
        
        # Concatenate all scales
        # [num_scales, B, H, W, C//num_scales] → [B, H, W, num_scales, C//num_scales]
        # → [B, H, W, C]
        x = outputs.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Convert back to MODEST format: [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return x


# ===== Test and validation =====
if __name__ == '__main__':
    print("="*60)
    print("Testing MSDA for MODEST Integration")
    print("="*60)
    
    # Test cases matching MODEST's 4-layer pyramid
    test_cases = [
        {
            'name': 'index=3 (Coarsest)',
            'H': 12, 'W': 12,
            'C': 2048,  # resample_dim * 8
            'dilation': [4, 8, 16],
            'expected_receptive_field': '100% coverage'
        },
        {
            'name': 'index=2',
            'H': 24, 'W': 24,
            'C': 1024,  # resample_dim * 4
            'dilation': [2, 4, 8],
            'expected_receptive_field': '67% coverage'
        },
        {
            'name': 'index=1',
            'H': 48, 'W': 48,
            'C': 512,  # resample_dim * 2
            'dilation': [1, 2, 4],
            'expected_receptive_field': '33% coverage'
        },
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"Resolution: {test['H']}×{test['W']}, Channels: {test['C']}")
        print(f"Dilation rates: {test['dilation']}")
        print(f"Expected receptive field: {test['expected_receptive_field']}")
        print('-'*60)
        
        # Create input matching MODEST's Fusion format
        x = torch.randn(2, test['C'], test['H'], test['W'])
        print(f"Input shape: {x.shape}")
        
        # Create MSDA module
        msda = MultiScaleDilatedAttention(
            dim=test['C'],
            num_heads=8,
            kernel_size=3,
            dilation=test['dilation']
        )
        
        # Forward pass
        try:
            with torch.no_grad():
                y = msda(x)
            
            print(f"Output shape: {y.shape}")
            print(f"✅ Success! Dimensions match: {y.shape == x.shape}")
            
            # Calculate parameters
            params = sum(p.numel() for p in msda.parameters())
            print(f"Parameters: {params/1e6:.2f}M")
            
            # Estimate FLOPs (rough)
            flops = 3 * test['C'] ** 2 * test['H'] * test['W']  # QKV projection
            flops += len(test['dilation']) * 9 * test['H'] * test['W'] * (test['C'] // len(test['dilation']))
            print(f"Estimated FLOPs: {flops/1e6:.1f}M")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print("="*60)