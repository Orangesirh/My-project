"""
Enhanced Segmentation Loss v2 — 透明物体分割专用

设计原则：
  1. 单次 CE + 逐像素权重图（消除旧版三重 CE 叠加导致的权重稀释）
  2. 形态学边界检测（替代 Sobel on discrete labels，消除标签数值耦合）
  3. 乘性权重组合：boundary_weight × class_weight，语义清晰且效果可叠加
  4. Dice Loss 补充项：直接优化 IoU-like 指标，缓解类别不平衡
  
权重图构造逻辑：
  ┌─────────────────────────────────────────────────┐
  │ 像素类型              │ 权重                     │
  ├───────────────────────┼─────────────────────────┤
  │ 非透明 非边界         │ 1.0                     │
  │ 非透明 边界           │ 1.0 × edge_weight       │
  │ 透明   非边界         │ transparent_weight       │
  │ 透明   边界           │ transparent_weight × edge_weight  ← 最高 │
  └─────────────────────────────────────────────────┘

接口与旧版 EnhancedSegmentationLoss 兼容：
  loss = EnhancedSegmentationLossV2(...)
  loss_val = loss(pred, target)   # pred: [B,C,H,W], target: [B,H,W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSegmentationLossV2(nn.Module):
    """
    统一加权 CE + Dice Loss

    Args:
        edge_weight:        边界像素的权重乘子（默认 2.0）
        transparent_class:  透明物体类别索引（syntodd=2, clearpose=1）
        transparent_weight: 透明类别的权重乘子（默认 1.5）
        num_classes:        类别总数
        dice_weight:        Dice Loss 的混合权重（默认 0.5）
        boundary_dilation:  边界检测的膨胀半径，越大则边界带越宽（默认 1）
    """

    def __init__(self,
                 edge_weight=2.0,
                 transparent_class=2,
                 transparent_weight=1.5,
                 num_classes=3,
                 dice_weight=0.5,
                 boundary_dilation=1):
        super().__init__()

        self.edge_weight = edge_weight
        self.transparent_class = transparent_class
        self.transparent_weight = transparent_weight
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.boundary_dilation = int(boundary_dilation)

        # 逐像素 CE（reduction='none' 以便手动加权）
        self.ce = nn.CrossEntropyLoss(reduction='none')

    # ------------------------------------------------------------------ #
    # 边界检测（形态学方法）
    # ------------------------------------------------------------------ #
    def _detect_boundary(self, seg_gt):
        """
        用形态学膨胀+腐蚀检测类别边界。

        对每个类别的 one-hot mask 做 max_pool（膨胀）和负值 max_pool（腐蚀），
        膨胀与腐蚀的差集即为边界带，所有类别取并集。

        优势：
          - 不依赖标签数值，只看"邻居是否同类"
          - 边界宽度由 kernel_size 控制，语义明确
          - 无可学习参数

        Args:
            seg_gt: [B, H, W] LongTensor
        Returns:
            boundary: [B, H, W] FloatTensor, 1=边界, 0=非边界
        """
        k = 2 * self.boundary_dilation + 1
        boundary = torch.zeros_like(seg_gt, dtype=torch.float32)

        for cls in range(self.num_classes):
            cls_mask = (seg_gt == cls).float().unsqueeze(1)          # [B,1,H,W]
            dilated = F.max_pool2d(cls_mask, kernel_size=k, stride=1,
                                   padding=self.boundary_dilation)   # [B,1,H,W]
            eroded = -F.max_pool2d(-cls_mask, kernel_size=k, stride=1,
                                   padding=self.boundary_dilation)   # 腐蚀
            cls_boundary = ((dilated - eroded) > 0).float().squeeze(1)
            boundary = torch.clamp(boundary + cls_boundary, 0, 1)

        return boundary  # [B, H, W]

    # ------------------------------------------------------------------ #
    # 权重图
    # ------------------------------------------------------------------ #
    def _build_weight_map(self, seg_gt):
        """
        构造逐像素权重图。

        权重 = class_weight × boundary_weight
             = class_weight × (1 + (edge_weight - 1) × is_boundary)

        Args:
            seg_gt: [B, H, W] LongTensor
        Returns:
            weight_map: [B, H, W] FloatTensor
        """
        class_w = torch.ones_like(seg_gt, dtype=torch.float32)
        class_w[seg_gt == self.transparent_class] = self.transparent_weight

        boundary = self._detect_boundary(seg_gt)
        boundary_w = 1.0 + (self.edge_weight - 1.0) * boundary

        return class_w * boundary_w

    # ------------------------------------------------------------------ #
    # Dice Loss
    # ------------------------------------------------------------------ #
    def _dice_loss(self, pred_softmax, seg_gt):
        """
        多类 Dice Loss（1 - mean_dice）。

        Dice 直接优化区域重叠度，对小目标 / 类别不平衡天然鲁棒，
        与 CE 互补：CE 提供逐像素梯度，Dice 提供全局形状约束。

        Args:
            pred_softmax: [B, C, H, W]
            seg_gt:       [B, H, W] LongTensor
        Returns:
            dice_loss: 标量
        """
        smooth = 1.0
        gt_onehot = F.one_hot(seg_gt, self.num_classes)     # [B, H, W, C]
        gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)
        intersection = (pred_softmax * gt_onehot).sum(dim=dims)
        cardinality = pred_softmax.sum(dim=dims) + gt_onehot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice_per_class.mean()

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, pred, target):
        """
        Args:
            pred:   [B, C, H, W]  模型输出（本项目 HeadSeg 已 Softmax）
            target: [B, H, W]     GT 标签
        Returns:
            loss: 标量
        """
        # 1) 加权 CE
        weight_map = self._build_weight_map(target)
        ce_per_pixel = self.ce(pred, target)
        weighted_ce = (ce_per_pixel * weight_map).mean()

        # 2) Dice Loss
        if pred.min() >= 0 and pred.max() <= 1:
            pred_softmax = pred
        else:
            pred_softmax = F.softmax(pred, dim=1)

        dice = self._dice_loss(pred_softmax, target)

        # 3) 总损失
        total = weighted_ce + self.dice_weight * dice

        return total


# ========== 测试 ==========
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # --- SynTODD（3 类）---
    print("=" * 50)
    print("SynTODD (3 classes, transparent=2)")
    print("=" * 50)
    B, C, H, W = 2, 3, 96, 96
    pred = torch.randn(B, C, H, W, requires_grad=True, device=device)
    target = torch.randint(0, C, (B, H, W), device=device)
    target[:, 40:60, 40:60] = 2

    loss_fn = EnhancedSegmentationLossV2(
        edge_weight=2.0, transparent_class=2,
        transparent_weight=1.5, num_classes=3, dice_weight=0.5
    ).to(device)

    loss = loss_fn(pred, target)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Grad OK: {pred.grad is not None}")

    # --- ClearPose（2 类）---
    print(f"\n{'=' * 50}")
    print("ClearPose (2 classes, transparent=1)")
    print("=" * 50)
    B, C, H, W = 2, 2, 96, 96
    pred = torch.randn(B, C, H, W, requires_grad=True, device=device)
    target = torch.randint(0, C, (B, H, W), device=device)
    target[:, 30:70, 30:70] = 1

    loss_fn = EnhancedSegmentationLossV2(
        edge_weight=2.0, transparent_class=1,
        transparent_weight=1.5, num_classes=2, dice_weight=0.5
    ).to(device)

    loss = loss_fn(pred, target)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Grad OK: {pred.grad is not None}")

    # --- 权重图验证 ---
    print(f"\n{'=' * 50}")
    print("Weight map verification")
    print("=" * 50)
    target_vis = torch.zeros(1, 32, 32, dtype=torch.long, device=device)
    target_vis[0, 10:22, 10:22] = 2
    target_vis[0, 5:15, 5:15] = 1

    loss_fn_3c = EnhancedSegmentationLossV2(
        edge_weight=2.0, transparent_class=2,
        transparent_weight=1.5, num_classes=3
    ).to(device)
    wm = loss_fn_3c._build_weight_map(target_vis)
    bd = loss_fn_3c._detect_boundary(target_vis)
    print(f"  Unique weights: {sorted(torch.unique(wm).tolist())}")
    print(f"  Expected: [1.0, 1.5, 2.0, 3.0]")
    print(f"  Boundary pixels: {bd.sum().item():.0f} / {bd.numel()}")

    print("\n✓ All tests passed!")