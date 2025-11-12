"""
增强分割损失模块（修复GPU设备问题 + 修复测试代码）
专门针对透明物体分割的边界增强和类别聚焦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareSegmentationLoss(nn.Module):
    """
    边界感知分割损失
    对分割边界区域施加更高惩罚权重
    """
    def __init__(self, edge_weight=2.0, num_classes=3):
        super().__init__()
        self.edge_weight = edge_weight
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # Sobel算子（边界检测）
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 使用register_buffer确保自动迁移
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def detect_edges(self, seg_mask):
        """
        使用Sobel算子检测边界
        
        Args:
            seg_mask: [B, H, W] 分割标签
        Returns:
            edge_mask: [B, 1, H, W] 边界mask
        """
        seg_float = seg_mask.float().unsqueeze(1)  # [B, 1, H, W]
        
        # 确保sobel算子在正确设备上（防御性编程）
        if seg_float.device != self.sobel_x.device:
            self.sobel_x = self.sobel_x.to(seg_float.device)
            self.sobel_y = self.sobel_y.to(seg_float.device)
        
        # 应用Sobel算子
        edge_x = F.conv2d(seg_float, self.sobel_x, padding=1)
        edge_y = F.conv2d(seg_float, self.sobel_y, padding=1)
        
        # 计算梯度幅值
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        
        # 二值化（阈值可调）
        edge_mask = (edge_magnitude > 0.1).float()
        
        return edge_mask
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] 预测logits
            target: [B, H, W] GT标签
        Returns:
            loss: 标量损失值
        """
        # 基础CE损失（逐像素）
        base_loss = self.ce_loss(pred, target)  # [B, H, W]
        
        # 检测边界
        edge_mask = self.detect_edges(target).squeeze(1)  # [B, H, W]
        non_edge_mask = 1.0 - edge_mask
        
        # 计算边界像素数和非边界像素数
        edge_count = edge_mask.sum() + 1e-6
        non_edge_count = non_edge_mask.sum() + 1e-6
        
        # 分别计算边界和非边界损失
        edge_loss = (base_loss * edge_mask).sum() / edge_count
        non_edge_loss = (base_loss * non_edge_mask).sum() / non_edge_count
        
        # 总损失：边界区域权重更高
        total_loss = non_edge_loss + self.edge_weight * edge_loss
        
        return total_loss


class TransparentFocusedLoss(nn.Module):
    """
    透明类别聚焦损失
    对透明类别（class=2）施加更高权重
    """
    def __init__(self, transparent_class=2, transparent_weight=1.5, num_classes=3):
        super().__init__()
        self.transparent_class = transparent_class
        
        # 构造类别权重
        class_weights = torch.ones(num_classes)
        class_weights[transparent_class] = transparent_weight
        
        # 注册为buffer，自动处理设备迁移
        self.register_buffer('class_weights', class_weights)
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, H, W]
        Returns:
            loss: 标量损失值
        """
        # 确保权重在正确设备上
        if pred.device != self.class_weights.device:
            self.class_weights = self.class_weights.to(pred.device)
        
        # 使用手动权重计算
        ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')
        return ce_loss_fn(pred, target)


class EnhancedSegmentationLoss(nn.Module):
    """
    组合增强分割损失
    = 基础CE + 边界加权 + 透明类别加权
    """
    def __init__(self, 
                 edge_weight=2.0,
                 transparent_class=2,
                 transparent_weight=1.5,
                 num_classes=3,
                 use_edge=True,
                 use_transparent_focus=True):
        super().__init__()
        
        self.use_edge = use_edge
        self.use_transparent_focus = use_transparent_focus
        
        # 基础CE损失
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 边界感知损失
        if use_edge:
            self.edge_loss = EdgeAwareSegmentationLoss(
                edge_weight=edge_weight,
                num_classes=num_classes
            )
        
        # 透明类别聚焦损失
        if use_transparent_focus:
            self.transparent_loss = TransparentFocusedLoss(
                transparent_class=transparent_class,
                transparent_weight=transparent_weight,
                num_classes=num_classes
            )
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, H, W]
        Returns:
            loss: 标量损失值（与原始CE接口一致）
        """
        # 基础CE损失
        loss_ce = self.ce_loss(pred, target)
        total_loss = loss_ce
        
        # 边界感知损失
        if self.use_edge:
            loss_edge = self.edge_loss(pred, target)
            total_loss = total_loss + loss_edge
        
        # 透明类别聚焦损失
        if self.use_transparent_focus:
            loss_transparent = self.transparent_loss(pred, target)
            total_loss = total_loss + loss_transparent
        
        return total_loss


# ========== 测试代码（修复版） ==========
if __name__ == '__main__':
    print("测试增强分割损失...")
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    B, H, W = 2, 96, 96
    num_classes = 3
    
    # 测试边界感知损失
    print("\n1. 边界感知损失")
    pred = torch.randn(B, num_classes, H, W, requires_grad=True).to(device)
    target = torch.randint(0, num_classes, (B, H, W)).to(device)
    
    edge_loss_fn = EdgeAwareSegmentationLoss(edge_weight=2.0, num_classes=3).to(device)
    loss = edge_loss_fn(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Device: pred={pred.device}, sobel_x={edge_loss_fn.sobel_x.device}")
    
    # 测试透明类别聚焦损失
    print("\n2. 透明类别聚焦损失")
    pred = torch.randn(B, num_classes, H, W, requires_grad=True).to(device)
    trans_loss_fn = TransparentFocusedLoss(transparent_class=2, transparent_weight=1.5, num_classes=3).to(device)
    loss = trans_loss_fn(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Device: pred={pred.device}, weights={trans_loss_fn.class_weights.device}")
    
    # 测试组合损失
    print("\n3. 组合增强损失")
    pred = torch.randn(B, num_classes, H, W, requires_grad=True).to(device)
    enhanced_loss_fn = EnhancedSegmentationLoss(
        edge_weight=2.0,
        transparent_class=2,
        transparent_weight=1.5,
        num_classes=3
    ).to(device)
    loss = enhanced_loss_fn(pred, target)
    print(f"   Total Loss: {loss.item():.4f}")
    
    # 测试反向传播
    print("\n4. 测试反向传播")
    loss.backward()
    print("   ✓ 反向传播成功")
    print(f"   ✓ 梯度shape: {pred.grad.shape}")
    
    print("\n✓ 所有测试通过！")