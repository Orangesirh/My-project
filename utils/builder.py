import os, errno
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.loss import *
from datasets.syntodd import SynTodd
from datasets.clearpose import ClearPose

from utils.enhanced_seg_loss import EnhancedSegmentationLoss
    

def multiview_collate(batch):
    targets = []
    for ii in range(len(batch[0])):
        targets.append([batch_element[ii] for batch_element in batch])

    stacked_images = torch.stack(targets[0])
    stacked_camera_poses = torch.stack(targets[1])
    stacked_intrinsics = torch.stack(targets[2])

    return stacked_images, stacked_camera_poses, targets[2], targets[3], targets[4], targets[5], targets[
        6], targets[9]



def get_dataloader(config, mode):
    dataset_name = config["Dataset"]["dataset_name"]
    dataset_config = config["Dataset"][dataset_name]

    if dataset_name == "syntodd":
        dataset = SynTodd(dataset_config, mode)
    elif (dataset_name == "clearpose"):
        dataset = ClearPose(dataset_config, mode)
    else:
        raise NotImplementedError(f'Invalid dataset type: {dataset_name}.')
    
    batch_size = dataset_config["batch_size"]
    num_workers = dataset_config["num_workers"]
    shuffle = dataset_config[f"{mode}_shuffle"]
    
    return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,  # ← 新增！
            prefetch_factor=4  # ← 新增！每个worker预取4个batch
            )


def get_losses(config):
    """获取损失函数"""
    def NoneFunction(a, b):
        return 0

    # 初始化所有损失函数为NoneFunction
    loss_depth = NoneFunction
    loss_segmentation = NoneFunction
    loss_depth_grad = NoneFunction
    loss_depth_normal = NoneFunction
    loss_seg_iou = NoneFunction

    # 获取配置信息
    loss_depth_type = config['Trainer']['loss_depth']
    loss_seg_type = config['Trainer']['loss_segmentation']
    model_type = config['Model']['type']
    dataset_name = config["Dataset"]["dataset_name"]
    num_classes = len(config['Dataset'][dataset_name]['classes'])
    
    print(f"Loss configuration debug:")
    print(f"  Model type: {model_type}")
    print(f"  Loss depth type: {loss_depth_type}")
    print(f"  Loss seg type: {loss_seg_type}")
    print(f"  Number of classes: {num_classes}")
    
    # 根据模型类型和配置初始化损失函数
    if model_type in ['full', 'depth']:
        print("Initializing depth losses...")
        
        # 主要深度损失
        if 'mse' in loss_depth_type:
            loss_depth = nn.MSELoss()
            print("  - MSE depth loss initialized")
        elif 'l1' in loss_depth_type:
            loss_depth = MaskedL1Loss()
            print("  - L1 depth loss initialized")
        elif 'smooth' in loss_depth_type:
            loss_depth = MaskedSmoothL1Loss()
            print("  - Smooth L1 depth loss initialized")
        
        # 梯度损失
        if 'grad' in loss_depth_type:
            loss_depth_grad = GradientLoss()
            print("  - Gradient loss initialized")
        
        # 法向量损失
        if 'normal' in loss_depth_type:
            loss_depth_normal = nn.L1Loss()
            print("  - Normal loss initialized")
        
    if model_type in ['full', 'seg']:
        print("Initializing segmentation losses...")
        
        # 主要分割损失
        # if 'ce' in loss_seg_type:
        #     loss_segmentation = nn.CrossEntropyLoss()
        #     print("  - CrossEntropy segmentation loss initialized")

        # ========== 关键修改：支持增强分割损失 ==========
        if 'enhanced' in loss_seg_type:
            # 使用增强分割损失
            loss_segmentation = EnhancedSegmentationLoss(
                edge_weight=config['Trainer'].get('seg_edge_weight', 2.0),
                transparent_class=2,  # syntodd数据集中透明物体是class 2
                transparent_weight=config['Trainer'].get('seg_transparent_weight', 1.5),
                num_classes=num_classes,
                use_edge=config['Trainer'].get('use_edge_aware', True),
                use_transparent_focus=config['Trainer'].get('use_transparent_focus', True)
            )
            print("  - Enhanced segmentation loss initialized (Edge + Transparent Focus)")
        elif 'ce' in loss_seg_type:
            # 使用基础CE损失
            loss_segmentation = nn.CrossEntropyLoss()
            print("  - CrossEntropy segmentation loss initialized")
        # ================================================

        # IoU损失
        if 'iou' in loss_seg_type:
            loss_seg_iou = SegIouLoss(num_classes)
            print("  - IoU segmentation loss initialized")
    
    # 验证至少有一个主要损失函数被初始化
    has_main_loss = (loss_depth != NoneFunction) or (loss_segmentation != NoneFunction)
    if not has_main_loss:
        print("WARNING: No main loss function initialized! Check your configuration.")
        print(f"  Model type: {model_type}")
        print(f"  Depth loss config: {loss_depth_type}")
        print(f"  Seg loss config: {loss_seg_type}")
        
        # 提供默认损失函数作为fallback
        if model_type in ['full', 'depth']:
            loss_depth = nn.MSELoss()
            print("  - Fallback: Using MSE as default depth loss")
        if model_type in ['full', 'seg']:
            loss_segmentation = nn.CrossEntropyLoss()
            print("  - Fallback: Using CrossEntropy as default seg loss")
    
    print(f"Final loss functions:")
    print(f"  - Depth: {type(loss_depth).__name__}")
    print(f"  - Depth Grad: {type(loss_depth_grad).__name__}")
    print(f"  - Depth Normal: {type(loss_depth_normal).__name__}")
    print(f"  - Segmentation: {type(loss_segmentation).__name__}")
    print(f"  - Seg IoU: {type(loss_seg_iou).__name__}")
        
    return loss_depth, loss_depth_grad, loss_depth_normal, loss_segmentation, loss_seg_iou


def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_optimizer(config, net):
    names = set([name.split('.')[0] for name, _ in net.named_modules()]) - set(['', 'transformer_encoders'])
    params_backbone = net.transformer_encoders.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net."+name).parameters())

    if config['Trainer']['optim'] == 'adam':
        optimizer_backbone = optim.Adam(params_backbone, lr=config['Trainer']['lr_backbone'])
        optimizer_scratch = optim.Adam(params_scratch, lr=config['Trainer']['lr_scratch'])
    elif config['Trainer']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(params_backbone, lr=config['Trainer']['lr_backbone'], momentum=config['Trainer']['momentum'])
        optimizer_scratch = optim.SGD(params_scratch, lr=config['Trainer']['lr_scratch'], momentum=config['Trainer']['momentum'])
    return optimizer_backbone, optimizer_scratch


def get_schedulers(optimizers):
    return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]