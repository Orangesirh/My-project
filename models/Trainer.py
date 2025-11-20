import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import cv2
import torch.nn.functional as F
import datetime

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from numpy.core.numeric import Inf
from models.Model import ISGNet
from utils.builder import get_losses, get_optimizer, get_schedulers, create_dir
from utils.loss import get_surface_normal
from utils.visualize import *
from utils.evaluate import compute_depth_metrics, compute_seg_metrics

from utils.progress import create_train_progress_bar, create_val_progress_bar, safe_write

# from utils.enhanced_seg_loss import EnhancedSegmentationLoss


class Trainer(object):

    def __init__(self, config):
        super().__init__()
        self.config = config
        ## type of predictions (full/seg/depth)
        self.type = config['Model']['type']
        ## choose the dataset (Syntodd or Clearpose)
        self.dataset_name = config["Dataset"]["dataset_name"]
        ## the number of semantic categories
        self.num_classes = len(config['Dataset'][self.dataset_name]['classes'])
        ## loss weights
        self.seg_multi = config['Trainer']['seg_multi']
        self.depth_multi = config['Trainer']['depth_multi']
        self.depth_scale_multi = config['Trainer']['depth_scale_multi']
        self.depth_grad_multi = config['Trainer']['depth_grad_multi']
        self.depth_normal_multi = config['Trainer']['depth_normal_multi']
        self.depth_error_multi = config['Trainer']['depth_error_multi']
        ## coefficient between different iterations
        self.gamma = config['Trainer']['gamma']
        ## resolution of multi-scale predictions
        self.resolutions = [48, 96, 192, 384]
        self.device = torch.device(config['Model']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        
        # 打印损失权重配置（调试用）
        print(f"损失权重配置: seg_multi={self.seg_multi}, depth_multi={self.depth_multi}")
        print(f"深度子损失权重: scale={self.depth_scale_multi}, grad={self.depth_grad_multi}, normal={self.depth_normal_multi}")
        
        im_resize = config['Dataset'][self.dataset_name]['transforms']['im_resize']
        
        ## define the model
        self.model = ISGNet(
                    image_size  =   (3,im_resize,im_resize),
                    emb_dim     =   config['Model']['emb_dim'],
                    resample_dim=   config['Model']['resample_dim'],
                    read        =   config['Model']['read'],
                    nclasses    =   self.num_classes,
                    hooks       =   config['Model']['hooks'],
                    model_timm  =   config['Model']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['Model']['patch_size'],
                    pretrain    =   config['Model']['pretrain'],
                    iterations  =   config['Model']['iterations'],
                    in_chans    =   config['Dataset'][self.dataset_name]["in_chans"],
                    coord_reduction = config['Model'].get('coord_reduction', 32),  # ← 新增，默认32
                    use_eds_at_finest=config['Model'].get('use_eds_at_finest', True)  # ← 添加这行
        )

        self.model.to(self.device)
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")

        ## loss functions
        self.loss_depth, self.loss_depth_grad, self.loss_depth_normal, self.loss_seg, self.loss_seg_iou = get_losses(config)

        # Debug: 打印损失函数类型和检查是否为NoneFunction
        print(f"损失函数初始化: depth={self.loss_depth}, seg={self.loss_seg}")
        
        # 检查损失函数是否有效
        self.has_depth_loss = hasattr(self.loss_depth, '__call__') and 'NoneFunction' not in str(self.loss_depth)
        self.has_depth_grad_loss = hasattr(self.loss_depth_grad, '__call__') and 'NoneFunction' not in str(self.loss_depth_grad)
        self.has_depth_normal_loss = hasattr(self.loss_depth_normal, '__call__') and 'NoneFunction' not in str(self.loss_depth_normal)
        self.has_seg_loss = hasattr(self.loss_seg, '__call__') and 'NoneFunction' not in str(self.loss_seg)
        self.has_seg_iou_loss = hasattr(self.loss_seg_iou, '__call__') and 'NoneFunction' not in str(self.loss_seg_iou)
        
        print(f"有效损失函数检查:")
        print(f"  - 深度损失: {'启用' if self.has_depth_loss else '禁用'}")
        print(f"  - 深度梯度损失: {'启用' if self.has_depth_grad_loss else '禁用'}")
        print(f"  - 深度法向量损失: {'启用' if self.has_depth_normal_loss else '禁用'}")
        print(f"  - 分割损失: {'启用' if self.has_seg_loss else '禁用'}")
        print(f"  - 分割IoU损失: {'启用' if self.has_seg_iou_loss else '禁用'}")
        
        if not self.has_depth_loss and not self.has_seg_loss:
            raise ValueError("至少需要启用一个主要损失函数（深度损失或分割损失）！请检查config.json中的损失函数配置。")
        
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])

        # 修正checkpoint路径配置
        self.ckpt_path = config['Trainer']['ckpt_path']
        
        # 创建更合理的模型保存路径
        model_save_dir = config['Model'].get('path_model', 'default_model')
        if model_save_dir == "The dir to save ckpt":  # 如果还是占位符，使用默认值
            model_save_dir = f"modest_{self.dataset_name}_{self.type}"
            
        self.path_model = os.path.join('ckpt', model_save_dir, self.model.__class__.__name__)
        self.path_statis = os.path.join('ckpt', model_save_dir, 'val_statis.txt')
        create_dir(os.path.dirname(self.path_model))
        create_dir(os.path.dirname(self.path_statis))

    def _safe_tensor_check(self, value):
        """安全地检查tensor是否为nan或inf"""
        if isinstance(value, (int, float)):
            return not (np.isnan(value) or np.isinf(value))
        elif torch.is_tensor(value):
            return not (torch.isnan(value).any() or torch.isinf(value).any())
        else:
            return False

    def _safe_item(self, value):
        """安全地获取数值"""
        if torch.is_tensor(value):
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0

    ############################ training ################################
    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['Trainer']['epochs']
        
        ## visualization
        use_wandb = self.config['wandb']['enable']
        if use_wandb:
            try:
                # 修正wandb配置
                project_name = self.config['wandb'].get('project_name', f'MODEST_{self.dataset_name}')
                username = self.config['wandb']['username']
                
                # 检查用户名是否还是占位符
                if username == "Modify here":
                    print("Warning: wandb username is still placeholder. Please update config.json")
                    username = "anonymous"
                
                run_name = f"{self.dataset_name}_{self.type}_{os.path.basename(self.path_model)}"
                
                wandb.init(
                    project=project_name, 
                    entity=username, 
                    name=run_name,
                    config={
                        "learning_rate_backbone": self.config['Trainer']['lr_backbone'],
                        "learning_rate_scratch": self.config['Trainer']['lr_scratch'],
                        "epochs": epochs,
                        "batch_size": self.config['Dataset'][self.dataset_name]['batch_size'],
                        "dataset": self.dataset_name,
                        "model_type": self.type
                    }
                )
                print("Wandb initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                use_wandb = False
        
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            print("="*50)
            print(f"Epoch {epoch+1}/{epochs}")
            print("="*50)
            
            ## training losses
            train_loss, train_depth_all_loss, train_seg_all_loss = 0.0, 0.0, 0.0
            train_grad_loss, train_normal_loss = 0.0, 0.0
            
            self.model.train()
            #pbar = tqdm(train_dataloader, desc="Training")
            ## ========== 使用优化的进度条 ========== 
            pbar = create_train_progress_bar(train_dataloader, epoch=epoch+1)
            ## ====================================

            for index, data in enumerate(pbar):
                try:
                    self.optimizer_backbone.zero_grad()
                    self.optimizer_scratch.zero_grad()

                    ## load the data
                    rgb = data['rgb'].to(self.device)
                    depth_gt = data['depth_gt'].to(self.device) 
                    seg_gt = data['seg_gt'].to(self.device)
                    zero_mask = data['zero_mask'].to(self.device)
                    loss_mask = data['loss_mask'].to(self.device)
                    depth_min = data['depth_min'][0].to(self.device) if torch.is_tensor(data['depth_min'][0]) else torch.tensor(data['depth_min'][0]).to(self.device)
                    depth_max = data['depth_max'][0].to(self.device) if torch.is_tensor(data['depth_max'][0]) else torch.tensor(data['depth_max'][0]).to(self.device)
                    
                    # Debug: 检查输入数据范围（只在第一个batch打印）
                    if index == 0:
                        print(f"\n首轮数据检查:")
                        print(f"RGB范围: {rgb.min().item():.4f} ~ {rgb.max().item():.4f}")
                        print(f"深度真值范围: {depth_gt.min().item():.4f} ~ {depth_gt.max().item():.4f}")
                        print(f"分割真值范围: {seg_gt.min().item()} ~ {seg_gt.max().item()} (类别数: {len(torch.unique(seg_gt))})")
                    
                    output_depths, output_segs = self.model(rgb)
                    
                    # Debug: 检查模型输出（只在第一个batch打印）
                    if index == 0:
                        print(f"模型输出深度范围: {output_depths[-1][-1].min().item():.4f} ~ {output_depths[-1][-1].max().item():.4f}")
                        print(f"模型输出分割范围: {output_segs[-1][-1].min().item():.4f} ~ {output_segs[-1][-1].max().item():.4f}")
                    
                    total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    depth_loss_total = torch.tensor(0.0, device=self.device)
                    seg_loss_total = torch.tensor(0.0, device=self.device)
                    
                    batch_depth_losses = []
                    batch_seg_losses = []
                    
                    ## calculate multi-scale depth losses
                    if self.has_depth_loss or self.has_depth_grad_loss or self.has_depth_normal_loss:
                        for iter_idx, output_depths_iter in enumerate(output_depths):
                            iter_weight = self.gamma * (iter_idx + 1)  # 修正迭代权重计算
                            
                            for scale_idx, (output_depth, resolution) in enumerate(zip(output_depths_iter, self.resolutions)):
                                # 准备对应尺度的ground truth
                                if resolution == 384:
                                    depth_gt_multiscale = depth_gt.clone()
                                else:
                                    depth_gt_multiscale = F.interpolate(depth_gt.unsqueeze(1), 
                                                                       size=(resolution, resolution), 
                                                                       mode="bilinear", align_corners=True).squeeze(1)
                                
                                # 主要深度损失
                                if self.has_depth_loss:
                                    output_depth_squeezed = output_depth.squeeze(1)
                                    depth_loss_main = self.loss_depth(output_depth_squeezed, depth_gt_multiscale)
                                    
                                    if self._safe_tensor_check(depth_loss_main):
                                        weighted_depth_loss = iter_weight * self.depth_scale_multi * depth_loss_main
                                        batch_depth_losses.append(weighted_depth_loss)
                                        
                                        if index == 0 and iter_idx == 0 and scale_idx == 0:
                                            print(f"深度尺度损失 (res={resolution}): {self._safe_item(depth_loss_main):.6f}")
                                
                                # 梯度损失
                                if self.has_depth_grad_loss:
                                    try:
                                        grad_loss = self.loss_depth_grad(output_depth, depth_gt_multiscale.unsqueeze(1))
                                        if self._safe_tensor_check(grad_loss):
                                            weighted_grad_loss = iter_weight * self.depth_grad_multi * grad_loss
                                            batch_depth_losses.append(weighted_grad_loss)
                                            train_grad_loss += self._safe_item(weighted_grad_loss)
                                            
                                            if index == 0 and iter_idx == 0 and scale_idx == 0:
                                                print(f"深度梯度损失 (res={resolution}): {self._safe_item(grad_loss):.6f}")
                                    except Exception as e:
                                        if index == 0:
                                            print(f"Warning: 梯度损失计算失败: {e}")
                                
                                # 法向量损失
                                if self.has_depth_normal_loss:
                                    try:
                                        output_normal, _, _ = get_surface_normal(output_depth, self.dataset_name)
                                        target_normal, _, _ = get_surface_normal(depth_gt_multiscale.unsqueeze(1), self.dataset_name)
                                        normal_loss = self.loss_depth_normal(output_normal, target_normal)
                                        if self._safe_tensor_check(normal_loss):
                                            weighted_normal_loss = iter_weight * self.depth_normal_multi * normal_loss
                                            batch_depth_losses.append(weighted_normal_loss)
                                            train_normal_loss += self._safe_item(weighted_normal_loss)
                                            
                                            if index == 0 and iter_idx == 0 and scale_idx == 0:
                                                print(f"深度法向量损失 (res={resolution}): {self._safe_item(normal_loss):.6f}")
                                    except Exception as e:
                                        if index == 0:
                                            print(f"Warning: 法向量损失计算失败: {e}")

                    ## calculate multi-scale segmentation losses
                    if self.has_seg_loss or self.has_seg_iou_loss:
                        for iter_idx, output_segs_iter in enumerate(output_segs):
                            iter_weight = self.gamma * (iter_idx + 1)
                            
                            for scale_idx, (output_seg, resolution) in enumerate(zip(output_segs_iter, self.resolutions)):
                                # 准备对应尺度的ground truth
                                if resolution == 384:
                                    seg_gt_multiscale = seg_gt.clone()
                                else:
                                    seg_gt_multiscale = F.interpolate(seg_gt.unsqueeze(1).float(), 
                                                                     size=(resolution, resolution), 
                                                                     mode="nearest").squeeze(1).long()
                                
                                # 分割损失
                                if self.has_seg_loss:
                                    seg_loss = self.loss_seg(output_seg, seg_gt_multiscale)
                                    if self._safe_tensor_check(seg_loss):
                                        weighted_seg_loss = iter_weight * seg_loss
                                        batch_seg_losses.append(weighted_seg_loss)
                                        
                                        if index == 0 and iter_idx == 0 and scale_idx == 0:
                                            print(f"分割损失 (res={resolution}): {self._safe_item(seg_loss):.6f}")
                                
                                # IoU损失 (如果启用)
                                if self.has_seg_iou_loss:
                                    try:
                                        seg_iou_loss = self.loss_seg_iou(output_seg, seg_gt_multiscale)
                                        if self._safe_tensor_check(seg_iou_loss):
                                            weighted_iou_loss = iter_weight * seg_iou_loss
                                            batch_seg_losses.append(weighted_iou_loss)
                                            
                                            if index == 0 and iter_idx == 0 and scale_idx == 0:
                                                print(f"分割IoU损失 (res={resolution}): {self._safe_item(seg_iou_loss):.6f}")
                                    except Exception as e:
                                        if index == 0:
                                            print(f"Warning: IoU损失计算失败: {e}")

                    # 组合所有损失
                    if batch_depth_losses:
                        depth_loss_total = sum(batch_depth_losses)
                        total_loss = total_loss + self.depth_multi * depth_loss_total
                        train_depth_all_loss += self._safe_item(self.depth_multi * depth_loss_total)
                    
                    if batch_seg_losses:
                        seg_loss_total = sum(batch_seg_losses)
                        total_loss = total_loss + self.seg_multi * seg_loss_total
                        train_seg_all_loss += self._safe_item(self.seg_multi * seg_loss_total)
                    
                    # Debug: 打印损失信息（只在第一个batch）
                    if index == 0:
                        print(f"损失组件汇总:")
                        print(f"  深度损失总计: {self._safe_item(depth_loss_total):.6f} (子损失数: {len(batch_depth_losses)})")
                        print(f"  分割损失总计: {self._safe_item(seg_loss_total):.6f} (子损失数: {len(batch_seg_losses)})")
                        print(f"  最终总损失: {self._safe_item(total_loss):.6f}")
                    
                    # 检查损失是否有效
                    loss_value = self._safe_item(total_loss)
                    if loss_value == 0 or not self._safe_tensor_check(total_loss):
                        if index == 0:
                            print(f"Warning: 无效损失 at batch {index}: {loss_value}")
                            print("这可能表明损失函数配置或数据存在问题")
                        continue
                    
                    # 反向传播
                    total_loss.backward()
                    
                    # 梯度裁剪（可选）
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0  # 将梯度范数限制在1.0以内
                    )

                    self.optimizer_scratch.step()
                    self.optimizer_backbone.step()

                    train_loss += loss_value
                    
                    # 更新进度条
                    # pbar.set_postfix({
                    #     'total_loss': f'{train_loss/(index+1):.6f}',
                    #     'depth_loss': f'{train_depth_all_loss/(index+1):.6f}',
                    #     'seg_loss': f'{train_seg_all_loss/(index+1):.6f}'
                    # })

                    if index % 10 == 0:
                        pbar.set_postfix({
                            'loss': f'{train_loss/(index+1):.6f}',
                            'depth': f'{train_depth_all_loss/(index+1):.6f}',
                            'seg': f'{train_seg_all_loss/(index+1):.6f}'
                        })

                  
                    ## debug
                    # if np.isnan(train_loss):
                    #     print(f'\nNaN detected at batch {index}:')
                    #     print(f'  RGB range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]')
                    #     print(f'  Depth GT range: [{depth_gt.min().item():.3f}, {depth_gt.max().item():.3f}]')
                    #     print(f'  Model output range: [{output_depths[-1][-1].min().item():.3f}, {output_depths[-1][-1].max().item():.3f}]')
                    #     print(f'  Total loss: {loss_value}')
                    #     raise ValueError("NaN loss detected")

                    if np.isnan(train_loss):
                        safe_write(pbar, f'\n⚠️  NaN detected at batch {index}:')
                        safe_write(pbar, f'  RGB range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]')
                        safe_write(pbar, f'  Depth GT range: [{depth_gt.min().item():.3f}, {depth_gt.max().item():.3f}]')
                        safe_write(pbar, f'  Model output range: [{output_depths[-1][-1].min().item():.3f}, {output_depths[-1][-1].max().item():.3f}]')
                        safe_write(pbar, f'  Total loss: {loss_value}')
                        raise ValueError("NaN loss detected")
                    
                    ## visualization
                    if use_wandb and ((index % 50 == 0 and index > 0) or index == len(train_dataloader)-1):
                        wandb.log({
                            "epoch": epoch + 1,
                            "batch": index,
                            "train_loss": train_loss / (index + 1),
                            "train_depth_all_loss": train_depth_all_loss / (index + 1),
                            "train_seg_all_loss": train_seg_all_loss / (index + 1),
                            "train_grad_loss": train_grad_loss / (index + 1),
                            "train_normal_loss": train_normal_loss / (index + 1),
                            "learning_rate_backbone": self.optimizer_backbone.param_groups[0]['lr'],
                            "learning_rate_scratch": self.optimizer_scratch.param_groups[0]['lr']
                        })
                
                except Exception as e:
                    print(f"Error in training batch {index}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            ## ========== 关闭进度条 ==========
            pbar.close()
            ## ==============================
            
            # 打印epoch总结
            avg_train_loss = train_loss / len(train_dataloader)
            avg_depth_loss = train_depth_all_loss / len(train_dataloader)
            avg_seg_loss = train_seg_all_loss / len(train_dataloader)
            
            print(f"\nEpoch {epoch+1} Training Summary:")
            print(f"  Average Total Loss: {avg_train_loss:.6f}")
            print(f"  Average Depth Loss: {avg_depth_loss:.6f}")
            print(f"  Average Seg Loss: {avg_seg_loss:.6f}")
            
            ## validation
            print("\nRunning validation...")
            new_val_loss, new_depth_eval, new_seg_eval = self.run_eval(val_dataloader)
            
            ## save best model
            if new_val_loss < best_val_loss:
                best_val_loss = new_val_loss
                best_epoch = epoch
                self.save_model('best')
                print(f"New best model saved at epoch {epoch+1} with val_loss: {new_val_loss:.6f}")
            
            ## save ckpts periodically
            if epoch % 5 == 0:
                self.save_model(f'epoch_{epoch}')

            # 学习率调度
            old_scratch_lr = [group['lr'] for group in self.optimizer_scratch.param_groups]
            old_backbone_lr = [group['lr'] for group in self.optimizer_backbone.param_groups]
            
            self.schedulers[0].step(new_val_loss)
            self.schedulers[1].step(new_val_loss)

            for i, group in enumerate(self.optimizer_backbone.param_groups):
                new_lr = group['lr']
                if new_lr != old_backbone_lr[i]:
                    print(f"Backbone Learning rate reduced from {old_backbone_lr[i]} to {new_lr} at epoch {epoch+1}")
            for i, group in enumerate(self.optimizer_scratch.param_groups):
                new_lr = group['lr']
                if new_lr != old_scratch_lr[i]:
                    print(f"Scratch Learning rate reduced from {old_scratch_lr[i]} to {new_lr} at epoch {epoch+1}")

        print(f'\nTraining completed!')
        print(f'Best epoch: {best_epoch+1}, Best val_loss: {best_val_loss:.6f}')
        if use_wandb:
            wandb.finish()

    ############################ validation ################################
    def run_eval(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        val_size = len(val_dataloader)
        val_loss = 0.
        val_depth_loss, val_seg_loss = 0.0, 0.0
        ## evaluation metrics
        MAE_all, RMSE_all, REL_all, DELTA105_all, DELTA110_all, DELTA125_all = [], [], [], [], [], []
        MAE_mask_all, RMSE_mask_all, REL_mask_all, DELTA105_mask_all, DELTA110_mask_all, DELTA125_mask_all = [], [], [], [], [], []
        IoU_all, mAP_all = [], []
        
        self.model.eval()
        use_wandb = self.config['wandb']['enable']
        
        with torch.no_grad():
            # pbar = tqdm(val_dataloader, desc="Validation")

            ## ========== 使用优化的验证进度条 ==========
            pbar = create_val_progress_bar(val_dataloader)
            ## ========================================

            for index, data in enumerate(pbar):
                try:
                    rgb = data['rgb'].to(self.device)
                    depth_gt = data['depth_gt'].to(self.device)
                    seg_gt = data['seg_gt'].to(self.device) 
                    zero_mask = data['zero_mask'].to(self.device)
                    loss_mask = data['loss_mask'].to(self.device)
                    depth_min = data['depth_min'][0].to(self.device) if torch.is_tensor(data['depth_min'][0]) else torch.tensor(data['depth_min'][0]).to(self.device)
                    depth_max = data['depth_max'][0].to(self.device) if torch.is_tensor(data['depth_max'][0]) else torch.tensor(data['depth_max'][0]).to(self.device)
                    
                    output_depths, output_segs = self.model(rgb)

                    total_loss = 0.0
                    depth_loss_total = 0.0
                    seg_loss_total = 0.0

                    # 计算验证损失 (只使用最后一次迭代的最高分辨率结果来简化)
                    if len(output_depths) > 0 and self.has_depth_loss:
                        final_depth = output_depths[-1][-1].squeeze(1)  # 最后一次迭代，最高分辨率
                        depth_loss_main = self.loss_depth(final_depth, depth_gt)
                        if self._safe_tensor_check(depth_loss_main):
                            depth_loss_total += self._safe_item(depth_loss_main)
                    
                    if len(output_segs) > 0 and self.has_seg_loss:
                        final_seg = output_segs[-1][-1]  # 最后一次迭代，最高分辨率
                        seg_loss_main = self.loss_seg(final_seg, seg_gt)
                        if self._safe_tensor_check(seg_loss_main):
                            seg_loss_total += self._safe_item(seg_loss_main)
                    
                    total_loss = self.depth_multi * depth_loss_total + self.seg_multi * seg_loss_total
                    val_loss += total_loss
                    val_depth_loss += self.depth_multi * depth_loss_total
                    val_seg_loss += self.seg_multi * seg_loss_total

                    # 计算评估指标
                    if len(output_depths) > 0:
                        MAE, RMSE, REL, DELTA105, DELTA110, DELTA125, MAE_, RMSE_, REL_, DELTA105_, DELTA110_, DELTA125_ \
                            = compute_depth_metrics(output_depths[-1][-1].squeeze(1), depth_gt, depth_min, depth_max, zero_masks=zero_mask, \
                                                    denorm=True, gt_masks=seg_gt, num_classes=self.num_classes)
                        MAE_all.append(MAE)
                        RMSE_all.append(RMSE)
                        REL_all.append(REL)
                        DELTA105_all.append(DELTA105)
                        DELTA110_all.append(DELTA110)
                        DELTA125_all.append(DELTA125)
                        MAE_mask_all.append(MAE_)
                        RMSE_mask_all.append(RMSE_)
                        REL_mask_all.append(REL_)
                        DELTA105_mask_all.append(DELTA105_)
                        DELTA110_mask_all.append(DELTA110_)
                        DELTA125_mask_all.append(DELTA125_)
                        
                    if len(output_segs) > 0:
                        IoU, mAP = compute_seg_metrics(output_segs[-1][-1], seg_gt, num_classes=self.num_classes)
                        IoU_all.append(IoU)
                        mAP_all.append(mAP)

                    # pbar.set_postfix({'val_loss': f'{val_loss/(index+1):.6f}'})

                    ## ========== 更新进度条（降低频率） ==========
                    if index % 10 == 0:
                        pbar.set_postfix({'val_loss': f'{val_loss/(index+1):.6f}'})
                    ## ==========================================
                    
                    ## 保存第一个batch用于可视化
                    if index == 0:
                        rgb_visual = rgb
                        depth_gt_visual = depth_gt
                        depth_pred_visual = output_depths
                        seg_gt_visual = seg_gt
                        seg_pred_visual = output_segs
                        zero_mask_visual = zero_mask
                
                except Exception as e:
                    print(f"Error in validation batch {index}: {e}")
                    continue
            
            ## ========== 关闭进度条 ==========
            pbar.close()
            ## ==============================

            # 计算平均指标
            val_loss_avg = val_loss / val_size
            val_depth_loss_avg = val_depth_loss / val_size
            val_seg_loss_avg = val_seg_loss / val_size
            
            ## visualization
            if use_wandb:
                log_dict = {
                    "val_loss": val_loss_avg,
                    "val_depth_loss": val_depth_loss_avg,
                    "val_seg_loss": val_seg_loss_avg
                }
                
                # 添加深度评估指标
                if len(MAE_all) > 0:
                    log_dict.update({
                        "val_MAE": sum(MAE_all) / len(MAE_all),
                        "val_RMSE": sum(RMSE_all) / len(RMSE_all),
                        "val_REL": sum(REL_all) / len(REL_all),
                        "val_MAE_mask": sum(MAE_mask_all) / len(MAE_mask_all),
                        "val_RMSE_mask": sum(RMSE_mask_all) / len(RMSE_mask_all),
                        "val_REL_mask": sum(REL_mask_all) / len(REL_mask_all)
                    })
                
                # 添加分割评估指标
                if len(IoU_all) > 0:
                    log_dict.update({
                        "val_IoU": sum(IoU_all) / len(IoU_all),
                        "val_mAP": sum(mAP_all) / len(mAP_all)
                    })
                
                wandb.log(log_dict)
                
                # 可视化结果
                try:
                    self.img_logger(rgb_visual, depth_gt_visual, seg_gt_visual, depth_pred_visual, seg_pred_visual, zero_mask_visual)
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")

            # 打印验证结果
            print(f"\nValidation Results:")
            print(f"  Average Loss: {val_loss_avg:.6f}")
            print(f"  Depth Loss: {val_depth_loss_avg:.6f}")
            print(f"  Seg Loss: {val_seg_loss_avg:.6f}")
            
            if len(MAE_all) > 0:
                MAE_mean = sum(MAE_all) / len(MAE_all)
                RMSE_mean = sum(RMSE_all) / len(RMSE_all)
                REL_mean = sum(REL_all) / len(REL_all)
                DELTA105_mean = sum(DELTA105_all) / len(DELTA105_all)
                DELTA110_mean = sum(DELTA110_all) / len(DELTA110_all)
                DELTA125_mean = sum(DELTA125_all) / len(DELTA125_all)
                MAE_mask_mean = sum(MAE_mask_all) / len(MAE_mask_all)
                RMSE_mask_mean = sum(RMSE_mask_all) / len(RMSE_mask_all)
                REL_mask_mean = sum(REL_mask_all) / len(REL_mask_all)
                DELTA105_mask_mean = sum(DELTA105_mask_all) / len(DELTA105_mask_all)
                DELTA110_mask_mean = sum(DELTA110_mask_all) / len(DELTA110_mask_all)
                DELTA125_mask_mean = sum(DELTA125_mask_all) / len(DELTA125_mask_all)
                
                depth_eval = MAE_mean + RMSE_mean + REL_mean if self.dataset_name == "syntodd" else MAE_mask_mean + RMSE_mask_mean + REL_mask_mean
                
                print(f"  Depth Metrics:")
                print(f"    RMSE: {RMSE_mean:.4f}, MAE: {MAE_mean:.4f}, REL: {REL_mean:.4f}")
                print(f"    δ<1.05: {DELTA105_mean:.1f}%, δ<1.10: {DELTA110_mean:.1f}%, δ<1.25: {DELTA125_mean:.1f}%")
                print(f"    Masked - RMSE: {RMSE_mask_mean:.4f}, MAE: {MAE_mask_mean:.4f}, REL: {REL_mask_mean:.4f}")
            else:
                depth_eval = val_loss_avg
            
            if len(IoU_all) > 0:
                IOU_mean = sum(IoU_all)/len(IoU_all)
                mAP_mean = sum(mAP_all)/len(mAP_all)
                seg_eval = IOU_mean + mAP_mean
                print(f"  Segmentation Metrics:")
                print(f"    mAP: {mAP_mean:.4f}, IoU: {IOU_mean:.4f}")
                
                with open(self.path_statis, 'a') as f:
                    f.write(f"Epoch validation - mAP: {mAP_mean:.5f} \t IoU: {IOU_mean:.5f} \n")
            else:
                seg_eval = val_loss_avg
            # ========== 新增：写入完整统计信息到文件 ==========
            import datetime
            with open(self.path_statis, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*80}\n")
                f.write(f"Validation @ {timestamp}\n")
                f.write(f"{'='*80}\n")
    
                # 损失
                f.write(f"Loss: Total={val_loss_avg:.6f}, Depth={val_depth_loss_avg:.6f}, Seg={val_seg_loss_avg:.6f}\n")
    
                # 深度指标
                if len(MAE_all) > 0:
                    f.write(f"\nDepth (Overall): RMSE={RMSE_mean:.4f}, MAE={MAE_mean:.4f}, REL={REL_mean:.4f}\n")
                    f.write(f"  Accuracy: δ<1.05={DELTA105_mean:.2f}%, δ<1.10={DELTA110_mean:.2f}%, δ<1.25={DELTA125_mean:.2f}%\n")
                    f.write(f"Depth (Transparent): RMSE={RMSE_mask_mean:.4f}, MAE={MAE_mask_mean:.4f}, REL={REL_mask_mean:.4f}\n")
                    f.write(f"  Accuracy: δ<1.05={DELTA105_mask_mean:.2f}%, δ<1.10={DELTA110_mask_mean:.2f}%, δ<1.25={DELTA125_mask_mean:.2f}%\n")
    
                # 分割指标
                if len(IoU_all) > 0:
                    f.write(f"Segmentation: mAP={mAP_mean:.4f}, IoU={IOU_mean:.4f}\n")
    
                f.write(f"{'='*80}\n")
            # ====================================================

        return val_loss_avg, depth_eval, seg_eval


    ############################ test ################################
    def test(self, test_dataloader):
        ## load model weights
        if not os.path.exists(self.ckpt_path):
            print(f"Checkpoint not found at {self.ckpt_path}")
            print("Available checkpoint files:")
            ckpt_dir = os.path.dirname(self.ckpt_path)
            if os.path.exists(ckpt_dir):
                for f in os.listdir(ckpt_dir):
                    if f.endswith('.p'):
                        print(f"  - {os.path.join(ckpt_dir, f)}")
            return
            
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
        self.model.eval()

        with torch.no_grad():
            MAE_all, RMSE_all, REL_all, DELTA105_all, DELTA110_all, DELTA125_all = [], [], [], [], [], []
            MAE_mask_all, RMSE_mask_all, REL_mask_all, DELTA105_mask_all, DELTA110_mask_all, DELTA125_mask_all = [], [], [], [], [], []
            IoU_all, mAP_all = [], []
            
            pbar = tqdm(test_dataloader, desc="Testing")
            for index, data in enumerate(pbar):
                try:
                    rgb = data['rgb'].to(self.device)
                    depth_gt = data['depth_gt'].to(self.device)
                    seg_gt = data['seg_gt'].to(self.device)
                    zero_mask = data['zero_mask'].to(self.device)
                    loss_mask = data['loss_mask'].to(self.device)
                    depth_min = data['depth_min'][0].to(self.device) if torch.is_tensor(data['depth_min'][0]) else torch.tensor(data['depth_min'][0]).to(self.device)
                    depth_max = data['depth_max'][0].to(self.device) if torch.is_tensor(data['depth_max'][0]) else torch.tensor(data['depth_max'][0]).to(self.device)
                    
                    output_depths, output_segs = self.model(rgb)

                    if len(output_depths) > 0:
                        MAE, RMSE, REL, DELTA105, DELTA110, DELTA125, MAE_, RMSE_, REL_, DELTA105_, DELTA110_, \
                                    DELTA125_ = compute_depth_metrics(output_depths[-1][-1].squeeze(1), depth_gt, depth_min, depth_max, \
                                                                    zero_masks=zero_mask, denorm=True, gt_masks=seg_gt, num_classes=self.num_classes)
                        MAE_all.append(MAE)
                        RMSE_all.append(RMSE)
                        REL_all.append(REL)
                        DELTA105_all.append(DELTA105)
                        DELTA110_all.append(DELTA110)
                        DELTA125_all.append(DELTA125)
                        MAE_mask_all.append(MAE_)
                        RMSE_mask_all.append(RMSE_)
                        REL_mask_all.append(REL_)
                        DELTA105_mask_all.append(DELTA105_)
                        DELTA110_mask_all.append(DELTA110_)
                        DELTA125_mask_all.append(DELTA125_)

                    if len(output_segs) > 0:
                        IoU, mAP = compute_seg_metrics(output_segs[-1][-1], seg_gt, num_classes=self.num_classes)
                        IoU_all.append(IoU)
                        mAP_all.append(mAP)
                
                except Exception as e:
                    print(f"Error in test batch {index}: {e}")
                    continue

            # 打印测试结果
            print("\n" + "="*50)
            print("TEST RESULTS")
            print("="*50)
                
            if len(MAE_all) > 0:
                print("Depth Estimation Results:")
                print(f"  Overall - MAE: {sum(MAE_all)/len(MAE_all):.4f}, RMSE: {sum(RMSE_all)/len(RMSE_all):.4f}, REL: {sum(REL_all)/len(REL_all):.4f}")
                print(f"  Overall - δ<1.05: {sum(DELTA105_all)/len(DELTA105_all):.1f}%, δ<1.10: {sum(DELTA110_all)/len(DELTA110_all):.1f}%, δ<1.25: {sum(DELTA125_all)/len(DELTA125_all):.1f}%")
                print(f"  Masked  - MAE: {sum(MAE_mask_all)/len(MAE_mask_all):.4f}, RMSE: {sum(RMSE_mask_all)/len(RMSE_mask_all):.4f}, REL: {sum(REL_mask_all)/len(REL_mask_all):.4f}")
                print(f"  Masked  - δ<1.05: {sum(DELTA105_mask_all)/len(DELTA105_mask_all):.1f}%, δ<1.10: {sum(DELTA110_mask_all)/len(DELTA110_mask_all):.1f}%, δ<1.25: {sum(DELTA125_mask_all)/len(DELTA125_mask_all):.1f}%")
            
            if len(IoU_all) > 0:
                print("Segmentation Results:")
                print(f"  IoU: {sum(IoU_all)/len(IoU_all):.4f}, mAP: {sum(mAP_all)/len(mAP_all):.4f}")


    ############################ inference ################################
    def inference(self, image_path):
        ## load model weights
        if not os.path.exists(self.ckpt_path):
            print(f"Checkpoint not found at {self.ckpt_path}")
            return
            
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")
        self.model.eval()

        if not os.path.exists(image_path):
            print(f"Image not found at {image_path}")
            return
            
        image = Image.open(image_path).convert('RGB')
        
        # 使用与训练时一致的预处理
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 与训练时保持一致
        ])
        image = transform(image)
        image = image.unsqueeze(0).to(self.device)

        # 确保输出目录存在
        os.makedirs('results', exist_ok=True)

        with torch.no_grad():
            output_depths, output_segs = self.model(image)
            
            # 使用最后一次迭代的最高分辨率结果
            final_depth = output_depths[-1][-1].squeeze(0).squeeze(0)
            final_seg = output_segs[-1][-1].squeeze(0)
            
            depth_visual = vis_depth(final_depth)
            seg_visual = vis_seg(final_seg, image.squeeze(0).cpu().permute(1, 2, 0), self.dataset_name)
            
            depth_image = Image.fromarray(depth_visual)
            seg_image = Image.fromarray(seg_visual)
            depth_image.save('results/depth.png')
            seg_image.save('results/seg.png')
            print("Results saved to results/depth.png and results/seg.png")


    ############################ save ckpts ################################
    def save_model(self, name=None):
        if name == None:
            save_path = self.path_model + '.p'
        else:
            save_path = self.path_model + '_' + str(name) + '.p'
            
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                    'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                    }, save_path)
        print('Model saved at : {}'.format(save_path))

    ############################ visualization ################################
    def img_logger(self, rgb, depth_gt, seg_gt, depth_pred, seg_pred, zero_mask):
        if not self.config['wandb']['enable']:
            return
            
        try:
            nb_to_show = min(self.config['wandb']['images_to_show'], len(rgb))
            seg_truths_all, depth_truths_all = [], []
            seg_preds_all, depth_preds_all = [], []
            
            tmp = rgb[:nb_to_show].detach()
            imgs = [vis_rgb(img, self.dataset_name) for img in tmp]

            # 处理真值深度图
            depth_truths = depth_gt[:nb_to_show]
            for depth_truth in depth_truths:
                depth_visual_gt = vis_depth(depth_truth)
                depth_truths_all.append(depth_visual_gt)

            # 处理预测深度图
            for output_depths_iter in depth_pred:
                depth_preds_iter = []
                output_depth_iter = output_depths_iter[-1][:nb_to_show].squeeze(1)
                for pred in output_depth_iter:
                    depth_visual_pred = vis_depth(pred)
                    depth_preds_iter.append(depth_visual_pred)
                depth_preds_all.append(depth_preds_iter)
                        
            # 处理真值分割图
            seg_truths = seg_gt[:nb_to_show]
            for truths, img in zip(seg_truths, imgs):
                seg_visual_gt = vis_seg_gt(truths, img, self.dataset_name)
                seg_truths_all.append(seg_visual_gt)
            
            # 处理预测分割图
            for output_segs_iter in seg_pred:
                seg_preds_iter = []
                preds = output_segs_iter[-1][:nb_to_show]
                for pred, img in zip(preds, imgs):
                    seg_visual_pred = vis_seg(pred, img, self.dataset_name)
                    seg_preds_iter.append(seg_visual_pred)
                seg_preds_all.append(seg_preds_iter)

            output_dim = (int(self.config['wandb']['im_w']), int(self.config['wandb']['im_h']))

            # 记录原始图像
            wandb.log({
                "img": [wandb.Image(Image.fromarray(im).resize(output_dim), caption='img_{}'.format(i+1)) for i, im in enumerate(imgs)]
            })

            # 记录深度图
            wandb.log({
                "depth_truths": [wandb.Image(Image.fromarray(im).resize(output_dim), caption=f'depth_truth_{i+1}') for i, im in enumerate(depth_truths_all)],
                "depth_preds": [wandb.Image(Image.fromarray(depth_preds_all[-1][i]).resize(output_dim), caption=f'depth_pred_{i+1}') for i in range(len(depth_preds_all[-1]))]
            })

            # 记录分割图
            wandb.log({
                "seg_truths": [wandb.Image(Image.fromarray(im).resize(output_dim), caption=f'seg_truth_{i+1}') for i, im in enumerate(seg_truths_all)],
                "seg_preds": [wandb.Image(Image.fromarray(seg_preds_all[-1][i]).resize(output_dim), caption=f'seg_pred_{i+1}') for i in range(len(seg_preds_all[-1]))]
            })
        
        except Exception as e:
            print(f"Warning: Image logging failed: {e}")


    def isnan(self, x):
        return x != x