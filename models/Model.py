"""
ISGNet Model - EDS-Fusion版本
完整版 - 可直接替换项目中的 models/Model.py
"""

import numpy as np
import torch
import torch.nn as nn
import timm

from models.Reassemble import Reassemble
from models.Fusion import Fusion
from models.Head import HeadDepth, HeadSeg, MultiscaleHead


class ISGNet(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [5, 11, 17, 23],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 nclasses           = 3,
                 type               = "full",
                 model_timm         = "vit_large_patch16_384",
                 pretrain           = True,
                 iterations         = 3,
                 in_chans           = 3,
                 coord_reduction    = 32,
                 use_eds_at_finest  = True,
                 use_msda           = False,
                 msda_config        = None):
        """
        ISGNet模型（集成EDS-Fusion）
        
        参数说明：
            type : {"full", "depth", "seg"}
            image_size : (c, h, w)
            patch_size : patch大小
            emb_dim : Transformer嵌入维度
            resample_dim : 特征重采样维度
            read : {"ignore", "add", "projection"}
            coord_reduction : CoordinateAttention降维比例 (default: 32)
            use_eds_at_finest : 是否只在最细尺度使用EDS-Fusion (default: True)
        """
        super().__init__()

        ## Transformer Encoder
        self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrain, in_chans=in_chans)
        print("✓ Vision Transformer loaded successfully")
        
        self.type_ = type
        self.iterations = iterations

        ## Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        ## Reassemble + Fusion
        self.reassembles = []
        self.fusions = []

        # 打印配置信息
        print("\n" + "="*60)
        print("ISGNet Fusion Configuration:")
        print(f"  - resample_dim: {resample_dim}")
        print(f"  - use_eds_at_finest: {use_eds_at_finest}")
        print(f"  - use_msda: {use_msda}")
        if msda_config:
            print(f"  - msda_dilation_rates: {msda_config.get('dilation_rates', {})}")
        print("="*60 + "\n")
      
        for idx, s in enumerate(reassemble_s):  # idx: 0,1,2,3
            # Reassemble模块（保持不变）
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            
            # ===== 确定当前层是否使用MSDA ===== ✅
            # 注意：reassemble_s = [4, 8, 16, 32]
            # 对应的index在forward中是：3, 2, 1, 0（倒序）
            # 所以 idx=0→index=3, idx=1→index=2, idx=2→index=1, idx=3→index=0
            
            fusion_index = 3 - idx  # 转换：idx→实际的pyramid index
            
            use_msda_current = False
            msda_dilation = None
            
            # 粗尺度（index=1,2,3）使用MSDA
            if use_msda and msda_config and fusion_index in [1, 2, 3]:
                use_msda_current = True
                msda_dilation = msda_config['dilation_rates'].get(str(fusion_index))
            
            # 最细尺度（index=0）使用EDS
            use_eds_current = (fusion_index == 0 and use_eds_at_finest)
            
            # Fusion模块（传入MSDA参数）✅
            self.fusions.append(Fusion(
                resample_dim=resample_dim, 
                nclasses=nclasses, 
                coord_reduction=coord_reduction,
                use_eds_at_finest=use_eds_current,
                use_msda=use_msda_current,        # ✅ 新增
                msda_dilation=msda_dilation       # ✅ 新增
            ))
            
            # 打印每层配置
            print(f"[Fusion {idx}] pyramid_index={fusion_index}, "
                  f"use_msda={use_msda_current}, dilation={msda_dilation}, "
                  f"use_eds={use_eds_current}")

        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        ## Prediction Head
        if type == "full":
            self.head_multiscale = MultiscaleHead(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        elif type == "seg":
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        else:
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
            
        print(f"✓ Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")


    def forward(self, img):
        """
        前向传播
        
        输入:
            img: (B, C, H, W) - RGB图像
        
        输出:
            out_depths: 多尺度深度预测列表
            out_segs: 多尺度分割预测列表
        """
        # Transformer编码
        t = self.transformer_encoders(img)
        
        out_depth, out_seg = None, None
        guide_depth, guide_seg = [], []
        out_depths, out_segs = [], []

        ## 多尺度迭代融合
        for iter in range(self.iterations):
            depth_feature, seg_feature = None, None
            multiscale_depth, multiscale_seg = [], []
            depth_features, seg_features = [], []
            
            # 从粗到细处理4个尺度
            for i in np.arange(len(self.fusions)-1, -1, -1):  # 3, 2, 1, 0
                # 获取对应层的特征
                hook_to_take = 't'+str(self.hooks[i])
                activation_result = self.activation[hook_to_take]
                
                # Reassemble
                reassemble_result = self.reassembles[i](activation_result)
                
                # Fusion（会根据index自动选择EDS-Fusion或简单SA）
                depth_feature, seg_feature = self.fusions[i](
                    reassemble_result, 
                    i,
                    depth_feature, 
                    seg_feature, 
                    guide_depth, 
                    guide_seg
                )
                
                # 预测头
                output_depth, output_seg = self.head_multiscale(depth_feature, seg_feature)
                
                multiscale_depth.append(output_depth)
                multiscale_seg.append(output_seg)
                depth_features.append(depth_feature)
                seg_features.append(seg_feature)
            
            # 记录本次迭代结果
            guide_depth.append(depth_features)
            guide_seg.append(seg_features)
            out_depths.append(multiscale_depth)
            out_segs.append(multiscale_seg)
        
        return out_depths, out_segs


    def _get_layers_from_hooks(self, hooks):
        """注册hook以提取中间层特征"""
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(
                get_activation('t'+str(h))
            )