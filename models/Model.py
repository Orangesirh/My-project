"""
    The code partly borrows from
    https://github.com/antocad/FocusOnDepth
"""

import numpy as np
import torch
import torch.nn as nn
import timm

from models.Reassemble import Reassemble
from models.Fusion import Fusion
from models.Head import HeadDepth, HeadSeg, MultiscaleHead

# torch.manual_seed(0)

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
                 use_triplet        = True,      # 新增：是否启用TripletAttention
                 use_final_sa       = False):    # 新增：是否使用最终SA
        """
        type : {"full", "depth", "seg"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        ## Transformer
        self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrain, in_chans=in_chans)
        print("load vit successfully")
        self.type_ = type
        self.iterations = iterations

        ## Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        ## Reassembles Fusion
        self.reassembles = []
        self.fusions = []
      
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            # 添加coord_reduction参数
            coord_reduction = 32  # 或从config中读取
            self.fusions.append(Fusion(
                resample_dim=resample_dim, 
                nclasses=nclasses, 
                coord_reduction=coord_reduction,
                use_triplet=use_triplet,      # 新增
                use_final_sa=use_final_sa     # 新增
            ))

        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        ## Head
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
            


    def forward(self, img):

        t = self.transformer_encoders(img)
        out_depth, out_seg = None, None
        guide_depth, guide_seg = [], []
        out_depths, out_segs = [], []

        ## multi-scale iterations
        for iter in range(self.iterations):
            depth_feature, seg_feature = None, None
            multiscale_depth, multiscale_seg = [], []
            depth_features, seg_features = [], []
            for i in np.arange(len(self.fusions)-1, -1, -1):                        # 3, 2, 1, 0
                hook_to_take = 't'+str(self.hooks[i])
                activation_result = self.activation[hook_to_take]
                reassemble_result = self.reassembles[i](activation_result)          # [256, 12, 12], [256, 24, 24], [256, 48, 48], [256, 96, 96]                             
                depth_feature, seg_feature = self.fusions[i](reassemble_result, i, depth_feature, seg_feature, guide_depth, guide_seg)     # [256, 24, 24], [256, 48, 48], [256, 96, 96], [256, 192, 192]      
                output_depth, output_seg = self.head_multiscale(depth_feature, seg_feature)     # [256, 48, 48], [256, 96, 96], [256, 192, 192], [256, 384, 384]
                multiscale_depth.append(output_depth)
                multiscale_seg.append(output_seg)
                depth_features.append(depth_feature)
                seg_features.append(seg_feature)
            guide_depth.append(depth_features)
            guide_seg.append(seg_features)
            out_depths.append(multiscale_depth)
            out_segs.append(multiscale_seg)
        
        return out_depths, out_segs


    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))