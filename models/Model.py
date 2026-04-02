"""
ISGNet Model

修复：
  1. EDS-Fusion 硬编码 bug
       原代码：use_eds_current 硬编码为 False，use_identity 硬编码给 idx=0,1
               导致 ISGNet.__init__ 接收的 use_eds_at_finest 参数从未真正传入 Fusion
       修复后：直接将 use_eds_at_finest 传入每个 Fusion，use_identity 全部置 False
               四个尺度（idx=0~3）均使用 EDS-Fusion 作为 Stage4 空间注意力

  2. 保留其余逻辑不变（iterations 参数、hooks、head 等）
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
                 image_size          = (3, 384, 384),
                 patch_size          = 16,
                 emb_dim             = 1024,
                 resample_dim        = 256,
                 read                = 'projection',
                 num_layers_encoder  = 24,
                 hooks               = [5, 11, 17, 23],
                 reassemble_s        = [4, 8, 16, 32],
                 transformer_dropout = 0,
                 nclasses            = 3,
                 type                = "full",
                 model_timm          = "vit_large_patch16_384",
                 pretrain            = True,
                 iterations          = 1,
                 in_chans            = 3,
                 coord_reduction     = 32,
                 use_eds_at_finest   = True):
        super().__init__()

        self.transformer_encoders = timm.create_model(
            model_timm, pretrained=pretrain, in_chans=in_chans)
        print("✓ Vision Transformer loaded successfully")

        self.type_      = type
        self.iterations = iterations

        self.activation = {}
        self.hooks      = hooks
        self._get_layers_from_hooks(self.hooks)

        self.reassembles = []
        self.fusions     = []

        print("\n" + "=" * 60)
        print("ISGNet Configuration:")
        print(f"  iterations       : {iterations}")
        print(f"  resample_dim     : {resample_dim}")
        print(f"  use_eds_at_finest: {use_eds_at_finest}")
        print(f"  Stage4 分配:")
        for idx, s in enumerate(reassemble_s):
            h = image_size[1] // s
            label = "EDS-Fusion" if use_eds_at_finest else "SpatialAttention"
            print(f"    idx={idx}  s={s:2d}  {h:3d}×{h:<3d}  → {label}")
        print("=" * 60 + "\n")

        for idx, s in enumerate(reassemble_s):
            self.reassembles.append(
                Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim)
            )

            # 修复：直接将外部传入的 use_eds_at_finest 传递给每个 Fusion
            #       不再按 idx 硬编码，四层全部统一策略
            self.fusions.append(Fusion(
                resample_dim      = resample_dim,
                coord_reduction   = coord_reduction,
                use_eds_at_finest = use_eds_at_finest,  # 真正使用外部参数
                use_identity      = False,               # 四层均不做恒等透传
            ))

        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions     = nn.ModuleList(self.fusions)

        if type == "full":
            self.head_multiscale = MultiscaleHead(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth        = HeadDepth(resample_dim)
            self.head_segmentation = None
        elif type == "seg":
            self.head_depth        = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        else:
            self.head_depth        = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Model initialized with {total_params:,} total parameters "
              f"({trainable_params:,} trainable)")

    def forward(self, img):
        t = self.transformer_encoders(img)

        depth_feature, seg_feature       = None, None
        multiscale_depth, multiscale_seg = [], []

        for i in np.arange(len(self.fusions) - 1, -1, -1):
            hook_to_take      = 't' + str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)

            depth_feature, seg_feature = self.fusions[i](
                reassemble_result, i,
                depth_feature, seg_feature,
                [], []
            )

            output_depth, output_seg = self.head_multiscale(
                depth_feature, seg_feature)

            multiscale_depth.append(output_depth)
            multiscale_seg.append(output_seg)

        return [multiscale_depth], [multiscale_seg]

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(
                get_activation('t' + str(h))
            )