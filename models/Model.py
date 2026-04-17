"""
ISGNet Model — 消融版本：CoordinateAttention + 混合 Stage4

消融设计：
  Stage3 : 原版 CoordinateAttention（替换 DCCA）
  Stage4 :
    idx=0  s=4   96×96  → EDS-Fusion
    idx=1  s=8   48×48  → EDS-Fusion
    idx=2  s=16  24×24  → SpatialAttention
    idx=3  s=32  12×12  → SpatialAttention
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

        # idx=0,1（96×96, 48×48）用 EDS-Fusion，idx=2,3（24×24, 12×12）用 SpatialAttention
        eds_indices = [0, 1]

        print("\n" + "=" * 60)
        print("ISGNet Configuration [Ablation: CA + EDS-Fusion(fine) + SAM(coarse)]:")
        print(f"  iterations       : {iterations}")
        print(f"  resample_dim     : {resample_dim}")
        print(f"  Stage3           : CoordinateAttention（替换 DCCA）")
        print(f"  Stage4 分配:")
        for idx, s in enumerate(reassemble_s):
            h = image_size[1] // s
            label = "EDS-Fusion" if idx in eds_indices else "SpatialAttention"
            print(f"    idx={idx}  s={s:2d}  {h:3d}×{h:<3d}  → {label}")
        print("=" * 60 + "\n")

        for idx, s in enumerate(reassemble_s):
            self.reassembles.append(
                Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim)
            )

            self.fusions.append(Fusion(
                resample_dim      = resample_dim,
                coord_reduction   = coord_reduction,
                use_eds_at_finest = (idx in eds_indices),
                use_identity      = False,
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