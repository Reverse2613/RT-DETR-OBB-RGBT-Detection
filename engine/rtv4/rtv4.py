"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from ..core import register


__all__ = ['RTv4', ]


@register()
class RTv4(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    # 检测器容器本身不实现融合：backbone 产三尺度特征，encoder 产金字塔，
    # decoder 产预测字典。默认 OBB 实验并未配置视觉基础模型教师。

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None, teacher_encoder_output=None):
        # targets 只在训练中供 denoising/损失使用；推理不需要 GT 框或 GT 条件。
        x_backbone = self.backbone(x)  # [S3, S4, S5] features from backbone

        encoder_output = self.encoder(x_backbone)
        # tuple: (fpn_features, student_distill_output) or fpn_features (list) if not training or distillation is off.

        student_distill_output = None
        if self.training and isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            x_fpn_features, student_distill_output = encoder_output
        else:
            x_fpn_features = encoder_output

        x_decoder_out = self.decoder(x_fpn_features, targets)

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:
            x_decoder_out['student_distill_output'] = student_distill_output
            x_decoder_out['teacher_encoder_output'] = teacher_encoder_output


        # ======================================================================
        # 【新增】：透传主干网络条件自感知头的训练期预测输出，供 Criterion 算损失
        # ======================================================================
        # 条件监督闭环：LGDense 临时属性 -> 输出字典 -> criterion.loss_conditions。
        # 普通 Add/Concat/IR 无此属性，self-driven 下属性值为 None。
        if self.training and hasattr(self.backbone, "latest_cond_preds"):
            x_decoder_out["cond_preds"] = self.backbone.latest_cond_preds

            
        return x_decoder_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
