import warnings

import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm


@BACKBONES.register_module()
class TimmModel(BaseModule):

    def __init__(self,
                 model_name,
                 pretrained=False,
                 features_only=True,
                 out_indices=(3, 4),
                 in_chans=3,
                 checkpoint_path='',
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super(TimmModel, self).__init__(init_cfg)
        self.model_name = model_name
        self.features_only = features_only
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        if isinstance(pretrained, str):
            warnings.warn('String pretrained is deprecated, use checkpoint_path instead.')
            checkpoint_path = pretrained
            pretrained = False

        try:
            import timm
        except ImportError as exc:
            raise ImportError('`timm` is required for TimmModel backbone.') from exc

        self.timm_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            out_indices=out_indices,
            in_chans=in_chans,
            checkpoint_path=checkpoint_path,
            **kwargs)

        feature_info = getattr(self.timm_model, 'feature_info', None)
        self.out_channels = feature_info.channels() if feature_info is not None else None
        self.out_indices = out_indices
        self.output_layernorms = None
        if self.out_channels is not None:
            self.output_layernorms = nn.ModuleList(
                [nn.LayerNorm(channel) for channel in self.out_channels]
            )

    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return

        stem_modules = ['conv_stem', 'bn1', 'act1']
        if self.frozen_stages >= 0:
            for module_name in stem_modules:
                module = getattr(self.timm_model, module_name, None)
                if module is None:
                    continue
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        blocks = getattr(self.timm_model, 'blocks', None)
        if blocks is None:
            return

        freeze_count = min(self.frozen_stages, len(blocks))
        for stage_idx in range(freeze_count):
            blocks[stage_idx].eval()
            for param in blocks[stage_idx].parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.timm_model(x)
        if isinstance(feats, (list, tuple)):
            if self.output_layernorms is not None:
                feats = [
                    norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
                    for feat, norm in zip(feats, self.output_layernorms)
                ]
            return tuple(feats)
        return feats

    def train(self, mode=True):
        super(TimmModel, self).train(mode)
        if mode:
            self._freeze_stages()
            if self.norm_eval:
                for module in self.modules():
                    if isinstance(module, _BatchNorm):
                        module.eval()
