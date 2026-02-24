"""Frozen FPN feature extractor wrapping a pretrained RetinaNet."""

from typing import List

import torch
import torch.nn as nn
from ovotools.params import AttrDict

from pytorch_retinanet.encoder import DataEncoder
from pytorch_retinanet.retinanet import RetinaNet

from .config import VisualFeatureConfig


class RetinaNetFeatureExtractor(nn.Module):
    """Loads a pretrained RetinaNet checkpoint and exposes its FPN features.

    All parameters are frozen. The detection heads are kept (they are part of
    the state_dict) but never called — only ``self._retinanet.fpn`` is used.
    """

    def __init__(self, config: VisualFeatureConfig):
        super().__init__()
        self.config = config
        self.fpn_levels = config.fpn_levels

        # Reconstruct the model from saved params (mirrors create_model_retinanet)
        params = AttrDict.load(config.param_path, verbose=0)
        encoder = DataEncoder(**params.model_params.encoder_params)
        num_classes = 1 if params.data.get("get_points", False) else (
            [1] * 6 if params.data.get("class_as_6pt", False) else 64
        )
        self._retinanet = RetinaNet(
            num_layers=encoder.num_layers(),
            num_anchors=encoder.num_anchors(),
            num_classes=num_classes,
            num_fpn_layers=params.model_params.get("num_fpn_layers", 0),
        )

        # Load weights
        state = torch.load(config.checkpoint_path, map_location="cpu")
        self._retinanet.load_state_dict(state)

        # The FPN may return fewer levels than we need (num_layers is set from
        # the number of anchor areas, which can be 1 for single-scale models).
        # The FPN *computes* all levels internally when num_fpn_layers >= needed,
        # but truncates the output to num_layers.  Override to return enough.
        min_needed = max(config.fpn_levels) + 1
        if self._retinanet.fpn.num_layers < min_needed:
            assert self._retinanet.fpn.num_fpn_layers >= min_needed, (
                f"FPN only computes {self._retinanet.fpn.num_fpn_layers} levels "
                f"but level index {max(config.fpn_levels)} was requested"
            )
            # num_layers is a Final[int] for JIT but we're not scripting, so
            # override via __dict__ to bypass the frozen attribute.
            object.__setattr__(self._retinanet.fpn, "num_layers", min_needed)
            object.__setattr__(self._retinanet, "num_layers", min_needed)

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "RetinaNetFeatureExtractor":
        """Override to keep the model permanently in eval mode."""
        return super().train(False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract selected FPN feature maps.

        Args:
            x: input images, shape (B, 3, H, W).

        Returns:
            List of tensors, one per selected FPN level, each (B, 256, H_i, W_i).
        """
        fpn_outputs = self._retinanet.fpn(x)
        return [fpn_outputs[i] for i in self.fpn_levels]
