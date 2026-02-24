"""Visual token adapter: pools FPN features and projects to LLM embedding space."""

from typing import List

import torch
import torch.nn as nn

from .config import VisualFeatureConfig


class VisualTokenAdapter(nn.Module):
    """Converts multi-scale FPN features into a flat sequence of LLM-ready tokens.

    Per FPN level:
        AdaptiveAvgPool2d  → (B, C, pool_h, pool_w)
        flatten spatial    → (B, pool_h*pool_w, C)
        + learnable level embedding
        + LayerNorm

    Concatenate levels → (B, num_visual_tokens, C)

    MLP projection:
        Linear(C → llm_dim) → GELU → Linear(llm_dim → llm_dim)

    Output → (B, num_visual_tokens, llm_dim)
    """

    def __init__(self, config: VisualFeatureConfig):
        super().__init__()
        self.config = config
        num_levels = len(config.fpn_levels)
        C = config.fpn_channels
        pool_h, pool_w = config.pool_size

        self.pool = nn.AdaptiveAvgPool2d(config.pool_size)

        # One learnable embedding per FPN level
        self.level_embeds = nn.Parameter(torch.zeros(num_levels, 1, C))
        nn.init.normal_(self.level_embeds, std=0.02)

        self.norm = nn.LayerNorm(C)

        # 2-layer MLP (LLaVA-1.5 style)
        D = config.llm_hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(C, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

    def forward(self, fpn_features: List[torch.Tensor]) -> torch.Tensor:
        """Project FPN features to LLM embedding space.

        Args:
            fpn_features: list of tensors from RetinaNetFeatureExtractor,
                          each (B, 256, H_i, W_i).

        Returns:
            (B, num_visual_tokens, llm_hidden_dim)
        """
        tokens_per_level = []
        for i, feat in enumerate(fpn_features):
            pooled = self.pool(feat)                        # (B, C, ph, pw)
            B, C, ph, pw = pooled.shape
            flat = pooled.flatten(2).transpose(1, 2)        # (B, ph*pw, C)
            flat = flat + self.level_embeds[i]               # broadcast level embed
            tokens_per_level.append(flat)

        tokens = torch.cat(tokens_per_level, dim=1)         # (B, N, C)
        tokens = self.norm(tokens)
        tokens = self.proj(tokens)                           # (B, N, D)
        return tokens
