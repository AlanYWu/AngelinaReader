#!/usr/bin/env python
"""End-to-end shape verification for the visual feature pipeline.

Usage:
    PYTHONPATH=. python vision_llm/test_pipeline.py \
        --checkpoint weights/model.t7 --params weights/param.txt \
        --image input_5.jpg --device mps
"""

import argparse
import sys

import torch
import numpy as np
from PIL import Image

from vision_llm.config import VisualFeatureConfig
from vision_llm.feature_extractor import RetinaNetFeatureExtractor
from vision_llm.visual_token_adapter import VisualTokenAdapter


def load_image(path: str, size: int = 1024) -> torch.Tensor:
    """Load an image and preprocess to (1, 3, size, size) float tensor."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    # Grayscale-normalize (same as BrailleInference preprocessing)
    mean = arr.mean()
    std = max(arr.std(), 1e-6)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def dummy_image(size: int = 1024) -> torch.Tensor:
    """Create a random (1, 3, size, size) tensor for testing without an image."""
    return torch.randn(1, 3, size, size)


def main():
    parser = argparse.ArgumentParser(description="Test vision_llm pipeline shapes")
    parser.add_argument("--checkpoint", required=True, help="Path to .t7 weights")
    parser.add_argument("--params", required=True, help="Path to param.txt")
    parser.add_argument("--image", default=None, help="Path to test image (optional; uses random tensor if omitted)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, mps")
    args = parser.parse_args()

    config = VisualFeatureConfig(
        checkpoint_path=args.checkpoint,
        param_path=args.params,
        device=args.device,
    )

    print(f"Config: fpn_levels={config.fpn_levels}, pool_size={config.pool_size}, "
          f"num_visual_tokens={config.num_visual_tokens}")

    # --- Feature extractor ---
    print("\nLoading RetinaNet feature extractor...")
    extractor = RetinaNetFeatureExtractor(config).to(config.device)

    trainable_extractor = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    total_extractor = sum(p.numel() for p in extractor.parameters())
    print(f"  Extractor params: {total_extractor:,} total, {trainable_extractor:,} trainable")
    assert trainable_extractor == 0, f"Extractor should have 0 trainable params, got {trainable_extractor}"

    # --- Adapter ---
    adapter = VisualTokenAdapter(config).to(config.device)
    trainable_adapter = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"  Adapter params:   {trainable_adapter:,} trainable")

    # --- Forward pass ---
    if args.image:
        print(f"\nLoading image: {args.image}")
        x = load_image(args.image).to(config.device)
    else:
        print("\nUsing random tensor (no --image provided)")
        x = dummy_image().to(config.device)

    print(f"  Input shape: {x.shape}")

    with torch.no_grad():
        fpn_features = extractor(x)
    print(f"  FPN outputs: {len(fpn_features)} levels")
    for i, feat in enumerate(fpn_features):
        print(f"    Level {config.fpn_levels[i]} (p{config.fpn_levels[i]+3}): {feat.shape}")

    with torch.no_grad():
        visual_tokens = adapter(fpn_features)
    print(f"\nOutput shape: {visual_tokens.shape}")

    expected = (1, config.num_visual_tokens, config.llm_hidden_dim)
    assert visual_tokens.shape == expected, f"Expected {expected}, got {tuple(visual_tokens.shape)}"

    print(f"\nAll assertions passed.")
    print(f"  - Extractor: 0 trainable params")
    print(f"  - Adapter: {trainable_adapter:,} trainable params")
    print(f"  - Output: {visual_tokens.shape} (ready for Qwen3-8B)")


if __name__ == "__main__":
    main()
