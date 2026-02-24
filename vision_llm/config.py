from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VisualFeatureConfig:
    """Configuration for the visual feature extraction + projection pipeline."""

    # RetinaNet checkpoint and params
    checkpoint_path: str = "weights/model.t7"
    param_path: str = "weights/param.txt"

    # Which FPN levels to use (indices into the FPN output tuple).
    # 0=p3, 1=p4, 2=p5, 3=p6, 4=p7.  p6/p7 are too coarse for Braille.
    fpn_levels: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Adaptive-pool each FPN level to this spatial size
    pool_size: Tuple[int, int] = (8, 8)

    # FPN always outputs 256 channels
    fpn_channels: int = 256

    # Qwen3-8B hidden dimension
    llm_hidden_dim: int = 4096

    device: str = "cpu"

    @property
    def num_visual_tokens(self) -> int:
        """Total visual tokens = num_levels * pool_h * pool_w."""
        return len(self.fpn_levels) * self.pool_size[0] * self.pool_size[1]
