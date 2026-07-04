"""Default configuration for LMCF-Net experiments."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    """Centralized experiment configuration.

    Command-line arguments in ``main.py`` can override the most common fields.
    """

    # CSV files should contain prior_path,current_path,label and optional mask_path.
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    test_csv: str = "data/test.csv"

    image_size: int = 1024
    num_classes: int = 2
    in_channels: int = 1

    # Optimization settings.
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    seed: int = 42

    # Shared hierarchical transformer encoder.
    embed_dims: Tuple[int, int, int, int] = (64, 128, 320, 512)
    num_heads: Tuple[int, int, int, int] = (2, 4, 8, 8)
    depths: Tuple[int, int, int, int] = (1, 1, 1, 1)
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Alignment and temporal fusion.
    alignment_dim: int = 256
    attention_dim: int = 256

    # Dynamic weak localization objective.
    class_weights: Tuple[float, float] = (1.0, 1.0)
    lambda_smooth: float = 1e-4
    gamma_l2: float = 1e-5
    beta: float = 0.15
    transition_epoch: int = 50

    # Inference-time BLM refinement.
    threshold: float = 0.5
    min_component_area: int = 100
    closing_kernel: int = 5
    use_classification_gate: bool = True
    cancer_probability_threshold: float = 0.20

    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    resume: Optional[str] = None
