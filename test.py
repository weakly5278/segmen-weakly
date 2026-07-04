"""Evaluation entry point for LMCF-Net.

This script loads a trained checkpoint, evaluates image-level classification
and localization metrics, and optionally saves score maps and refined Binary
Localization Masks (BLMs) for each test case.
"""

import argparse
import json
import os

import torch

from config import Config
from datasets.dataloader import build_dataloader
from models import LMCFNet
from trainer.evaluator import evaluate
from utils.checkpoint import load_checkpoint
from utils.logger import build_logger
from utils.seed import set_seed


def parse_args():
    """Parse command-line arguments for testing."""
    parser = argparse.ArgumentParser(description="Evaluate LMCF-Net on a private paired mammogram dataset")
    parser.add_argument("--test_csv", type=str, required=True, help="Test CSV with prior_path,current_path,label columns")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained LMCF-Net checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for logs, metrics, and saved masks")

    parser.add_argument("--image_size", type=int, default=1024, help="Input image size after preprocessing")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting score maps into binary masks")
    parser.add_argument("--min_component_area", type=int, default=100, help="Minimum connected-component area retained in BLM refinement")
    parser.add_argument("--closing_kernel", type=int, default=5, help="Kernel size for morphological closing")
    parser.add_argument("--disable_classification_gate", action="store_true", help="Disable classification-guided suppression of normal cases")
    parser.add_argument("--cancer_probability_threshold", type=float, default=0.20, help="Cancer-probability threshold for classification-guided BLM refinement")
    parser.add_argument("--save_outputs", action="store_true", help="Save score maps and refined BLMs")
    return parser.parse_args()


def build_config(args) -> Config:
    """Create an evaluation configuration from command-line arguments."""
    cfg = Config()
    cfg.test_csv = args.test_csv
    cfg.output_dir = args.output_dir
    cfg.image_size = args.image_size
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed

    cfg.threshold = args.threshold
    cfg.min_component_area = args.min_component_area
    cfg.closing_kernel = args.closing_kernel
    cfg.use_classification_gate = not args.disable_classification_gate
    cfg.cancer_probability_threshold = args.cancer_probability_threshold
    return cfg


def build_model(cfg: Config) -> LMCFNet:
    """Construct the LMCF-Net architecture for evaluation."""
    return LMCFNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        embed_dims=cfg.embed_dims,
        num_heads=cfg.num_heads,
        depths=cfg.depths,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        alignment_dim=cfg.alignment_dim,
        attention_dim=cfg.attention_dim,
    )


def main():
    args = parse_args()
    cfg = build_config(args)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)
    if args.save_outputs:
        os.makedirs(os.path.join(cfg.output_dir, "masks"), exist_ok=True)

    logger = build_logger(os.path.join(cfg.output_dir, "logs", "test.log"))
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating LMCF-Net on device: {device}")
    logger.info(f"Test CSV: {cfg.test_csv}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    test_loader = build_dataloader(
        cfg.test_csv,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment=False,
        shuffle=False,
    )

    model = build_model(cfg).to(device)
    state = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state["model"] if "model" in state else state)

    metrics = evaluate(
        model,
        test_loader,
        device,
        cfg,
        save_outputs=args.save_outputs,
        output_dir=os.path.join(cfg.output_dir, "masks"),
    )

    metrics_path = os.path.join(cfg.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Test metrics: {metrics}")
    logger.info(f"Saved metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
