"""Training entry point for LMCF-Net.

This script trains LMCF-Net on paired prior-current mammograms using only
image-level labels. Optional pixel-level masks in the validation CSV are used
only for reporting localization metrics and are not used by the optimizer.
"""

import argparse
import os
from typing import Optional

import torch

from config import Config
from datasets.dataloader import build_dataloader
from models import LMCFNet
from trainer import Trainer
from utils.checkpoint import load_checkpoint
from utils.logger import build_logger
from utils.seed import set_seed


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train LMCF-Net on a private paired mammogram dataset")
    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV with prior_path,current_path,label columns")
    parser.add_argument("--val_csv", type=str, required=True, help="Validation CSV with prior_path,current_path,label columns")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for logs and generated outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint for resuming model weights")

    parser.add_argument("--image_size", type=int, default=1024, help="Input image size after preprocessing")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for cosine schedule")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--lambda_smooth", type=float, default=1e-4, help="Weight for spatial smoothness regularization")
    parser.add_argument("--gamma_l2", type=float, default=1e-5, help="Weight for L2 regularization in the dynamic objective")
    parser.add_argument("--beta", type=float, default=0.15, help="Steepness of the dynamic coefficient schedule")
    parser.add_argument("--transition_epoch", type=int, default=50, help="Transition epoch for dynamic regularization")

    return parser.parse_args()


def build_config(args) -> Config:
    """Create an experiment configuration from command-line arguments."""
    cfg = Config()
    cfg.train_csv = args.train_csv
    cfg.val_csv = args.val_csv
    cfg.output_dir = args.output_dir
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.resume = args.resume

    cfg.image_size = args.image_size
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.num_workers = args.num_workers
    cfg.learning_rate = args.lr
    cfg.min_learning_rate = args.min_lr
    cfg.seed = args.seed

    cfg.lambda_smooth = args.lambda_smooth
    cfg.gamma_l2 = args.gamma_l2
    cfg.beta = args.beta
    cfg.transition_epoch = args.transition_epoch
    return cfg


def build_model(cfg: Config) -> LMCFNet:
    """Construct LMCF-Net using the active experiment configuration."""
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


def load_model_weights(model: torch.nn.Module, checkpoint_path: Optional[str], device: torch.device) -> None:
    """Load model parameters when a resume checkpoint is provided."""
    if checkpoint_path is None:
        return
    state = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state["model"] if "model" in state else state)


def main():
    args = parse_args()
    cfg = build_config(args)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    logger = build_logger(os.path.join(cfg.output_dir, "logs", "train.log"))
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training LMCF-Net on device: {device}")
    logger.info(f"Training CSV: {cfg.train_csv}")
    logger.info(f"Validation CSV: {cfg.val_csv}")

    train_loader = build_dataloader(
        cfg.train_csv,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment=True,
        shuffle=True,
    )
    val_loader = build_dataloader(
        cfg.val_csv,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment=False,
        shuffle=False,
    )

    model = build_model(cfg)
    load_model_weights(model, cfg.resume, device)

    trainer = Trainer(model, train_loader, val_loader, cfg, device, logger)
    trainer.fit()


if __name__ == "__main__":
    main()
