"""Entry point for training and evaluating LMCF-Net."""

import argparse
import os

import torch

from config import Config
from datasets.dataloader import build_dataloader
from models import LMCFNet
from trainer import Trainer, evaluate
from utils.checkpoint import load_checkpoint
from utils.logger import build_logger
from utils.seed import set_seed


def build_model(cfg: Config) -> LMCFNet:
    """Construct LMCF-Net from the active configuration."""
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


def parse_args():
    parser = argparse.ArgumentParser(description="LMCF-Net training and evaluation")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def update_config(cfg: Config, args) -> Config:
    """Apply command-line overrides to the default configuration."""
    if args.train_csv is not None:
        cfg.train_csv = args.train_csv
    if args.val_csv is not None:
        cfg.val_csv = args.val_csv
    if args.test_csv is not None:
        cfg.test_csv = args.test_csv
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    return cfg


def main():
    args = parse_args()
    cfg = update_config(Config(), args)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    logger = build_logger(os.path.join(cfg.output_dir, "logs", f"{args.mode}.log"))
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = build_model(cfg)

    if args.mode == "train":
        train_loader = build_dataloader(
            cfg.train_csv,
            cfg.image_size,
            cfg.batch_size,
            cfg.num_workers,
            augment=True,
            shuffle=True,
        )
        val_loader = build_dataloader(
            cfg.val_csv,
            cfg.image_size,
            cfg.batch_size,
            cfg.num_workers,
            augment=False,
            shuffle=False,
        )
        trainer = Trainer(model, train_loader, val_loader, cfg, device, logger)
        trainer.fit()
    else:
        checkpoint_path = args.checkpoint or os.path.join(cfg.checkpoint_dir, "best_lmcf_net.pth")
        state = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(state["model"])
        model.to(device)

        test_loader = build_dataloader(
            cfg.test_csv,
            cfg.image_size,
            cfg.batch_size,
            cfg.num_workers,
            augment=False,
            shuffle=False,
        )
        metrics = evaluate(
            model,
            test_loader,
            device,
            cfg,
            save_outputs=True,
            output_dir=os.path.join(cfg.output_dir, "masks"),
        )
        logger.info(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
