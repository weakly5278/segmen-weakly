"""Training loop for LMCF-Net."""

import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses import DynamicWeakLocalizationObjective
from utils.checkpoint import save_checkpoint
from .evaluator import evaluate


class Trainer:
    """Encapsulates optimization, validation, logging, and checkpointing."""

    def __init__(self, model, train_loader, val_loader, cfg, device, logger):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.optimizer = Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.min_learning_rate,
        )
        self.criterion = DynamicWeakLocalizationObjective(
            class_weights=cfg.class_weights,
            lambda_smooth=cfg.lambda_smooth,
            gamma_l2=cfg.gamma_l2,
            beta=cfg.beta,
            transition_epoch=cfg.transition_epoch,
        )
        self.best_dice = -1.0

    def fit(self):
        """Train for the configured number of epochs and save the best model."""
        for epoch in range(1, self.cfg.epochs + 1):
            train_log = self._train_one_epoch(epoch)
            val_metrics = evaluate(self.model, self.val_loader, self.device, self.cfg)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch:03d}/{self.cfg.epochs} | lr={lr:.6e} | "
                f"loss={train_log['loss']:.4f} | wce={train_log['wce']:.4f} | "
                f"smooth={train_log['smooth']:.4f} | alpha={train_log['alpha']:.4f} | "
                f"val_acc={val_metrics.get('accuracy', 0):.4f} | "
                f"val_dice={val_metrics.get('dice', 0):.4f} | "
                f"val_iou={val_metrics.get('iou', 0):.4f}"
            )

            score = val_metrics.get("dice", val_metrics.get("f1", 0.0))
            if score > self.best_dice:
                self.best_dice = score
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_score": self.best_dice,
                        "config": self.cfg.__dict__,
                    },
                    os.path.join(self.cfg.checkpoint_dir, "best_lmcf_net.pth"),
                )

    def _train_one_epoch(self, epoch: int):
        """Run one training epoch."""
        self.model.train()
        running = {"loss": 0.0, "wce": 0.0, "smooth": 0.0, "l2": 0.0, "alpha": 0.0}
        total = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
            prior = batch["prior"].to(self.device, non_blocking=True)
            current = batch["current"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            outputs = self.model(prior, current)
            loss, logs = self.criterion(outputs, labels, self.model, epoch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_size = prior.size(0)
            total += batch_size
            for key in running:
                running[key] += logs[key] * batch_size

        for key in running:
            running[key] /= max(total, 1)
        return running
