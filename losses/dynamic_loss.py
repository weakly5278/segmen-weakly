"""Dynamic weak localization objective for LMCF-Net."""

from typing import Sequence

import torch
from torch import nn

from .regularization import l2_regularization
from .smoothness_loss import smoothness_loss
from .weighted_ce import weighted_cross_entropy


class DynamicWeakLocalizationObjective(nn.Module):
    """Combine WCE, spatial smoothness, and L2 with a dynamic schedule.

    Early in training, regularization has a stronger effect to suppress unstable
    localization responses. As training progresses, the schedule reduces the
    regularization contribution and allows image-level classification to guide
    discriminative learning.
    """

    def __init__(
        self,
        class_weights: Sequence[float],
        lambda_smooth: float,
        gamma_l2: float,
        beta: float,
        transition_epoch: int,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.lambda_smooth = lambda_smooth
        self.gamma_l2 = gamma_l2
        self.beta = beta
        self.transition_epoch = transition_epoch

    def forward(self, outputs, labels, model: nn.Module, epoch: int):
        logits = outputs["logits"]
        score_low = outputs["score_low"]

        wce = weighted_cross_entropy(logits, labels, self.class_weights)
        smooth = smoothness_loss(score_low)
        l2 = l2_regularization(model).to(logits.device)

        alpha = 1.0 / (1.0 + torch.exp(torch.tensor(-self.beta * (epoch - self.transition_epoch), device=logits.device)))
        reg_weight = 1.0 - alpha

        loss = wce + reg_weight * self.lambda_smooth * smooth + reg_weight * self.gamma_l2 * l2
        logs = {
            "loss": float(loss.detach().item()),
            "wce": float(wce.detach().item()),
            "smooth": float(smooth.detach().item()),
            "l2": float(l2.detach().item()),
            "alpha": float(alpha.detach().item()),
        }
        return loss, logs
