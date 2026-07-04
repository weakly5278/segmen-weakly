"""Inference script for generating Binary Localization Masks with LMCF-Net."""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from datasets.dataloader import build_dataloader
from main import build_model, update_config
from postprocessing import refine_blm
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


@torch.no_grad()
def run_inference(cfg: Config, checkpoint: str, csv_path: str, output_dir: str):
    """Generate score maps and refined BLMs for all pairs in a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    model = build_model(cfg).to(device)
    state = load_checkpoint(checkpoint, device)
    model.load_state_dict(state["model"])
    model.eval()

    loader = build_dataloader(
        csv_path,
        image_size=cfg.image_size,
        batch_size=1,
        num_workers=cfg.num_workers,
        augment=False,
        shuffle=False,
    )

    for index, batch in enumerate(loader):
        prior = batch["prior"].to(device)
        current = batch["current"].to(device)
        outputs = model(prior, current)

        probs = F.softmax(outputs["logits"], dim=1)
        cancer_probability = float(probs[0, 1].detach().cpu().item())
        score_map = outputs["score_map"][0, 0].detach().cpu().numpy()

        refined = refine_blm(
            score_map,
            threshold=cfg.threshold,
            min_component_area=cfg.min_component_area,
            closing_kernel=cfg.closing_kernel,
            cancer_probability=cancer_probability if cfg.use_classification_gate else None,
            cancer_probability_threshold=cfg.cancer_probability_threshold,
        )

        case_name = str(batch.get("patient_id", [f"case_{index:04d}"])[0])
        cv2.imwrite(os.path.join(output_dir, f"{case_name}_score.png"), (score_map * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, f"{case_name}_blm.png"), (refined * 255).astype(np.uint8))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LMCF-Net localization masks")
    parser.add_argument("--csv", required=True, help="CSV file with prior_path,current_path,label columns")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--output_dir", default="outputs/masks")
    parser.add_argument("--image_size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    if args.image_size is not None:
        cfg.image_size = args.image_size
    run_inference(cfg, args.checkpoint, args.csv, args.output_dir)
