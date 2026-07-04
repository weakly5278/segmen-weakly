"""Checkpoint loading and saving utilities."""

import os

import torch


def save_checkpoint(state, path: str):
    """Save a training checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device):
    """Load a checkpoint onto the requested device."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)
