"""Dataset utilities for paired prior-current mammograms.

The loader expects a CSV file with at least three columns:

    prior_path,current_path,label

Optional columns such as ``mask_path``, ``patient_id``, and ``view`` are
preserved when available. Pixel-level masks are never required for training;
when supplied, they are used only by evaluation utilities.
"""

from pathlib import Path
from typing import Any, Dict

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import PairTransform


class PrivatePairedMammogramDataset(Dataset):
    """Paired mammogram dataset for weakly supervised temporal localization.

    Each sample contains a prior mammogram, a current mammogram, and an
    image-level label. The dataset keeps the interface intentionally simple so
    the same loader can be used for training, validation, testing, and
    inference on a private longitudinal mammography cohort.
    """

    def __init__(self, csv_path: str, image_size: int = 1024, augment: bool = False):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        required = {"prior_path", "current_path", "label"}
        missing = required.difference(set(self.df.columns))
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        self.transform = PairTransform(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Read paired grayscale mammograms. All image resizing, normalization,
        # and synchronized augmentation are handled by PairTransform.
        prior = self._read_grayscale(row["prior_path"])
        current = self._read_grayscale(row["current_path"])

        # Masks are optional and are used only for evaluation of localization
        # quality. Training relies exclusively on image-level labels.
        mask = None
        if "mask_path" in self.df.columns:
            mask_value = row.get("mask_path", "")
            if isinstance(mask_value, str) and len(mask_value) > 0:
                mask_path = Path(mask_value)
                if mask_path.exists():
                    mask = self._read_grayscale(str(mask_path))

        prior, current, mask = self.transform(prior, current, mask)

        sample: Dict[str, Any] = {
            "prior": torch.from_numpy(prior),
            "current": torch.from_numpy(current),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "prior_path": str(row["prior_path"]),
            "current_path": str(row["current_path"]),
        }

        if mask is not None:
            sample["mask"] = torch.from_numpy(mask)
        if "patient_id" in self.df.columns:
            sample["patient_id"] = str(row["patient_id"])
        if "view" in self.df.columns:
            sample["view"] = str(row["view"])

        return sample

    @staticmethod
    def _read_grayscale(path: str):
        """Load an image as a single-channel mammogram array."""
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image
