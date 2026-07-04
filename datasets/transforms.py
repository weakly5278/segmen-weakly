"""Preprocessing and synchronized augmentation for paired mammograms."""

from typing import Optional, Tuple

import cv2
import numpy as np


class PairTransform:
    """Apply identical geometric transforms to prior/current pairs.

    Mammogram pairs must remain spatially aligned after augmentation. This
    transform therefore applies the same random flip and rotation to both
    images, and to the optional mask when available.
    """

    def __init__(self, image_size: int = 1024, augment: bool = False):
        self.image_size = image_size
        self.augment = augment

    def __call__(
        self,
        prior: np.ndarray,
        current: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Resize to the network input size used in the paper.
        prior = self._resize_image(prior)
        current = self._resize_image(current)
        if mask is not None:
            mask = self._resize_mask(mask)

        if self.augment:
            prior, current, mask = self._augment_pair(prior, current, mask)

        prior = self._normalize(prior)
        current = self._normalize(current)
        if mask is not None:
            mask = (mask > 127).astype(np.float32)[None, ...]

        return prior, current, mask

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        return cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image[None, ...]

    def _augment_pair(self, prior, current, mask):
        # Horizontal flip is applied consistently across the pair.
        if np.random.rand() < 0.5:
            prior = np.ascontiguousarray(np.fliplr(prior))
            current = np.ascontiguousarray(np.fliplr(current))
            if mask is not None:
                mask = np.ascontiguousarray(np.fliplr(mask))

        # Small rotations reflect acquisition variability while keeping the
        # temporal relationship between prior and current images intact.
        angle = float(np.random.uniform(-10.0, 10.0))
        prior = self._rotate(prior, angle, interpolation=cv2.INTER_LINEAR)
        current = self._rotate(current, angle, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = self._rotate(mask, angle, interpolation=cv2.INTER_NEAREST)

        # Mild intensity perturbations are applied independently, since prior
        # and current mammograms may differ in exposure and acquisition setting.
        prior = self._intensity_jitter(prior)
        current = self._intensity_jitter(current)
        return prior, current, mask

    def _rotate(self, image: np.ndarray, angle: float, interpolation: int) -> np.ndarray:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=interpolation, borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def _intensity_jitter(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        contrast = float(np.random.uniform(0.9, 1.1))
        brightness = float(np.random.uniform(-0.05, 0.05))
        gamma = float(np.random.uniform(0.9, 1.1))
        image = np.clip(image * contrast + brightness, 0.0, 1.0)
        image = np.power(image, gamma)
        return (image * 255.0).astype(np.uint8)
