"""Binary Localization Mask refinement used during inference."""

import numpy as np

from .largest_component import keep_largest_component
from .morphology import morphological_closing, remove_small_components
from .threshold import threshold_map


def refine_blm(
    score_map: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 100,
    closing_kernel: int = 5,
    cancer_probability: float = None,
    cancer_probability_threshold: float = 0.20,
) -> np.ndarray:
    """Convert a localization score map into a refined binary mask.

    When a cancer probability is supplied, confidently normal cases can be
    suppressed before morphological processing. This classification-guided gate
    is useful for reducing residual activations in normal mammograms.
    """
    if cancer_probability is not None and cancer_probability < cancer_probability_threshold:
        return np.zeros_like(score_map, dtype=np.uint8)

    mask = threshold_map(score_map, threshold)
    mask = remove_small_components(mask, min_component_area)
    mask = morphological_closing(mask, closing_kernel)
    mask = keep_largest_component(mask)
    return mask.astype(np.uint8)
