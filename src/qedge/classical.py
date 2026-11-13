"""Classical edge detectors used as the ground-truth reference.

The quantum edge maps (NEQR / QHED) are scored against these. We provide:

* ``sobel_edges``  - gradient magnitude (continuous, in [0, 1]).
* ``canny_edges``  - binary edge map (the primary F1 ground truth).
* ``binarize``     - Otsu / fixed-threshold helper to turn any magnitude map
  into a binary edge mask for F1 scoring.
"""

from __future__ import annotations

import numpy as np
from skimage import filters, feature
from skimage.filters import threshold_otsu


def sobel_edges(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, normalised to [0, 1]."""
    mag = filters.sobel(image.astype(float))
    mx = mag.max()
    return mag / mx if mx > 1e-12 else mag


def canny_edges(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Binary Canny edge map (bool array)."""
    img = image.astype(float)
    # For tiny images, Canny's Gaussian smoothing can erase everything; fall
    # back to a thresholded Sobel in that regime.
    if min(img.shape) < 8:
        return binarize(sobel_edges(img))
    try:
        return feature.canny(img, sigma=sigma)
    except Exception:
        return binarize(sobel_edges(img))


def binarize(mag: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Binarise a magnitude map. Uses Otsu when no threshold is given."""
    m = np.asarray(mag, dtype=float)
    if m.max() - m.min() < 1e-12:
        return np.zeros_like(m, dtype=bool)
    if threshold is None:
        try:
            threshold = threshold_otsu(m)
        except Exception:
            threshold = 0.5 * (m.max() + m.min())
    return m > threshold


def ground_truth_edges(image: np.ndarray, kind: str = "sobel") -> np.ndarray:
    """Binary ground-truth edge mask used for F1 scoring."""
    if kind == "canny":
        return canny_edges(image)
    return binarize(sobel_edges(image))
