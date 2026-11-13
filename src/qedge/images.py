"""Image sources for the benchmark.

Provides three families of identical-resolution greyscale images so NEQR and
QHED are compared on exactly the same inputs:

1. ``synthetic`` - deterministic shapes (step, cross, disk, checkerboard) with
   crisp ground-truth edges. Ideal for sanity checks at 4x4 .. 32x32.
2. ``standard``  - classic greyscale test images bundled with scikit-image
   (camera, coins, ...), resized to the target resolution.
3. ``bsds500``   - natural images from the Berkeley BSDS500 dataset. Downloaded
   on demand; falls back gracefully to the two families above if unavailable.

All images are returned as float arrays in [0, 1], square, with side length a
power of two (required by the quantum encoders).
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage import data as skdata
from skimage.color import rgb2gray
from skimage.transform import resize

from .config import CONFIG


@dataclass
class TestImage:
    name: str
    family: str  # "synthetic" | "standard" | "bsds500"
    size: int
    image: np.ndarray  # float array in [0, 1], shape (size, size)

    def to_uint(self, bits: int = 8) -> np.ndarray:
        levels = (1 << bits) - 1
        return np.round(np.clip(self.image, 0.0, 1.0) * levels).astype(np.int64)


# --------------------------------------------------------------------------- #
# Synthetic images
# --------------------------------------------------------------------------- #
def _synthetic(size: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    out: dict[str, np.ndarray] = {}

    # Vertical step edge (left dark, right bright)
    step = np.where(xx >= size // 2, 0.9, 0.1)
    out["step"] = step

    # Centred bright square on a dark field
    q = size // 4
    square = np.full((size, size), 0.1)
    square[q : size - q, q : size - q] = 0.9
    out["square"] = square

    # Disk
    cy, cx = (size - 1) / 2.0, (size - 1) / 2.0
    r = size / 3.0
    disk = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r, 0.85, 0.15)
    out["disk"] = disk

    # Checkerboard (2x2 cells)
    cell = max(1, size // 4)
    checker = (((xx // cell) + (yy // cell)) % 2).astype(float)
    checker = 0.1 + 0.8 * checker
    out["checker"] = checker

    return out


# --------------------------------------------------------------------------- #
# Standard scikit-image greyscale images
# --------------------------------------------------------------------------- #
def _standard_sources() -> dict[str, np.ndarray]:
    sources: dict[str, np.ndarray] = {}
    loaders = {
        "camera": skdata.camera,
        "coins": skdata.coins,
        "moon": skdata.moon,
    }
    for name, loader in loaders.items():
        try:
            img = np.asarray(loader(), dtype=float)
        except Exception:
            continue
        if img.ndim == 3:
            img = rgb2gray(img)
        img = img / (img.max() + 1e-12)
        sources[name] = img
    return sources


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    # Center-crop to square first, then resize (anti-aliased) to size x size.
    h, w = img.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    img = img[top : top + s, left : left + s]
    out = resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    out = out.astype(float)
    mn, mx = out.min(), out.max()
    if mx - mn > 1e-12:
        out = (out - mn) / (mx - mn)
    return out


# --------------------------------------------------------------------------- #
# BSDS500 (optional download)
# --------------------------------------------------------------------------- #
def _ensure_bsds500() -> Path | None:
    """Return the directory containing BSDS500 test images, or None."""
    if not CONFIG.use_bsds500:
        return None

    target_dir = CONFIG.bsds500_dir
    images_root = target_dir / "BSR" / "BSDS500" / "data" / "images"
    if images_root.exists() and any(images_root.rglob("*.jpg")):
        return images_root

    if not CONFIG.bsds500_url:
        return None

    archive = CONFIG.data_dir / "BSR_bsds500.tgz"
    try:
        if not archive.exists():
            import requests  # local import so the package imports without network

            with requests.get(CONFIG.bsds500_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                archive.parent.mkdir(parents=True, exist_ok=True)
                with open(archive, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        target_dir.mkdir(parents=True, exist_ok=True)
        # The mirror sometimes serves the archive already decompressed, so let
        # tarfile auto-detect the compression ("r:*") instead of forcing gzip.
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(target_dir)
    except Exception:
        return None

    if images_root.exists() and any(images_root.rglob("*.jpg")):
        return images_root
    return None


def _load_bsds500_sources(limit: int) -> dict[str, np.ndarray]:
    root = _ensure_bsds500()
    sources: dict[str, np.ndarray] = {}
    if root is None:
        return sources
    from skimage.io import imread

    paths = sorted(root.rglob("*.jpg"))[: max(0, limit)]
    for p in paths:
        try:
            img = np.asarray(imread(p), dtype=float)
        except Exception:
            continue
        if img.ndim == 3:
            img = rgb2gray(img)
        img = img / (img.max() + 1e-12)
        sources[f"bsds_{p.stem}"] = img
    return sources


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_images(
    sizes: list[int] | None = None,
    families: tuple[str, ...] = ("synthetic", "standard", "bsds500"),
    bsds_limit: int | None = None,
) -> list[TestImage]:
    """Build the full set of benchmark images across requested sizes."""
    sizes = sizes or CONFIG.image_sizes
    bsds_limit = CONFIG.bsds500_sample if bsds_limit is None else bsds_limit

    standard_src = _standard_sources() if "standard" in families else {}
    bsds_src = _load_bsds500_sources(bsds_limit) if "bsds500" in families else {}

    out: list[TestImage] = []
    for size in sizes:
        if "synthetic" in families:
            for name, img in _synthetic(size).items():
                out.append(TestImage(name, "synthetic", size, img.astype(float)))
        for name, img in standard_src.items():
            out.append(TestImage(name, "standard", size, _resize_square(img, size)))
        for name, img in bsds_src.items():
            out.append(TestImage(name, "bsds500", size, _resize_square(img, size)))
    return out


def bsds500_available() -> bool:
    return _ensure_bsds500() is not None
