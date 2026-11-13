"""Quantum image edge detection with NEQR and QHED.

Accuracy, shot requirements, and noise analysis. Head-to-head comparison of the
NEQR (basis-encoded) and QHED (amplitude-encoded) quantum edge detectors.
"""

from .config import CONFIG, Config
from .detectors import DetectorResult, run_neqr, run_qhed
from .images import TestImage, load_images
from .metrics import F1Result, f1_score, resource_report, ssim_score
from .noise import ReadoutMitigator, build_noise_model

__all__ = [
    "CONFIG",
    "Config",
    "DetectorResult",
    "run_neqr",
    "run_qhed",
    "TestImage",
    "load_images",
    "F1Result",
    "f1_score",
    "ssim_score",
    "resource_report",
    "ReadoutMitigator",
    "build_noise_model",
]

__version__ = "0.1.0"
