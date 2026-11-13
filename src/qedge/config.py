"""Central configuration, loaded from the project .env file.

All tunables (seed, shot budgets, image sizes, noise sweep, dataset paths)
live in ``.env`` so experiments are reproducible and easy to re-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/qedge/..)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_list(name: str, default: str) -> list[float]:
    raw = os.getenv(name, default)
    return [float(x) for x in raw.split(",") if x.strip() != ""]


def _int_list(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    return [int(x) for x in raw.split(",") if x.strip() != ""]


@dataclass
class Config:
    seed: int = field(default_factory=lambda: _int("QEDGE_SEED", 12345))
    shots: int = field(default_factory=lambda: _int("QEDGE_SHOTS", 4096))
    shot_sweep: list[int] = field(
        default_factory=lambda: _int_list(
            "QEDGE_SHOT_SWEEP", "256,512,1024,2048,4096,8192,16384,32768"
        )
    )
    image_sizes: list[int] = field(
        default_factory=lambda: _int_list("QEDGE_IMAGE_SIZES", "4,8,16,32")
    )
    noise_sweep: list[float] = field(
        default_factory=lambda: _float_list(
            "QEDGE_NOISE_SWEEP", "0.0,0.001,0.005,0.01,0.02"
        )
    )
    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("QEDGE_DATA_DIR", "data")
    )
    results_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("QEDGE_RESULTS_DIR", "results")
    )
    bsds500_url: str = field(
        default_factory=lambda: os.getenv("QEDGE_BSDS500_URL", "")
    )
    bsds500_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / os.getenv("QEDGE_BSDS500_DIR", "data/BSDS500")
    )
    use_bsds500: bool = field(
        default_factory=lambda: _int("QEDGE_USE_BSDS500", 1) == 1
    )
    bsds500_sample: int = field(
        default_factory=lambda: _int("QEDGE_BSDS500_SAMPLE", 4)
    )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
CONFIG.ensure_dirs()
