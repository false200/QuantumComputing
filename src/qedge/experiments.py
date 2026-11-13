"""Experiment orchestration: shot sweep, noise sweep, and size/resource sweep.

Each experiment writes a tidy CSV into ``results/`` and returns a pandas
DataFrame. Figures are produced separately in :mod:`qedge.figures`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .classical import ground_truth_edges, sobel_edges
from .config import CONFIG
from .detectors import (
    DetectorResult,
    neqr_n_bits,
    qhed_n_bits,
    run_neqr,
    run_qhed,
)
from .images import TestImage, load_images
from .metrics import f1_score, ssim_score
from .noise import ReadoutMitigator, build_noise_model

# Keep NEQR tractable: cap total measured bits (position + colour) and the
# image size we actually simulate (state-prep gate count explodes with size).
NEQR_MAX_BITS = 18
NEQR_MAX_SIZE = 16


def color_bits_for(size: int) -> int:
    """Adaptive NEQR colour depth so the circuit stays simulable."""
    if size <= 4:
        return 8
    if size <= 8:
        return 6
    if size <= 16:
        return 4
    return 2


def _score(res: DetectorResult, image: np.ndarray, gt_kind: str = "sobel") -> dict:
    gt = ground_truth_edges(image, gt_kind)
    sob = sobel_edges(image)
    f1 = f1_score(res.edge_map, gt)
    ssim = ssim_score(res.edge_map, sob)
    return {
        "precision": f1.precision,
        "recall": f1.recall,
        "f1": f1.f1,
        "ssim": ssim,
        "num_qubits": res.resources.num_qubits,
        "two_qubit_gates": res.resources.two_qubit_gates,
        "depth_basis": res.resources.depth_basis,
        "size_basis": res.resources.size_basis,
        "circuits": res.resources.circuits,
        "total_shots": res.resources.total_shots,
    }


def _run_one(
    method: str,
    image: np.ndarray,
    shots: int,
    color_bits: int,
    noise_model=None,
    mitigator=None,
    seed: int | None = None,
) -> DetectorResult:
    if method == "QHED":
        return run_qhed(image, shots, noise_model, mitigator, seed)
    return run_neqr(image, shots, color_bits, noise_model, mitigator, seed)


def _neqr_feasible(size: int) -> bool:
    return (
        size <= NEQR_MAX_SIZE
        and neqr_n_bits(size, color_bits_for(size)) <= NEQR_MAX_BITS
    )


# --------------------------------------------------------------------------- #
# 1. Shot-requirement sweep (noise-free)
# --------------------------------------------------------------------------- #
def shot_sweep(
    images: list[TestImage],
    shots_list: list[int] | None = None,
    methods: tuple[str, ...] = ("QHED", "NEQR"),
    save: bool = True,
) -> pd.DataFrame:
    shots_list = shots_list or CONFIG.shot_sweep
    rows = []
    for ti in images:
        cbits = color_bits_for(ti.size)
        for method in methods:
            if method == "NEQR" and not _neqr_feasible(ti.size):
                continue
            for shots in shots_list:
                res = _run_one(method, ti.image, shots, cbits, seed=CONFIG.seed)
                row = {
                    "experiment": "shot_sweep",
                    "method": method,
                    "image": ti.name,
                    "family": ti.family,
                    "size": ti.size,
                    "color_bits": cbits if method == "NEQR" else None,
                    "shots": shots,
                    "noise": 0.0,
                    "mitigation": False,
                }
                row.update(_score(res, ti.image))
                rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(CONFIG.results_dir / "shot_sweep.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 2. Noise-tolerance sweep (fixed shots, +/- mitigation)
# --------------------------------------------------------------------------- #
def noise_sweep(
    images: list[TestImage],
    noise_levels: list[float] | None = None,
    shots: int | None = None,
    methods: tuple[str, ...] = ("QHED", "NEQR"),
    neqr_color_bits: int | None = None,
    save: bool = True,
) -> pd.DataFrame:
    noise_levels = noise_levels or CONFIG.noise_sweep
    shots = shots or CONFIG.shots
    rows = []
    for ti in images:
        cbits = neqr_color_bits or color_bits_for(ti.size)
        for method in methods:
            if method == "NEQR" and not _neqr_feasible(ti.size):
                continue
            n_bits = (
                qhed_n_bits(ti.size)
                if method == "QHED"
                else neqr_n_bits(ti.size, cbits)
            )
            for level in noise_levels:
                nm = build_noise_model(level)
                for mitigate in (False, True):
                    if mitigate and (nm is None or n_bits > 20):
                        continue
                    mit = (
                        ReadoutMitigator.calibrate(n_bits, nm, seed=CONFIG.seed)
                        if mitigate
                        else None
                    )
                    res = _run_one(
                        method, ti.image, shots, cbits, nm, mit, CONFIG.seed
                    )
                    row = {
                        "experiment": "noise_sweep",
                        "method": method,
                        "image": ti.name,
                        "family": ti.family,
                        "size": ti.size,
                        "color_bits": cbits if method == "NEQR" else None,
                        "shots": shots,
                        "noise": level,
                        "mitigation": mitigate,
                    }
                    row.update(_score(res, ti.image))
                    rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(CONFIG.results_dir / "noise_sweep.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 3. Size / resource sweep
# --------------------------------------------------------------------------- #
def shot_size_sweep(
    sizes: list[int],
    shots_list: list[int] | None = None,
    image_name: str = "disk",
    methods: tuple[str, ...] = ("QHED", "NEQR"),
    save: bool = True,
) -> pd.DataFrame:
    """Shots needed to reconstruct the edge map as image size (N pixels) grows.

    This is the core measurement-bottleneck experiment: with more pixels the
    probability mass per outcome shrinks, so more shots are needed for the same
    accuracy - the O(N) readout cost that erodes the asymptotic advantage.
    """
    shots_list = shots_list or CONFIG.shot_sweep
    rows = []
    for size in sizes:
        imgs = [
            ti
            for ti in load_images(sizes=[size], families=("synthetic",))
            if ti.name == image_name
        ]
        if not imgs:
            continue
        ti = imgs[0]
        cbits = color_bits_for(size)
        for method in methods:
            if method == "NEQR" and not _neqr_feasible(size):
                continue
            for shots in shots_list:
                res = _run_one(method, ti.image, shots, cbits, seed=CONFIG.seed)
                row = {
                    "experiment": "shot_size_sweep",
                    "method": method,
                    "image": image_name,
                    "size": size,
                    "pixels": size * size,
                    "shots": shots,
                }
                row.update(_score(res, ti.image))
                rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(CONFIG.results_dir / "shot_size_sweep.csv", index=False)
    return df


def size_sweep(
    sizes: list[int] | None = None,
    image_name: str = "disk",
    shots: int | None = None,
    methods: tuple[str, ...] = ("QHED", "NEQR"),
    save: bool = True,
) -> pd.DataFrame:
    sizes = sizes or CONFIG.image_sizes
    shots = shots or CONFIG.shots
    rows = []
    for size in sizes:
        imgs = [
            ti
            for ti in load_images(sizes=[size], families=("synthetic",))
            if ti.name == image_name
        ]
        if not imgs:
            continue
        ti = imgs[0]
        cbits = color_bits_for(size)
        for method in methods:
            if method == "NEQR" and not _neqr_feasible(size):
                rows.append(
                    {
                        "experiment": "size_sweep",
                        "method": method,
                        "image": image_name,
                        "size": size,
                        "shots": shots,
                        "feasible": False,
                        "predicted_qubits": neqr_n_bits(size, cbits),
                    }
                )
                continue
            res = _run_one(method, ti.image, shots, cbits, seed=CONFIG.seed)
            row = {
                "experiment": "size_sweep",
                "method": method,
                "image": image_name,
                "size": size,
                "shots": shots,
                "feasible": True,
                "color_bits": cbits if method == "NEQR" else None,
            }
            row.update(_score(res, ti.image))
            rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(CONFIG.results_dir / "size_sweep.csv", index=False)
    return df
