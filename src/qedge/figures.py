"""Figure generation from experiment CSVs + qualitative edge-map montages."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classical import ground_truth_edges, sobel_edges
from .config import CONFIG
from .detectors import run_neqr, run_qhed
from .experiments import color_bits_for, _neqr_feasible
from .images import TestImage

_COLORS = {"QHED": "#1f77b4", "NEQR": "#d62728"}


def fig_shot_sweep(csv: Path | None = None) -> Path | None:
    csv = csv or (CONFIG.results_dir / "shot_sweep.csv")
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, sub in df.groupby("method"):
        g = sub.groupby("shots")[["f1", "ssim"]].mean().reset_index()
        ax1.plot(g["shots"], g["f1"], "o-", label=method, color=_COLORS.get(method))
        ax2.plot(g["shots"], g["ssim"], "o-", label=method, color=_COLORS.get(method))
    for ax, title, ylab in (
        (ax1, "Edge-map F1 vs shot budget", "F1 (BSDS-style, tol=1)"),
        (ax2, "Edge-map SSIM vs shot budget", "SSIM vs Sobel"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("shots per circuit")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = CONFIG.results_dir / "fig_shot_sweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_noise_sweep(csv: Path | None = None) -> Path | None:
    csv = csv or (CONFIG.results_dir / "noise_sweep.csv")
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    for (method, mit), sub in df.groupby(["method", "mitigation"]):
        g = sub.groupby("noise")["f1"].mean().reset_index()
        style = "-" if not mit else "--"
        label = f"{method} {'(mitigated)' if mit else '(raw)'}"
        ax.plot(
            g["noise"], g["f1"], marker="o", linestyle=style,
            color=_COLORS.get(method), label=label,
        )
    ax.set_xlabel("noise level (2-qubit depolarising prob.)")
    ax.set_ylabel("F1 (BSDS-style, tol=1)")
    ax.set_title("Noise tolerance with / without readout mitigation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = CONFIG.results_dir / "fig_noise_sweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_resources(csv: Path | None = None, neqr_q: int = 8) -> Path | None:
    csv = csv or (CONFIG.results_dir / "size_sweep.csv")
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: theoretical qubit requirement (honest asymptotic scaling with a
    # fixed q-bit colour depth), independent of what we could simulate.
    sizes = sorted(df["size"].unique())
    nlog = np.array([int(round(np.log2(s))) for s in sizes])
    qhed_qubits = 2 * nlog
    neqr_qubits = 2 * nlog + 2 * neqr_q  # two colour registers for differencing
    ax1.plot(sizes, qhed_qubits, "o-", label="QHED (2·log₂N)",
             color=_COLORS["QHED"])
    ax1.plot(sizes, neqr_qubits, "o-",
             label=f"NEQR (2·log₂N + 2·{neqr_q})", color=_COLORS["NEQR"])

    for method, sub in df.groupby("method"):
        sub = sub.sort_values("size")
        feas = sub[sub["feasible"] == True]  # noqa: E712
        if "two_qubit_gates" in feas:
            ax2.plot(
                feas["size"], feas["two_qubit_gates"], "o-",
                label=method, color=_COLORS.get(method),
            )
    ax1.set_title("Theoretical qubit requirement vs image size")
    ax1.set_ylabel("qubits")
    ax2.set_title("Two-qubit gates vs image size")
    ax2.set_ylabel("CX count (after transpile)")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xlabel("image side length")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = CONFIG.results_dir / "fig_resources.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_shot_size(csv: Path | None = None) -> Path | None:
    csv = csv or (CONFIG.results_dir / "shot_size_sweep.csv")
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    sizes = sorted(df["size"].unique())
    for method in sorted(df["method"].unique()):
        for i, size in enumerate(sizes):
            sub = df[(df["method"] == method) & (df["size"] == size)]
            if sub.empty:
                continue
            sub = sub.sort_values("shots")
            style = "-" if method == "QHED" else "--"
            ax.plot(
                sub["shots"], sub["f1"], marker="o", linestyle=style,
                color=cmap(i / max(len(sizes) - 1, 1)),
                label=f"{method} {size}x{size}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("shots per circuit")
    ax.set_ylabel("F1 (BSDS-style, tol=1)")
    ax.set_title("Shot budget needed grows with image size (O(N) readout)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = CONFIG.results_dir / "fig_shot_size.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def montage(
    images: list[TestImage], shots: int = 8192, seed: int | None = None
) -> Path:
    seed = CONFIG.seed if seed is None else seed
    rows = len(images)
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.6 * rows))
    if rows == 1:
        axes = axes[None, :]
    col_titles = ["input", "Sobel (ground truth)", "QHED", "NEQR"]
    for r, ti in enumerate(images):
        cbits = color_bits_for(ti.size)
        qhed = run_qhed(ti.image, shots, seed=seed).edge_map
        if _neqr_feasible(ti.size):
            neqr = run_neqr(ti.image, shots, cbits, seed=seed).edge_map
        else:
            neqr = np.zeros_like(ti.image)
        panels = [ti.image, sobel_edges(ti.image), qhed, neqr]
        for c, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"{ti.name}\n{ti.size}x{ti.size}", fontsize=9)
    fig.tight_layout()
    out = CONFIG.results_dir / "fig_montage.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
