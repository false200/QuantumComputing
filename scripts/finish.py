"""Finish the benchmark: noise sweep (representative subset) + size sweep,
then figures + summary. The full-scope shot sweep is already in
results/shot_sweep.csv."""
import json

import pandas as pd

from qedge.config import CONFIG
from qedge.experiments import noise_sweep, size_sweep
from qedge.figures import (
    fig_noise_sweep,
    fig_resources,
    fig_shot_sweep,
    montage,
)
from qedge.images import bsds500_available, load_images
from qedge.cli import _build_summary


def noise_subset():
    imgs = []
    for ti in load_images(sizes=[4], families=("synthetic",)):
        if ti.name in ("step", "square", "disk"):
            imgs.append(ti)
    for ti in load_images(sizes=[4], families=("standard",)):
        if ti.name == "camera":
            imgs.append(ti)
    if bsds500_available():
        b = [t for t in load_images(sizes=[4], families=("bsds500",))]
        if b:
            imgs.append(b[0])
    return imgs


def main():
    print("[noise] subset sweep ...", flush=True)
    imgs = noise_subset()
    print("  images:", [i.name for i in imgs], flush=True)
    # 2-bit colour keeps the deep NEQR state-prep at 8 qubits so the *noisy*
    # density-matrix simulations stay fast; synthetic shapes are ~2-level so
    # this is near-lossless for them and still edge-bearing for natural images.
    df_noise = noise_sweep(imgs, shots=4096, neqr_color_bits=2)

    print("[size] resource sweep ...", flush=True)
    df_size = size_sweep(sizes=[4, 8, 16, 32], image_name="disk")

    print("[figures] ...", flush=True)
    fig_shot_sweep()
    fig_noise_sweep()
    fig_resources()
    montage([t for t in load_images(sizes=[8], families=("synthetic",))
             if t.name in ("step", "square", "disk", "checker")])

    df_shots = pd.read_csv(CONFIG.results_dir / "shot_sweep.csv")
    sspath = CONFIG.results_dir / "shot_size_sweep.csv"
    df_shot_size = pd.read_csv(sspath) if sspath.exists() else None
    summary = _build_summary(df_shots, df_noise, df_size, df_shot_size)
    with open(CONFIG.results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    main()
