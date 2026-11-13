"""Command-line entry point: ``qedge <command>``.

Commands
--------
smoke     quick noise-free correctness check on tiny images
shots     shot-requirement sweep  -> results/shot_sweep.csv
noise     noise-tolerance sweep   -> results/noise_sweep.csv
size      size/resource sweep     -> results/size_sweep.csv
figures   render figures from existing CSVs
montage   qualitative input/Sobel/QHED/NEQR montage
all       run the full benchmark + figures + summary
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import CONFIG
from .detectors import run_neqr, run_qhed
from .images import bsds500_available, load_images
from .metrics import f1_score, ssim_score
from .classical import ground_truth_edges, sobel_edges


def _benchmark_images(sizes, bsds_limit=None):
    families = ("synthetic", "standard")
    if CONFIG.use_bsds500 and bsds500_available():
        families = families + ("bsds500",)
    return load_images(sizes=sizes, families=families, bsds_limit=bsds_limit)


def cmd_smoke(_args):
    imgs = load_images(sizes=[4, 8], families=("synthetic",))
    imgs = [t for t in imgs if t.name in ("step", "disk")]
    for ti in imgs:
        gt = ground_truth_edges(ti.image)
        sob = sobel_edges(ti.image)
        q = run_qhed(ti.image, shots=16384, seed=CONFIG.seed)
        n = run_neqr(ti.image, shots=16384, color_bits=4, seed=CONFIG.seed)
        print(
            f"{ti.name:6s} {ti.size}x{ti.size} | "
            f"QHED F1={f1_score(q.edge_map, gt).f1:.2f} "
            f"SSIM={ssim_score(q.edge_map, sob):.2f} "
            f"q={q.resources.num_qubits} 2q={q.resources.two_qubit_gates} | "
            f"NEQR F1={f1_score(n.edge_map, gt).f1:.2f} "
            f"SSIM={ssim_score(n.edge_map, sob):.2f} "
            f"q={n.resources.num_qubits} 2q={n.resources.two_qubit_gates}"
        )


def cmd_shots(args):
    from .experiments import shot_sweep

    imgs = _benchmark_images(sizes=args.sizes)
    df = shot_sweep(imgs)
    print(df.groupby(["method", "shots"])[["f1", "ssim"]].mean())


def cmd_noise(args):
    from .experiments import noise_sweep

    imgs = _benchmark_images(sizes=args.sizes)
    df = noise_sweep(imgs)
    print(df.groupby(["method", "noise", "mitigation"])["f1"].mean())


def cmd_size(args):
    from .experiments import size_sweep

    df = size_sweep(sizes=args.sizes)
    print(df)


def cmd_figures(_args):
    from .figures import fig_noise_sweep, fig_resources, fig_shot_sweep

    for fn in (fig_shot_sweep, fig_noise_sweep, fig_resources):
        out = fn()
        print("wrote", out)


def cmd_montage(args):
    from .figures import montage

    imgs = load_images(sizes=[args.size], families=("synthetic",))
    imgs = [t for t in imgs if t.name in ("step", "square", "disk", "checker")]
    out = montage(imgs)
    print("wrote", out)


def cmd_all(args):
    from .experiments import (
        noise_sweep,
        shot_size_sweep,
        shot_sweep,
        size_sweep,
    )
    from .figures import (
        fig_noise_sweep,
        fig_resources,
        fig_shot_size,
        fig_shot_sweep,
        montage,
    )

    print(f"BSDS500 available: {bsds500_available()}")
    sizes_sweep = args.sizes or [4, 8]
    imgs = _benchmark_images(sizes=sizes_sweep)
    print(f"benchmark images: {len(imgs)}")

    print("[1/4] shot sweep ...")
    df_shots = shot_sweep(imgs)
    print("[2/4] noise sweep ...")
    df_noise = noise_sweep(
        [t for t in imgs if t.size == min(sizes_sweep)], neqr_color_bits=2
    )
    print("[3/4] size/resource sweep ...")
    df_size = size_sweep(sizes=CONFIG.image_sizes)
    print("[4/4] shots-vs-size (O(N) readout) sweep ...")
    df_shot_size = shot_size_sweep(sizes=[s for s in CONFIG.image_sizes if s >= 8])

    fig_shot_sweep()
    fig_noise_sweep()
    fig_resources()
    fig_shot_size()
    montage([t for t in imgs if t.size == min(sizes_sweep)
             and t.name in ("step", "square", "disk", "checker")])

    summary = _build_summary(df_shots, df_noise, df_size, df_shot_size)
    with open(CONFIG.results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nAll results written to", CONFIG.results_dir)


def _build_summary(df_shots, df_noise, df_size, df_shot_size=None):
    summary = {}
    # shots needed to first reach F1 >= 0.9 (mean over images), per method
    need = {}
    for method, sub in df_shots.groupby("method"):
        g = sub.groupby("shots")["f1"].mean().sort_index()
        hit = g[g >= 0.9]
        need[method] = int(hit.index[0]) if len(hit) else None
    summary["shots_to_f1_0.9_small_images"] = need

    # Core O(N) finding: shots to reach F1 >= 0.99 as a function of image size.
    if df_shot_size is not None and len(df_shot_size):
        scale = {}
        for (method, size), sub in df_shot_size.groupby(["method", "size"]):
            g = sub.groupby("shots")["f1"].mean().sort_index()
            hit = g[g >= 0.99]
            scale.setdefault(method, {})[int(size)] = (
                int(hit.index[0]) if len(hit) else None
            )
        summary["shots_to_f1_0.99_by_size"] = scale

    # Noise tolerance: F1 retained at the highest swept noise level, raw vs
    # mitigated. (A sharper discriminator than a fixed F1 threshold.)
    max_noise = float(df_noise["noise"].max()) if len(df_noise) else 0.0
    retained = {}
    for (method, mit), sub in df_noise.groupby(["method", "mitigation"]):
        g = sub.groupby("noise")["f1"].mean()
        key = f"{method}_{'mitigated' if mit else 'raw'}"
        retained[key] = round(float(g.get(max_noise, float("nan"))), 4)
    summary["f1_at_max_noise"] = {"noise_level": max_noise, "f1": retained}

    # resource scaling
    res = {}
    for _, row in df_size.iterrows():
        if not row.get("feasible", False):
            continue
        res.setdefault(row["method"], {})[int(row["size"])] = {
            "qubits": int(row["num_qubits"]),
            "two_qubit_gates": int(row["two_qubit_gates"]),
        }
    summary["resources"] = res
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="qedge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_sizes(sp, default):
        sp.add_argument(
            "--sizes", type=int, nargs="+", default=default,
            help="image side lengths (powers of two)",
        )

    sp = sub.add_parser("smoke"); sp.set_defaults(func=cmd_smoke)
    sp = sub.add_parser("shots"); add_sizes(sp, [4, 8]); sp.set_defaults(func=cmd_shots)
    sp = sub.add_parser("noise"); add_sizes(sp, [8]); sp.set_defaults(func=cmd_noise)
    sp = sub.add_parser("size"); add_sizes(sp, CONFIG.image_sizes); sp.set_defaults(func=cmd_size)
    sp = sub.add_parser("figures"); sp.set_defaults(func=cmd_figures)
    sp = sub.add_parser("montage"); sp.add_argument("--size", type=int, default=8); sp.set_defaults(func=cmd_montage)
    sp = sub.add_parser("all"); add_sizes(sp, [4, 8]); sp.set_defaults(func=cmd_all)

    args = p.parse_args(argv)
    np.random.seed(CONFIG.seed)
    args.func(args)


if __name__ == "__main__":
    main()
