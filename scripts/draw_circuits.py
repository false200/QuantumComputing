"""Draw the NEQR / QHED / Draper circuit diagrams into results/."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qedge.arithmetic import draper_subtractor
from qedge.config import CONFIG
from qedge.images import load_images
from qedge.neqr import build_neqr_edge_circuit
from qedge.qhed import build_qhed_circuits


def main() -> None:
    out = CONFIG.results_dir
    out.mkdir(parents=True, exist_ok=True)

    img = next(
        t
        for t in load_images(sizes=[4], families=("synthetic",))
        if t.name == "step"
    )
    levels = 15
    img_u = np.round(np.clip(img.image, 0, 1) * levels).astype(np.int64)

    neqr = build_neqr_edge_circuit(img_u, color_bits=2, direction="h")
    fig = neqr.circuit.draw(
        output="mpl", fold=-1, idle_wires=False, style="iqp", scale=0.85
    )
    path = out / "fig_circuit_neqr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)

    qh = build_qhed_circuits(img.image, direction="h")
    for name, qc in (("qhed", qh.even), ("qhed_odd", qh.odd)):
        fig = qc.draw(
            output="mpl", fold=-1, idle_wires=False, style="iqp", scale=1.0
        )
        path = out / f"fig_circuit_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("wrote", path)

    sub = draper_subtractor(3).decompose(reps=1)
    fig = sub.draw(output="mpl", fold=55, style="iqp", scale=0.85)
    path = out / "fig_circuit_draper.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
