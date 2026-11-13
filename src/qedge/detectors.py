"""End-to-end detector pipelines for NEQR and QHED.

Each ``run_*`` builds the per-axis circuits, executes them on Aer (optionally
noisy, optionally with readout mitigation), decodes the counts into an edge
map, and returns the combined map together with an aggregated resource report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit_aer.noise import NoiseModel

from .images import TestImage
from .metrics import ResourceReport, combine_resource_reports, resource_report
from .neqr import (
    build_neqr_edge_circuit,
    decode_neqr_edges,
    neqr_edge_magnitude,
)
from .noise import ReadoutMitigator
from .qhed import (
    build_qhed_circuits,
    decode_qhed_axis,
    qhed_edge_magnitude,
)
from .simulate import run_counts


@dataclass
class DetectorResult:
    method: str
    edge_map: np.ndarray
    resources: ResourceReport
    h_map: np.ndarray
    v_map: np.ndarray


# --------------------------------------------------------------------------- #
# QHED
# --------------------------------------------------------------------------- #
def run_qhed(
    image: np.ndarray,
    shots: int,
    noise_model: NoiseModel | None = None,
    mitigator: ReadoutMitigator | None = None,
    seed: int | None = None,
) -> DetectorResult:
    size = image.shape[0]
    reports: list[ResourceReport] = []
    axis_maps = {}
    for direction in ("h", "v"):
        circuits = build_qhed_circuits(image, direction)
        decoded = {}
        for name, qc in (("even", circuits.even), ("odd", circuits.odd)):
            counts = run_counts(qc, shots, noise_model, seed)
            if mitigator is not None:
                counts = mitigator.apply(counts)
            decoded[name] = counts
            reports.append(resource_report(qc, shots, circuits=1, seed=seed or 0))
        axis_maps[direction] = decode_qhed_axis(
            decoded["even"], decoded["odd"], size, direction
        )

    edge = qhed_edge_magnitude(axis_maps["h"], axis_maps["v"])
    res = combine_resource_reports(reports)
    return DetectorResult("QHED", edge, res, axis_maps["h"], axis_maps["v"])


# --------------------------------------------------------------------------- #
# NEQR
# --------------------------------------------------------------------------- #
def run_neqr(
    image: np.ndarray,
    shots: int,
    color_bits: int = 8,
    noise_model: NoiseModel | None = None,
    mitigator: ReadoutMitigator | None = None,
    seed: int | None = None,
) -> DetectorResult:
    size = image.shape[0]
    levels = (1 << color_bits) - 1
    image_u = np.round(np.clip(image, 0, 1) * levels).astype(np.int64)

    reports: list[ResourceReport] = []
    axis_maps = {}
    for direction in ("h", "v"):
        built = build_neqr_edge_circuit(image_u, color_bits, direction)
        counts = run_counts(built.circuit, shots, noise_model, seed)
        if mitigator is not None:
            counts = mitigator.apply(counts)
        axis_maps[direction] = decode_neqr_edges(counts, size, color_bits)
        reports.append(
            resource_report(built.circuit, shots, circuits=1, seed=seed or 0)
        )

    edge = neqr_edge_magnitude(axis_maps["h"], axis_maps["v"])
    res = combine_resource_reports(reports)
    return DetectorResult("NEQR", edge, res, axis_maps["h"], axis_maps["v"])


def neqr_n_bits(size: int, color_bits: int) -> int:
    n = int(round(np.log2(size)))
    return color_bits + 2 * n


def qhed_n_bits(size: int) -> int:
    n = int(round(np.log2(size)))
    return 2 * n
