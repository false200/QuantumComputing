"""Accuracy and resource metrics.

* F1 (precision/recall) between a binarised quantum edge map and the classical
  ground-truth edge mask.
* SSIM between the continuous quantum edge-magnitude map and the classical
  gradient magnitude.
* End-to-end resource counts: qubits, circuit depth/size, gate breakdown,
  two-qubit-gate count after transpilation to a hardware basis, and the total
  shot budget actually consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity

from .classical import binarize


@dataclass
class F1Result:
    precision: float
    recall: float
    f1: float


def f1_score(
    pred_mag: np.ndarray, gt_binary: np.ndarray, tol: int = 1
) -> F1Result:
    """BSDS-style boundary F1 with a ``tol``-pixel matching tolerance.

    A predicted edge pixel is a true positive if a ground-truth edge lies
    within ``tol`` pixels, and a ground-truth edge is recalled if a predicted
    edge lies within ``tol`` pixels. ``tol=0`` reduces to exact matching.
    """
    pred = binarize(np.asarray(pred_mag, dtype=float))
    gt = np.asarray(gt_binary, dtype=bool)

    if tol > 0:
        struct = np.ones((3, 3), dtype=bool)
        gt_match = binary_dilation(gt, structure=struct, iterations=tol)
        pred_match = binary_dilation(pred, structure=struct, iterations=tol)
    else:
        gt_match = gt
        pred_match = pred

    pred_count = int(np.sum(pred))
    gt_count = int(np.sum(gt))
    tp_precision = int(np.sum(pred & gt_match))
    tp_recall = int(np.sum(gt & pred_match))

    precision = tp_precision / pred_count if pred_count > 0 else 0.0
    recall = tp_recall / gt_count if gt_count > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return F1Result(precision, recall, f1)


def ssim_score(pred_mag: np.ndarray, gt_mag: np.ndarray) -> float:
    a = np.asarray(pred_mag, dtype=float)
    b = np.asarray(gt_mag, dtype=float)
    size = min(a.shape)
    win = min(7, size if size % 2 == 1 else size - 1)
    if win < 3:
        # Too small for windowed SSIM: fall back to correlation-based proxy.
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 1.0 if np.allclose(a, b) else 0.0
        return float(np.clip(np.corrcoef(a.ravel(), b.ravel())[0, 1], -1, 1))
    return float(
        structural_similarity(a, b, data_range=1.0, win_size=win)
    )


@dataclass
class ResourceReport:
    num_qubits: int
    depth_logical: int
    size_logical: int
    depth_basis: int
    size_basis: int
    two_qubit_gates: int
    ops: dict
    shots: int
    circuits: int
    total_shots: int

    def to_dict(self) -> dict:
        return asdict(self)


_BASIS = ["rz", "sx", "x", "cx"]


def resource_report(
    circuit: QuantumCircuit,
    shots: int,
    circuits: int = 1,
    seed: int = 0,
) -> ResourceReport:
    """Compute logical + transpiled resource counts for a circuit."""
    logical_depth = circuit.depth()
    logical_size = circuit.size()
    try:
        tqc = transpile(
            circuit, basis_gates=_BASIS, optimization_level=1, seed_transpiler=seed
        )
        ops = dict(tqc.count_ops())
        two_q = int(ops.get("cx", 0) + ops.get("cz", 0))
        depth_basis = tqc.depth()
        size_basis = tqc.size()
    except Exception:
        ops = dict(circuit.count_ops())
        two_q = int(ops.get("cx", 0) + ops.get("cz", 0))
        depth_basis = logical_depth
        size_basis = logical_size

    return ResourceReport(
        num_qubits=circuit.num_qubits,
        depth_logical=logical_depth,
        size_logical=logical_size,
        depth_basis=depth_basis,
        size_basis=size_basis,
        two_qubit_gates=two_q,
        ops=ops,
        shots=shots,
        circuits=circuits,
        total_shots=shots * circuits,
    )


def combine_resource_reports(reports: list[ResourceReport]) -> ResourceReport:
    """Aggregate several per-circuit reports into one end-to-end report."""
    if not reports:
        raise ValueError("no reports to combine")
    ops: dict[str, int] = {}
    for r in reports:
        for k, v in r.ops.items():
            ops[k] = ops.get(k, 0) + v
    return ResourceReport(
        num_qubits=max(r.num_qubits for r in reports),
        depth_logical=max(r.depth_logical for r in reports),
        size_logical=sum(r.size_logical for r in reports),
        depth_basis=max(r.depth_basis for r in reports),
        size_basis=sum(r.size_basis for r in reports),
        two_qubit_gates=sum(r.two_qubit_gates for r in reports),
        ops=ops,
        shots=max(r.shots for r in reports),
        circuits=sum(r.circuits for r in reports),
        total_shots=sum(r.total_shots for r in reports),
    )
