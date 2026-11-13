"""QHED (Quantum Hadamard Edge Detection) on a QPIE-encoded image.

QPIE (Quantum Probability Image Encoding) packs the whole 2**n x 2**n image into
the *amplitudes* of a single 2n-qubit register, so the image uses only
``2*log2(N)`` qubits - the source of the "exponential" qubit advantage.

A Hadamard on the least-significant qubit of an axis maps neighbouring pixel
amplitudes to their sum / difference:

    (a_2b , a_{2b+1})  ->  ( (a_2b + a_{2b+1})/sqrt2 , (a_2b - a_{2b+1})/sqrt2 )

so the odd outcomes carry |a_2b - a_{2b+1}|, i.e. the edge between a pair. A
cyclic decrement before the Hadamard shifts the pairing to recover the
in-between boundaries, giving the full gradient along that axis. Two passes
(even + odd) per axis are combined. Reading the full N-pixel edge map still
needs O(N) shots - the measurement bottleneck under study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def _log2_int(n: int) -> int:
    r = int(round(np.log2(n)))
    if 2 ** r != n:
        raise ValueError(f"size {n} is not a power of two")
    return r


def _increment(qubits: list) -> QuantumCircuit:
    n = len(qubits)
    qc = QuantumCircuit(n, name="incr")
    for i in range(n - 1, 0, -1):
        qc.mcx(list(range(i)), i)
    qc.x(0)
    return qc


@dataclass
class QHEDCircuits:
    even: QuantumCircuit
    odd: QuantumCircuit
    size: int
    direction: str
    n_axis: int


def _amplitudes(image: np.ndarray) -> np.ndarray:
    a = image.astype(float).flatten(order="C")  # index = y*W + x
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        a = np.ones_like(a)
        norm = np.linalg.norm(a)
    return a / norm


def build_qhed_circuits(image: np.ndarray, direction: str = "h") -> QHEDCircuits:
    """Build the even/odd QHED circuits for one axis ('h' or 'v')."""
    H, W = image.shape
    if H != W:
        raise ValueError("image must be square")
    n = _log2_int(W)
    ntot = 2 * n

    amps = _amplitudes(image)

    if direction == "h":
        axis_qubits = list(range(n))        # x register (LSB = qubit 0)
    elif direction == "v":
        axis_qubits = list(range(n, 2 * n)) # y register (LSB = qubit n)
    else:
        raise ValueError("direction must be 'h' or 'v'")
    lsb = axis_qubits[0]

    def make(shift: bool) -> QuantumCircuit:
        qr = QuantumRegister(ntot, "q")
        cr = ClassicalRegister(ntot, "m")
        qc = QuantumCircuit(qr, cr)
        qc.prepare_state(amps, qr[:])
        if shift:
            dec = _increment([qr[i] for i in axis_qubits]).inverse()
            qc.append(dec.to_gate(), [qr[i] for i in axis_qubits])
        qc.h(qr[lsb])
        qc.measure(qr[:], cr[:])
        return qc

    return QHEDCircuits(make(False), make(True), W, direction, n)


def _accumulate(counts: dict[str, int], n: int, direction: str):
    """Group odd-outcome counts by (pair_high_bits, orthogonal_index)."""
    mask = (1 << n) - 1
    shots = sum(counts.values())
    groups: dict[tuple[int, int], int] = {}
    for key, c in counts.items():
        i = int(key.replace(" ", ""), 2)
        x = i & mask
        y = (i >> n) & mask
        if direction == "h":
            axis_val, ortho = x, y
        else:
            axis_val, ortho = y, x
        if axis_val & 1:  # odd outcome carries the difference
            b = axis_val >> 1
            groups[(ortho, b)] = groups.get((ortho, b), 0) + c
    return groups, shots


def decode_qhed_axis(
    counts_even: dict[str, int], counts_odd: dict[str, int], size: int, direction: str
) -> np.ndarray:
    """Combine even + odd passes into a gradient-magnitude map along one axis."""
    n = _log2_int(size)
    edge = np.zeros((size, size), dtype=float)

    g_even, s_even = _accumulate(counts_even, n, direction)
    g_odd, s_odd = _accumulate(counts_odd, n, direction)

    for (ortho, b), c in g_even.items():
        diff = np.sqrt(2.0 * c / max(s_even, 1))
        pos = 2 * b  # boundary between 2b and 2b+1
        if direction == "h":
            edge[ortho, pos] = diff
        else:
            edge[pos, ortho] = diff

    for (ortho, b), c in g_odd.items():
        diff = np.sqrt(2.0 * c / max(s_odd, 1))
        pos = 2 * b + 1  # boundary between 2b+1 and 2b+2
        if pos >= size - 1:
            # pos == size-1 is the non-physical cyclic wrap boundary
            # (between the last and first pixel) introduced by the decrement.
            continue
        if direction == "h":
            edge[ortho, pos] = diff
        else:
            edge[pos, ortho] = diff

    mx = edge.max()
    return edge / mx if mx > 1e-12 else edge


def qhed_edge_magnitude(h_map: np.ndarray, v_map: np.ndarray) -> np.ndarray:
    mag = np.sqrt(h_map.astype(float) ** 2 + v_map.astype(float) ** 2)
    mx = mag.max()
    return mag / mx if mx > 1e-12 else mag
