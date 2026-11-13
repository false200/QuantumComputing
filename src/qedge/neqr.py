"""NEQR (Novel Enhanced Quantum Representation) encoding + edge detection.

NEQR stores a 2**n x 2**n greyscale image as

    |I> = (1/2**n) * sum_{y,x} |c(y,x)> |y> |x>

with a ``q``-bit colour register holding the exact intensity of each pixel in
the computational basis. State preparation costs many multi-controlled-X gates
(one group per pixel) - precisely the expensive step the project accounts for.

Edge detection (quantum-arithmetic differencing)
-------------------------------------------------
To obtain a horizontal gradient we entangle *two* colour registers with the
same position register: ``c1`` holds ``I(y, x)`` and ``c2`` holds the neighbour
``I(y, x+1)`` (the neighbour image is a classically shifted copy, which is a
legitimate NEQR input). A Draper subtractor then computes

    c1 <- (I(y, x) - I(y, x+1))  (mod 2**q)

so that measuring the position + ``c1`` registers yields, per position, the
exact neighbour difference. Reading the whole edge map therefore needs enough
shots to visit all O(N) positions - the measurement bottleneck under study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .arithmetic import draper_subtractor


def _log2_int(n: int) -> int:
    r = int(round(np.log2(n)))
    if 2 ** r != n:
        raise ValueError(f"size {n} is not a power of two")
    return r


def _bits_lsb(value: int, width: int) -> list[int]:
    return [(value >> i) & 1 for i in range(width)]


@dataclass
class NEQRCircuit:
    circuit: QuantumCircuit
    size: int
    color_bits: int
    direction: str
    n_pos_x: int
    n_pos_y: int


def _shifted_image(img_u: np.ndarray, direction: str) -> np.ndarray:
    """Neighbour image with edge-replicate (clamp) boundaries."""
    if direction == "h":
        shifted = np.empty_like(img_u)
        shifted[:, :-1] = img_u[:, 1:]
        shifted[:, -1] = img_u[:, -1]
    elif direction == "v":
        shifted = np.empty_like(img_u)
        shifted[:-1, :] = img_u[1:, :]
        shifted[-1, :] = img_u[-1, :]
    else:
        raise ValueError("direction must be 'h' or 'v'")
    return shifted


def _load_image(
    qc: QuantumCircuit,
    img_u: np.ndarray,
    color: QuantumRegister,
    px: QuantumRegister,
    py: QuantumRegister,
    color_bits: int,
) -> None:
    """Load a classical image into ``color`` controlled on position (px, py)."""
    H, W = img_u.shape
    nx, ny = len(px), len(py)
    controls = list(px) + list(py)
    for y in range(H):
        ybits = _bits_lsb(y, ny)
        for x in range(W):
            value = int(img_u[y, x])
            if value == 0:
                continue
            xbits = _bits_lsb(x, nx)
            pattern = xbits + ybits  # aligns with `controls`
            zero_ctrls = [controls[i] for i, b in enumerate(pattern) if b == 0]
            if zero_ctrls:
                qc.x(zero_ctrls)
            for bit in range(color_bits):
                if (value >> bit) & 1:
                    qc.mcx(controls, color[bit])
            if zero_ctrls:
                qc.x(zero_ctrls)


def build_neqr_edge_circuit(
    image_u: np.ndarray, color_bits: int = 8, direction: str = "h"
) -> NEQRCircuit:
    """Build the full NEQR encode + neighbour-difference circuit for one axis."""
    H, W = image_u.shape
    if H != W:
        raise ValueError("image must be square")
    n = _log2_int(W)

    px = QuantumRegister(n, "px")
    py = QuantumRegister(n, "py")
    c1 = QuantumRegister(color_bits, "c1")
    c2 = QuantumRegister(color_bits, "c2")
    # classical bits: [c1 (q)] [px (n)] [py (n)]
    creg = ClassicalRegister(color_bits + 2 * n, "m")
    qc = QuantumCircuit(px, py, c1, c2, creg)

    # Uniform superposition over all positions (shared by both colour regs).
    qc.h(px)
    qc.h(py)

    neighbour = _shifted_image(image_u, direction)
    _load_image(qc, image_u, c1, px, py, color_bits)
    _load_image(qc, neighbour, c2, px, py, color_bits)

    # c1 <- (c1 - c2) mod 2**q
    sub = draper_subtractor(color_bits)
    qc.append(sub.to_gate(), list(c1) + list(c2))

    # Measure c1 (diff) + position.
    for i in range(color_bits):
        qc.measure(c1[i], creg[i])
    for i in range(n):
        qc.measure(px[i], creg[color_bits + i])
    for i in range(n):
        qc.measure(py[i], creg[color_bits + n + i])

    return NEQRCircuit(qc, W, color_bits, direction, n, n)


def _parse_key(key: str, color_bits: int, n: int) -> tuple[int, int, int]:
    """Parse a counts key into (diff_value, x, y)."""
    s = key.replace(" ", "")
    # classical index i corresponds to character s[-(i+1)]
    def val(start: int, width: int) -> int:
        v = 0
        for i in range(width):
            idx = start + i
            if s[-(idx + 1)] == "1":
                v |= 1 << i
        return v

    diff = val(0, color_bits)
    x = val(color_bits, n)
    y = val(color_bits + n, n)
    return diff, x, y


def decode_neqr_edges(
    counts: dict[str, int], size: int, color_bits: int
) -> np.ndarray:
    """Turn measurement counts into a (size, size) edge-magnitude map in [0,1].

    For each visited position we take the most frequently observed diff value
    (robust to readout noise) and convert the two's-complement result to a true
    magnitude |I(x) - I(x+1)|.
    """
    n = _log2_int(size)
    levels = 1 << color_bits
    # position -> {diff_value: count}
    acc: dict[tuple[int, int], dict[int, int]] = {}
    for key, c in counts.items():
        diff, x, y = _parse_key(key, color_bits, n)
        if 0 <= x < size and 0 <= y < size:
            acc.setdefault((y, x), {})
            acc[(y, x)][diff] = acc[(y, x)].get(diff, 0) + c

    edge = np.zeros((size, size), dtype=float)
    for (y, x), hist in acc.items():
        best = max(hist.items(), key=lambda kv: kv[1])[0]
        signed = best if best < levels // 2 else best - levels
        edge[y, x] = abs(signed)
    mx = edge.max()
    return edge / mx if mx > 1e-12 else edge


def neqr_edge_magnitude(h_map: np.ndarray, v_map: np.ndarray) -> np.ndarray:
    mag = np.sqrt(h_map.astype(float) ** 2 + v_map.astype(float) ** 2)
    mx = mag.max()
    return mag / mx if mx > 1e-12 else mag
