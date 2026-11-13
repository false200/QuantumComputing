"""Parameterised device noise model + measurement-error mitigation.

``build_noise_model(level)`` creates a single-knob, realistic noise model:
    * 1-qubit depolarising error      p1 = level / 10
    * 2-qubit depolarising error      p2 = level            (gates: cx, cz)
    * symmetric readout (bit-flip)    pr = min(level * 2, 0.5)
plus optional thermal relaxation on 1q gates. Sweeping ``level`` gives the
noise-tolerance curves the project requires.

Measurement-error mitigation uses an empirically calibrated *tensored*
(per-qubit) assignment model. Correction is applied by contracting each qubit's
inverse 2x2 response against the dense probability vector, which scales to the
QHED (<=10 qubit) and small-NEQR (<=~18 qubit) registers without ever forming
the full 2^k x 2^k matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

from .simulate import run_counts


def build_noise_model(
    level: float, thermal: bool = True
) -> NoiseModel | None:
    """Return a noise model for a given scalar noise ``level`` (0 => None)."""
    if level <= 0:
        return None

    nm = NoiseModel()
    p1 = level / 10.0
    p2 = level
    pr = min(level * 2.0, 0.5)

    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)

    if thermal:
        # Representative superconducting timescales.
        t1, t2, gate_time = 60e3, 40e3, 100.0  # ns
        relax = thermal_relaxation_error(t1, t2, gate_time)
        err1 = err1.compose(relax)

    nm.add_all_qubit_quantum_error(err1, ["u1", "u2", "u3", "x", "h", "rz", "sx"])
    nm.add_all_qubit_quantum_error(err2, ["cx", "cz"])

    ro = ReadoutError([[1 - pr, pr], [pr, 1 - pr]])
    nm.add_all_qubit_readout_error(ro)
    return nm


@dataclass
class ReadoutMitigator:
    """Tensored (per-qubit) readout-error mitigator."""

    flip_01: np.ndarray  # P(measure 1 | prepared 0) per classical bit
    flip_10: np.ndarray  # P(measure 0 | prepared 1) per classical bit
    n_bits: int

    @classmethod
    def calibrate(
        cls,
        n_bits: int,
        noise_model: NoiseModel | None,
        shots: int = 8192,
        seed: int | None = None,
    ) -> "ReadoutMitigator":
        qr = QuantumRegister(n_bits, "q")
        cr = ClassicalRegister(n_bits, "c")

        c0 = QuantumCircuit(qr, cr)
        c0.measure(qr, cr)

        c1 = QuantumCircuit(qr, cr)
        c1.x(qr)
        c1.measure(qr, cr)

        counts0 = run_counts(c0, shots, noise_model, seed)
        counts1 = run_counts(c1, shots, noise_model, seed)

        flip_01 = np.zeros(n_bits)
        flip_10 = np.zeros(n_bits)
        tot0 = sum(counts0.values())
        tot1 = sum(counts1.values())
        for key, c in counts0.items():
            s = key.replace(" ", "")
            for b in range(n_bits):
                if s[-(b + 1)] == "1":
                    flip_01[b] += c
        for key, c in counts1.items():
            s = key.replace(" ", "")
            for b in range(n_bits):
                if s[-(b + 1)] == "0":
                    flip_10[b] += c
        flip_01 /= max(tot0, 1)
        flip_10 /= max(tot1, 1)
        return cls(flip_01, flip_10, n_bits)

    def _counts_to_vector(self, counts: dict[str, int]) -> np.ndarray:
        vec = np.zeros(1 << self.n_bits)
        for key, c in counts.items():
            idx = int(key.replace(" ", ""), 2)
            vec[idx] += c
        total = vec.sum()
        return vec / total if total > 0 else vec

    def apply(self, counts: dict[str, int]) -> dict[str, int]:
        """Mitigate a counts dict; returns pseudo-counts (non-negative)."""
        if self.n_bits > 20:  # keep memory bounded
            return counts
        total = sum(counts.values())
        vec = self._counts_to_vector(counts)
        tensor = vec.reshape([2] * self.n_bits)

        # classical bit b corresponds to tensor axis (n_bits-1-b) because the
        # bitstring is MSB-first while index bit b has weight 2**b.
        for b in range(self.n_bits):
            p01 = min(max(self.flip_01[b], 0.0), 0.49)
            p10 = min(max(self.flip_10[b], 0.0), 0.49)
            A = np.array([[1 - p01, p10], [p01, 1 - p10]])
            Ainv = np.linalg.inv(A)
            axis = self.n_bits - 1 - b
            tensor = np.tensordot(Ainv, tensor, axes=([1], [axis]))
            tensor = np.moveaxis(tensor, 0, axis)

        vec = tensor.reshape(-1)
        vec = np.clip(vec, 0.0, None)  # project back to valid distribution
        s = vec.sum()
        if s > 0:
            vec = vec / s
        out: dict[str, int] = {}
        for idx in np.nonzero(vec)[0]:
            cnt = int(round(vec[idx] * total))
            if cnt > 0:
                out[format(idx, f"0{self.n_bits}b")] = cnt
        return out
