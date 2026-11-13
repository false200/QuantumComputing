"""Draper QFT-based quantum adder / subtractor.

Used by the NEQR edge detector to compute the (mod 2**q) difference of two
colour registers entangled with the same position register. The Draper adder is
ancilla-free: it adds register ``b`` into register ``a`` purely with QFT phase
kicks, so it keeps the already large NEQR qubit budget as small as possible.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


def draper_adder(n: int) -> QuantumCircuit:
    """Return a circuit on 2n qubits computing ``a += b (mod 2**n)``.

    Qubit order: first ``n`` qubits are register ``a`` (target, LSB first),
    next ``n`` qubits are register ``b`` (addend, LSB first).
    """
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    qc = QuantumCircuit(a, b, name="draper_add")

    qft = QFT(n, do_swaps=False)
    qc.append(qft.to_gate(), a[:])
    for j in range(n):           # target qubit (significance j)
        for k in range(j + 1):   # addend qubit (significance k)
            angle = 2 * np.pi / (2 ** (j - k + 1))
            qc.cp(angle, b[k], a[j])
    qc.append(qft.inverse().to_gate(), a[:])
    return qc


def draper_subtractor(n: int) -> QuantumCircuit:
    """Circuit on 2n qubits computing ``a -= b (mod 2**n)``."""
    sub = draper_adder(n).inverse()
    sub.name = "draper_sub"
    return sub
