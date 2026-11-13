import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from qedge.arithmetic import draper_adder, draper_subtractor
from qiskit_aer import AerSimulator


def _eval(op, n, a_val, b_val):
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    c = ClassicalRegister(n, "c")
    qc = QuantumCircuit(a, b, c)
    for i in range(n):
        if (a_val >> i) & 1:
            qc.x(a[i])
        if (b_val >> i) & 1:
            qc.x(b[i])
    qc.append(op(n).to_gate(), list(a) + list(b))
    qc.measure(a, c)
    sim = AerSimulator()
    tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=1)
    counts = sim.run(tqc, shots=256, seed_simulator=1).result().get_counts()
    key = max(counts, key=counts.get)
    return int(key, 2)


@pytest.mark.parametrize("a_val,b_val", [(3, 5), (7, 1), (0, 0), (6, 6), (2, 7)])
def test_adder(a_val, b_val):
    n = 3
    assert _eval(draper_adder, n, a_val, b_val) == (a_val + b_val) % (2 ** n)


@pytest.mark.parametrize("a_val,b_val", [(5, 3), (1, 7), (0, 1), (6, 6), (2, 7)])
def test_subtractor(a_val, b_val):
    n = 3
    assert _eval(draper_subtractor, n, a_val, b_val) == (a_val - b_val) % (2 ** n)
