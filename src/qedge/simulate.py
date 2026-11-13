"""Thin wrapper around Aer for running circuits with/without noise."""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .config import CONFIG


# Above this qubit count an exact density-matrix simulation of a *noisy*
# circuit becomes too memory-hungry, so we fall back to trajectory sampling.
_DENSITY_MATRIX_MAX_QUBITS = 13


def get_simulator(
    noise_model: NoiseModel | None = None, method: str = "automatic"
) -> AerSimulator:
    if noise_model is None:
        return AerSimulator(method=method)
    return AerSimulator(noise_model=noise_model, method=method)


def run_counts(
    circuit: QuantumCircuit,
    shots: int,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    method: str | None = None,
) -> dict[str, int]:
    """Transpile + run a measured circuit, returning a counts dict.

    For noisy circuits small enough to fit, we use the exact ``density_matrix``
    method so a single simulation yields the full noisy distribution (sampling
    thousands of shots is then free). Larger noisy circuits fall back to the
    default trajectory sampler.
    """
    if method is None:
        if noise_model is not None and circuit.num_qubits <= _DENSITY_MATRIX_MAX_QUBITS:
            method = "density_matrix"
        else:
            method = "automatic"
    sim = get_simulator(noise_model, method=method)
    seed = CONFIG.seed if seed is None else seed
    tqc = transpile(circuit, sim, optimization_level=1, seed_transpiler=seed)
    result = sim.run(tqc, shots=shots, seed_simulator=seed).result()
    return result.get_counts()
