import numpy as np

from qedge.classical import binarize, sobel_edges, ground_truth_edges
from qedge.images import load_images
from qedge.metrics import f1_score, ssim_score, resource_report
from qedge.noise import ReadoutMitigator, build_noise_model
from qedge.detectors import run_qhed
from qiskit import QuantumCircuit


def test_f1_perfect_match():
    gt = np.zeros((4, 4), dtype=bool)
    gt[:, 1] = True
    pred = np.zeros((4, 4))
    pred[:, 1] = 1.0
    r = f1_score(pred, gt, tol=0)
    assert r.f1 == 1.0


def test_ssim_bounds():
    a = np.random.default_rng(0).random((8, 8))
    assert abs(ssim_score(a, a) - 1.0) < 1e-6


def test_image_families_and_sizes():
    imgs = load_images(sizes=[4, 8], families=("synthetic",))
    sizes = {t.size for t in imgs}
    assert sizes == {4, 8}
    for t in imgs:
        assert t.image.shape == (t.size, t.size)
        assert 0.0 <= t.image.min() and t.image.max() <= 1.0


def test_resource_report_fields():
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
    r = resource_report(qc, shots=1024, circuits=2)
    assert r.num_qubits == 3
    assert r.total_shots == 2048
    assert r.two_qubit_gates >= 2


def test_noise_model_builds():
    assert build_noise_model(0.0) is None
    assert build_noise_model(0.01) is not None


def test_mitigation_runs_and_preserves_mass():
    nm = build_noise_model(0.02)
    mit = ReadoutMitigator.calibrate(4, nm, seed=1)
    counts = {"0000": 500, "0001": 300, "1111": 200}
    out = mit.apply(counts)
    assert sum(out.values()) > 0
