import numpy as np
import pytest

from qedge.classical import ground_truth_edges, sobel_edges
from qedge.detectors import run_neqr, run_qhed
from qedge.images import load_images
from qedge.metrics import f1_score, ssim_score


def _img(name, size=4):
    for ti in load_images(sizes=[size], families=("synthetic",)):
        if ti.name == name:
            return ti.image
    raise KeyError(name)


def test_qhed_step_perfect():
    img = _img("step")
    gt = ground_truth_edges(img)
    res = run_qhed(img, shots=50000, seed=1)
    assert res.resources.num_qubits == 4
    assert f1_score(res.edge_map, gt).f1 >= 0.99


def test_neqr_step_perfect():
    img = _img("step")
    gt = ground_truth_edges(img)
    res = run_neqr(img, shots=50000, color_bits=4, seed=1)
    assert f1_score(res.edge_map, gt).f1 >= 0.99


def test_methods_agree_on_square():
    img = _img("square")
    q = run_qhed(img, shots=80000, seed=1).edge_map
    n = run_neqr(img, shots=80000, color_bits=4, seed=1).edge_map
    # Binarised edge maps should match closely.
    from qedge.classical import binarize

    assert np.mean(binarize(q) == binarize(n)) >= 0.9


def test_qhed_fewer_qubits_than_neqr():
    img = _img("disk", size=8)
    q = run_qhed(img, shots=4096, seed=1)
    n = run_neqr(img, shots=4096, color_bits=6, seed=1)
    assert q.resources.num_qubits < n.resources.num_qubits
    assert q.resources.two_qubit_gates < n.resources.two_qubit_gates


def test_constant_image_no_edges():
    img = np.full((4, 4), 0.5)
    res = run_qhed(img, shots=8192, seed=1)
    assert res.edge_map.max() < 1e-6
