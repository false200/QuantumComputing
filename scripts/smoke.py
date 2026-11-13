"""Quick correctness smoke-test (noise-free, high shots)."""
import numpy as np
from qedge.images import load_images
from qedge.classical import sobel_edges, ground_truth_edges
from qedge.detectors import run_qhed, run_neqr
from qedge.metrics import f1_score, ssim_score

np.set_printoptions(precision=2, suppress=True)

imgs = load_images(sizes=[4], families=("synthetic",))
for ti in imgs:
    if ti.name not in ("step", "square"):
        continue
    print("=" * 60)
    print(f"{ti.name} {ti.size}x{ti.size}")
    print("image:\n", ti.image)
    sob = sobel_edges(ti.image)
    gt = ground_truth_edges(ti.image)
    print("sobel:\n", sob)

    q = run_qhed(ti.image, shots=200000, seed=1)
    print("QHED edge:\n", q.edge_map)
    print("QHED F1:", round(f1_score(q.edge_map, gt).f1, 3),
          "SSIM:", round(ssim_score(q.edge_map, sob), 3),
          "qubits:", q.resources.num_qubits, "2q:", q.resources.two_qubit_gates,
          "total_shots:", q.resources.total_shots)

    nq = run_neqr(ti.image, shots=200000, color_bits=4, seed=1)
    print("NEQR edge:\n", nq.edge_map)
    print("NEQR F1:", round(f1_score(nq.edge_map, gt).f1, 3),
          "SSIM:", round(ssim_score(nq.edge_map, sob), 3),
          "qubits:", nq.resources.num_qubits, "2q:", nq.resources.two_qubit_gates,
          "total_shots:", nq.resources.total_shots)
