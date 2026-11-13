# QuantumComputing — Quantum Image Edge Detection (NEQR & QHED)

**Quantum Image Edge Detection Using NEQR and QHED: accuracy, shot requirements
and noise analysis.**

NEQR and QHED quantum edge detectors claim exponential speed-ups over classical
`O(N)` methods, but these claims ignore **state preparation** and the `O(N)`
**measurement shots** needed to read out an edge map. They have typically been
validated only in noise-free simulation or on tiny 2×2–4×4 hardware, and the two
have rarely been compared head-to-head. **This project closes that gap.**

We implement **both** detectors in Qiskit on **identical** 4×4–32×32 greyscale
and **BSDS500** images, sweep shot counts, and simulate realistic device noise
with measurement-error mitigation. Using **F1**, **SSIM** and **end-to-end
resource counts** (qubits, two-qubit gates, depth, and total shots) we establish
how many shots each method needs, and how much noise it tolerates, for an
accurate edge map.

---

## TL;DR findings

Generated automatically into [`results/summary.json`](results/summary.json) and
the figures below (numbers from our run; reproducible with `qedge all`).

* **Qubits (theoretical scaling).** QHED (QPIE/amplitude encoding) uses
  `2·log₂(N_side)` qubits — **4, 6, 8, 10** for 4→32. NEQR (basis encoding)
  uses `2·log₂(N_side) + 2q` — **20, 22, 24, 26** at `q=8` (two colour
  registers for the neighbour difference). QHED wins qubits by a wide margin.
* **State preparation (the hidden cost).** NEQR pays a brutal two-qubit-gate
  price — **4.1k (4×4) → 19.6k (8×8) → 98.5k (16×16) CX** — because every pixel
  needs multi-controlled loads. QHED stays at **46 → 242 → 1030 → 4146 CX**
  for 4→32. This state-prep blow-up is exactly what the speed-up claims ignore.
* **Shots — the O(N) readout bottleneck.** For QHED, the shots to reach
  **F1 ≥ 0.99** grow with image size: **256 (8×8) → 512 (16×16) → 1024 (32×32)**
  — doubling with each side-length doubling. NEQR's basis readout is shot-cheap
  per pixel on simple shapes, but it still must visit all `O(N)` positions and
  pays the gate cost above. The asymptotic advantage is eroded either way.
* **Noise.** QHED is markedly more noise-tolerant: at a 2% two-qubit
  depolarising level it retains **F1 = 0.978**, while deep NEQR drops to
  **≈ 0.91**. Tensored readout mitigation barely changes NEQR
  (**0.905 → 0.905**) because its dominant errors are *gate* errors, not
  readout — a direct consequence of its circuit depth.

Figures: [`results/fig_shot_size.png`](results/fig_shot_size.png) (O(N) shots),
[`results/fig_shot_sweep.png`](results/fig_shot_sweep.png),
[`results/fig_noise_sweep.png`](results/fig_noise_sweep.png),
[`results/fig_resources.png`](results/fig_resources.png),
[`results/fig_montage.png`](results/fig_montage.png).

---

## Methods

### NEQR — Novel Enhanced Quantum Representation (basis encoding)

A `2ⁿ × 2ⁿ` image is stored as

```
|I⟩ = (1/2ⁿ) · Σ_{y,x} |c(y,x)⟩ |y⟩ |x⟩
```

with a `q`-bit colour register holding each pixel's **exact** intensity in the
computational basis. State preparation applies, per pixel, a group of
multi-controlled-X gates — the dominant cost.

**Edge detection (quantum-arithmetic differencing).** To get a horizontal
gradient we entangle *two* colour registers with the same position register:
`c1 = I(y,x)` and `c2 = I(y,x+1)` (the neighbour image is a classically shifted
copy — a legitimate NEQR input). A **Draper QFT subtractor** computes
`c1 ← (I(y,x) − I(y,x+1)) mod 2^q`, so measuring position + `c1` yields the exact
neighbour difference per pixel. Vertical edges use a vertically shifted copy.
Boundaries use edge-replicate (clamp) padding. Implemented in
[`src/qedge/neqr.py`](src/qedge/neqr.py) and
[`src/qedge/arithmetic.py`](src/qedge/arithmetic.py).

### QHED — Quantum Hadamard Edge Detection (amplitude encoding)

The whole image is packed into the amplitudes of a single `2·log₂(N)`-qubit
register (QPIE). A Hadamard on the least-significant qubit of an axis maps
neighbouring amplitudes to their sum/difference:

```
(a₂ᵦ , a₂ᵦ₊₁)  →  ( (a₂ᵦ + a₂ᵦ₊₁)/√2 , (a₂ᵦ − a₂ᵦ₊₁)/√2 )
```

so the odd measurement outcomes carry the pairwise edge `|a₂ᵦ − a₂ᵦ₊₁|`. A
cyclic **decrement** before the Hadamard shifts the pairing to recover the
in-between boundaries; the even+odd passes are combined per axis. The
non-physical wrap boundary (last↔first pixel) is discarded. Implemented in
[`src/qedge/qhed.py`](src/qedge/qhed.py).

### Classical ground truth & metrics

* Ground truth: **Sobel** gradient magnitude and **Canny** binary edges
  ([`src/qedge/classical.py`](src/qedge/classical.py)).
* **F1**: BSDS-style boundary F-measure with a 1-pixel matching tolerance.
* **SSIM**: structural similarity between the quantum edge-magnitude map and the
  Sobel magnitude.
* **Resources**: qubits, logical/transpiled depth & size, two-qubit-gate count
  (transpiled to `rz, sx, x, cx`), circuit count and total shots
  ([`src/qedge/metrics.py`](src/qedge/metrics.py)).

### Noise & mitigation

`build_noise_model(level)` ([`src/qedge/noise.py`](src/qedge/noise.py)) builds a
single-knob model: 1q/2q depolarising errors, thermal relaxation, and symmetric
readout error. `ReadoutMitigator` performs empirically calibrated **tensored
(per-qubit) readout mitigation** by contracting each qubit's inverse 2×2
response against the probability vector (no dense `2ᵏ×2ᵏ` matrix).

---

## Install & run (uv)

```bash
# 1. Create the environment (Python 3.12) and install everything
uv venv --python 3.12
uv sync --extra dev

# 2. Sanity check
uv run qedge smoke

# 3. Individual experiments (CSVs land in results/)
uv run qedge shots --sizes 4 8
uv run qedge noise --sizes 8
uv run qedge size  --sizes 4 8 16 32

# 4. Full benchmark + figures + results/summary.json
uv run qedge all --sizes 4 8

# 5. Tests
uv run pytest
```

Configuration (seed, shot/noise sweeps, image sizes, BSDS500 toggle) lives in
[`.env`](.env) and is loaded by [`src/qedge/config.py`](src/qedge/config.py).

BSDS500 is downloaded on demand to `data/` the first time it is needed; if the
download fails the pipeline falls back to synthetic + scikit-image test images.

---

## Repository layout

```
src/qedge/
  config.py       # .env-driven configuration
  images.py       # synthetic + standard + BSDS500 image sources
  classical.py    # Sobel / Canny ground truth
  arithmetic.py   # Draper QFT adder / subtractor
  neqr.py         # NEQR encoding + neighbour-difference edge detector
  qhed.py         # QPIE encoding + Hadamard edge detector
  noise.py        # Aer noise model + tensored readout mitigation
  simulate.py     # Aer run wrapper
  metrics.py      # F1, SSIM, resource reports
  detectors.py    # end-to-end NEQR / QHED pipelines
  experiments.py  # shot / noise / size sweeps -> CSV
  figures.py      # plots + qualitative montage
  cli.py          # `qedge` command-line entry point
tests/            # pytest suite (arithmetic, detectors, metrics, images, noise)
scripts/          # smoke.py and helpers
results/          # CSVs, figures, summary.json (generated)
data/             # BSDS500 (downloaded) + cache
```

---

## Limitations & honest caveats

* NEQR state preparation is simulated with full multi-controlled loads; its
  gate count grows so fast that we simulate NEQR up to **16×16** and report
  **32×32** as a resource *projection* (QHED is simulated at all sizes).
* The "exponential speed-up" is in **qubit count**, not wall-clock: reading a
  full N-pixel edge map needs `O(N)` shots for **both** methods, which is the
  central finding.
* The noise model is representative, not a calibrated replica of a specific
  device; mitigation is tensored (assumes independent readout errors).

## References

* Zhang et al., *NEQR: a novel enhanced quantum representation of digital
  images* (2013).
* Yao et al., *Quantum image processing and its application to edge detection*
  (QHED), Phys. Rev. X (2017).
* Arbeláez et al., *Contour Detection and Hierarchical Image Segmentation*
  (BSDS500), IEEE TPAMI (2011).

## License

MIT.
