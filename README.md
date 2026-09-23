# Monte Carlo Greeks by Algorithmic Differentiation

Monte Carlo option pricing in JAX, with Greeks computed by automatic differentiation
and compared against finite differences, likelihood-ratio estimators and Black-Scholes.

## Status

- [x] European call/put pricer (GBM, vectorised, jit-compiled), validated against Black-Scholes (E1)
- [ ] Autodiff delta, vega, rho
- [ ] Finite-difference baseline (CRN and independent seeds)
- [ ] Gamma: naive AD vs LR vs pathwise-LR vs smoothing
- [ ] Digital options
- [ ] Basket option and cost scaling

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python experiments/e1_validation.py
```

## Timing protocol

Every timing is measured after compilation, over repeated runs (at least 3 for the
slow cases, up to 200 for the fast ones), with Python's GC off; the median and
interquartile range (IQR) are reported (`src/mcgreeks/bench.py`). Operation counts
come from XLA's `compile().cost_analysis()`, which works on CPU; it counts a loop
body once regardless of trip count, so counts for chunked programs are read from
the equivalent loop-free program. Hardware for the numbers below: Intel Core
i7-13700HX (`os.cpu_count()` = 24), Windows 11, Python 3.12.10, JAX 0.11.2, CPU backend
(`results/e1_hardware.json`, `results/e6_hardware.json`, `results/e6b_hardware.json`).
Timings are hardware-dependent; the flop, transcendental and byte counts are not.
Each figure can be redrawn from its CSV with `--plot-only` (E6, E6b).

## Forward vs reverse mode: which one should win

Cost model (Griewank & Walther 2008; Capriotti 2011, sec. 3; Giles & Glasserman 2006):
for a program with n inputs and m outputs, forward (tangent) mode needs one sweep per
**input**, each ~1-2x one evaluation, while reverse (adjoint) mode needs one sweep per
**output**, ~3-4x one evaluation in total. A price is one scalar output, so **reverse
mode is expected to beat forward mode for any n >= 2 inputs** (close at n = 2, clear
from n = 3). Forward mode wins when outputs outnumber inputs (E6b). The experiments
below confirm both regimes; neither result is a surprise.

## E1: pricer validation and the cost of 3 Greeks

Price + delta, vega, rho of one call, N = 1,000,000 (`results/e1_timing.csv`):

| Method | Median ms [IQR] | x one pricing (time) | x one pricing (flops) | exp/log evaluations |
|---|---|---|---|---|
| Price only | 0.50 [0.47, 0.56] | 1.0 | 1.0 | 1x |
| Forward-mode AD (3 jvps) | 5.03 [4.44, 5.57] | 10.1 | 6.6 | 1x |
| Reverse-mode AD | 4.15 [3.92, 4.84] | 8.4 | 4.3 | 1x |
| Bumping, Python loop (7 pricings) | 4.42 [4.31, 4.56] | 8.9 | 7.0 | 7x |
| Bumping, vmapped (7 pricings) | 5.06 [4.79, 5.52] | 10.2 | 7.0 | 7x |

- **Reverse beats forward with 3 inputs and one output, as the cost model predicts:**
  4.3x vs 6.6x one pricing in flops (~1.9 pricings per forward tangent). None of four
  forward formulations did better: vmapped jvps and `jacfwd` were within noise of
  each other, and 3 separate or scalar jvps were slower.
- The flop ratios are exact and machine-independent. The time ratios are not stable:
  the denominator is a 0.5 ms pricing that moves by 20-25% between runs. A second run
  on the same machine gave price 0.63 ms and reverse 6.7x, forward 8.6x, loop 7.3x,
  vmap 8.4x. Before this change (mean of 50 runs, two methods): reverse 7.9x, bumping 7.4x.
- With only 3 inputs, reverse AD and bumping are within noise in wall-clock. The
  wall-clock ratios sit above the flop ratios because a European pricing is only ~7
  flops per path. It is limited by memory traffic, not arithmetic (see the roofline below).

## E6: basket call, forward vs reverse vs bumping

For d assets there are P = d deltas + d vegas + d(d-1)/2 correlation sensitivities of
one output. `python experiments/e6_basket.py`, N = 100,000 paths, h = 1e-4. Forward
mode and vectorised bumping run in chunks. The budget (16 MB per batched array) is the
fastest of 16 / 64 / 256 MB in `experiments/e6_chunk_sweep.py` (table below).

![E6](results/e6_basket.png)

Cost in units of one pricing, median wall-clock [IQR] and XLA flops:

| d | P | Reverse AD | Forward AD | Bump, loop | Bump, vmap | Flops: rev / fwd / bump |
|---|---|---|---|---|---|---|
| 2 | 5 | 2.9 [2.8, 3.0] | 5.4 [5.2, 5.7] | 11.0 [10.8, 11.2] | 8.3 [8.0, 8.6] | 2.9 / 8.4 / 11 |
| 5 | 20 | 2.8 [2.8, 2.9] | 18 [18, 19] | 44 [42, 46] | 49 [49, 51] | 2.4 / 27 / 41 |
| 10 | 65 | 3.4 [3.1, 3.5] | 55 [54, 55] | 167 [163, 171] | 170 [167, 173] | 2.2 / 79 / 131 |
| 20 | 230 | 3.7 [3.6, 3.7] | 202 [201, 203] | 484 [479, 487] | 764 [759, 771] | 2.1 / 256 / 461 |
| 35 | 665 | 3.8 [3.8, 3.9] | 638 [636, 639] | 1,313 [1,301, 1,313] | 2,236 [2,233, 2,237] | 2.1 / 710 / 1,331 |
| 50 | 1,325 | 4.2 [4.0, 4.3] | 1,359 [1,359, 1,359] | 2,666 [2,662, 2,676] | 4,629 [4,597, 4,637] | 2.05 / 1,389 / 2,650 |

- **Reverse mode is flat; forward mode and bumping grow linearly in P**, reproducing the
  shape of Giles & Glasserman (2006), Fig. 4. Reverse AD costs 2.8-4.2x one pricing in
  time and 2.05-2.9x in flops for any P. Forward AD costs ~1.05-1.7 pricings of
  arithmetic per input, about half of bumping's 2 per input.
- At d = 50: reverse AD 27 ms; forward AD 8.8 s (325x slower); loop bumping 17.3 s
  (637x; 633x before Phase 1); vmapped bumping 30.0 s (1,106x).
- All four give the same gradient: forward vs reverse to <= 4.5e-14, vmapped vs loop
  bumping to <= 8.9e-12 (tested in `tests/test_basket.py`). Bumping vs AD differs by
  <= 2.2e-4, the O(h^2) and kink-path error of central differences.
- **AD evaluates every exp once; bumping evaluates it 2P + 1 times.** XLA's
  transcendental count is exactly 1.00x one pricing for both reverse and forward AD
  at every d. Bumping's count is 2P + 1 (2,651x at d = 50). The derivative of exp
  reuses the value already computed on the forward sweep (cf. Giles & Glasserman 2006,
  sec. 7).
- **Reverse-mode flops match the literature; its wall-clock is set by memory traffic.**
  The flop ratio falls to 2.05x at d = 50. That is consistent with Capriotti's (2011,
  Fig. 9b) ~2.3x for basket deltas and vegas, though his figure is a timing of
  hand-coded or tool-generated adjoints and ours is a flop count that also includes the
  correlation sensitivities through the Cholesky factor. The wall-clock ratio is higher
  (4.2x at d = 50) because reverse mode stores the forward sweep's intermediates and
  reads them back. XLA's bytes-accessed count is 3.80x one pricing at every d, and the
  wall-clock ratio tracks it for d >= 10 (3.4-4.2x). At d <= 5 the N x d arrays
  (<= 4 MB) fit in cache and the ratio is lower (2.8-2.9x). Reducing that stored
  state is Phase 2.
- What the benchmark does not claim: the bumps reprice from scratch (generic black-box
  bumping). A hand-written bump could reuse `Z @ L.T` for the delta and vega bumps;
  that is a smarter algorithm, not a fairer benchmark of the same one.

### Vectorised bumping, chunk sizes and the roofline

`python experiments/e6_chunk_sweep.py` (`results/e6_chunk_sweep.csv`,
`results/e6_roofline.csv`). Cost in units of one pricing, median [IQR]:

| d | Loop bumping | vmap, 16 MB | vmap, 64 MB | vmap, 256 MB | Forward, 16 MB | Forward, 256 MB |
|---|---|---|---|---|---|---|
| 20 | 445 | 696 [694, 698] (chunk 1) | 696 [693, 699] (chunk 4) | 954 [953, 954] (chunk 16) | 202 (chunk 1) | 329 (chunk 16) |
| 50 | 2,657 | 4,608 [4,601, 4,611] (chunk 1) | same program as 16 MB | 4,888 [4,880, 4,892] (chunk 6) | 1,378 (chunk 1) | 1,733 (chunk 6) |

Smaller chunks are faster for both chunked methods. Even at its best, **vectorised
bumping is slower than the Python loop for d >= 20** (0.58-0.64x the loop's speed).
It is faster only at d = 2 (1.32x) and roughly equal at d = 5-10 (0.90-0.98x).

Why: on this CPU the pricing is memory-bandwidth-bound. Throughput from XLA's flop
and bytes-accessed counts, divided by median time:

| d | Program | GFLOP/s (% of GEMM peak) | GB/s (% of triad) | flops per byte |
|---|---|---|---|---|
| 20 | one pricing | 31 (7%) | 28 (113%) | 1.1 |
| 20 | loop bumping | 32 (7%) | 29 (117%) | 1.1 |
| 20 | vmap bumping | 21 (5%) | 15 (60%) | 1.4 |
| 20 | reverse AD | 19 (4%) | 31 (124%) | 0.6 |
| 50 | one pricing | 82 (18%) | 31 (125%) | 2.6 |
| 50 | loop bumping | 81 (18%) | 31 (125%) | 2.6 |
| 50 | vmap bumping | 47 (11%) | 14 (58%) | 3.3 |
| 50 | reverse AD | 40 (9%) | 28 (114%) | 1.4 |

The machine reference is a float64 streaming triad `y = 2x + z` at 25 GB/s (the
bandwidth JAX achieves here) and a 4096 x 4096 GEMM at 446 GFLOP/s. The pricing and
the loop run at the streaming bandwidth (above 100% because XLA's byte count is a
model, and some traffic hits cache) and at 7-18% of peak arithmetic.
**Batching the bumps cannot help a bandwidth-bound kernel.** Each bump still has to
stream its own N x d arrays (bytes drop only to 0.8x the loop's). On XLA:CPU the
batched program also reaches only ~60% of the streaming bandwidth: a batched
`Z @ L.T` ran ~1.9x slower than the same number of single GEMMs (99 ms vs 53 ms for
12 at d = 50). This is specific to this backend; on a GPU the balance may differ. In
all cases vectorised bumping does 2P + 1 pricings of work, so its cost grows linearly
in P.

## E6b: the forward-mode regime (many outputs, one input)

`python experiments/e6b_forward_regime.py`: K European calls with strikes over
70..130, one spot, N = 100,000. The K deltas form one Jacobian column: forward mode
gets it from one jvp, and reverse mode needs K pullbacks (what `jax.jacrev` does).
Cost is in units of one pricing of all K strikes; forward and reverse agree to
<= 3.3e-16.

![E6b](results/e6b_forward_regime.png)

| K | Forward AD, time [IQR] | Reverse AD, time [IQR] | Flops: fwd / rev | Temp memory fwd / rev |
|---|---|---|---|---|
| 1 | 5.1 [5.0, 5.4] | 5.6 [5.2, 5.8] | 2.3 / 2.5 | 3 / 3 MB |
| 10 | 10.5 [9.6, 11.0] | 51 [50, 53] | 3.2 / 8.9 | 25 / 25 MB |
| 100 | 31 [30, 32] | 1,269 [1,257, 1,269] | 3.3 / 69 | 241 / 241 MB |
| 1,000 | 43 [43, 44] | 17,261 [16,566, 17,818] | 3.3 / 669 | 2.4 / 2.4 GB |

- **The mirror image of E6:** forward-mode flops are flat (2.3-3.3x), and reverse-mode
  flops grow linearly in K (~0.67 K). At K = 1,000, forward mode takes 0.23 s and
  reverse mode 92 s. With 1 output and 1 input (K = 1) the two are equal within noise.
- **Negative result: forward mode is flat in flops but not in wall-clock** (5x to 43x).
  The K-strike price fuses into one pass and never stores the N x K payoff matrix
  (temp memory ~0). The jvp materialises N x K intermediates (2.4 GB at K = 1,000;
  XLA bytes accessed 4.8 GB vs 0.8 MB for the price). Its time is again set by memory
  traffic, not arithmetic.
- Reverse mode's time grows faster than its flops (1,269x time vs 69x flops at
  K = 100). Each chunked pullback re-reads the stored N x K residuals.
