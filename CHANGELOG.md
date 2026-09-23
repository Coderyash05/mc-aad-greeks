# Changelog

How results moved between versions of the code. The current results are in the
[README](README.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); this file keeps the
before/after comparisons that used to be inline there. Entries are newest first.
Accuracy experiments are deterministic (fixed seeds), so "unchanged" means identical up
to floating-point rounding. Timings are hardware-dependent and are compared only
within one machine.

## Documentation pass (after aee5d2a)

- **E3 rerun with LR (parity) and pathwise-LR (parity), same batches.** Every existing
  row's RMSE and CI is identical: bootstrap resamples depend only on the seed and the
  number of batches. Pathwise-LR (parity) becomes the ATM best, RMSE 0.00017
  [0.00016, 0.00018], so every other method's ratio to the best moves:

  | Estimator | RMSE / best, before | RMSE / best, after |
  |---|---|---|
  | Pathwise-LR (mixed) | 1 (best) | 1.58 [1.45, 1.71] |
  | FD, CRN (h = 6.31) | 1.11 [1.03, 1.21] | 1.76 [1.61, 1.92] |
  | Autodiff, smoothed (eps = 1.58) | 1.25 [1.15, 1.35] | 1.97 [1.81, 2.15] |
  | Likelihood ratio | 3.57 [3.36, 3.78] | 5.63 [5.15, 6.16] |

  The pairwise ratios among the original methods (e.g. FD / pathwise-LR 1.11) are
  unchanged.
- The "E3 not rerun with the parity estimator" limitation was removed.
- Before/after notes were moved from the README and docs/EXPERIMENTS.md to this file.

## Phase 7: README rewrite (aee5d2a)

- The experiment sections were moved verbatim from the README to docs/EXPERIMENTS.md.
- **Figures recoloured so a method keeps one colour across figures.** No numbers
  changed; the E2 CSVs are byte-identical after the rerun.
  - E2: CRN finite differences orange → amber; independent seeds blue → red.
  - E6: vectorised bumping orange → amber, dashed with squares, sharing the loop's
    colour because it is the same estimator.

## Phase 5: test hygiene (9eed4ca, aee5d2a)

- Four absolute floors and twelve ad-hoc tolerances were removed or replaced; see
  [docs/TESTING.md](docs/TESTING.md#what-changed-in-phase-5).
- **Suite false-alarm budget:** about 4% (bound 13.4%) at 4 SE per comparison, then
  ≤ 2.0% with family-wise thresholds for the geometric-basket families.
- **E9 z-scores** are computed from per-batch arrays in NumPy instead of JAX: maximum
  relative change 4e-11 in `e9_zscores.csv`, 1e-12 in `e9_calibration.csv`.

## Phases 3b + 4: parity estimators, geometric basket (96bb16f)

- **E8: parity methods added.** Every existing RMSE and CI is unchanged (max relative
  change 0). Best-or-tied counts moved where a parity method became the new best:

  | Greek | Method | Best or tied before | After |
  |---|---|---|---|
  | Gamma | Pathwise-LR | 10/20 | 6/20 |
  | Gamma | FD, CRN | 12/20 | 8/20 |
  | Gamma | Smoothed autodiff | 9/20 (worst 1.5x) | 6/20 (worst 2.4x) |
  | Digital delta | LR | 11/20 | 11/20 |
  | Digital delta | Smoothed autodiff | 8/20 (worst 2.0x) | 7/20 (worst 2.0x) |
  | Digital delta | FD, CRN | 6/20 | 7/20 |

  Smoothed autodiff's worst case rose from 1.5x to 2.4x only because the best method
  in those cells improved; its own RMSEs are unchanged.
- **E8 bump grid.** Two FD widths had landed at the top of the grid (h = 15.8). With
  the grid extended to 63.1, both choices and RMSEs are unchanged (1.542e-4,
  1.028e-4).

## Phase 3: paired bootstrap and out-of-sample tuning

- **E2/E3/E5:** h and eps are chosen on separate calibration batches instead of on the
  evaluation batches. The choices came out the same in every case (E3 h = 6.31), so
  every RMSE is unchanged.
- **E2/E3/E5/E8:** other sweep CSVs changed only at the 1e-13 level.

## Phase 2: memory (chunked adjoint, √M checkpointing)

- **Normals generated inside each chunk** (`fold_in(key, b)`) instead of precomputed.
  - Memory before, with the normals precomputed and an input:
    - basket d = 50, chunked B = 1e4 / 1e3: 16.4 / 5.7 MB (N = 1e4), 56 / 42 MB
      (N = 1e5), 416 / 402 MB (N = 1e6);
    - Asian, chunked / per-step checkpoint: 12.3 / 12.3 MB (M = 12), 50.7 / 50.7
      (M = 52), 243 / 243 (M = 252), 960 / 960 (M = 1,000).
  - After: 20.3 / 2.1 MB for the basket at any N, and 1.5-11.4 MB for the Asian with
    √M checkpointing. Old tables: `results/e7_memory_before_precomputedZ.csv`.
  - New random numbers give new estimates. One-shot prices on the old precomputed Z vs
    the `fold_in` stream differ by z = +0.84, +0.46, -2.49 (basket) and +0.79, -1.15,
    +1.14, -0.79 (Asian), z being the difference over its combined SE
    (`results/e7_streams.csv`). One of seven beyond 2 is about a 9% event.
- **Per-step checkpointing replaced by √M block checkpointing.** Per-step saved
  nothing (960 MB at M = 1,000 with or without it).
- **Rejected hypothesis.** Reverse mode's wall-clock gap over its flops (4-6x vs 2x)
  had been attributed to memory footprint. A pricing-only timing script, run with the
  normals precomputed and a 2-20 MB footprint, measured flops 2.03-2.25x and time
  3.8-10.5x, so the gap is not memory footprint. That script was replaced by the
  end-to-end `e7_wallclock.py`; its CSV was not kept.

## Phase 1: forward mode and vectorised bumping (17d15a9)

- **E1:** the previous timing (mean of 50 runs, two methods) gave reverse 7.9x and
  bumping 7.4x one pricing. It was replaced by medians and IQRs over repeated runs for
  five methods.
- **E6:** loop bumping at d = 50 was 633x one pricing, against 637x after re-timing,
  i.e. within noise.
