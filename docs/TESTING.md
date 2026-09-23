# Testing statistical code

Most tests in this repository compare a Monte Carlo estimate with an exact value. Such a
test can fail with correct code, purely by chance. This page gives the rules the
tests follow, the chance that the whole suite fails with correct code, and the
tolerances that were removed or replaced in Phase 5.

Run `pytest -q`. The end of the output prints the false-alarm budget of the run and
every rare-event branch taken (from `tests/helpers.py` and `tests/conftest.py`).

## Rules

1. **Compare within 4 standard errors.** Use `assert_within_se(estimate, se, exact, k=4, reason, dof=None)`.
   - For a correct estimator with a correct SE, z = (estimate - exact) / SE is ≈ N(0, 1),
     so one comparison fails by chance with probability 2(1 - Φ(4)) = 6.3e-5.
   - The SE comes from per-path samples (z is ≈ normal), or from batch means over B
     batches (z ~ t with B - 1 dof; pass `dof`).
2. **`exact` is the expectation of the *estimator*, not always the true Greek.**
   - For a biased estimator, use its own target, computed exactly:
     - central differences with common random numbers have expectation equal to the
       same finite difference of the exact Black-Scholes prices;
     - a smoothed payoff's autodiff Greek has expectation equal to the Greek of the
       smoothed price, computed by 1-d quadrature (`smoothed_*_expectation` in
       `tests/helpers.py`, checked against a 4M-path Monte Carlo and against
       eps → 0).
   - The bias itself (target - true Greek) is deterministic and is tested
     separately, by its known order: halving h or eps must quarter an O(h²) bias.
     The observed ratios are 3.99 - 4.00, and the test allows 4 ± 0.05.
3. **No absolute floors.** `4 * se + 1e-4` passes anything whose SE is 0.
   `assert_within_se` raises an error if SE ≤ 0 and points to rule 4.
4. **Rare events: check structure, not the SE.**
   - When fewer than 100 paths (`RARE_THRESHOLD`) are on the minority side of the
     payoff, the sample SE is estimated from a handful of values and is unreliable,
     usually too small. The minority side means paths with a nonzero payoff, or, for
     an estimator that depends only on the exercise indicator (pathwise rho), the
     smaller of the exercised and unexercised counts.
   - In those cases the tests check instead:
     - the count is consistent with Binomial(N, p), with p the exact exercise
       probability N(±d2) (`assert_count_consistent`: exact two-sided test at the
       6.3e-5 level). This checks the tail of the path generator, which is what the
       estimate depends on here;
     - structural facts:
       - no payers means estimate = 0 and SE = 0 exactly;
       - per-path pathwise derivatives vanish off the exercise region, so all Greeks
         are then 0;
       - delta and rho carry the option's sign;
       - pathwise rho is exactly ±K T e^{-rT} n_exercised / N.
5. **Deterministic comparisons are not statistical.** Tests that compare two
   computations on the same random numbers are identities up to rounding or a known
   discretisation, so they use small numerical tolerances with the reason in the
   docstring. Examples: forward vs reverse mode, grad-of-mean vs mean-of-grads, vmap
   vs loop, chunked vs one-shot.
6. **Never change a seed or loosen a tolerance to make a failing test pass.** Seeds
   are fixed (`zlib.crc32` of a case tag), so the suite is deterministic. A failure
   means the numbers changed, for example after a code change or a new JAX random
   stream. Investigate first. To tell a real bias from a false alarm, rerun the
   comparison on many fresh seeds and look at the z distribution (E9 does this for
   the basket).

## Families of correlated comparisons

`test_mc_greeks_match_closed_form[d]` compares every basket sensitivity with the
closed form: 1 + 2d + d(d-1)/2 comparisons, which is 6, 66 and 1,326 at d = 2, 10, 50.

**Why family-wise control.**
- At 4 SE per comparison, the d = 50 family failed on 2.5% [0.8%, 5.7%] of 200 fresh
  seeds with a correct estimator (E9 (b)). That is 400 times the rate of one 4-SE
  check.
- The individual thresholds are therefore Šidák-adjusted, k = t₁₉₉ quantile at
  1 - (1 - 6.3e-5)^(1/P), so that the whole family has the false-alarm rate of one
  4-SE check (`mcgreeks.stats.k_familywise`):
  - k = 5.06 at d = 10;
  - k = 5.68 at d = 50;
  - d = 2 (6 comparisons) keeps 4.
- Šidák is exact for independent comparisons and conservative for these positively
  dependent ones (Šidák 1967). Measured over 200 seeds: 0 failures at every d.
- The cost is power: a bias in any one sensitivity must exceed 5.7 SE, not 4, to be
  caught on its own. At d = 50 the SE relative to the Greek it measures is 0.3% for
  the price, 0.2% for the deltas, 1.1% (median) for the vegas and 8.5% (median) for
  the correlation sensitivities (more for those near 0).

**The aggregate tests: what individual thresholds miss.** A bias of 3 SE in many
sensitivities at once passes every individual check. Two statistics of the whole
family catch it (`mcgreeks.stats.family_zscores`, `tests/helpers.py::assert_family`):
- **mean z over the family, within 4 of its own SE.** This detects a bias with a
  common sign.
  - Mean z is a fixed linear combination of the estimates (weights 1/(P SE_k)), so its
    SE comes from the batch values of that combination. That accounts for every
    correlation between the z's.
  - The naive SE 1/√P is 3-5x too small here, because the z's share noise (all d delta
    z's are identical: d log G / d S0_i = w_i / S0_i on every path). At d = 50 it is
    0.027, against a correct 0.127.
- **mean z² over the family, in the two-sided band of a scaled chi-square.** This
  detects SEs that are systematically too small or too large, and biases of mixed
  sign that cancel in mean z.
  - E[z²] = ν/(ν - 2) for t_ν, and Var(mean z²) = (P Var(t_ν²) + 2 Σ_{i≠j} ρ_ij²) / P²
    (Isserlis), with ρ_ij² estimated from the batches and bias-corrected.
  - The statistic is matched to a·χ²_f with the same mean and variance (Satterthwaite).
    Effective f ≈ 2.3, 8.2 and 39 at d = 2, 10, 50.
  - At d = 50 the test seed's mean z² = 0.854 (expected 1.010). With the naive
    independent sd (0.039) that is -4.0 sd, a false alarm; with the correct sd
    (0.230) it is -0.68 sd.

**Validation (E9 (c), 200 seeds per d).** Each aggregate statistic should be ≈ N(0, 1)
(`results/e9_aggregate.csv`):

| d | individual failures at k | mean z / SE: sd, max \|.\| | mean z², scaled χ² score: sd, max \|.\| | mean z², normal approx.: max \|.\| |
|---|---|---|---|---|
| 2 | 0 / 200 (k = 4) | 1.03, 2.66 | 1.03, 3.26 | **4.55** |
| 10 | 0 / 200 (k = 5.06) | 0.99, 2.48 | 0.96, 2.85 | **4.01** |
| 50 | 0 / 200 (k = 5.68) | 0.96, 3.34 | 0.99, 2.97 | 3.57 |

The normal approximation for mean z² was tried first and rejected. Mean z² is
positive and right-skewed, and with few effective degrees of freedom its normal
approximation crossed 4 sd twice in 600 seeds, where 0.04 crossings are expected. The
scaled chi-square has no such excursions.

## Family-wise false-alarm probability

The chance that at least one comparison fails for a correct implementation, if all
seeds were redrawn. `pytest -q` output, 216 tests:

| Test file | Comparisons | Expected false alarms |
|---|---|---|
| test_greeks_ad.py | 159 | 0.010 |
| test_pricer.py | 58 | 0.0037 |
| test_basket_geometric.py (3 families + aggregates) | 1,410 | 0.0017 |
| test_greeks_lr.py | 25 | 0.0016 |
| test_stats.py (bootstrap coverage band) | 1 | 0.0016 |
| test_asian.py | 18 | 0.0011 |
| test_basket.py, test_digital.py, test_greeks_fd.py | 8 | 0.0006 |
| **Total** | **1,679** | **0.020** |

- **Whole suite:** P(at least one false alarm) ≤ 2.0% if the comparisons were
  independent, about 1 fresh seed set in 50. Positive dependence makes the true value
  lower.
- **Before this change** it was about 4% (≤ 13.4% bound), dominated by the d = 50
  family's measured 2.5%.
- **The remaining budget is spread over 58-159 comparisons per file,** each at 6.3e-5.

## What changed in Phase 5

### Absolute floors removed

| Test | Before | After | What the floor was hiding |
|---|---|---|---|
| test_pricer::test_matches_black_scholes | `4*se + 1e-4` | rare-event branch (2 of 54 cases) | put K=80, T=0.25, σ=0.1: 0 payers, estimate 0 and SE 0 vs true 1.1e-6 |
| test_greeks_ad::test_greeks_match_black_scholes | `4*se + 1e-3` | rare-event branch (4 of 54 cases) | **call K=80, T=0.25, σ=0.1: 0 non-payers, so pathwise rho is constant, SE ≈ 7e-18 (rounding), z ≈ 4e12.** The exact rho is slightly smaller, since N(d2) < 1. The estimator is right (rho = K T e^{-rT} n_ex / N exactly), but no SE test can show it. |
| test_digital::test_smoothed_delta… | `4*se + 1e-4` | 4 SE vs smoothed target (quadrature) + O(eps²) check | the smoothing bias, -0.10% at eps = 0.5 |
| test_greeks_lr::test_smoothed_gamma… | `4*se + 1e-4` | same | the smoothing bias, -0.10% at eps = 0.5 |

### Ad-hoc tolerances on random quantities replaced

| Test | Before | After |
|---|---|---|
| test_digital::test_crn_fd_delta… | `rtol=0.05` | 4 SE vs exact FD expectation + O(h²) check |
| test_greeks_fd::test_crn_gamma… | relative error < 5% | same |
| test_greeks_fd::test_crn_delta_converges… | gap < 1e-4 | exact path-wise bound e^{-rT} mean(a/2 · 1{\|S_T - K\| < h a}), a = S_T/S0 |
| test_greeks_fd::test_independent_seeds… | \|estimate - delta\| > 1 (itself failed ~3.4% of seeds: SE = 23) | SE > 10 × delta, and CRN SE < 0.01 × delta |
| test_greeks_lr::test_smoothed_price… | gap < 1e-3 | exact bound 0 ≤ gap ≤ e^{-rT} eps log 2, ordering in eps exact |
| test_basket::test_one_asset… | `rtol=5e-3` | 4 batch-means SE (t₁₉₉) |
| test_basket::test_higher_correlation… | sign > 0 | z > 4 (observed min z = 13.6) |
| test_pricer::test_z_scores… | \|mean\| < 0.5, 0.6 < sd < 1.4 | mean within 4 SE; Σz² in the χ²₆₀ band of the same size |
| test_pricer::test_put_call_parity | 3 × the *call's* SE (wrong quantity) | exact path-wise identity + martingale test of e^{-rT} mean(S_T) within 4 of its own SE |
| test_pricer::test_standard_error_scales… | ratio within 5% of 10 | 4 × delta-method sd of the ratio, sqrt((κ-1)/4 · (1/n₁ + 1/n₂)) |
| test_greeks_ad::test_delta_parity | \|c - p - 1\| < 0.01 (≈ 22 SE) | 4 SE |
| test_greeks_ad::test_pathwise_delta_is_bounded | < 1.05 | ≤ e^{-rT} mean(S_T)/S0, exact per-path bound |

All tests pass after the change (215, then 216 with the family-wise threshold test).
No seed was changed.

### Numerical tolerances kept (deterministic comparisons)

| Tolerance | Where | Why |
|---|---|---|
| rtol 1e-10 … 1e-14, atol 1e-12 | forward = reverse, grad-of-mean = mean-of-grads, vmap = loop, chunked = one-shot, unrolled = rolled | same arithmetic, different order: rounding only |
| atol 1e-10 | vectorised vs loop bumping (h = 1e-4) | price rounding ~1e-15 amplified by 1/(2h) |
| rtol 1e-4 (h = 1e-4), 1e-5 (Asian, h = 1e-5) | CRN FD vs AD on the same paths | same estimator up to O(h²) and paths within ~h of the kink (~1e-5 relative) |
| rtol 1e-7 | geometric-basket closed form vs its own central difference (h = 1e-5) | smooth function: O(h²) ≈ 1e-10, rounding ~1e-16/h |
| rtol 1e-12 | closed forms vs Black-Scholes in limiting cases | exact identities |
| 4 ± 0.05 | O(h²), O(eps²) bias ratios | the next-order term moves the ratio by ~0.2%; observed 3.99 - 4.00 |
| [0.90, 0.99] of 200 | bootstrap CI coverage (test_stats) | ±3 binomial sd; logged in the budget at its exact false-alarm rate, 0.16% |
