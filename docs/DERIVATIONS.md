# Derivations

The maths behind every estimator and test in this repository, in the order the code
uses it. Notation: S_T = S0 exp(μT + σ√T Z) with μ = r - σ²/2 and Z ~ N(0, 1), f is
the payoff, and V = e^{-rT} E[f(S_T)]. Each section names the file that implements it
and the experiment that measures it.

## 1. Pathwise estimators and when they fail

`src/mcgreeks/greeks_ad.py` (E1, E2)

Differentiate inside the expectation, path by path, holding Z fixed:

    ∂V/∂θ = e^{-rT} E[ f'(S_T) ∂S_T/∂θ ] + E[f(S_T)] ∂e^{-rT}/∂θ

- **The path derivatives:** ∂S_T/∂S0 = S_T/S0, ∂S_T/∂σ = S_T(-σT + √T Z), and
  ∂S_T/∂r = T S_T.
- **The call:** f'(S) = 1{S > K} gives the pathwise delta e^{-rT} E[1{S_T > K} S_T/S0].
  This is what `jax.grad` of the Monte Carlo price computes, and Giles & Glasserman
  (2006) show the adjoint gives exactly this estimator.
- **Why it is valid:** exchanging derivative and expectation needs f to be Lipschitz
  and differentiable almost everywhere (Glasserman 2004, ch. 7, pp. 393-395). A call
  is Lipschitz. A digital, 1{S > K}, is not; its derivative is 0 almost everywhere, so
  the pathwise digital delta is **identically 0** although the true delta is positive
  (E5).
- **Gamma** differentiates the call's pathwise delta, whose f' is a step, so the same
  failure gives pathwise (autodiff) gamma ≡ 0 (E2, E3). AD is not wrong here: it
  differentiates the program exactly, and the program's second derivative is 0 on
  every path. The Dirac mass at the kink has probability 0 of being sampled.

## 2. Finite differences with common random numbers

`src/mcgreeks/greeks_fd.py` (E2; tested in `tests/test_greeks_fd.py`)

The central difference with the same Z at both bumps, D_h = [V̂(S0+h) - V̂(S0-h)]/(2h),
where V̂ is the Monte Carlo price:
- **Its expectation** is the finite difference of the exact prices,
  [V(S0+h) - V(S0-h)]/(2h). So its bias is V'''(S0) h²/6 + O(h⁴), with no Monte
  Carlo term (the tests use this as the exact target).
- **Its limit as h → 0 is the pathwise estimator.** With a = S_T/S0, each path
  contributes [max(S_T + ha - K, 0) - max(S_T - ha - K, 0)]/(2h). That equals
  a 1{S_T > K} unless |S_T - K| < ha, and differs from it by at most a/2 on those
  paths. So

      |D_h - pathwise| ≤ e^{-rT} mean(a/2 · 1{|S_T - K| < h a}) = O(h)

  (Capriotti 2011, eq. 2.8: "exactly the same estimators … and the same variance").
- **With independent seeds** the two prices are independent, so
  Var(D_h) = [Var V̂(S0+h) + Var V̂(S0-h)]/(4h²) = O(1/h²). At h = 1e-3 the standard
  error is 23, against a true delta of 0.64 (E2).

## 3. Likelihood ratio (LR)

`src/mcgreeks/greeks_lr.py` (E3, E5, E8)

Move the parameter into the density instead of the payoff. log S_T has density
g(x; S0) = φ(ζ)/(σ√T) with ζ = (x - log S0 - μT)/(σ√T), and at x = log S_T, ζ = Z.
Then

    ∂V/∂S0 = e^{-rT} E[ f(S_T) ∂ log g/∂S0 ],     ∂ log g/∂S0 = Z/(S0 σ√T).

For gamma use ∂²g/g = (∂ log g)² + ∂² log g. Since ∂Z/∂S0 = -1/(S0σ√T),

    ∂/∂S0 [Z/(S0σ√T)] = -1/(S0²σ²T) - Z/(S0²σ√T),

so the LR gamma weight is

    w_Γ = (Z² - 1 - σ√T Z) / (S0² σ² T),      Γ = e^{-rT} E[f(S_T) w_Γ].

- **Unbiased for any integrable payoff,** digitals included: nothing is differentiated
  but the density.
- **Its variance is high.** The weight has standard deviation ~1/(S0²σ²T), so it grows
  as T → 0. For a call, though, the payoff it multiplies near the strike is
  ~S0 σ√T, so the ATM error grows like 1/√T, the same rate as the true gamma. The
  LR / pathwise-LR ratio is therefore flat at the money (3.65 at T = 0.1, 3.72 at
  T = 2; E8).

## 4. Mixed pathwise-LR gamma

`pwlr_gamma_samples` (E3, E8)

Take the pathwise delta, then apply LR once. Delta = e^{-rT} E[f'(S_T) S_T]/S0, so

    Γ = -Δ/S0 + (e^{-rT}/S0) ∂/∂S0 E[f'(S_T) S_T]
      = -Δ/S0 + (e^{-rT}/S0) E[f'(S_T) S_T · Z/(S0σ√T)]
      = e^{-rT}/S0² · E[ f'(S_T) S_T (Z/(σ√T) - 1) ].

- **Only one derivative falls on the density,** so the weight is O(1/√T) rather than
  O(1/T) and the variance is much lower than LR's (3.57x smaller RMSE ATM, E3).
- **It needs f' only almost everywhere,** which a call has.

## 5. Parity switching

`*_parity_samples` (E8)

- **Call gamma equals put gamma.** C - P = S0 - K e^{-rT} is linear in S0, so the two
  have the same gamma exactly.
- **Digital deltas are opposite.** 1{S_T > K} + 1{S_T < K} = 1 almost surely, so
  Δ(digital call) = -Δ(digital put).
- **Either leg gives an unbiased LR or pathwise-LR estimator; their variances differ.**
  The LR second moment is E[f(S_T)² w²]. In the money a call pays on most paths, and
  pays S_T - K, whose linear part has true gamma 0 but a large LR estimate; the put pays
  only in the tail.
- **So use the put leg when K < S0 e^{rT},** i.e. the out-of-the-money leg in forward
  terms. For pathwise-LR, f'(S) = -1{S < K}.
- **This is parity used as a control variate inside the estimator** (Glasserman 2004,
  ch. 4). At K = 80, T = 0.1 the pathwise-LR RMSE falls to 0.045 [0.038, 0.053] of the
  plain estimator's (E8).

## 6. Payoff smoothing, and why its bias is O(ε²)

`call_payoff_smooth`, `digital_payoff_smooth` (E3, E4, E5)

- **The smoothed payoffs:** softplus call, ε log(1 + e^{(S-K)/ε}); sigmoid digital,
  1/(1 + e^{-(S-K)/ε}). Both are smooth, so autodiff gamma and digital delta become
  nonzero.
- **Both are convolutions.** d²/dx² [ε softplus(x/ε)] is the logistic density k_ε with
  scale ε, which is symmetric with mean 0. E_Y[max(x - Y, 0)] has the same second
  derivative, so the two differ by an affine function a + bx. Both tend to 0 as
  x → -∞, which forces a = b = 0, so

      ε softplus(x/ε) = E_Y[ max(x - Y, 0) ],   Y ~ logistic(0, ε),  Var Y = π²ε²/3,

  and the sigmoid is the logistic CDF, so sigmoid((S-K)/ε) = E_Y[1{S - K > Y}].
- **So the smoothed price is an average of prices over a random strike K + Y.** A
  Taylor expansion in Y:

      V_ε = E_Y[V(K + Y)] = V(K) + V_K E[Y] + ½ V_KK E[Y²] + O(ε⁴) = V(K) + (π²ε²/6) V_KK + O(ε⁴),

  because E[Y] = 0 by symmetry. The same holds for every S0-derivative. **The bias is
  O(ε²):** halving ε quarters it. Measured ratios are 3.974 at ε = 1, 3.994 at 0.5 and
  3.998 at 0.25; the tests assert 4 ± 0.05.
- **Variance grows as ε shrinks.** The smoothed gamma is e^{-rT} k_ε(S_T - K)(S_T/S0)²,
  whose second moment scales like 1/ε. The RMSE-optimal ε balances ε⁴ against 1/(εN):
  ε* ∝ N^{-1/5}. E4 finds ε ≈ 1.6 at N = 20,000.
- **Exact targets for tests.** E[smoothed estimator] is a one-dimensional integral over
  Z, computed by adaptive quadrature (`tests/helpers.py`). That is what the smoothed
  Monte Carlo estimate is tested against, not the Black-Scholes value.

## 7. Forward vs reverse mode: cost

`greeks_ad.py`, `basket.py` (E1, E6, E6b)

Let F: R^n → R^m cost W.
- **Forward (tangent) mode** propagates one direction v: it computes J·v for ~c_f·W,
  with c_f ≈ 1-2.5. The full Jacobian costs n·c_f·W.
- **Reverse (adjoint) mode** propagates one output weight u: it computes uᵀJ for
  ~c_r·W, with c_r ≲ 4-5 (the cheap-gradient principle; Griewank & Walther 2008). The
  full Jacobian costs m·c_r·W, plus memory for the stored forward sweep (the "tape").
- **A price is m = 1 output of n inputs,** so reverse mode wins for n ≥ 2 and its cost
  does not depend on n. E6: 2.05x flops at n = 1,325.
- **Strike-sweep deltas are m = K outputs of n = 1 input,** so forward mode wins. E6b:
  3.3x vs 669x flops at K = 1,000.
- **Transcendentals.** Both modes evaluate each exp once, since d/dx e^x reuses the
  value already computed. Bumping evaluates it 2P + 1 times.
- **Why wall-clock exceeds flops.** Reverse mode stores and re-reads the forward
  sweep's intermediates. XLA counts 3.8x the bytes of one pricing, and this Monte Carlo
  kernel is bandwidth-bound (7-18% of peak arithmetic), so time tracks bytes: 4.2x at
  d = 50.

## 8. Chunked adjoint and √M checkpointing

`src/mcgreeks/chunked.py`, `asian.py` (E7)

- **Chunking.** V = (1/N) Σ_i v_i(θ) = (1/n_b) Σ_b V_b(θ), with V_b the mean over batch
  b, so ∇V = (1/n_b) Σ_b ∇V_b.
  - A `lax.scan` over batches keeps one batch's tape at a time: memory O(B), not O(N).
  - Each batch draws its normals from `fold_in(key, b)`, so nothing N-sized exists.
  - Parameter-only work p(θ), such as the Cholesky factor L(ρ), is hoisted: with
    V_b = F_b(p(θ)), ∇V = J_pᵀ (1/n_b) Σ_b ∇_p F_b, which needs one vjp through the
    Cholesky factorisation instead of n_b of them.
- **Checkpointing M time steps.** Storing every step costs O(M) memory. Split the path
  into M/b blocks of b steps and keep only block-boundary states. In the backward pass,
  recompute one block's forward sweep at a time, which holds b states.
  - Memory ∝ M/b + b, minimised at b = √M (2√M).
  - Cost: one extra forward sweep, ~2x flops.
  - E7: 172 → 11.4 MB at M = 1,000; total MB / √M ≈ 0.36-0.44 for M = 12 to 1,000.
  - Griewank & Walther (2008, ch. 12) give the optimal (binomial) schedules; √M is the
    simplest two-level version.

## 9. Geometric basket closed form

`geometric_basket_price` (E9)

log G_T = Σ_i w_i log S_T^i with Σ w_i = 1. Each log S_T^i = log S0_i + (r - σ_i²/2)T +
σ_i √T X_i with X ~ N(0, C), so log G_T is normal with

    m = Σ_i w_i [log S0_i + (r - σ_i²/2) T],        s² = T (w∘σ)ᵀ C (w∘σ).

For any lognormal G the call price is

    e^{-rT} [ e^{m + s²/2} N(d1) - K N(d2) ],   d1 = (m - log K + s²)/s,  d2 = d1 - s.

- **Exact Greeks:** `jax.grad` of this formula gives exact deltas, vegas and
  correlation sensitivities.
- **Correlation convention:** ρ_k moves both symmetric entries C_ij = C_ji, in the
  closed form and in Monte Carlo alike.
- **Special cases:** d = 1 is Black-Scholes. C = all ones with equal σ and S0 gives
  G_T = S_T, so the price is Black-Scholes, Δ_i = w_i Δ_BS, and Σ_i vega_i = vega_BS.
- **All deltas share one z-score.** ∂ log G/∂S0_i = w_i/S0_i on every path, so all d
  pathwise deltas are (w_i/S0_i) times one common random variable. That is why they
  share a single z-score (E9).

## 10. Statistics

`src/mcgreeks/stats.py`, `tests/helpers.py` (E2-E5, E8, E9; docs/TESTING.md)

- **Standard errors.**
  - Per-path samples: SE = s/√N and z ≈ N(0, 1).
  - Batch means over B batches: SE = s_batch/√B and z ~ t_{B-1} (Glasserman 2004,
    App. A). Batch means are used when per-path gradients would be too large to store
    (N x 1,325).
- **Paired bootstrap.** Methods evaluated on the same R batches have correlated errors,
  so the R batch indices are resampled jointly and every method's RMSE is recomputed
  on the same resample.
  - The CI of RMSE_A / RMSE_B then reflects the pairing (Efron & Tibshirani 1993).
  - "A beats B" is claimed only if that CI excludes 1.
- **Out-of-sample tuning.** h and ε are chosen on separate calibration batches, then
  evaluated on the evaluation batches. Choosing on the evaluation batches selects
  favourable noise and biases RMSE down. E8 measures the gap at 0-4%.
- **Family-wise control.**
  - P comparisons each at level α fail together with probability up to
    1 - (1 - α)^P.
  - Šidák's per-comparison level 1 - (1 - α)^{1/P} restores α for the family. This is
    exact for independent comparisons and conservative for positively dependent ones.
    At α = 6.3e-5 it gives k = 5.06 (P = 66) and 5.68 (P = 1,326) for t₁₉₉.
- **Aggregate tests, with dependence.**
  - Mean z = mean_b u_b with u_b = (1/P) Σ_k (G_bk - exact_k)/SE_k, a linear
    combination of the estimates. So its SE is sd(u_b)/√B, which includes every
    correlation.
  - For t_ν z's, E[z²] = ν/(ν - 2). By Isserlis' theorem,
    Var(mean z²) = [P Var(t_ν²) + 2 Σ_{i≠j} ρ_ij²]/P². Here ρ_ij² comes from the
    batches, corrected for the 1/(B - 1) bias of r².
  - Mean z² is skewed, so it is matched to a scaled a·χ²_f with the same two moments
    (Satterthwaite 1946; Box 1954): f = 2E²/Var, a = Var/(2E).

## Interview questions

<details>
<summary><b>1. Why does autodiff return a gamma of exactly zero for a Monte Carlo call, and what are the fixes?</b></summary>

AD differentiates the simulation program exactly, path by path. The call payoff is
piecewise linear in S_T, so its second derivative is 0 on every path. The curvature
is a Dirac mass at the strike, which a sample path hits with probability 0. So the
pathwise gamma estimator is identically 0: not noisy, but structurally blind. Fixes:

- move one or both derivatives onto the density: likelihood ratio, or mixed
  pathwise-LR;
- smooth the payoff (softplus), trading an O(ε²) bias for variance ~1/ε;
- finite-difference the pathwise delta with common random numbers.

At the money here (E3), pathwise-LR on the put leg ("parity", since K < S0 e^{rT}) is
best. Relative to its RMSE, plain pathwise-LR is 1.58x [1.45, 1.71], CRN FD 1.76x
[1.61, 1.92], smoothed AD 1.97x [1.81, 2.15], LR (parity) 3.50x and LR 5.63x.
</details>

<details>
<summary><b>2. Pathwise vs likelihood-ratio estimators: when is each unbiased, and which has lower variance?</b></summary>

- **Pathwise:** needs the payoff to be Lipschitz and differentiable almost everywhere,
  so that derivative and expectation can be swapped (Glasserman 2004, ch. 7). That
  holds for calls and puts, not for digitals or for gamma.
- **LR:** needs only a differentiable density of the state, so it handles any
  integrable payoff.
- **Variance:** pathwise is usually far lower, because LR multiplies the whole payoff
  by a score weight whose variance explodes as T → 0 (the gamma weight is ~1/T).
- **Where LR wins:** where pathwise does not apply. For the ATM digital delta, LR is
  best: 0.61x the RMSE of CRN FD (E5).
- **Beyond GBM:** LR needs the transition density. Under Heston this is not available
  in closed form for the discretised path, which is why the fixes here are shown for
  GBM only.
</details>

<details>
<summary><b>3. Tuned bump-and-reprice beat autodiff for delta. Isn't AD supposed to be exact?</b></summary>

AD is exact for the derivative of the simulated price, and that is the pathwise
estimator, which has its own Monte Carlo variance.

CRN central differences converge to the same estimator as h → 0 (Capriotti 2011,
eq. 2.8; here the gap is bounded path by path by the fraction of paths within ~h of
the strike). At a finite h the difference quotient averages the kink over ±h. That
adds an O(h²) bias but slightly reduces variance, and at the RMSE-optimal h = 2.5 the
trade is favourable: RMSE ratio 0.978 [0.962, 0.994] (E2).

So bumping is 2% more accurate for one Greek, given an h tuned against a known answer.
It also costs 2P + 1 pricings, versus a constant ~2-4 for reverse AD, and needs no
tuning. The honest statement is that AD wins on cost and robustness, not on accuracy.
</details>

<details>
<summary><b>4. When should you use forward mode rather than reverse mode?</b></summary>

- **Forward mode** costs ~1-2.5 function evaluations per input direction.
- **Reverse mode** costs ~2-4 evaluations per output (the cheap-gradient bound), plus
  memory for the tape.
- **One price, many sensitivities:** reverse. E6: 1,325 basket sensitivities (+ price) for 2.05x
  flops, against forward's 1,389x.
- **Many outputs of one input** (a delta ladder across strikes, a whole vector of
  prices): forward. E6b: 1,000 strikes cost 3.3x in forward flops vs 669x in reverse.
- **Even at 3 inputs,** reverse already wins (4.3x vs 6.6x flops, E1). That is exactly
  what the cost model predicts, not a surprise.
</details>

<details>
<summary><b>5. Reverse mode costs 2.05x a pricing in flops but 4.2x in wall-clock. Why?</b></summary>

- **Reverse mode stores the forward sweep's intermediates** (the N x d path arrays) and
  reads them back in the backward sweep.
- **XLA's bytes-accessed count** is 3.8x one pricing at every d, and the wall-clock
  ratio tracks bytes, not flops.
- **The kernel is memory-bandwidth-bound:** a pricing streams at ~25 GB/s (the machine's
  triad bandwidth) while using 7-18% of peak arithmetic.
- **When the arrays fit in cache (d ≤ 5),** the time ratio drops to 2.8-2.9x.

On a GPU, or with a hand-written adjoint that recomputes cheap values instead of
storing them, the balance would differ.
</details>

<details>
<summary><b>6. How do you keep the memory of an adjoint bounded for a million paths or a thousand time steps?</b></summary>

- **Paths: chunk them.** The gradient of a mean is the mean of the batch gradients, so
  a `lax.scan` over batches keeps one batch's tape at a time.
  - Each batch draws its own normals with `fold_in(key, b)`.
  - Parameter-only work such as the Cholesky factor is hoisted and pulled back once.
  - Basket, d = 50, N = 1e6: 2,024 MB → 20 MB at B = 1e4, same flops, 1.03x the time.
- **Time steps: checkpoint.** Keep states only at block boundaries (blocks of √M steps)
  and recompute each block's forward sweep during the backward pass.
  - Memory ~2√M instead of M, for ~2x the flops: 172 → 11 MB at M = 1,000.
- **Checkpointing every single step saves nothing for GBM:** a step's residual is one
  array, the same size as the carry you store instead.
</details>

<details>
<summary><b>7. How would you check that 1,325 basket sensitivities (+ price) are all correct?</b></summary>

- **Change the payoff to the geometric basket,** which has a closed form (log G_T is
  normal). Everything else, including the Cholesky and correlation code, is shared
  with the arithmetic basket. Then compare every Monte Carlo sensitivity with
  `jax.grad` of the closed form, as z = error / SE.
- **Do not trust max |z| from one seed.** At 4 SE per comparison the d = 50 family
  would fail on 2.5% of seeds with correct code. Yet the observed max is only 2.57,
  because the z's are strongly correlated: all deltas share one z.
- **Use family-wise thresholds** (Šidák: 5.68 SE for 1,326 comparisons).
- **Add two aggregate tests with dependence-aware SEs,** to catch many small biases:
  - mean z, whose SE comes from the batches;
  - mean z², against a moment-matched scaled chi-square.
- **Calibrate the whole procedure over 200 seeds:** mean z² = 1.03 [1.00, 1.06], and
  the aggregate statistics have sd ≈ 1.
</details>

<details>
<summary><b>8. Plain pathwise-LR beats finite differences for gamma at the money. Why does it fail in the money, and how do you fix it?</b></summary>

- **The failure.** Pathwise-LR multiplies f'(S_T) S_T by a score weight. In the money
  a call's f' = 1 on most paths, so the estimator is dominated by the linear part of
  the payoff: its true gamma is 0, but it contributes large variance. At K = 80,
  T = 0.1 it is 28x worse than the best method (E8).
- **The fix is put-call parity.** C - P is linear in S0, so the put has the same gamma,
  and the put's f' = -1{S_T < K} is nonzero only in the tail. Switching to the put leg
  when K < S0 e^{rT} cuts the RMSE to 0.045x.
- **Result:** best or tied in 17 of 20 strike/maturity cells, worst case 2.9x. It helps
  at the money too, because K = 100 is below the forward (105.1): 0.63x [0.58, 0.69]
  the RMSE of plain pathwise-LR (E3).
- **The same idea applies to the digital delta** (1{S > K} = 1 - 1{S < K}): the worst
  case falls from 23x to 1.7x.
</details>

<details>
<summary><b>9. How large is the bias from payoff smoothing, and how do you choose the width?</b></summary>

- **The bias is second order.** A softplus-smoothed call is exactly the call averaged
  over a logistic strike shift Y with mean 0 and variance π²ε²/3. The first-order term
  vanishes by symmetry, leaving (π²ε²/6)·∂²V/∂K² + O(ε⁴).
  - Halving ε quarters the bias: measured ratio 3.994 at ε = 0.5.
  - The same holds for the sigmoid digital.
- **The variance of the smoothed gamma scales like 1/ε,** so the RMSE-optimal
  ε ∝ N^{-1/5}.
- **Choose ε out of sample:** minimise RMSE on calibration batches, then report on
  separate evaluation batches.
- **Smoothing is the most robust fix** (never worse than 2.4x the best across 20
  cells) but rarely the best (6/20).
</details>

<details>
<summary><b>10. How do you claim that one Monte Carlo estimator beats another?</b></summary>

- **Evaluate all methods on the same independent batches** and record each batch's
  error.
- **Use a paired bootstrap.** Resample batch indices jointly and recompute every
  method's RMSE on the same resample. Common random numbers make the errors
  correlated, and pairing turns that into a much tighter CI for the RMSE ratio.
- **Claim "A beats B" only if the ratio's CI excludes 1.** For example, smoothed AD vs
  CRN FD for the digital delta is 1.03x [0.99, 1.07]: not distinguishable.
- **Tune any free parameter (h, ε) on separate calibration batches,** and report the
  in-sample choice too, so selection bias is visible (0-4% here).
- **Never compare a naive AD gamma of exactly 0 in a ratio.** Report its bias.
</details>
