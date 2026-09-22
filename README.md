# Monte Carlo Greeks by Algorithmic Differentiation

Monte Carlo option pricing in JAX, with Greeks computed by automatic differentiation
and compared against finite differences, likelihood-ratio estimators and Black-Scholes.

## Status

- [x] European call/put pricer (GBM, vectorised, jit-compiled), validated against Black-Scholes (E1)
- [x] Autodiff delta, vega, rho with standard errors, validated against Black-Scholes
- [x] Finite-difference baseline (CRN and independent seeds), error vs bump size (E2)
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

## E1: pricer validation

54 cases (call/put, K in {80, 100, 120}, T in {0.25, 1, 2}, sigma in {0.1, 0.2, 0.4}),
S0 = 100, r = 5%, 1,000,000 paths each, independent seeds per case.
Result: all 54 within 3 standard errors of Black-Scholes. Full table in `results/e1_prices.csv`.

## E1: autodiff Greeks

Delta, vega and rho from one `jax.value_and_grad` call (pathwise estimator), with
standard errors from per-path gradients (`jax.vmap`). Same 54-case grid, 1,000,000 paths,
independent seeds. Every Greek within 3 SE of Black-Scholes; mean z between -0.26 and +0.04.
Cases where the per-path estimator is constant (every path in or out of the money,
e.g. deep ITM call rho = K T e^{-rT}) have zero variance and are reported separately.
Full table: `results/e1_greeks.csv`.

Cost with 3 Greeks (Windows, 24-thread CPU, N = 1,000,000): autodiff 7.9x one pricing,
central-difference bumping 7.4x. With three parameters the two cost about the same;
reverse mode's advantage is that its cost does not grow with the number of parameters
(see the basket experiment). Timings are machine-dependent.

## E2: finite differences vs autodiff

![E2](results/e2_fd_vs_ad.png)

ATM call, 500 independent batches of 20,000 paths, RMSE against Black-Scholes.

| Estimator | Best h | RMSE |
|---|---|---|
| Delta, FD independent seeds | 10 | 0.0113 |
| Delta, FD common random numbers | 2.5 | 0.0040 |
| Delta, autodiff (pathwise) | none | 0.0041 |
| Gamma, FD independent seeds | 16 | 0.0013 |
| Gamma, FD common random numbers | 6.3 | 0.0003 |
| Gamma, autodiff (naive) | none | 0.0188 (returns exactly 0) |

- Independent seeds: error grows like 1/h (delta) and 1/h^2 (gamma) as h shrinks,
  because noise that does not cancel is divided by the bump. The optimum is a large,
  biased bump, and the error there is ~3x worse than autodiff.
- CRN: delta converges to the pathwise (autodiff) estimator as h -> 0, with the same
  error. It needs a tuned h and two repricings per parameter; autodiff needs neither.
- Gamma with CRN grows like h^(-1/2) as h shrinks: only paths within h of the strike
  contribute, each with a 1/h-sized term.
- Naive autodiff gamma is exactly 0 on every batch (the payoff's second derivative is
  zero almost everywhere). Fixed in the next step.
