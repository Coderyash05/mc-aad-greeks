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

## E1: pricer validation

54 cases (call/put, K in {80, 100, 120}, T in {0.25, 1, 2}, sigma in {0.1, 0.2, 0.4}),
S0 = 100, r = 5%, 1,000,000 paths each, independent seeds per case.
Result: all 54 within 3 standard errors of Black-Scholes. Full table in `results/e1_prices.csv`.
