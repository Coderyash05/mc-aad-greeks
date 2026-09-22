"""E3 + E4: fixing autodiff gamma.

E3 compares gamma estimators over REPS independent batches of N paths
(the same random numbers as E2, so results are directly comparable):
  naive autodiff, FD with common random numbers (best h from E2),
  likelihood ratio (LR), mixed pathwise-LR, and softplus-smoothed autodiff.
E4 sweeps the smoothing width eps: bias falls, variance rises, RMSE has a minimum.

Run:  python experiments/e3_gamma.py
Writes results/e3_gamma.csv, results/e4_smoothing.csv, results/e3_e4_gamma.png.
"""
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks
from mcgreeks.greeks_lr import (lr_gamma_samples, pwlr_gamma_samples,
                                smooth_gamma_samples)
from mcgreeks.pricer import discounted_payoffs, mc_price

S0, SIGMA, R, T, K = 100.0, 0.2, 0.05, 1.0, 100.0
REPS, N = 500, 20_000
H_FD = 6.31                        # best CRN bump from E2
EPS_GRID = np.logspace(-2, 1.2, 17)
RESULTS = Path(__file__).resolve().parents[1] / "results"

C_SMOOTH, C_LR, C_PWLR, C_FD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def per_batch(samples):
    return samples.reshape(REPS, N).mean(axis=-1)


def stats(est, exact):
    bias = float(jnp.mean(est) - exact)
    std = float(jnp.std(est, ddof=1))
    rmse = float(jnp.sqrt(jnp.mean((est - exact) ** 2)))
    return {"mean": float(jnp.mean(est)), "bias": bias, "std": std, "rmse": rmse}


def main():
    g_ex = float(bs_greeks(S0, K, R, SIGMA, T)["gamma"])
    key = jax.random.split(jax.random.PRNGKey(2026), 4)[0]   # same Z as E2 (CRN key)
    Z = jax.random.normal(key, (REPS, N), dtype=jnp.float64)
    Zf = Z.reshape(-1)

    naive = jax.vmap(jax.grad(jax.grad(mc_price, 0), 0), in_axes=(None,) * 5 + (0,))(
        S0, SIGMA, R, T, K, Z)
    price = jax.jit(lambda s: discounted_payoffs(s, SIGMA, R, T, K, Z).mean(-1))
    fd = (price(S0 + H_FD) - 2 * price(S0) + price(S0 - H_FD)) / H_FD**2
    lr = per_batch(lr_gamma_samples(S0, SIGMA, R, T, K, Zf))
    pwlr = per_batch(pwlr_gamma_samples(S0, SIGMA, R, T, K, Zf))

    sweep = []
    for eps in EPS_GRID:
        s = per_batch(smooth_gamma_samples(S0, SIGMA, R, T, K, Zf, float(eps)))
        sweep.append({"eps": float(eps), **stats(s, g_ex)})
    best = min(sweep, key=lambda r: r["rmse"])

    table = [
        ("Autodiff, naive", stats(naive, g_ex)),
        (f"FD, CRN (h = {H_FD})", stats(fd, g_ex)),
        ("Likelihood ratio", stats(lr, g_ex)),
        ("Pathwise-LR (mixed)", stats(pwlr, g_ex)),
        (f"Autodiff, smoothed (eps = {best['eps']:.2g})",
         {k: best[k] for k in ("mean", "bias", "std", "rmse")}),
    ]

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "e3_gamma.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimator", "mean", "bias", "std", "rmse"])
        for name, s in table:
            w.writerow([name, s["mean"], s["bias"], s["std"], s["rmse"]])
    with (RESULTS / "e4_smoothing.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)

    print(f"E3: gamma of an ATM call, exact = {g_ex:.5f}; {REPS} batches x {N:,} paths\n")
    print(f"{'estimator':<34}{'mean':>10}{'bias':>11}{'std':>10}{'RMSE':>10}")
    for name, s in table:
        print(f"{name:<34}{s['mean']:>10.5f}{s['bias']:>11.2e}{s['std']:>10.5f}{s['rmse']:>10.5f}")
    print("\nE4: smoothing width sweep")
    print(f"{'eps':>8}{'bias':>12}{'std':>10}{'RMSE':>10}")
    for r in sweep:
        print(f"{r['eps']:>8.3g}{r['bias']:>12.2e}{r['std']:>10.5f}{r['rmse']:>10.5f}")

    # ---- figure ----------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
    eps = [r["eps"] for r in sweep]

    a1.loglog(eps, [abs(r["bias"]) for r in sweep], "--o", color=C_SMOOTH, lw=2, ms=4,
              label="|Bias| (falls as eps shrinks)")
    a1.loglog(eps, [r["std"] for r in sweep], "-o", color=C_SMOOTH, lw=2, ms=4,
              label="Standard deviation (grows as eps shrinks)")
    # Below ~2 std / sqrt(REPS) a bias cannot be told apart from sampling noise.
    a1.loglog(eps, [2 * r["std"] / np.sqrt(REPS) for r in sweep], ":", color=MUTED, lw=1.5,
              label="Bias detection limit (2 SE of the mean)")
    a1.set_title("Smoothed autodiff gamma: bias vs noise", loc="left", fontsize=11)
    a1.set_ylabel("Error component")

    a2.loglog(eps, [r["rmse"] for r in sweep], "-o", color=C_SMOOTH, lw=2, ms=4,
              label="Autodiff, softplus-smoothed")
    for (name, s), c, ls in zip(table[1:4], (C_FD, C_LR, C_PWLR), (":", "-.", "--")):
        a2.axhline(s["rmse"], color=c, lw=2, ls=ls, label=name)
    a2.set_title("Gamma RMSE vs smoothing width", loc="left", fontsize=11)
    a2.set_ylabel("RMSE vs Black-Scholes")
    ticks = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    a2.set_yticks(ticks, [f"{t:g}" for t in ticks])
    a2.minorticks_off()

    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        ax.set_xlabel("Smoothing width eps (spot units, S0 = 100)")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"Fixing autodiff gamma: ATM call, {REPS} batches of {N:,} paths "
                 f"(naive autodiff RMSE = {table[0][1]['rmse']:.4f}, always 0)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_e4_gamma.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e3_e4_gamma.png'}")


if __name__ == "__main__":
    main()
