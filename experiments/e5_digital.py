"""E5: delta of a digital (cash-or-nothing) call.

The payoff jumps from 0 to 1 at the strike, so its derivative is 0 on every path:
autodiff (pathwise) delta is exactly 0. Compared: FD with CRN (bump sweep),
sigmoid-smoothed autodiff (width sweep), and the likelihood ratio (no parameter).
Same random numbers as E2/E3.

Run:  python experiments/e5_digital.py
Writes results/e5_digital.csv and results/e5_digital.png.
"""
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_digital_delta
from mcgreeks.greeks_lr import lr_delta_samples, smooth_digital_delta_samples
from mcgreeks.pricer import discounted_payoffs, mc_price

S0, SIGMA, R, T, K = 100.0, 0.2, 0.05, 1.0, 100.0
REPS, N = 500, 20_000
WIDTHS = np.logspace(-2, 1.2, 17)   # bump size h or smoothing width eps, spot units
RESULTS = Path(__file__).resolve().parents[1] / "results"

C_SMOOTH, C_LR, C_FD, C_NAIVE = "#2a78d6", "#eb6834", "#eda100", "#1baf7a"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def stats(est, exact):
    return {"mean": float(jnp.mean(est)), "bias": float(jnp.mean(est) - exact),
            "std": float(jnp.std(est, ddof=1)),
            "rmse": float(jnp.sqrt(jnp.mean((est - exact) ** 2)))}


def main():
    d_ex = float(bs_digital_delta(S0, K, R, SIGMA, T))
    key = jax.random.split(jax.random.PRNGKey(2026), 4)[0]
    Z = jax.random.normal(key, (REPS, N), dtype=jnp.float64)
    Zf = Z.reshape(-1)

    def per_batch(x):
        return x.reshape(REPS, N).mean(axis=-1)

    naive = jax.vmap(jax.grad(lambda s, z: mc_price(s, SIGMA, R, T, K, z, kind="digital")),
                     in_axes=(None, 0))(S0, Z)
    lr = per_batch(lr_delta_samples(S0, SIGMA, R, T, K, Zf, kind="digital"))
    price = jax.jit(lambda s: discounted_payoffs(s, SIGMA, R, T, K, Z, kind="digital").mean(-1))

    sweep = []
    for w in WIDTHS:
        fd = (price(S0 + w) - price(S0 - w)) / (2 * w)
        sm = per_batch(smooth_digital_delta_samples(S0, SIGMA, R, T, K, Zf, float(w)))
        sweep.append({"width": float(w),
                      **{f"fd_{k}": v for k, v in stats(fd, d_ex).items()},
                      **{f"smooth_{k}": v for k, v in stats(sm, d_ex).items()}})
    best_fd = min(sweep, key=lambda r: r["fd_rmse"])
    best_sm = min(sweep, key=lambda r: r["smooth_rmse"])

    table = [
        ("Autodiff, naive (pathwise)", stats(naive, d_ex)),
        (f"FD, CRN (best h = {best_fd['width']:.2g})",
         {k: best_fd[f"fd_{k}"] for k in ("mean", "bias", "std", "rmse")}),
        (f"Autodiff, sigmoid (best eps = {best_sm['width']:.2g})",
         {k: best_sm[f"smooth_{k}"] for k in ("mean", "bias", "std", "rmse")}),
        ("Likelihood ratio", stats(lr, d_ex)),
    ]

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "e5_digital.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)

    print(f"E5: delta of an ATM digital call, exact = {d_ex:.5f}; {REPS} batches x {N:,} paths\n")
    print(f"{'estimator':<38}{'mean':>10}{'bias':>11}{'std':>10}{'RMSE':>10}")
    for name, s in table:
        print(f"{name:<38}{s['mean']:>10.5f}{s['bias']:>11.2e}{s['std']:>10.5f}{s['rmse']:>10.5f}")

    # ---- figure ----------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    x = [r["width"] for r in sweep]
    ax.loglog(x, [r["smooth_rmse"] for r in sweep], "-o", color=C_SMOOTH, lw=2, ms=4,
              label="Autodiff, sigmoid-smoothed (x = eps)")
    ax.loglog(x, [r["fd_rmse"] for r in sweep], "-o", color=C_FD, lw=2, ms=4,
              label="Finite difference, CRN (x = h)")
    ax.axhline(table[3][1]["rmse"], color=C_LR, lw=2, ls="-.", label="Likelihood ratio")
    ax.axhline(table[0][1]["rmse"], color=C_NAIVE, lw=2, ls="--",
               label="Autodiff, naive (returns exactly 0)")
    ax.set_title(f"Digital call delta: RMSE vs width, {REPS} batches of {N:,} paths",
                 loc="left", fontsize=11)
    ax.set_xlabel("Bump size h / smoothing width eps (spot units, S0 = 100)")
    ax.set_ylabel("RMSE vs Black-Scholes")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ticks = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout()
    fig.savefig(RESULTS / "e5_digital.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e5_digital.png'}")


if __name__ == "__main__":
    main()
