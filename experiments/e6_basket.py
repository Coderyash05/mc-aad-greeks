"""E6: cost of all sensitivities of a basket call, autodiff vs bump-and-reprice.

For d assets there are P = d deltas + d vegas + d(d-1)/2 correlation
sensitivities. Central-difference bumping needs 2P + 1 pricings. One reverse-mode
pass returns all P for a roughly constant multiple of one pricing.
Everything below is measured, nothing extrapolated.

Run:  python experiments/e6_basket.py        (a few minutes: bumping d = 50 alone is ~2,650 pricings)
Writes results/e6_basket.csv and results/e6_basket.png.
"""
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.basket import basket_greeks, basket_price

R, T, K, N = 0.05, 1.0, 100.0, 100_000
DIMS = [2, 5, 10, 20, 35, 50]
RESULTS = Path(__file__).resolve().parents[1] / "results"

C_AD, C_FD = "#1baf7a", "#eb6834"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def timeit(fn, reps):
    fn()  # warm-up / compile
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def setup(d):
    S0 = jnp.full(d, 100.0)
    sigma = jnp.linspace(0.15, 0.35, d)
    rho = jnp.full(d * (d - 1) // 2, 0.5)
    w = jnp.full(d, 1.0 / d)
    Z = jax.random.normal(jax.random.PRNGKey(d), (N, d), dtype=jnp.float64)
    return S0, sigma, rho, w, Z


def bump_all(S0, sigma, rho, w, Z, h=1e-4):
    """Full central-difference gradient: 2P + 1 pricings. Returns the last result."""
    out = basket_price(S0, sigma, rho, R, T, K, w, Z)
    for idx, x in enumerate((S0, sigma, rho)):
        for i in range(x.shape[0]):
            for s in (h, -h):
                args = [S0, sigma, rho]
                args[idx] = x.at[i].add(s)
                out = basket_price(*args, R, T, K, w, Z)
    return out.block_until_ready()


def main():
    rows = []
    for d in DIMS:
        S0, sigma, rho, w, Z = setup(d)
        P = 2 * d + d * (d - 1) // 2
        t_price = timeit(lambda: basket_price(S0, sigma, rho, R, T, K, w, Z).block_until_ready(), 20)
        t_ad = timeit(lambda: basket_greeks(S0, sigma, rho, R, T, K, w, Z)["corr"].block_until_ready(), 20)
        t_fd = timeit(lambda: bump_all(S0, sigma, rho, w, Z), 1)
        rows.append({"d": d, "n_sens": P, "price_ms": t_price * 1e3, "ad_ms": t_ad * 1e3,
                     "fd_ms": t_fd * 1e3, "ad_x": t_ad / t_price, "fd_x": t_fd / t_price,
                     "speedup": t_fd / t_ad})
        r = rows[-1]
        print(f"d={d:>3}  sensitivities={P:>5}  price {r['price_ms']:7.2f} ms   "
              f"AD {r['ad_ms']:8.2f} ms ({r['ad_x']:5.1f}x)   "
              f"bump {r['fd_ms']:10.1f} ms ({r['fd_x']:7.1f}x)   AD speedup {r['speedup']:6.1f}x")

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "e6_basket.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # ---- figure: cost in units of one pricing, vs number of sensitivities ----------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    P = [r["n_sens"] for r in rows]
    ax.loglog(P, [r["fd_x"] for r in rows], "-o", color=C_FD, lw=2, ms=5)
    ax.loglog(P, [r["ad_x"] for r in rows], "-o", color=C_AD, lw=2, ms=5)
    ax.text(P[-1] * 0.8, rows[-1]["fd_x"], "Bump-and-reprice", color=INK, fontsize=9,
            ha="right", va="bottom")
    ax.loglog(P, [2 * p + 1 for p in P], ":", color=MUTED, lw=1.5)
    ax.text(P[3] * 1.15, (2 * P[3] + 1) * 0.55, "2P + 1 pricings (theory)", color=MUTED,
            fontsize=8.5, ha="left", va="top")
    ax.text(P[2], rows[2]["ad_x"] * 2.4, "Autodiff (one reverse pass)", color=INK,
            fontsize=9, ha="center", va="bottom")
    ax.set_ylim(1, None)
    for r in rows:
        ax.annotate(f"d={int(r['d'])}", (r["n_sens"], r["ad_x"]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=MUTED)
    ax.set_title(f"Basket call: cost of all sensitivities, {N:,} paths",
                 loc="left", fontsize=11)
    ax.set_xlabel("Number of sensitivities (deltas + vegas + correlations)")
    ax.set_ylabel("Cost, in units of one pricing")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "e6_basket.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e6_basket.png'}")


if __name__ == "__main__":
    main()
