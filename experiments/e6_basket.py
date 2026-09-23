"""E6: cost of all sensitivities of a basket call: reverse AD, forward AD, bumping.

For d assets there are P = d deltas + d vegas + d(d-1)/2 correlation
sensitivities of one output (the price). Four ways to get all P:
  - reverse-mode AD: one forward + one backward sweep, cost ~constant in P
  - forward-mode AD: one jvp per input direction, cost ~linear in P
  - bump-and-reprice, Python loop: 2P + 1 calls of the jit-compiled pricer
  - bump-and-reprice, vectorised: the same 2P + 1 pricings vmapped in one program
This is the forward-vs-adjoint comparison of Giles & Glasserman (2006), Fig. 4,
with bumping added. Forward mode and the vectorised bumps are chunked
(lax.map with batch_size) so no intermediate exceeds MEM_BUDGET bytes.

Reported: wall-clock (median and IQR over repeated runs after compilation, see
mcgreeks.bench) and XLA's static operation count (cost_analysis), both in units
of one pricing. Flops for chunked programs are read from the unchunked (pure vmap)
program, because XLA counts a loop body once; that program is compiled, not run.
Everything is measured, nothing extrapolated.

Run:  python experiments/e6_basket.py        (~6 minutes on a 24-thread CPU)
      python experiments/e6_basket.py --plot-only   (redraw the figure from the CSV)
Writes results/e6_basket.csv, results/e6_hardware.json, results/e6_basket.png.
"""
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

import mcgreeks  # noqa: F401
from mcgreeks.basket import (basket_greeks, basket_greeks_fwd, basket_price, bump_grad_loop,
                             bump_grad_vmap)
from mcgreeks.bench import cost, print_hardware, summary, temp_bytes, time_runs

R, T, K, N, H = 0.05, 1.0, 100.0, 100_000, 1e-4
DIMS = [2, 5, 10, 20, 35, 50]
# Bytes per batched (chunk, N, d) float64 intermediate, per chunked method. Chosen
# as the fastest of 16 / 64 / 256 MB in experiments/e6_chunk_sweep.py: 16 MB was
# fastest or tied (within IQR) for both methods at d = 20 and d = 50. Smaller
# chunks win because the pricing is memory-bandwidth-bound (see e6_roofline.csv).
MEM_BUDGET = {"fwd": 16e6, "vec": 16e6}
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Project-wide method colours: forward AD blue, reverse AD aqua, CRN bumping amber.
# Loop and vectorised bumping are the same estimator (same arithmetic), so they share
# the amber and differ by line style and marker.
C_FWD, C_REV, C_BUMP = "#2a78d6", "#1baf7a", "#eda100"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

METHODS = ("price", "rev", "fwd", "loop", "vec")
LABELS = {"fwd": "Forward-mode AD (one jvp per input)",
          "vec": "Bump-and-reprice, vectorised (vmap)",
          "rev": "Reverse-mode AD (one backward sweep)",
          "loop": "Bump-and-reprice, Python loop"}
COLORS = {"fwd": C_FWD, "vec": C_BUMP, "rev": C_REV, "loop": C_BUMP}
STYLE = {"fwd": "-o", "vec": "--s", "rev": "-o", "loop": "-o"}


def setup(d):
    S0 = jnp.full(d, 100.0)
    sigma = jnp.linspace(0.15, 0.35, d)
    rho = jnp.full(d * (d - 1) // 2, 0.5)
    w = jnp.full(d, 1.0 / d)
    Z = jax.random.normal(jax.random.PRNGKey(d), (N, d), dtype=jnp.float64)
    return S0, sigma, rho, w, Z


def chunk_for(d, rows, budget):
    """Largest chunk with chunk x N x d x 8 bytes <= budget; None if all rows fit."""
    c = max(1, int(budget // (N * d * 8)))
    return None if c >= rows else c


def main():
    hw = print_hardware()
    print(f"N = {N:,} paths, h = {H:g}, memory budget per batched array: "
          f"forward {MEM_BUDGET['fwd'] / 1e6:.0f} MB, vectorised bumping "
          f"{MEM_BUDGET['vec'] / 1e6:.0f} MB\n")
    rows = []
    for d in DIMS:
        S0, sigma, rho, w, Z = setup(d)
        a = (S0, sigma, rho, R, T, K, w, Z)
        P = 2 * d + d * (d - 1) // 2
        c_fwd = chunk_for(d, P, MEM_BUDGET["fwd"])
        c_vec = chunk_for(d, 2 * P + 1, MEM_BUDGET["vec"])

        # ---- correctness first: all four give the same gradient ----------------
        rev = basket_greeks(*a)
        g_rev = jnp.concatenate([rev["delta"], rev["vega"], rev["corr"]])
        _, g_fwd = basket_greeks_fwd(*a, chunk=c_fwd)
        _, g_loop = bump_grad_loop(*a, h=H)
        _, g_vec = bump_grad_vmap(*a, h=H, chunk=c_vec)
        err_fwd = float(jnp.max(jnp.abs(g_fwd - g_rev)))
        err_vec = float(jnp.max(jnp.abs(g_vec - g_loop)))
        err_fd = float(jnp.max(jnp.abs(g_loop - g_rev)))   # O(h^2) + kink paths, not rounding
        assert err_fwd < 1e-10 and err_vec < 1e-10, (d, err_fwd, err_vec)

        # ---- timing -------------------------------------------------------------
        fns = {
            "price": lambda: basket_price(*a).block_until_ready(),
            "rev": lambda: basket_greeks(*a)["corr"].block_until_ready(),
            "fwd": lambda: basket_greeks_fwd(*a, chunk=c_fwd)[1].block_until_ready(),
            "loop": lambda: bump_grad_loop(*a, h=H)[1].block_until_ready(),
            "vec": lambda: bump_grad_vmap(*a, h=H, chunk=c_vec)[1].block_until_ready(),
        }
        row = {"d": d, "n_sens": P, "chunk_fwd": c_fwd or P, "chunk_vec": c_vec or 2 * P + 1}
        for m, fn in fns.items():
            med, q25, q75, n = summary(time_runs(fn, min_reps=3, max_reps=100, budget_s=2.0))
            row.update({f"{m}_ms": med * 1e3, f"{m}_q25_ms": q25 * 1e3,
                        f"{m}_q75_ms": q75 * 1e3, f"{m}_runs": n})

        # ---- static operation counts (loop-free programs only, see docstring) ----
        f_price = cost(basket_price, *a)
        f_rev = cost(basket_greeks, *a)
        f_fwd = cost(basket_greeks_fwd, *a, chunk=None)
        f_vec = cost(bump_grad_vmap, *a, h=H, chunk=None)
        f_loop = {k: (2 * P + 1) * v for k, v in f_price.items()}  # 2P + 1 identical calls
        for m, fc in (("price", f_price), ("rev", f_rev), ("fwd", f_fwd), ("loop", f_loop),
                      ("vec", f_vec)):
            row[f"{m}_flops"] = fc["flops"]
            row[f"{m}_transc"] = fc["transcendentals"]
            row[f"{m}_bytes"] = fc["bytes"]
        row["price_temp_mb"] = temp_bytes(basket_price, *a) / 1e6
        row["rev_temp_mb"] = temp_bytes(basket_greeks, *a) / 1e6
        row["fwd_temp_mb"] = temp_bytes(basket_greeks_fwd, *a, chunk=c_fwd) / 1e6
        row["vec_temp_mb"] = temp_bytes(bump_grad_vmap, *a, h=H, chunk=c_vec) / 1e6

        for m in METHODS:
            row[f"{m}_x"] = row[f"{m}_ms"] / row["price_ms"]
            row[f"{m}_flops_x"] = row[f"{m}_flops"] / row["price_flops"]
            row[f"{m}_transc_x"] = row[f"{m}_transc"] / row["price_transc"]
            row[f"{m}_bytes_x"] = row[f"{m}_bytes"] / row["price_bytes"]
        row.update({"max_err_fwd_vs_rev": err_fwd, "max_err_vec_vs_loop": err_vec,
                    "max_err_bump_vs_rev": err_fd})
        rows.append(row)

        print(f"d={d:>2} P={P:>4} | price {row['price_ms']:6.2f} ms | x one pricing (median [IQR]):"
              f"  rev {row['rev_x']:5.2f}  fwd {row['fwd_x']:7.1f}  loop {row['loop_x']:7.1f}"
              f"  vec {row['vec_x']:7.1f} | flops x: rev {row['rev_flops_x']:4.2f}"
              f"  fwd {row['fwd_flops_x']:6.1f}  bump {row['vec_flops_x']:6.1f}")
        print(f"        exp/log x: rev {row['rev_transc_x']:.2f}  fwd {row['fwd_transc_x']:.2f}  "
              f"bump {row['vec_transc_x']:.0f} | bytes accessed x: rev {row['rev_bytes_x']:.2f}  "
              f"fwd {row['fwd_bytes_x']:.1f}  bump {row['vec_bytes_x']:.1f}")
        print(f"        chunks: fwd {row['chunk_fwd']}, vec {row['chunk_vec']} | temp MB: "
              f"price {row['price_temp_mb']:.0f}, "
              f"rev {row['rev_temp_mb']:.0f}, fwd {row['fwd_temp_mb']:.0f}, "
              f"vec {row['vec_temp_mb']:.0f} | max|fwd-rev| {err_fwd:.1e}, "
              f"max|vec-loop| {err_vec:.1e}, max|bump-rev| {err_fd:.1e}")

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "e6_basket.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (RESULTS / "e6_hardware.json").write_text(json.dumps(hw, indent=2))

    print(f"\n{'d':>3}{'P':>6}  {'wall-clock, x one pricing: median [IQR]':<60}")
    for r in rows:
        cells = "  ".join(
            f"{m} {r[f'{m}_x']:.1f} [{r[f'{m}_q25_ms'] / r['price_ms']:.1f}, "
            f"{r[f'{m}_q75_ms'] / r['price_ms']:.1f}]" for m in ("rev", "fwd", "loop", "vec"))
        print(f"{r['d']:>3}{r['n_sens']:>6}  {cells}")
    print(f"\n{'d':>3}{'P':>6}{'rev vs loop':>13}{'rev vs vec':>12}{'rev vs fwd':>12}"
          f"{'vec vs loop':>13}")
    for r in rows:
        print(f"{r['d']:>3}{r['n_sens']:>6}{r['loop_ms'] / r['rev_ms']:>12.0f}x"
              f"{r['vec_ms'] / r['rev_ms']:>11.0f}x{r['fwd_ms'] / r['rev_ms']:>11.0f}x"
              f"{r['loop_ms'] / r['vec_ms']:>12.2f}x")
    plot(rows)


def plot(rows):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURF, sharey=True)
    P = np.array([r["n_sens"] for r in rows])
    price = np.array([r["price_ms"] for r in rows])
    for ax in (a1, a2):
        ax.plot(P, 2 * P + 1, ":", color=MUTED, lw=1.5, label="2P + 1 pricings")

    for m in ("fwd", "vec", "rev", "loop"):
        y = np.array([r[f"{m}_x"] for r in rows])
        lo = y - np.array([r[f"{m}_q25_ms"] for r in rows]) / price
        hi = np.array([r[f"{m}_q75_ms"] for r in rows]) / price - y
        a1.errorbar(P, y, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt=STYLE[m],
                    color=COLORS[m], lw=2, ms=4.5, elinewidth=1.2, capsize=0, label=LABELS[m])
    a1.set_title("Wall-clock time", loc="left", fontsize=11)

    for m, label in (("fwd", LABELS["fwd"]), ("rev", LABELS["rev"]),
                     ("loop", "Bump-and-reprice, loop or vmap (same arithmetic)")):
        a2.plot(P, [r[f"{m}_flops_x"] for r in rows], STYLE[m], color=COLORS[m], lw=2,
                ms=4.5, label=label)
    a2.set_title("Arithmetic (XLA flop count)", loc="left", fontsize=11)

    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor(SURF)
        ax.xaxis.set_major_locator(FixedLocator(P))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{r['n_sens']:,}\nd={r['d']}" for r in rows])
        ax.set_xlabel("Number of sensitivities P (deltas + vegas + correlations), d assets")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    yt = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    a1.yaxis.set_major_locator(FixedLocator(yt))
    a1.yaxis.set_minor_locator(NullLocator())
    a1.set_yticklabels([f"{t:,}" for t in yt])
    a1.set_ylim(1, 6e4)     # headroom above the data so the legend never touches a line
    a1.set_ylabel("Cost, in units of one pricing")
    fig.suptitle(f"Basket call: cost of all P sensitivities, {N:,} paths "
                 f"(IQR error bars are drawn but smaller than the markers)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "e6_basket.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e6_basket.png'}")


def plot_from_csv():
    with (RESULTS / "e6_basket.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["d"], r["n_sens"] = int(r["d"]), int(r["n_sens"])
    plot(rows)


if __name__ == "__main__":
    import sys

    plot_from_csv() if "--plot-only" in sys.argv else main()
