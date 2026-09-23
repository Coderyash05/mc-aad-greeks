"""E6b: the forward-mode regime, many outputs and one input (the mirror image of E6).

K European calls with strikes spread over 70..130 (K = 1: strike 100), one spot
S0, N paths shared by all strikes. The K deltas dV_k/dS0 form one Jacobian column:
  - forward mode: ONE jvp with tangent dS0 = 1 gives all K, cost ~constant in K
  - reverse mode: one pullback per output (cotangent e_k), as jax.jacrev does,
    cost ~linear in K
Costs are in units of one pricing of all K strikes (itself ~linear in K), so
"flat" means "a constant multiple of just pricing the strikes".
Reverse pullbacks are chunked like E6 (lax.map with batch_size) so no batched
cotangent array exceeds MEM_BUDGET; flops come from the unchunked program.

Run:  python experiments/e6b_forward_regime.py      (~7 minutes; K = 1000 reverse ~90 s/run)
      python experiments/e6b_forward_regime.py --plot-only
Writes results/e6b_forward_regime.csv, results/e6b_hardware.json, results/e6b_forward_regime.png.
"""
import csv
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

import mcgreeks  # noqa: F401
from mcgreeks.bench import cost, print_hardware, summary, temp_bytes, time_runs
from mcgreeks.greeks_ad import strike_deltas_fwd, strike_deltas_rev
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price_strikes

S0, SIGMA, R, T, N = 100.0, 0.2, 0.05, 1.0, 100_000
KS = [1, 10, 100, 1000]
MEM_BUDGET = 256e6          # bytes per batched (chunk, N, K) float64 cotangent array
RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "e6b_forward_regime.csv"

# Same method colours as E6: forward AD slot 1, reverse AD the autodiff green.
C_FWD, C_REV = "#2a78d6", "#1baf7a"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def strikes_for(K):
    return jnp.array([100.0]) if K == 1 else jnp.linspace(70.0, 130.0, K)


def chunk_for(K):
    c = max(1, int(MEM_BUDGET // (N * K * 8)))
    return None if c >= K else c


def main():
    hw = print_hardware()
    Z = normals(jax.random.PRNGKey(6), N)
    print(f"N = {N:,} paths, strikes 70..130, memory budget {MEM_BUDGET / 1e6:.0f} MB\n")
    rows = []
    for K in KS:
        a = (S0, SIGMA, R, T, strikes_for(K), Z)
        chunk = chunk_for(K)

        # Correctness first; these calls also compile, so timing skips its warm-up.
        _, d_fwd = strike_deltas_fwd(*a)
        _, d_rev = strike_deltas_rev(*a, chunk=chunk)
        err = float(jnp.max(jnp.abs(d_fwd - d_rev)))
        assert err < 1e-12, (K, err)
        mc_price_strikes(*a).block_until_ready()

        row = {"K": K, "chunk_rev": chunk or K}
        for m, fn in (("price", lambda: mc_price_strikes(*a).block_until_ready()),
                      ("fwd", lambda: strike_deltas_fwd(*a)[1].block_until_ready()),
                      ("rev", lambda: strike_deltas_rev(*a, chunk=chunk)[1].block_until_ready())):
            med, q25, q75, n = summary(time_runs(fn, min_reps=3, max_reps=100, budget_s=2.0,
                                                 warm=False))
            row.update({f"{m}_ms": med * 1e3, f"{m}_q25_ms": q25 * 1e3, f"{m}_q75_ms": q75 * 1e3,
                        f"{m}_runs": n})
        for m, fc in (("price", cost(mc_price_strikes, *a)), ("fwd", cost(strike_deltas_fwd, *a)),
                      ("rev", cost(strike_deltas_rev, *a, chunk=None))):
            row[f"{m}_flops"], row[f"{m}_bytes"] = fc["flops"], fc["bytes"]
        row["fwd_temp_mb"] = temp_bytes(strike_deltas_fwd, *a) / 1e6
        row["rev_temp_mb"] = temp_bytes(strike_deltas_rev, *a, chunk=chunk) / 1e6
        for m in ("fwd", "rev"):
            row[f"{m}_x"] = row[f"{m}_ms"] / row["price_ms"]
            row[f"{m}_flops_x"] = row[f"{m}_flops"] / row["price_flops"]
        row["max_err_fwd_vs_rev"] = err
        rows.append(row)
        print(f"K={K:>4} | price {row['price_ms']:8.2f} ms | x one pricing, median [IQR]: "
              f"fwd {row['fwd_x']:6.1f} [{row['fwd_q25_ms'] / row['price_ms']:.1f}, "
              f"{row['fwd_q75_ms'] / row['price_ms']:.1f}]  rev {row['rev_x']:8.1f} "
              f"[{row['rev_q25_ms'] / row['price_ms']:.1f}, {row['rev_q75_ms'] / row['price_ms']:.1f}]"
              f" | flops x: fwd {row['fwd_flops_x']:5.2f}  rev {row['rev_flops_x']:7.1f}"
              f" | temp MB: fwd {row['fwd_temp_mb']:.0f}, rev {row['rev_temp_mb']:.0f}"
              f" | max|fwd-rev| {err:.1e}")

    RESULTS.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "e6b_hardware.json").write_text(json.dumps(hw, indent=2))
    plot(rows)


def plot(rows):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF, sharey=True)
    K = np.array([r["K"] for r in rows])
    price = np.array([r["price_ms"] for r in rows])
    for ax in (a1, a2):
        ax.plot(K, K, ":", color=MUTED, lw=1.5, label="K pricings")
    for m, c, lab in (("fwd", C_FWD, "Forward-mode AD (one jvp)"),
                      ("rev", C_REV, "Reverse-mode AD (K vjps, = jax.jacrev)")):
        y = np.array([r[f"{m}_x"] for r in rows])
        lo = y - np.array([r[f"{m}_q25_ms"] for r in rows]) / price
        hi = np.array([r[f"{m}_q75_ms"] for r in rows]) / price - y
        a1.errorbar(K, y, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="-o", color=c, lw=2,
                    ms=4.5, elinewidth=1.2, capsize=0, label=lab)
        a2.plot(K, [r[f"{m}_flops_x"] for r in rows], "-o", color=c, lw=2, ms=4.5, label=lab)
    a1.set_title("Wall-clock time", loc="left", fontsize=11)
    a2.set_title("Arithmetic (XLA flop count)", loc="left", fontsize=11)
    yt = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor(SURF)
        ax.xaxis.set_major_locator(FixedLocator(K))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{k:,}" for k in K])
        ax.set_xlabel("Number of outputs K (strikes), one input (S0)")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    a1.yaxis.set_major_locator(FixedLocator(yt))
    a1.yaxis.set_minor_locator(NullLocator())
    a1.set_yticklabels([f"{t:,}" for t in yt])
    top = max(max(r["rev_x"], r["rev_flops_x"]) for r in rows)
    a1.set_ylim(0.8, top * 30)   # headroom for the legend
    a1.set_ylabel("Cost, in units of one pricing of all K strikes")
    fig.suptitle(f"Many outputs, one input: all K call deltas, {N:,} paths "
                 f"(IQR error bars are smaller than the markers)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "e6b_forward_regime.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e6b_forward_regime.png'}")


def plot_from_csv():
    with OUT.open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["K"] = int(r["K"])
    plot(rows)


if __name__ == "__main__":
    plot_from_csv() if "--plot-only" in sys.argv else main()
