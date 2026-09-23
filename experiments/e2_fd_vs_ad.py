"""E2: finite-difference error vs bump size h, against autodiff.

For each h, estimate delta and gamma REPS times on independent batches of N paths,
then measure RMSE against the closed form. Two finite-difference variants:
  - independent seeds: fresh random numbers for every repricing
  - CRN: the same random numbers for every repricing
Autodiff delta has no h, so it is a flat line. Autodiff gamma is exactly 0
(the trap we fix next), so its error equals the true gamma: also a flat line.

The headline (best-h FD delta vs autodiff delta) chooses h OUT OF SAMPLE, on
separate calibration batches, and reports 95% paired-bootstrap CIs.

Run:  python experiments/e2_fd_vs_ad.py
Writes results/e2_fd_vs_ad.csv, results/e2_headline.csv and results/e2_fd_vs_ad.png.
"""
import csv
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks
from mcgreeks.pricer import discounted_payoffs, mc_price
from mcgreeks.stats import paired_bootstrap, tune, verdict

CAL_KEY = jax.random.PRNGKey(zlib.crc32(b"e2-e5-calibration"))
S0, SIGMA, R, T, K = 100.0, 0.2, 0.05, 1.0, 100.0
REPS, N = 500, 20_000
H_GRID = np.logspace(-4, 1.6, 29)  # 1e-4 .. ~40 (price units, S0 = 100)
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Colors: fixed categorical order, validated palette (slot 1 blue, 2 orange, 3 aqua).
C_INDEP, C_CRN, C_AD = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


@jax.jit
def batch_price(s0, Z):
    """Price for every row of Z (shape REPS x N) at spot s0 -> shape (REPS,)."""
    return discounted_payoffs(s0, SIGMA, R, T, K, Z).mean(axis=-1)


# Pathwise (autodiff) delta and gamma, one per batch
ad_delta_fn = jax.jit(jax.vmap(jax.grad(mc_price, argnums=0), in_axes=(None,) * 5 + (0,)))
ad_gamma_fn = jax.jit(jax.vmap(jax.grad(jax.grad(mc_price, argnums=0), argnums=0),
                               in_axes=(None,) * 5 + (0,)))


def rmse(est, exact):
    return float(jnp.sqrt(jnp.mean((est - exact) ** 2)))


def per_h(Z, Zu, Zm, Zd):
    """Per-batch FD estimates for every h: arrays (len(H_GRID), REPS)."""
    mid_c, mid_i = batch_price(S0, Z), batch_price(S0, Zm)
    out = {"delta_crn": [], "delta_indep": [], "gamma_crn": [], "gamma_indep": []}
    for h in H_GRID:
        up_c, dn_c = batch_price(S0 + h, Z), batch_price(S0 - h, Z)
        up_i, dn_i = batch_price(S0 + h, Zu), batch_price(S0 - h, Zd)
        out["delta_crn"].append((up_c - dn_c) / (2 * h))
        out["delta_indep"].append((up_i - dn_i) / (2 * h))
        out["gamma_crn"].append((up_c - 2 * mid_c + dn_c) / h**2)
        out["gamma_indep"].append((up_i - 2 * mid_i + dn_i) / h**2)
    return {k: np.asarray(jnp.stack(v)) for k, v in out.items()}


def headline(Z, Zu, Zm, Zd, ad_d, d_ex, g_ex):
    """Best-h FD vs autodiff, with h chosen OUT OF SAMPLE on separate calibration
    batches, and 95% paired-bootstrap CIs (mcgreeks.stats). Writes e2_headline.csv."""
    cal = per_h(*(jax.random.normal(k, (REPS, N), dtype=jnp.float64)
                  for k in jax.random.split(CAL_KEY, 4)))
    ev = per_h(Z, Zu, Zm, Zd)
    t = {k: tune(H_GRID, cal[k], ev[k], d_ex if k.startswith("delta") else g_ex) for k in ev}
    errs = {f"FD CRN (h = {t['delta_crn']['value']:.3g})": t["delta_crn"]["errors"],
            f"FD independent (h = {t['delta_indep']['value']:.3g})": t["delta_indep"]["errors"],
            "Autodiff (pathwise)": np.asarray(ad_d) - d_ex}
    boot = paired_bootstrap(errs, seed=2, reference="Autodiff (pathwise)")
    rows = []
    print(f"E2 headline, delta (h chosen on calibration batches; 95% paired-bootstrap CIs):")
    for name, r in boot["methods"].items():
        rows.append({"greek": "delta", "estimator": name, "rmse": r["rmse"], "ci_lo": r["ci"][0],
                     "ci_hi": r["ci"][1], "ratio_vs_ad": r["ratio"], "ratio_ci_lo": r["ratio_ci"][0],
                     "ratio_ci_hi": r["ratio_ci"][1], "verdict": verdict(boot, name)})
        print(f"  {name:<30} RMSE {r['rmse']:.5f} [{r['ci'][0]:.5f}, {r['ci'][1]:.5f}]   "
              f"x AD {r['ratio']:.3f} [{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}]   "
              f"{verdict(boot, name)}")
    for k, v in t.items():
        rows.append({"greek": k, "estimator": "tuning", "rmse": v["rmse_out"],
                     "h_out_of_sample": v["value"], "h_in_sample": v["value_in"],
                     "rmse_in_sample": v["rmse_in"]})
        print(f"  tuning {k:<12} h out-of-sample {v['value']:.3g} -> RMSE {v['rmse_out']:.5f};"
              f"  in-sample {v['value_in']:.3g} -> RMSE {v['rmse_in']:.5f}")
    print()
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (RESULTS / "e2_headline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    exact = bs_greeks(S0, K, R, SIGMA, T)
    d_ex, g_ex = float(exact["delta"]), float(exact["gamma"])

    k_crn, k_up, k_mid, k_dn = jax.random.split(jax.random.PRNGKey(2026), 4)
    shape = (REPS, N)
    Z = jax.random.normal(k_crn, shape, dtype=jnp.float64)       # shared (CRN)
    Zu = jax.random.normal(k_up, shape, dtype=jnp.float64)       # independent
    Zm = jax.random.normal(k_mid, shape, dtype=jnp.float64)
    Zd = jax.random.normal(k_dn, shape, dtype=jnp.float64)

    ad_d = ad_delta_fn(S0, SIGMA, R, T, K, Z)
    ad_g = ad_gamma_fn(S0, SIGMA, R, T, K, Z)
    ad_delta_rmse, ad_gamma_rmse = rmse(ad_d, d_ex), rmse(ad_g, g_ex)

    v_mid_crn, v_mid_ind = batch_price(S0, Z), batch_price(S0, Zm)
    rows = []
    for h in H_GRID:
        up_c, dn_c = batch_price(S0 + h, Z), batch_price(S0 - h, Z)
        up_i, dn_i = batch_price(S0 + h, Zu), batch_price(S0 - h, Zd)
        rows.append({
            "h": h,
            "delta_indep": rmse((up_i - dn_i) / (2 * h), d_ex),
            "delta_crn": rmse((up_c - dn_c) / (2 * h), d_ex),
            "gamma_indep": rmse((up_i - 2 * v_mid_ind + dn_i) / h**2, g_ex),
            "gamma_crn": rmse((up_c - 2 * v_mid_crn + dn_c) / h**2, g_ex),
        })

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "e2_fd_vs_ad.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["ad_delta", "ad_gamma"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "ad_delta": ad_delta_rmse, "ad_gamma": ad_gamma_rmse})

    headline(Z, Zu, Zm, Zd, ad_d, d_ex, g_ex)

    # ---- summary -------------------------------------------------------------
    print(f"Setup: S0=K=100, sigma=0.2, r=0.05, T=1; {REPS} batches x {N:,} paths")
    print(f"Exact delta {d_ex:.4f}, exact gamma {g_ex:.5f}\n")
    print(f"{'estimator':<24}{'best h':>10}{'RMSE':>12}")
    for key, label in (("delta_indep", "delta, FD independent"),
                       ("delta_crn", "delta, FD CRN")):
        best = min(rows, key=lambda r: r[key])
        print(f"{label:<24}{best['h']:>10.4g}{best[key]:>12.5f}")
    print(f"{'delta, autodiff':<24}{'-':>10}{ad_delta_rmse:>12.5f}")
    for key, label in (("gamma_indep", "gamma, FD independent"),
                       ("gamma_crn", "gamma, FD CRN")):
        best = min(rows, key=lambda r: r[key])
        print(f"{label:<24}{best['h']:>10.4g}{best[key]:>12.5f}")
    print(f"{'gamma, autodiff (naive)':<24}{'-':>10}{ad_gamma_rmse:>12.5f}"
          f"   <- mean estimate {float(ad_g.mean()):.1f}: the gamma trap")

    # ---- figure ----------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#fcfcfb")
    h = [r["h"] for r in rows]
    panels = (
        (axes[0], "delta", ad_delta_rmse, "Autodiff (pathwise)", "Delta"),
        (axes[1], "gamma", ad_gamma_rmse, "Autodiff, naive (returns 0)", "Gamma"),
    )
    for ax, g, ad_err, ad_label, title in panels:
        ax.set_facecolor("#fcfcfb")
        ax.loglog(h, [r[f"{g}_indep"] for r in rows], "-o", color=C_INDEP, lw=2, ms=4,
                  label="Finite difference, independent seeds")
        ax.loglog(h, [r[f"{g}_crn"] for r in rows], "-o", color=C_CRN, lw=2, ms=4,
                  label="Finite difference, common random numbers")
        ax.axhline(ad_err, color=C_AD, lw=2, ls="--", label=ad_label)
        ax.set_title(f"{title}: RMSE vs bump size", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Bump size h (spot units, S0 = 100)")
        ax.set_ylabel("RMSE vs Black-Scholes")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(f"Monte Carlo Greeks, ATM call, {REPS} batches of {N:,} paths",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "e2_fd_vs_ad.png", dpi=160, facecolor=fig.get_facecolor())
    print(f"\nSaved {RESULTS / 'e2_fd_vs_ad.png'}")


if __name__ == "__main__":
    main()
