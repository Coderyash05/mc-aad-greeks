"""E5: delta of a digital (cash-or-nothing) call, with paired-bootstrap CIs.

The payoff jumps from 0 to 1 at the strike, so its derivative is 0 on every path:
autodiff (pathwise) delta is exactly 0. Compared: FD with CRN (bump h), sigmoid-
smoothed autodiff (width eps) and the likelihood ratio (no parameter). Same
evaluation random numbers as E2/E3.
h and eps are tuned OUT OF SAMPLE (chosen on separate calibration batches), with
the in-sample choice reported alongside. RMSEs get 95% paired-bootstrap CIs, and
ratios vs the best method get CIs; "A beats B" only if the ratio CI excludes 1.
Naive autodiff is identically 0: reported by its bias, not ranked.

Run:  python experiments/e5_digital.py
Writes results/e5_digital.csv (width sweep), results/e5_summary.csv,
results/e5_pairwise.csv, results/e5_tuning.csv, results/e5_digital.png.
"""
import csv
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_digital_delta
from mcgreeks.greeks_lr import lr_delta_samples, smooth_digital_delta_samples
from mcgreeks.pricer import discounted_payoffs, mc_price
from mcgreeks.stats import paired_bootstrap, tune, verdict

S0, SIGMA, R, T, K = 100.0, 0.2, 0.05, 1.0, 100.0
REPS, N = 500, 20_000
WIDTHS = np.logspace(-2, 1.2, 17)   # bump size h or smoothing width eps, spot units
BOOT_SEED = 5
CAL_KEY = jax.random.PRNGKey(zlib.crc32(b"e2-e5-calibration"))
RESULTS = Path(__file__).resolve().parents[1] / "results"

C_SMOOTH, C_LR, C_FD, C_NAIVE = "#2a78d6", "#eb6834", "#eda100", "#1baf7a"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def stats(est, exact):
    est = np.asarray(est)
    return {"mean": float(est.mean()), "bias": float(est.mean() - exact),
            "std": float(est.std(ddof=1)), "rmse": float(np.sqrt(np.mean((est - exact) ** 2)))}


def estimates(Z):
    Zf = Z.reshape(-1)

    def per_batch(x):
        return np.asarray(x.reshape(REPS, N).mean(axis=-1))

    price = jax.jit(lambda s: discounted_payoffs(s, SIGMA, R, T, K, Z, kind="digital").mean(-1))
    fd = np.stack([np.asarray((price(S0 + w) - price(S0 - w)) / (2 * w)) for w in WIDTHS])
    sm = np.stack([per_batch(smooth_digital_delta_samples(S0, SIGMA, R, T, K, Zf, float(w)))
                   for w in WIDTHS])
    naive = np.asarray(jax.vmap(jax.grad(lambda s, z: mc_price(s, SIGMA, R, T, K, z,
                                                               kind="digital")),
                                in_axes=(None, 0))(S0, Z))
    return {"fd": fd, "smooth": sm, "naive": naive,
            "lr": per_batch(lr_delta_samples(S0, SIGMA, R, T, K, Zf, kind="digital"))}


def main():
    d_ex = float(bs_digital_delta(S0, K, R, SIGMA, T))
    key = jax.random.split(jax.random.PRNGKey(2026), 4)[0]
    ev = estimates(jax.random.normal(key, (REPS, N), dtype=jnp.float64))
    cal = estimates(jax.random.normal(CAL_KEY, (REPS, N), dtype=jnp.float64))

    t_fd = tune(WIDTHS, cal["fd"], ev["fd"], d_ex)
    t_sm = tune(WIDTHS, cal["smooth"], ev["smooth"], d_ex)
    names = {"fd": f"FD, CRN (h = {t_fd['value']:.2g})",
             "smooth": f"Autodiff, sigmoid (eps = {t_sm['value']:.2g})", "lr": "Likelihood ratio"}
    est = {"fd": ev["fd"][t_fd["index"]], "smooth": ev["smooth"][t_sm["index"]], "lr": ev["lr"]}
    errs = {names[m]: est[m] - d_ex for m in est}
    boot = paired_bootstrap(errs, seed=BOOT_SEED)

    pairwise = []
    for ref in names.values():
        b = paired_bootstrap(errs, seed=BOOT_SEED, reference=ref)
        for m, r in b["methods"].items():
            if m != ref:
                pairwise.append({"method": m, "vs": ref, "ratio": r["ratio"],
                                 "ci_lo": r["ratio_ci"][0], "ci_hi": r["ratio_ci"][1],
                                 "verdict": verdict(b, m)})
    summary = [{"estimator": "Autodiff, naive (pathwise)", **stats(ev["naive"], d_ex),
                "rmse_ci_lo": "", "rmse_ci_hi": "", "ratio_vs_best": "", "ratio_ci_lo": "",
                "ratio_ci_hi": "", "verdict": "identically 0 (bias = -delta); not ranked"}]
    for m, name in names.items():
        r = boot["methods"][name]
        summary.append({"estimator": name, **stats(est[m], d_ex), "rmse_ci_lo": r["ci"][0],
                        "rmse_ci_hi": r["ci"][1], "ratio_vs_best": r["ratio"],
                        "ratio_ci_lo": r["ratio_ci"][0], "ratio_ci_hi": r["ratio_ci"][1],
                        "verdict": verdict(boot, name)})
    tuning = [{"method": meth, "parameter": par, "chosen_out_of_sample": t["value"],
               "rmse_out_of_sample": t["rmse_out"], "chosen_in_sample": t["value_in"],
               "rmse_in_sample": t["rmse_in"], "rmse_on_calibration": t["rmse_cal"]}
              for meth, par, t in (("FD, CRN", "h", t_fd), ("Autodiff, sigmoid", "eps", t_sm))]
    sweep = [{"width": float(w), **{f"fd_{k}": v for k, v in stats(ev["fd"][i], d_ex).items()},
              **{f"smooth_{k}": v for k, v in stats(ev["smooth"][i], d_ex).items()}}
             for i, w in enumerate(WIDTHS)]

    RESULTS.mkdir(exist_ok=True)
    for fname, data in (("e5_digital.csv", sweep), ("e5_summary.csv", summary),
                        ("e5_pairwise.csv", pairwise), ("e5_tuning.csv", tuning)):
        with (RESULTS / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"E5: delta of an ATM digital call, exact = {d_ex:.5f}; {REPS} batches x {N:,} paths; "
          f"95% paired-bootstrap CIs\n")
    print(f"{'estimator':<34}{'bias':>11}{'RMSE [95% CI]':>28}{'x best [95% CI]':>24}   verdict")
    for r in summary:
        if r["rmse_ci_lo"] == "":
            print(f"{r['estimator']:<34}{r['bias']:>11.2e}{r['rmse']:>12.5f}{'':>40}   {r['verdict']}")
            continue
        print(f"{r['estimator']:<34}{r['bias']:>11.2e}{r['rmse']:>12.5f} [{r['rmse_ci_lo']:.5f}, "
              f"{r['rmse_ci_hi']:.5f}]{r['ratio_vs_best']:>9.2f} [{r['ratio_ci_lo']:.2f}, "
              f"{r['ratio_ci_hi']:.2f}]   {r['verdict']}")
    print("\nPairwise RMSE ratios (method / vs):")
    for p in pairwise:
        print(f"  {p['method']:<34} / {p['vs']:<34} {p['ratio']:5.2f} [{p['ci_lo']:.2f}, "
              f"{p['ci_hi']:.2f}]  {p['verdict']}")
    print("\nTuning (chosen on calibration batches, RMSE on evaluation batches):")
    for t in tuning:
        print(f"  {t['method']:<20} {t['parameter']}: out-of-sample {t['chosen_out_of_sample']:.3g} "
              f"-> RMSE {t['rmse_out_of_sample']:.5f}   in-sample {t['chosen_in_sample']:.3g} "
              f"-> RMSE {t['rmse_in_sample']:.5f}")

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
    ax.loglog([t_sm["value"], t_fd["value"]], [t_sm["rmse_out"], t_fd["rmse_out"]], "o",
              color=INK, ms=10, mfc="none", mew=1.5, label="Chosen on calibration batches")
    lr = boot["methods"][names["lr"]]
    ax.axhline(lr["rmse"], color=C_LR, lw=2, ls="-.", label="Likelihood ratio (band: 95% CI)")
    ax.axhspan(lr["ci"][0], lr["ci"][1], color=C_LR, alpha=0.15, lw=0)
    ax.axhline(summary[0]["rmse"], color=C_NAIVE, lw=2, ls="--",
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
