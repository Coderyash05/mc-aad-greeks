"""E3 + E4: fixing autodiff gamma, with paired-bootstrap confidence intervals.

E3 compares gamma estimators over REPS independent batches of N paths (the same
evaluation random numbers as E2, so untuned results are unchanged):
  naive autodiff, FD with common random numbers, likelihood ratio (LR), mixed
  pathwise-LR, and softplus-smoothed autodiff.
The two tuned methods (FD bump h, smoothing width eps) are tuned OUT OF SAMPLE:
h and eps are chosen by RMSE on separate calibration batches, then evaluated on the
evaluation batches. The in-sample choice (best on the evaluation batches) is also
reported, to show the selection bias.
Every RMSE gets a 95% paired-bootstrap CI (mcgreeks.stats, 2,000 resamples of the
500 batches, jointly for all methods), and every method an RMSE ratio vs the best
method with its CI. "A beats B" is claimed only if that ratio CI excludes 1.
Naive autodiff is identically 0, so it is reported by its bias, not in a ratio.
E4 sweeps eps on the evaluation batches: bias falls, variance rises.

Run:  python experiments/e3_gamma.py
Writes results/e3_gamma.csv, results/e3_pairwise.csv, results/e3_tuning.csv,
results/e4_smoothing.csv, results/e3_e4_gamma.png.
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
from mcgreeks.greeks_lr import (lr_gamma_samples, pwlr_gamma_samples,
                                smooth_gamma_samples)
from mcgreeks.pricer import discounted_payoffs, mc_price
from mcgreeks.stats import paired_bootstrap, tune, verdict

S0, SIGMA, R, T, K = 100.0, 0.2, 0.05, 1.0, 100.0
REPS, N = 500, 20_000
H_GRID = np.logspace(-4, 1.6, 29)       # same bump grid as E2
EPS_GRID = np.logspace(-2, 1.2, 17)
BOOT_SEED = 3
CAL_KEY = jax.random.PRNGKey(zlib.crc32(b"e2-e5-calibration"))
RESULTS = Path(__file__).resolve().parents[1] / "results"

C_SMOOTH, C_LR, C_PWLR, C_FD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def per_batch(samples):
    return np.asarray(samples.reshape(REPS, N).mean(axis=-1))


def stats(est, exact):
    est = np.asarray(est)
    return {"mean": float(est.mean()), "bias": float(est.mean() - exact),
            "std": float(est.std(ddof=1)), "rmse": float(np.sqrt(np.mean((est - exact) ** 2)))}


def estimates(Z):
    """Per-batch estimates of every method on the batches Z (REPS, N)."""
    Zf = Z.reshape(-1)
    price = jax.jit(lambda s: discounted_payoffs(s, SIGMA, R, T, K, Z).mean(-1))
    p0 = price(S0)
    fd = np.stack([np.asarray((price(S0 + h) - 2 * p0 + price(S0 - h)) / h**2) for h in H_GRID])
    sm = np.stack([per_batch(smooth_gamma_samples(S0, SIGMA, R, T, K, Zf, float(e)))
                   for e in EPS_GRID])
    naive = np.asarray(jax.vmap(jax.grad(jax.grad(mc_price, 0), 0), in_axes=(None,) * 5 + (0,))(
        S0, SIGMA, R, T, K, Z))
    return {"fd": fd, "smooth": sm, "naive": naive,
            "lr": per_batch(lr_gamma_samples(S0, SIGMA, R, T, K, Zf)),
            "pwlr": per_batch(pwlr_gamma_samples(S0, SIGMA, R, T, K, Zf))}


def main():
    g_ex = float(bs_greeks(S0, K, R, SIGMA, T)["gamma"])
    key = jax.random.split(jax.random.PRNGKey(2026), 4)[0]   # same Z as E2 (CRN key)
    ev = estimates(jax.random.normal(key, (REPS, N), dtype=jnp.float64))
    cal = estimates(jax.random.normal(CAL_KEY, (REPS, N), dtype=jnp.float64))

    t_fd = tune(H_GRID, cal["fd"], ev["fd"], g_ex)
    t_sm = tune(EPS_GRID, cal["smooth"], ev["smooth"], g_ex)
    names = {"fd": f"FD, CRN (h = {t_fd['value']:.3g})", "lr": "Likelihood ratio",
             "pwlr": "Pathwise-LR (mixed)", "smooth": f"Autodiff, smoothed (eps = {t_sm['value']:.2g})"}
    est = {"fd": ev["fd"][t_fd["index"]], "lr": ev["lr"], "pwlr": ev["pwlr"],
           "smooth": ev["smooth"][t_sm["index"]]}
    boot = paired_bootstrap({names[m]: est[m] - g_ex for m in est}, seed=BOOT_SEED)

    # all pairwise ratios, same resamples (same seed) for every reference
    pairwise = []
    for ref in names.values():
        b = paired_bootstrap({names[m]: est[m] - g_ex for m in est}, seed=BOOT_SEED, reference=ref)
        for m, r in b["methods"].items():
            if m != ref:
                pairwise.append({"method": m, "vs": ref, "ratio": r["ratio"],
                                 "ci_lo": r["ratio_ci"][0], "ci_hi": r["ratio_ci"][1],
                                 "verdict": verdict(b, m)})

    rows = [{"estimator": "Autodiff, naive", **stats(ev["naive"], g_ex), "rmse_ci_lo": "",
             "rmse_ci_hi": "", "ratio_vs_best": "", "ratio_ci_lo": "", "ratio_ci_hi": "",
             "verdict": "identically 0 (bias = -gamma); not ranked"}]
    for m, name in names.items():
        r = boot["methods"][name]
        rows.append({"estimator": name, **stats(est[m], g_ex), "rmse_ci_lo": r["ci"][0],
                     "rmse_ci_hi": r["ci"][1], "ratio_vs_best": r["ratio"],
                     "ratio_ci_lo": r["ratio_ci"][0], "ratio_ci_hi": r["ratio_ci"][1],
                     "verdict": verdict(boot, name)})
    tuning = [{"method": "FD, CRN", "parameter": "h", "chosen_out_of_sample": t_fd["value"],
               "rmse_out_of_sample": t_fd["rmse_out"], "chosen_in_sample": t_fd["value_in"],
               "rmse_in_sample": t_fd["rmse_in"], "rmse_on_calibration": t_fd["rmse_cal"]},
              {"method": "Autodiff, smoothed", "parameter": "eps",
               "chosen_out_of_sample": t_sm["value"], "rmse_out_of_sample": t_sm["rmse_out"],
               "chosen_in_sample": t_sm["value_in"], "rmse_in_sample": t_sm["rmse_in"],
               "rmse_on_calibration": t_sm["rmse_cal"]}]
    sweep = [{"eps": float(e), **stats(ev["smooth"][i], g_ex)} for i, e in enumerate(EPS_GRID)]

    RESULTS.mkdir(exist_ok=True)
    for fname, data in (("e3_gamma.csv", rows), ("e3_pairwise.csv", pairwise),
                        ("e3_tuning.csv", tuning), ("e4_smoothing.csv", sweep)):
        with (RESULTS / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"E3: gamma of an ATM call, exact = {g_ex:.5f}; {REPS} batches x {N:,} paths; "
          f"95% paired-bootstrap CIs (2,000 resamples)\n")
    print(f"{'estimator':<32}{'bias':>11}{'RMSE [95% CI]':>28}{'x best [95% CI]':>24}   verdict")
    for r in rows:
        if r["rmse_ci_lo"] == "":
            print(f"{r['estimator']:<32}{r['bias']:>11.2e}{r['rmse']:>12.5f}{'':>16}{'':>24}   "
                  f"{r['verdict']}")
            continue
        print(f"{r['estimator']:<32}{r['bias']:>11.2e}{r['rmse']:>12.5f} [{r['rmse_ci_lo']:.5f}, "
              f"{r['rmse_ci_hi']:.5f}]{r['ratio_vs_best']:>9.2f} [{r['ratio_ci_lo']:.2f}, "
              f"{r['ratio_ci_hi']:.2f}]   {r['verdict']}")
    print("\nPairwise RMSE ratios (method / vs):")
    for p in pairwise:
        if p["method"] < p["vs"]:
            print(f"  {p['method']:<32} / {p['vs']:<32} {p['ratio']:5.2f} [{p['ci_lo']:.2f}, "
                  f"{p['ci_hi']:.2f}]  {p['verdict']}")
    print("\nTuning (chosen on calibration batches, RMSE on evaluation batches):")
    for t in tuning:
        print(f"  {t['method']:<20} {t['parameter']}: out-of-sample {t['chosen_out_of_sample']:.3g} "
              f"-> RMSE {t['rmse_out_of_sample']:.5f}   in-sample {t['chosen_in_sample']:.3g} "
              f"-> RMSE {t['rmse_in_sample']:.5f}")
    print("\nE4: smoothing width sweep (evaluation batches)")
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
    a2.loglog([t_sm["value"]], [t_sm["rmse_out"]], "o", color=C_SMOOTH, ms=10, mfc="none",
              mew=2, label="eps chosen on calibration batches")
    for m, c, ls in (("fd", C_FD, ":"), ("lr", C_LR, "-."), ("pwlr", C_PWLR, "--")):
        r = boot["methods"][names[m]]
        a2.axhline(r["rmse"], color=c, lw=2, ls=ls, label=names[m])
        a2.axhspan(r["ci"][0], r["ci"][1], color=c, alpha=0.12, lw=0)
    a2.set_title("Gamma RMSE vs smoothing width (bands: 95% CI)", loc="left", fontsize=11)
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
                 f"(naive autodiff RMSE = {rows[0]['rmse']:.4f}, always 0)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_e4_gamma.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e3_e4_gamma.png'}")


if __name__ == "__main__":
    main()
