"""E9: accuracy of every Monte Carlo basket sensitivity against the geometric-basket closed form.

For each sensitivity, z = (MC - exact) / SE with a batch-means SE (200 batches), so
under a correct estimator with a correct SE, z ~ t_199 ~ N(0, 1). Two parts:

(a) The test seeds. Same problems and random numbers as tests/test_basket_geometric.py
    (example_basket, crc32("geo-basket-{d}") / crc32("geo-basket-Z-{d}")), d = 2, 10,
    50, 200,000 paths: max |z| per group and overall.

(b) Calibration over R = 200 independent replications (only Z changes; crc32
    "e9-rep-Z-{d}-{j}"). Within one seed the z's are strongly correlated. For the
    geometric payoff the pathwise derivative is e^{-rT} 1{G_T > K} G_T d(log G_T)/d(theta)
    and d(log G_T)/d(S0_i) = w_i / S0_i does not depend on the path, so all d delta
    z-scores are IDENTICAL. So "P comparisons" overstates the multiple testing, and the
    spread of z within one seed says little about calibration. Across replications
    the z's are independent: mean z^2 (expected 199/197 = 1.010), share |z| > 2
    (expected 4.7%) and mean z (bias), with 95% CIs from a bootstrap over
    replications, and the EMPIRICAL family-wise false-alarm rate of the test "all
    |z| < 4" (and < 3), which accounts for the correlation, next to the independence
    bound 1 - (1 - p)^P.

Run:  python experiments/e9_geometric_basket.py      (about 1 minute)
Writes results/e9_geometric_basket.csv (a), results/e9_calibration.csv (b),
results/e9_zscores.csv (all z of part a), results/e9_qq.png (b).
"""
import csv
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import mcgreeks  # noqa: F401
from mcgreeks.basket import basket_greeks_batches, example_basket, geometric_basket_greeks
from mcgreeks.stats import family_zscores, k_familywise

P4 = 2 * stats.norm.sf(4.0)   # false-alarm probability of one 4-SE comparison

R, T, K = 0.05, 1.0, 100.0
N, N_BATCHES, DS, REPS = 200_000, 200, (2, 10, 50), 200
DOF = N_BATCHES - 1
RESULTS = Path(__file__).resolve().parents[1] / "results"
GROUPS = ("price", "delta", "vega", "corr")

# d is ordinal, so a sequential single-hue ramp (steps 300 / 500 / 700 of the blue).
C_D = {2: "#6da7ec", 10: "#256abf", 50: "#0d366b"}
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def seed(tag):
    return zlib.crc32(tag.encode())


def zscores(d, z_tag):
    """z per group, for the fixed problem example_basket(d) and normals from z_tag."""
    S0, sigma, rho, w = example_basket(d, seed(f"geo-basket-{d}"))
    Z = jax.random.normal(jax.random.PRNGKey(seed(z_tag)), (N, d), dtype=jnp.float64)
    gb = basket_greeks_batches(S0, sigma, rho, R, T, K, w, Z, N_BATCHES, True)
    ex = geometric_basket_greeks(S0, sigma, rho, R, T, K, w)
    out = {}
    for g in GROUPS:
        b = np.asarray(gb[g]).reshape(N_BATCHES, -1)
        m, se = b.mean(0), b.std(0, ddof=1) / np.sqrt(N_BATCHES)
        e = np.atleast_1d(np.asarray(ex[g]))
        out[g] = {"z": (m - e) / se, "exact": e, "mc": m, "se": se, "batches": b}
    out["family"] = family_zscores(np.concatenate([out[g]["batches"] for g in GROUPS], axis=1),
                                   np.concatenate([out[g]["exact"] for g in GROUPS]))
    return out


def p_indep(m, P):
    """P(max of P independent |t_DOF| >= m)."""
    return 1.0 - (1.0 - 2 * stats.t.sf(m, DOF)) ** P


def boot_ci(per_rep, n_boot=2000, rng_seed=0):
    """95% percentile CI of the mean of per-replication statistics (replications are iid)."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(per_rep), (n_boot, len(per_rep)))
    return tuple(np.percentile(per_rep[idx].mean(axis=1), [2.5, 97.5]))


def part_a():
    summary, rows = [], []
    for d in DS:
        zs = zscores(d, f"geo-basket-Z-{d}")
        allz = np.concatenate([zs[g]["z"] for g in GROUPS])
        names = [f"{g}[{i}]" for g in GROUPS for i in range(len(zs[g]["z"]))]
        s = {"d": d, "P": len(allz)}
        s.update({f"max_abs_z_{g}": float(np.max(np.abs(zs[g]["z"]))) for g in GROUPS})
        s.update(max_abs_z=float(np.max(np.abs(allz))), worst=names[int(np.argmax(np.abs(allz)))],
                 distinct_delta_z=len(np.unique(np.round(zs["delta"]["z"], 10))))
        summary.append(s)
        for g in GROUPS:
            for i, z in enumerate(zs[g]["z"]):
                rows.append({"d": d, "greek": g, "index": i, "exact": zs[g]["exact"][i],
                             "mc": zs[g]["mc"][i], "se": zs[g]["se"][i], "z": z})
    return summary, rows


def part_b():
    calib, pooled, agg = [], {}, []
    for d in DS:
        Zr = {g: [] for g in GROUPS}
        t_mean, z_msq, z_norm, chi_f, ind_fail = [], [], [], [], []
        for j in range(REPS):
            zs = zscores(d, f"e9-rep-Z-{d}-{j}")
            for g in GROUPS:
                Zr[g].append(zs[g]["z"])
            f = zs["family"]
            k = 4.0 if d == 2 else k_familywise(f["P"], P4, DOF)
            t_mean.append(f["mean_z"] / f["se_mean_z"])
            z_norm.append((f["mean_z2"] - f["expected_mean_z2"]) / f["sd_mean_z2"])
            z_msq.append(f["mean_z2_score"])
            chi_f.append(f["chi2_dof"])
            ind_fail.append(np.max(np.abs(f["z"])) >= k)
        agg.append(aggregate_row(d, k, np.array(t_mean), np.array(z_msq), np.array(z_norm),
                                 np.array(ind_fail), float(np.median(chi_f))))
        Zr = {g: np.stack(v) for g, v in Zr.items()}            # (REPS, P_g)
        allz = np.concatenate([Zr[g] for g in GROUPS], axis=1)   # (REPS, P)
        pooled[d] = allz.ravel()
        P = allz.shape[1]
        for g in GROUPS + ("all",):
            z = allz if g == "all" else Zr[g]
            z2, gt2, mz = (z**2).mean(1), (np.abs(z) > 2).mean(1), z.mean(1)
            row = {"d": d, "group": g, "P": z.shape[1], "reps": REPS,
                   "mean_z2": z2.mean(), "mean_z2_lo": boot_ci(z2)[0], "mean_z2_hi": boot_ci(z2)[1],
                   "share_gt2": gt2.mean(), "share_gt2_lo": boot_ci(gt2)[0],
                   "share_gt2_hi": boot_ci(gt2)[1],
                   "mean_z": mz.mean(), "mean_z_lo": boot_ci(mz)[0], "mean_z_hi": boot_ci(mz)[1]}
            if g == "all":
                for m in (3.0, 4.0):
                    k = int(np.sum(np.abs(allz).max(1) >= m))
                    lo, hi = stats.beta.ppf([0.025, 0.975], [k, k + 1], [REPS - k + 1, REPS - k])
                    row.update({f"fwer_{m:g}": k / REPS, f"fwer_{m:g}_lo": 0.0 if k == 0 else lo,
                                f"fwer_{m:g}_hi": 1.0 if k == REPS else hi,
                                f"fwer_{m:g}_indep": p_indep(m, P)})
            calib.append(row)
    return calib, pooled, agg


def aggregate_row(d, k, t_mean, z_msq, z_norm, ind_fail, chi_f):
    """Calibration of the family tests over REPS seeds: each aggregate statistic should
    be ~N(0, 1) (t_199 for mean z), so sd ~ 1 and |.| > 2 in ~4.6% of seeds.
    mean_z2_stat is the scaled-chi-square score used by the tests; mean_z2_normal is
    the rejected normal approximation, kept for the before/after."""
    row = {"d": d, "k_individual": k, "reps": REPS, "chi2_dof_median": chi_f,
           "individual_fail_share": float(ind_fail.mean())}
    for name, x in (("mean_z_stat", t_mean), ("mean_z2_stat", z_msq), ("mean_z2_normal", z_norm)):
        row.update({f"{name}_mean": float(x.mean()), f"{name}_sd": float(x.std(ddof=1)),
                    f"{name}_sd_lo": float(x.std(ddof=1) * np.sqrt((REPS - 1) / stats.chi2.ppf(0.975, REPS - 1))),
                    f"{name}_sd_hi": float(x.std(ddof=1) * np.sqrt((REPS - 1) / stats.chi2.ppf(0.025, REPS - 1))),
                    f"{name}_gt2": float(np.mean(np.abs(x) > 2)),
                    f"{name}_max_abs": float(np.max(np.abs(x))),
                    f"{name}_ge4": int(np.sum(np.abs(x) >= 4))})
    return row


def main():
    summary, rows = part_a()
    calib, pooled, agg = part_b()
    RESULTS.mkdir(exist_ok=True)
    for name, data in (("e9_geometric_basket.csv", summary), ("e9_zscores.csv", rows),
                       ("e9_calibration.csv", calib), ("e9_aggregate.csv", agg)):
        with (RESULTS / name).open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            wr.writeheader()
            wr.writerows(data)

    print(f"E9 (a): test seeds; {N:,} paths, {N_BATCHES} batches, z ~ t_{DOF}\n")
    print(f"{'d':>3}{'P':>6}" + "".join(f"{'max|z| ' + g:>15}" for g in GROUPS)
          + f"{'max|z|':>9}{'worst':>11}{'distinct delta z':>18}")
    for s in summary:
        print(f"{s['d']:>3}{s['P']:>6}" + "".join(f"{s['max_abs_z_' + g]:>15.2f}" for g in GROUPS)
              + f"{s['max_abs_z']:>9.2f}{s['worst']:>11}{s['distinct_delta_z']:>18}")

    print(f"\nE9 (b): calibration over {REPS} independent replications per d "
          f"(expected: mean z^2 {DOF / (DOF - 2):.3f}, |z|>2 {2 * stats.t.sf(2, DOF):.1%}, "
          f"mean z 0)\n")
    print(f"{'d':>3} {'group':<6}{'P':>6}{'mean z^2 [95% CI]':>26}{'|z|>2 [95% CI]':>24}"
          f"{'mean z [95% CI]':>26}")
    for c in calib:
        print(f"{c['d']:>3} {c['group']:<6}{c['P']:>6}{c['mean_z2']:>10.3f} [{c['mean_z2_lo']:.3f}, "
              f"{c['mean_z2_hi']:.3f}]{c['share_gt2']:>9.1%} [{c['share_gt2_lo']:.1%}, "
              f"{c['share_gt2_hi']:.1%}]{c['mean_z']:>10.3f} [{c['mean_z_lo']:+.3f}, "
              f"{c['mean_z_hi']:+.3f}]")
    print("\nFamily-wise false alarms (share of replications with any |z| >= m):")
    for c in (c for c in calib if c["group"] == "all"):
        print(f"  d={c['d']:>2}, P={c['P']:>4}:  m=3: {c['fwer_3']:.1%} [{c['fwer_3_lo']:.1%}, "
              f"{c['fwer_3_hi']:.1%}] (independence {c['fwer_3_indep']:.1%})   m=4: "
              f"{c['fwer_4']:.1%} [{c['fwer_4_lo']:.1%}, {c['fwer_4_hi']:.1%}] "
              f"(independence {c['fwer_4_indep']:.2%})")

    print(f"\nE9 (c): family tests over {REPS} seeds (each aggregate statistic should be "
          f"~N(0, 1): sd 1, |.|>2 in 4.6%)")
    for a in agg:
        print(f"  d={a['d']:>2}: individual |z| >= {a['k_individual']:.2f} in "
              f"{a['individual_fail_share']:.1%} of seeds; scaled chi^2 dof (median) "
              f"{a['chi2_dof_median']:.1f}")
        for n, label in (("mean_z_stat", "mean z / SE (t_199)"),
                         ("mean_z2_stat", "mean z^2, chi^2 score"),
                         ("mean_z2_normal", "mean z^2, normal approx.")):
            print(f"        {label:<26} sd {a[n + '_sd']:.2f} [{a[n + '_sd_lo']:.2f}, "
                  f"{a[n + '_sd_hi']:.2f}]  |.|>2 {a[n + '_gt2']:5.1%}  max |.| "
                  f"{a[n + '_max_abs']:.2f}  |.|>=4: {a[n + '_ge4']}")

    # ---- QQ plot of the replication z's --------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(6.6, 5.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    lim = 4.6
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=2, ls="--",
            label="Ideal: z ~ t$_{199}$", zorder=1)
    q = np.linspace(-3.0, 3.0, 37)        # evenly spaced theoretical quantiles
    probs = stats.t.cdf(q, DOF)
    for d, mk, ms in ((2, "o", 8), (10, "s", 6), (50, "D", 4.5)):
        c = next(c for c in calib if c["d"] == d and c["group"] == "all")
        ax.plot(q, np.quantile(pooled[d], probs), mk, ms=ms, color=C_D[d], mec=SURF, mew=0.6,
                label=f"d = {d}: {c['P']} sensitivities x {REPS} seeds, "
                      f"mean z$^2$ {c['mean_z2']:.2f}", zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Theoretical quantile (t, 199 dof)")
    ax.set_ylabel("Empirical quantile of z = (MC - exact) / SE")
    ax.set_title(f"Geometric basket: z-scores of all sensitivities, {REPS} seeds",
                 loc="left", fontsize=11)
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "e9_qq.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e9_qq.png'}")


if __name__ == "__main__":
    main()
