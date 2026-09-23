"""E8: robustness of the E3 (gamma) and E5 (digital delta) comparisons across K and T.

Grid: K in {80, 90, 100, 110, 120} x T in {0.1, 0.25, 1, 2}; sigma = 0.2, S0 = 100,
r = 0.05. Per cell: 200 evaluation batches of 20,000 paths, plus 200 independent
calibration batches used only to choose the FD bump h and the smoothing width eps
(out of sample; the in-sample choice is recorded too). All methods in a cell share
the same batches, so they are compared with a paired bootstrap (mcgreeks.stats,
2,000 resamples): RMSE with 95% CI, and RMSE / best-in-cell with 95% CI. A method
"ties" the best if that ratio CI contains 1.

  Vanilla call gamma: CRN finite differences, likelihood ratio (LR), pathwise-LR,
                      softplus-smoothed autodiff, and the parity-switched LR and
                      pathwise-LR (put leg when K < S0 e^{rT}; mcgreeks.greeks_lr).
  Digital call delta: CRN finite differences, sigmoid-smoothed autodiff, LR, and
                      LR (parity) (complementary digital when K < S0 e^{rT}).
The parity methods reuse the same evaluation batches and bootstrap seeds, so every
existing RMSE and CI is unchanged; only "ratio to best" moves where a parity method
becomes the best. For each parity method the paired ratio parity / plain (with CI)
is recorded as well.

Width grid: 17 widths 0.01-15.8. If the calibration choice lands on the top edge,
the grid for that cell is extended with EXTRA_WIDTHS (25.1, 39.8, 63.1; h < S0 so
S0 - h stays positive) and the method is re-tuned; the edge-of-grid result is kept
in the CSV (width_base, rmse_base) for a before/after.

LR variance vs T: the LR gamma weight (Z^2 - 1 - sigma sqrt(T) Z) / (S0^2 sigma^2 T)
has standard deviation ~ 1/T as T -> 0 (Glasserman 2004, ch. 7). For a CALL, though,
the payoff it multiplies near the strike is only ~ S0 sigma sqrt(T), so at the money
the LR gamma error grows like 1/sqrt(T): the same rate as the pathwise-LR error and
as the true gamma. The LR / pathwise-LR RMSE ratio at T = 0.1 vs T = 2 checks this;
the relative errors (RMSE / exact) show where LR really fails (short-dated, in the
money: small true gamma, large payoff times a large weight).

Run:  python experiments/e8_sweep.py --estimate    (times one cell, prints an estimate)
      python experiments/e8_sweep.py               (full sweep)
      python experiments/e8_sweep.py --plot-only
Writes results/e8_sweep.csv, results/e8_winners.csv, results/e8_lr_vs_T.csv,
results/e8_summary.csv, results/e8_parity.csv, results/e8_sweep.png.
"""
import csv
import sys
import time
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_digital_delta, bs_greeks
from mcgreeks.greeks_lr import (lr_delta_samples, lr_digital_delta_parity_samples,
                                lr_gamma_parity_samples, lr_gamma_samples,
                                pwlr_gamma_parity_samples, pwlr_gamma_samples,
                                smooth_digital_delta_samples, smooth_gamma_samples)
from mcgreeks.pricer import discounted_payoffs
from mcgreeks.stats import paired_bootstrap, tune

S0, SIGMA, R = 100.0, 0.2, 0.05
KS, TS = [80.0, 90.0, 100.0, 110.0, 120.0], [0.1, 0.25, 1.0, 2.0]
REPS, N = 200, 20_000
GRID_WIDTHS = np.logspace(-2, 1.2, 17)   # candidate h and eps, spot units (as in E5)
EXTRA_WIDTHS = np.logspace(1.4, 1.8, 3)  # 25.1, 39.8, 63.1: only if the choice hits 15.8
RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "e8_sweep.csv"
GREEKS = {"gamma": ["FD, CRN", "LR", "LR (parity)", "Pathwise-LR", "Pathwise-LR (parity)",
                    "Smoothed AD"],
          "digital delta": ["FD, CRN", "Smoothed AD", "LR", "LR (parity)"]}
PARITY = {"LR (parity)": "LR", "Pathwise-LR (parity)": "Pathwise-LR"}   # parity -> plain

# Sequential single-hue (blue) ramp, steps 100 -> 700 of the palette's blue.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"


@jax.jit
def _prices(s, K, T, Z):
    """Per-batch call and digital prices at spot s: two arrays (REPS,)."""
    return (discounted_payoffs(s, SIGMA, R, T, K, Z, kind="call").mean(-1),
            discounted_payoffs(s, SIGMA, R, T, K, Z, kind="digital").mean(-1))


def _pb(x):
    return np.asarray(x.reshape(REPS, N).mean(axis=-1))


def estimates(K, T, Z, widths=GRID_WIDTHS):
    """Per-batch estimates, tuned methods over the whole width grid."""
    Zf = Z.reshape(-1)
    c0, _ = _prices(S0, K, T, Z)
    fd_g, fd_d, sm_g, sm_d = [], [], [], []
    for w in widths:
        cu, du = _prices(S0 + w, K, T, Z)
        cd, dd = _prices(S0 - w, K, T, Z)
        fd_g.append(np.asarray((cu - 2 * c0 + cd) / w**2))
        fd_d.append(np.asarray((du - dd) / (2 * w)))
        sm_g.append(_pb(smooth_gamma_samples(S0, SIGMA, R, T, K, Zf, float(w))))
        sm_d.append(_pb(smooth_digital_delta_samples(S0, SIGMA, R, T, K, Zf, float(w))))
    return {"gamma": {"FD, CRN": np.stack(fd_g), "Smoothed AD": np.stack(sm_g),
                      "LR": _pb(lr_gamma_samples(S0, SIGMA, R, T, K, Zf)),
                      "LR (parity)": _pb(lr_gamma_parity_samples(S0, SIGMA, R, T, K, Zf)),
                      "Pathwise-LR": _pb(pwlr_gamma_samples(S0, SIGMA, R, T, K, Zf)),
                      "Pathwise-LR (parity)": _pb(pwlr_gamma_parity_samples(S0, SIGMA, R, T,
                                                                            K, Zf))},
            "digital delta": {"FD, CRN": np.stack(fd_d), "Smoothed AD": np.stack(sm_d),
                              "LR": _pb(lr_delta_samples(S0, SIGMA, R, T, K, Zf, kind="digital")),
                              "LR (parity)": _pb(lr_digital_delta_parity_samples(S0, SIGMA, R, T,
                                                                                 K, Zf))}}


def normals(tag):
    return jax.random.normal(jax.random.PRNGKey(zlib.crc32(tag.encode())), (REPS, N),
                             dtype=jnp.float64)


def run_cell(K, T):
    exact = {"gamma": float(bs_greeks(S0, K, R, SIGMA, T)["gamma"]),
             "digital delta": float(bs_digital_delta(S0, K, R, SIGMA, T))}
    Z_ev, Z_cal = normals(f"e8-eval-{K}-{T}"), normals(f"e8-cal-{K}-{T}")
    ev, cal = estimates(K, T, Z_ev), estimates(K, T, Z_cal)
    extra = {}   # estimates at EXTRA_WIDTHS, computed only if some choice hits the top edge
    rows = []
    for greek, methods in GREEKS.items():
        ex, errs, tuned, base = exact[greek], {}, {}, {}
        for m in methods:
            if ev[greek][m].ndim != 2:
                errs[m] = ev[greek][m] - ex
                continue
            t = tune(GRID_WIDTHS, cal[greek][m], ev[greek][m], ex)   # tuned over the width grid
            if t["index"] == len(GRID_WIDTHS) - 1:                   # top edge: extend and retune
                if not extra:
                    extra.update(ev=estimates(K, T, Z_ev, EXTRA_WIDTHS),
                                 cal=estimates(K, T, Z_cal, EXTRA_WIDTHS))
                base[m] = t
                t = tune(np.concatenate([GRID_WIDTHS, EXTRA_WIDTHS]),
                         np.concatenate([cal[greek][m], extra["cal"][greek][m]]),
                         np.concatenate([ev[greek][m], extra["ev"][greek][m]]), ex)
            tuned[m], errs[m] = t, t["errors"]
        seed = zlib.crc32(f"e8-boot-{greek}-{K}-{T}".encode())
        boot = paired_bootstrap(errs, seed=seed)
        for m in methods:
            r, e = boot["methods"][m], errs[m]
            t, b0 = tuned.get(m), base.get(m)
            n_grid = len(GRID_WIDTHS) + (len(EXTRA_WIDTHS) if b0 else 0)
            rows.append({
                "greek": greek, "K": K, "T": T, "exact": ex, "method": m,
                "rmse": r["rmse"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
                "bias": float(np.mean(e)), "std": float(np.std(e, ddof=1)),
                "ratio_to_best": r["ratio"], "ratio_ci_lo": r["ratio_ci"][0],
                "ratio_ci_hi": r["ratio_ci"][1], "best": m == boot["reference"],
                "tie_with_best": m != boot["reference"] and not r["beaten_by_reference"],
                "width_out": t["value"] if t else "", "width_in": t["value_in"] if t else "",
                "rmse_in_sample": t["rmse_in"] if t else "",
                "width_at_grid_edge": (t["index"] in (0, n_grid - 1)) if t else "",
                "grid_extended": bool(b0) if t else "",
                "width_base": b0["value"] if b0 else "", "rmse_base": b0["rmse_out"] if b0 else ""})
            if m in PARITY:   # parity / plain, same resamples
                p = paired_bootstrap(errs, seed=seed, reference=PARITY[m])["methods"][m]
                rows[-1].update(parity_over_plain=p["ratio"], parity_over_plain_lo=p["ratio_ci"][0],
                                parity_over_plain_hi=p["ratio_ci"][1],
                                parity_leg="put" if K < S0 * np.exp(R * T) else "call")
        if greek == "gamma":   # LR vs pathwise-LR, same resamples
            b = paired_bootstrap(errs, seed=seed, reference="Pathwise-LR")["methods"]["LR"]
            for row in rows[-len(methods):]:
                row.update(lr_over_pwlr=b["ratio"], lr_over_pwlr_lo=b["ratio_ci"][0],
                           lr_over_pwlr_hi=b["ratio_ci"][1])
    return rows


def winners(rows):
    out = []
    for greek in GREEKS:
        for T in TS:
            for K in KS:
                cell = [r for r in rows if r["greek"] == greek and r["K"] == K and r["T"] == T]
                best = next(r["method"] for r in cell if r["best"])
                ties = [r["method"] for r in cell if r["tie_with_best"]]
                out.append({"greek": greek, "K": K, "T": T, "best": best, "ties": "; ".join(ties),
                            "winner": best if not ties else "tie: " + ", ".join([best] + ties)})
    return out


def main():
    if "--plot-only" in sys.argv:
        with OUT.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in ("K", "T", "ratio_to_best"):
                r[k] = float(r[k])
            for k in ("best", "tie_with_best"):
                r[k] = r[k] == "True"
        plot(rows)
        return
    if "--estimate" in sys.argv:
        _prices(S0, 100.0, 1.0, normals("warm"))   # compile outside the timing
        t0 = time.perf_counter()
        run_cell(100.0, 1.0)
        dt = time.perf_counter() - t0
        print(f"one cell (both seeds, {len(GRID_WIDTHS)} widths): {dt:.1f} s "
              f"-> full sweep of {len(KS) * len(TS)} cells ~ {dt * len(KS) * len(TS) / 60:.1f} min")
        return

    rows, t0 = [], time.perf_counter()
    for T in TS:
        for K in KS:
            rows += run_cell(K, T)
            print(f"  cell K={K:.0f} T={T}: done ({time.perf_counter() - t0:.0f} s)", flush=True)
    win, summ, par = winners(rows), summary(rows), parity_table(rows)
    lr_T = [{"K": K, **{f"T={T}_{k}": next(r[k2] for r in rows if r["greek"] == "gamma"
                                           and r["K"] == K and r["T"] == T)
                        for T in (0.1, 2.0) for k, k2 in (("ratio", "lr_over_pwlr"),
                                                          ("lo", "lr_over_pwlr_lo"),
                                                          ("hi", "lr_over_pwlr_hi"))}}
            for K in KS]
    RESULTS.mkdir(exist_ok=True)
    for path, data in ((OUT, rows), (RESULTS / "e8_winners.csv", win),
                       (RESULTS / "e8_lr_vs_T.csv", lr_T), (RESULTS / "e8_summary.csv", summ),
                       (RESULTS / "e8_parity.csv", par)):
        fields = list(dict.fromkeys(k for r in data for k in r))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    report(rows, win, lr_T, summ, par)
    plot(rows)


def _cell(rows, greek, m, K, T):
    return next(r for r in rows if r["greek"] == greek and r["method"] == m
                and r["K"] == K and r["T"] == T)


def summary(rows):
    """Per method: cells where it is best or tied (all / ITM K <= 90 / K >= 100), and
    the median and worst RMSE / best-in-cell."""
    out = []
    for greek, methods in GREEKS.items():
        for m in methods:
            cells = [_cell(rows, greek, m, K, T) for T in TS for K in KS]
            ok = [r for r in cells if r["best"] or r["tie_with_best"]]
            worst = max(cells, key=lambda r: r["ratio_to_best"])
            out.append({"greek": greek, "method": m, "best_or_tied": len(ok),
                        "best_or_tied_itm": sum(r["K"] <= 90 for r in ok),
                        "best_or_tied_atm_otm": sum(r["K"] >= 100 for r in ok),
                        "n_cells": len(cells), "n_itm": sum(r["K"] <= 90 for r in cells),
                        "median_ratio": float(np.median([r["ratio_to_best"] for r in cells])),
                        "worst_ratio": worst["ratio_to_best"],
                        "worst_cell": f"K={worst['K']:g}, T={worst['T']:g}"})
    return out


def parity_table(rows):
    """Parity method vs its plain version and vs the best method, cell by cell."""
    out = []
    for greek, methods in GREEKS.items():
        for m in (x for x in methods if x in PARITY):
            for T in TS:
                for K in KS:
                    r, p = _cell(rows, greek, m, K, T), _cell(rows, greek, PARITY[m], K, T)
                    out.append({"greek": greek, "method": m, "K": K, "T": T,
                                "leg": r["parity_leg"],
                                "parity_over_plain": r["parity_over_plain"],
                                "parity_over_plain_lo": r["parity_over_plain_lo"],
                                "parity_over_plain_hi": r["parity_over_plain_hi"],
                                "ratio_to_best": r["ratio_to_best"],
                                "ratio_to_best_lo": r["ratio_ci_lo"],
                                "ratio_to_best_hi": r["ratio_ci_hi"],
                                "best_or_tied": r["best"] or r["tie_with_best"],
                                "plain_ratio_to_best": p["ratio_to_best"]})
    return out


def report(rows, win, lr_T, summ, par):
    for greek in GREEKS:
        print(f"\n{greek}: RMSE / best in cell; * = best, = = tie with best")
        for m in GREEKS[greek]:
            cells = []
            for T in TS:
                for K in KS:
                    r = _cell(rows, greek, m, K, T)
                    mark = "*" if r["best"] else ("=" if r["tie_with_best"] else " ")
                    cells.append(f"{r['ratio_to_best']:5.2f}{mark}")
            print(f"  {m:<21}" + " ".join(cells))
        print(f"  {'(K, T)':<21}" + " ".join(f"{int(K):>3}/{T:<2g}" for T in TS for K in KS))
        print(f"\n  winners ({greek}), rows T, columns K = {[int(k) for k in KS]}:")
        for T in TS:
            print(f"   T={T:<5}" + " | ".join(f"{w['winner']:<28}" for w in win
                                           if w["greek"] == greek and w["T"] == T))
    print("\nSummary: best or tied (all / ITM K<=90 / K>=100), median and worst RMSE / best:")
    for s in summ:
        print(f"  {s['greek']:<14}{s['method']:<22}{s['best_or_tied']:>3}/{s['n_cells']}"
              f"{s['best_or_tied_itm']:>4}/{s['n_itm']}{s['best_or_tied_atm_otm']:>4}/"
              f"{s['n_cells'] - s['n_itm']}   median {s['median_ratio']:5.2f}   worst "
              f"{s['worst_ratio']:6.2f} ({s['worst_cell']})")
    print("\nParity / plain RMSE ratio [95% CI], and parity RMSE / best in cell:")
    for p in par:
        print(f"  {p['greek']:<14}{p['method']:<22}K={p['K']:>4.0f} T={p['T']:<5g}{p['leg']:<5}"
              f"{p['parity_over_plain']:6.3f} [{p['parity_over_plain_lo']:.3f}, "
              f"{p['parity_over_plain_hi']:.3f}]   x best {p['ratio_to_best']:5.2f} "
              f"[{p['ratio_to_best_lo']:.2f}, {p['ratio_to_best_hi']:.2f}]"
              f"{'  best/tie' if p['best_or_tied'] else ''}   (plain x best "
              f"{p['plain_ratio_to_best']:.2f})")
    ext = [r for r in rows if r.get("grid_extended") is True]
    print(f"\nGrid extended (choice at 15.8) for {len(ext)} tuned case(s):")
    for r in ext:
        print(f"  {r['greek']:<14}{r['method']:<10}K={r['K']:.0f} T={r['T']:g}: width "
              f"{r['width_base']:.3g} -> {r['width_out']:.3g}, RMSE {r['rmse_base']:.4g} -> "
              f"{r['rmse']:.4g}, best in cell: {r['best']}")
    print("\nLR / pathwise-LR gamma RMSE ratio [95% CI], T = 0.1 vs T = 2:")
    for r in lr_T:
        print(f"  K={r['K']:>5.0f}: T=0.1 {r['T=0.1_ratio']:5.2f} [{r['T=0.1_lo']:.2f}, "
              f"{r['T=0.1_hi']:.2f}]   T=2 {r['T=2.0_ratio']:5.2f} [{r['T=2.0_lo']:.2f}, "
              f"{r['T=2.0_hi']:.2f}]")
    edge = [r for r in rows if r["width_at_grid_edge"] is True]
    print(f"\nTuned widths at the edge of the grid: {len(edge)} of "
          f"{sum(1 for r in rows if r['width_out'] != '')}")
    changed = [r for r in rows if r["width_out"] != "" and r["width_out"] != r["width_in"]]
    print(f"Cells where out-of-sample and in-sample widths differ: {len(changed)}; mean RMSE "
          f"out-of-sample / in-sample = "
          f"{np.mean([r['rmse'] / r['rmse_in_sample'] for r in changed]) if changed else 1:.3f}")


def plot(rows):
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    norm = LogNorm(vmin=1.0, vmax=20.0)
    cells = [(K, T) for T in TS for K in KS]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.4), facecolor=SURF,
                             gridspec_kw={"height_ratios": [len(m) for m in GREEKS.values()]})
    for ax, greek in zip(axes, GREEKS):
        methods = GREEKS[greek]
        V = np.array([[next(r["ratio_to_best"] for r in rows if r["greek"] == greek
                            and r["method"] == m and r["K"] == K and r["T"] == T)
                       for (K, T) in cells] for m in methods])
        im = ax.imshow(np.clip(V, 1.0, 20.0), cmap=cmap, norm=norm, aspect="auto")
        for i, m in enumerate(methods):
            for j, (K, T) in enumerate(cells):
                r = next(x for x in rows if x["greek"] == greek and x["method"] == m
                         and x["K"] == K and x["T"] == T)
                txt = "best" if r["best"] else (f"tie\n{V[i, j]:.2f}" if r["tie_with_best"]
                                                 else (f"{V[i, j]:.2f}" if V[i, j] < 10
                                                       else f"{V[i, j]:.0f}"))
                light = norm(min(V[i, j], 20.0)) < 0.45
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color=INK if light else "#ffffff")
        for x in (4.5, 9.5, 14.5):
            ax.axvline(x, color=SURF, lw=3)
        ax.set_yticks(range(len(methods)), methods)
        ax.set_xticks(range(len(cells)), [f"K={int(K)}" for K, _ in cells], fontsize=7.5)
        for j, T in enumerate(TS):
            ax.text(2 + 5 * j, -0.6, f"T = {T:g}", ha="center", va="bottom", fontsize=9,
                    color=INK)
        ax.set_title(f"{greek.capitalize()}: RMSE / best method in each cell "
                     f"(\"tie\" = 95% CI of the ratio contains 1)", loc="left", fontsize=10.5,
                     pad=26)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_ticks([1, 2, 5, 10, 20], labels=["1", "2", "5", "10", "20+"])
        cb.outline.set_visible(False)
    fig.suptitle(f"E8: estimator robustness across strikes and maturities, sigma = {SIGMA}, "
                 f"{REPS} batches x {N:,} paths per cell (h, eps tuned out of sample)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "e8_sweep.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e8_sweep.png'}")


if __name__ == "__main__":
    main()
