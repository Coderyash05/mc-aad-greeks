"""E7 wall-clock: whole pipelines, timed fairly.

Every program starts from a PRNG key and does all of its own work, so one-shot and
chunked are compared on the same job: generate the normals + price (+ gradient).
  - one-shot:  Z = normal(key, all N paths) in one call, then price / value_and_grad
               of the mean over all N (Z lives in temporaries, O(N)).
  - chunked:   lax.scan over batches; batch b draws normal(fold_in(key, b)) and
               prices / differentiates just its B paths (mcgreeks.chunked), O(B).
  - rng:       the same pipeline's random-number generation alone (the draws are
               summed so XLA cannot drop them), to get the RNG share of pricing time.
Asian chunked batches draw their normals per time block of ~sqrt(M) steps
(asian_price_blocked); "grad + ckpt" adds jax.checkpoint per block.

Two views are reported:
  (a) end-to-end: chunked gradient vs one-shot gradient, wall-clock and flops (both
      include the RNG, the pricing and the gradient);
  (b) AD overhead: gradient / price of the SAME pipeline, wall-clock and flops.
Plus the chunk-size trade-off for the basket at N = 1e6, B in {1e3, 1e4, 1e5}.

Timing: median [IQR] after compilation; at least 10 runs per basket program and 30
per Asian program. A result whose IQR exceeds 20% of its median is flagged "noisy",
and for it flops (XLA, lowered HLO, loop bodies x trip count) are the primary metric.
Close other programs before running: this is a laptop CPU shared with the desktop.

Run:  python experiments/e7_wallclock.py             (~10-12 minutes)
      python experiments/e7_wallclock.py --smoke     (tiny sizes, checks the code runs)
      python experiments/e7_wallclock.py --bsweep TAG (basket N = 1e5 B sweep only,
            for the single-core check; see README)
Writes results/e7_wallclock.csv (or results/e7_bsweep_TAG.csv) and results/e7_wallclock_hardware.json.
"""
import csv
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

import mcgreeks  # noqa: F401
from mcgreeks.asian import asian_price, asian_price_blocked, sqrt_block
from mcgreeks.basket import basket_cholesky, basket_price, basket_price_L
from mcgreeks.bench import cost_lowered, print_hardware, summary, temp_bytes, time_runs
from mcgreeks.chunked import chunked_mean_keyed, chunked_value_and_grad_keyed

R, T, K, D = 0.05, 1.0, 100.0, 50
KEY = jax.random.PRNGKey(2029)
RESULTS = Path(__file__).resolve().parents[1] / "results"
N_BASKET, BS_BASKET, REPS_BASKET = 1_000_000, [1_000, 10_000, 100_000], 10
N_ASIAN, MS, B_ASIAN, REPS_ASIAN = 100_000, [12, 52, 252, 1000], 10_000, 30
NOISY_IQR = 0.20


def timed(fn, min_reps):
    med, q25, q75, n = summary(time_runs(fn, min_reps=min_reps, max_reps=max(100, min_reps),
                                         budget_s=2.0))
    return {"ms": med * 1e3, "q25_ms": q25 * 1e3, "q75_ms": q75 * 1e3, "runs": n,
            "iqr_pct": 100 * (q75 - q25) / med, "noisy": (q75 - q25) / med > NOISY_IQR}


def scan_total(whole, body, n):
    """Flops of a program whose lax.scan runs `body` n times (XLA counts it once)."""
    return whole["flops"] + (n - 1) * body["flops"]


def entry(base, pipeline, program, fn, args, flops, reps):
    row = {**base, "pipeline": pipeline, "program": program, "flops": flops,
           "total_mb": temp_bytes(fn, *args) / 1e6}   # input is only a key
    jax.block_until_ready(fn(*args))                   # compile + warm-up
    row.update(timed(lambda: jax.block_until_ready(fn(*args)), reps))
    tag = "  NOISY" if row["noisy"] else ""
    print(f"   {base['part']:<6} N={base['N']:>9,} M={str(base['M']):>4} B={str(base['B']):>7} "
          f"{pipeline:<8} {program:<11} {row['ms']:9.1f} ms [{row['q25_ms']:.1f}, "
          f"{row['q75_ms']:.1f}] IQR {row['iqr_pct']:4.0f}% (n={row['runs']}) "
          f"{row['total_mb']:8.1f} MB{tag}")
    return row


# ---- basket --------------------------------------------------------------------------
def basket_rows(N, Bs, reps, oneshot=True):
    d = D
    S0, sigma = jnp.full(d, 100.0), jnp.linspace(0.15, 0.35, d)
    rho, w = jnp.full(d * (d - 1) // 2, 0.5), jnp.full(d, 1.0 / d)
    params = (S0, sigma, rho)

    def prepare(p):
        return p[0], p[1], basket_cholesky(p[2], d)

    rows = []
    if oneshot:
        base = {"part": "basket", "N": N, "M": "", "B": ""}
        rng = jax.jit(lambda k: jax.random.normal(k, (N, d)).sum())
        price = jax.jit(lambda p, k: basket_price(*p, R, T, K, w, jax.random.normal(k, (N, d))))
        grad = jax.jit(jax.value_and_grad(
            lambda p, k: basket_price(*p, R, T, K, w, jax.random.normal(k, (N, d)))))
        for prog, f, args in (("rng", rng, (KEY,)), ("price", price, (params, KEY)),
                              ("grad", grad, (params, KEY))):
            rows.append(entry(base, "one-shot", prog, f, args, cost_lowered(f, *args)["flops"],
                              reps))
    for B in Bs:
        nb = N // B
        base = {"part": "basket", "N": N, "M": "", "B": B}

        def batch(a, k, B=B):
            return basket_price_L(*a, R, T, K, w, jax.random.normal(k, (B, d)))

        def rng_all(k, B=B, nb=nb):
            def body(acc, i):
                return acc + jax.random.normal(jax.random.fold_in(k, i), (B, d)).sum(), None
            return jax.lax.scan(body, 0.0, jnp.arange(nb))[0]

        rng = jax.jit(rng_all)
        rng_one = jax.jit(lambda k, B=B: jax.random.normal(k, (B, d)).sum())
        mean_k = chunked_mean_keyed(batch, nb)
        price = jax.jit(lambda p, k, m=mean_k: m(prepare(p), k))
        grad = chunked_value_and_grad_keyed(batch, nb, prepare=prepare)
        kb = jax.random.fold_in(KEY, 0)
        body_price = cost_lowered(jax.jit(batch), prepare(params), kb)
        body_grad = cost_lowered(jax.jit(jax.value_and_grad(batch)), prepare(params), kb)
        for prog, f, args, body in (("rng", rng, (KEY,), cost_lowered(rng_one, kb)),
                                    ("price", price, (params, KEY), body_price),
                                    ("grad", grad, (params, KEY), body_grad)):
            rows.append(entry(base, "chunked", prog, f, args,
                              scan_total(cost_lowered(f, *args), body, nb), reps))
    return rows


# ---- Asian ---------------------------------------------------------------------------
def asian_rows(reps):
    params = (jnp.float64(100.0), jnp.float64(0.2))
    B, nb, N = B_ASIAN, N_ASIAN // B_ASIAN, N_ASIAN
    rows = []
    for M in MS:
        m = sqrt_block(M)
        base = {"part": "asian", "N": N, "M": M, "B": ""}

        def z_all(k, M=M):
            return jax.random.normal(k, (M, N))

        rng = jax.jit(lambda k: z_all(k).sum())
        price = jax.jit(lambda p, k, u=1: asian_price(p[0], p[1], R, T, K, z_all(k), unroll=u),
                        static_argnums=2)
        grad = jax.jit(jax.value_and_grad(
            lambda p, k, u=1: asian_price(p[0], p[1], R, T, K, z_all(k), unroll=u)),
            static_argnums=2)
        rows.append(entry(base, "one-shot", "rng", rng, (KEY,), cost_lowered(rng, KEY)["flops"],
                          reps))
        rows.append(entry(base, "one-shot", "price", price, (params, KEY),
                          cost_lowered(price, params, KEY, True)["flops"], reps))
        rows.append(entry(base, "one-shot", "grad", grad, (params, KEY),
                          cost_lowered(grad, params, KEY, True)["flops"], reps))

        base = {**base, "B": B}

        def batch(p, k, ckpt=False, unroll=1, M=M, m=m):
            return asian_price_blocked(p[0], p[1], R, T, K, k, B, M, m, ckpt, unroll)

        def rng_batch(k, unroll=1, M=M, m=m):
            def blk(acc, j):
                return acc + jax.random.normal(jax.random.fold_in(k, j), (m, B)).sum(), None
            return jax.lax.scan(blk, 0.0, jnp.arange(M // m), unroll=unroll)[0]

        def rng_all(k, unroll=1):
            return jax.lax.scan(lambda a, i: (a + rng_batch(jax.random.fold_in(k, i), unroll), None),
                                0.0, jnp.arange(nb))[0]

        kb = jax.random.fold_in(KEY, 0)
        rng_c = jax.jit(rng_all)
        rows.append(entry(base, "chunked", "rng", rng_c, (KEY,),
                          scan_total(cost_lowered(jax.jit(lambda k: rng_all(k, True)), KEY),
                                     cost_lowered(jax.jit(lambda k: rng_batch(k, True)), kb), nb),
                          reps))
        price_c = chunked_mean_keyed(batch, nb)
        rows.append(entry(base, "chunked", "price", price_c, (params, KEY),
                          scan_total(cost_lowered(chunked_mean_keyed(
                              lambda p, k: batch(p, k, unroll=True), nb), params, KEY),
                              cost_lowered(jax.jit(lambda p, k: batch(p, k, unroll=True)),
                                           params, kb), nb), reps))
        for ckpt, prog in ((False, "grad"), (True, "grad + ckpt")):
            g = chunked_value_and_grad_keyed(lambda p, k, c=ckpt: batch(p, k, c), nb)
            fl = scan_total(
                cost_lowered(chunked_value_and_grad_keyed(
                    lambda p, k, c=ckpt: batch(p, k, c, True), nb), params, KEY),
                cost_lowered(jax.jit(jax.value_and_grad(
                    lambda p, k, c=ckpt: batch(p, k, c, True))), params, kb), nb)
            rows.append(entry(base, "chunked", prog, g, (params, KEY), fl, reps))
    return rows


# ---- views ---------------------------------------------------------------------------
def pick(rows, part, N, M, B, pipeline, program):
    for r in rows:
        if (r["part"], r["N"], r["M"], r["B"], r["pipeline"], r["program"]) == \
                (part, N, M, B, pipeline, program):
            return r
    raise KeyError((part, N, M, B, pipeline, program))


def fmt(a, b, key):
    return a[key] / b[key]


def views(rows):
    out = []
    configs = sorted({(r["part"], r["N"], r["M"], r["B"]) for r in rows
                      if r["pipeline"] == "chunked"}, key=str)
    print("\n(a) End-to-end, chunked vs one-shot (same work: RNG + pricing + gradient)")
    print(f"{'part':<7}{'N':>10}{'M':>6}{'B':>8}  {'grad time: chunked / one-shot':>30}"
          f"{'flops':>8}{'MB: one-shot -> chunked':>26}")
    for part, N, M, B in configs:
        o = pick(rows, part, N, M, "", "one-shot", "grad")
        progs = ["grad"] + (["grad + ckpt"] if part == "asian" else [])
        for prog in progs:
            c = pick(rows, part, N, M, B, "chunked", prog)
            v = {"view": "a", "part": part, "N": N, "M": M, "B": B, "program": prog,
                 "time_ratio": fmt(c, o, "ms"), "flops_ratio": fmt(c, o, "flops"),
                 "oneshot_mb": o["total_mb"], "chunked_mb": c["total_mb"],
                 "noisy": o["noisy"] or c["noisy"]}
            out.append(v)
            print(f"{part:<7}{N:>10,}{str(M):>6}{B:>8,}  {prog:<12}{v['time_ratio']:>17.2f}"
                  f"{'  (noisy)' if v['noisy'] else '':<9}{v['flops_ratio']:>6.2f}"
                  f"{o['total_mb']:>14.1f} -> {c['total_mb']:.1f}")

    print("\n(b) AD overhead: gradient / price of the SAME pipeline; RNG share of pricing")
    print(f"{'part':<7}{'N':>10}{'M':>6}{'B':>8}  {'pipeline':<9}{'program':<12}"
          f"{'time x':>8}{'flops x':>9}{'RNG share of price time':>25}{'RNG share of flops':>20}")
    keys = sorted({(r["part"], r["N"], r["M"], r["B"], r["pipeline"]) for r in rows}, key=str)
    for part, N, M, B, pipe in keys:
        p = pick(rows, part, N, M, B, pipe, "price")
        g_rng = pick(rows, part, N, M, B, pipe, "rng")
        progs = ["grad"] + (["grad + ckpt"] if part == "asian" and pipe == "chunked" else [])
        for prog in progs:
            g = pick(rows, part, N, M, B, pipe, prog)
            v = {"view": "b", "part": part, "N": N, "M": M, "B": B, "pipeline": pipe,
                 "program": prog, "time_ratio": fmt(g, p, "ms"), "flops_ratio": fmt(g, p, "flops"),
                 "rng_share_time": g_rng["ms"] / p["ms"], "rng_share_flops": g_rng["flops"] / p["flops"],
                 "noisy": g["noisy"] or p["noisy"]}
            out.append(v)
            print(f"{part:<7}{N:>10,}{str(M):>6}{str(B):>8}  {pipe:<9}{prog:<12}"
                  f"{v['time_ratio']:>8.2f}{v['flops_ratio']:>9.2f}{v['rng_share_time']:>20.0%}"
                  f"{'  (noisy)' if v['noisy'] else '':<5}{v['rng_share_flops']:>18.0%}")
    return out


def write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    global N_BASKET, BS_BASKET, REPS_BASKET, N_ASIAN, MS, B_ASIAN, REPS_ASIAN
    hw = print_hardware()
    if "--bsweep" in sys.argv:
        tag = sys.argv[sys.argv.index("--bsweep") + 1]
        rows = basket_rows(100_000, [1_000, 10_000, 100_000], reps=3)
        write(RESULTS / f"e7_bsweep_{tag}.csv", rows)
        return
    if "--smoke" in sys.argv:
        N_BASKET, BS_BASKET, REPS_BASKET = 20_000, [1_000, 10_000], 2
        N_ASIAN, MS, B_ASIAN, REPS_ASIAN = 20_000, [12], 10_000, 2
    rows = basket_rows(N_BASKET, BS_BASKET, REPS_BASKET) + asian_rows(REPS_ASIAN)
    v = views(rows)
    if "--smoke" in sys.argv:
        return
    write(RESULTS / "e7_wallclock.csv", rows)
    write(RESULTS / "e7_wallclock_views.csv", v)
    (RESULTS / "e7_wallclock_hardware.json").write_text(json.dumps(hw, indent=2))


if __name__ == "__main__":
    main()
