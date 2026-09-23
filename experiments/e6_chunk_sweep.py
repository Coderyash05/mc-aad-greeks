"""E6 follow-up: chunk-size sweep for the chunked methods, and a roofline check.

1. Vectorised bumping and forward-mode AD are run in chunks (lax.map with
   batch_size) so no batched (chunk, N, d) float64 array exceeds a memory budget.
   Here the budget is swept over 16, 64 and 256 MB at d = 20 and d = 50, next to
   the loop bumping and reverse AD from the same session. Budgets that give the
   same chunk size run the same program and are timed once.
2. Is the pricing memory-bandwidth-bound? Each program's achieved throughput,
   XLA flops / time and XLA bytes accessed / time, is compared with what this
   machine achieves on a streaming kernel (triad y = a x + z, bandwidth-bound)
   and on a large square GEMM (compute-bound). A program running near the triad's
   GB/s and far below the GEMM's GFLOP/s is bandwidth-bound, and vice versa.

Run:  python experiments/e6_chunk_sweep.py        (~6-8 minutes)
Writes results/e6_chunk_sweep.csv, results/e6_roofline.csv.
"""
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcgreeks  # noqa: F401,E402
from e6_basket import H, K, N, R, T, setup  # noqa: E402
from mcgreeks.basket import (basket_greeks, basket_greeks_fwd, basket_price,  # noqa: E402
                             bump_grad_loop, bump_grad_vmap)
from mcgreeks.bench import cost, print_hardware, summary, time_runs  # noqa: E402

DIMS = [20, 50]
BUDGETS_MB = [16, 64, 256]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def chunk_for(d, rows, budget):
    c = max(1, int(budget // (N * d * 8)))
    return None if c >= rows else c


def timed(fn, first_call_done=True):
    med, q25, q75, n = summary(time_runs(fn, min_reps=3, max_reps=100, budget_s=2.0,
                                         warm=not first_call_done))
    return {"ms": med * 1e3, "q25_ms": q25 * 1e3, "q75_ms": q75 * 1e3, "runs": n}


def machine_reference():
    """Achievable bandwidth (triad) and compute (GEMM) on this machine, float64."""
    n = 50_000_000                                    # 400 MB per vector
    x = jnp.ones(n)
    z = jnp.ones(n)
    triad = jax.jit(lambda x, z: 2.0 * x + z)
    t = timed(lambda: triad(x, z).block_until_ready(), first_call_done=False)["ms"] / 1e3
    gbs = 3 * n * 8 / t / 1e9                         # read x, read z, write y
    m = 4096
    A = jnp.ones((m, m))
    gemm = jax.jit(lambda A: A @ A)
    t = timed(lambda: gemm(A).block_until_ready(), first_call_done=False)["ms"] / 1e3
    gflops = 2 * m**3 / t / 1e9
    return gbs, gflops


def main():
    print_hardware()
    gbs_peak, gflops_peak = machine_reference()
    print(f"\nMachine reference (float64): triad {gbs_peak:.1f} GB/s, "
          f"{4096}x{4096} GEMM {gflops_peak:.1f} GFLOP/s\n")

    sweep, roof = [], []
    for d in DIMS:
        S0, sigma, rho, w, Z = setup(d)
        a = (S0, sigma, rho, R, T, K, w, Z)
        P = 2 * d + d * (d - 1) // 2
        c_price, c_rev = cost(basket_price, *a), cost(basket_greeks, *a)
        c_fwd = cost(basket_greeks_fwd, *a, chunk=None)
        c_vec = cost(bump_grad_vmap, *a, h=H, chunk=None)

        basket_price(*a).block_until_ready()
        t_price = timed(lambda: basket_price(*a).block_until_ready())
        basket_greeks(*a)["corr"].block_until_ready()
        t_rev = timed(lambda: basket_greeks(*a)["corr"].block_until_ready())
        g_loop = bump_grad_loop(*a, h=H)[1].block_until_ready()
        t_loop = timed(lambda: bump_grad_loop(*a, h=H)[1].block_until_ready())
        base = {"d": d, "n_sens": P, "price_ms": t_price["ms"]}
        for name, t in (("price", t_price), ("rev", t_rev), ("loop", t_loop)):
            sweep.append({**base, "method": name, "budget_mb": "", "chunk": "", **t,
                          "x_price": t["ms"] / t_price["ms"]})
        print(f"d={d} P={P}: price {t_price['ms']:.2f} ms, rev {t_rev['ms'] / t_price['ms']:.2f}x, "
              f"loop {t_loop['ms'] / t_price['ms']:.0f}x")

        best = {}
        for method, rows in (("vec", 2 * P + 1), ("fwd", P)):
            done = {}
            for b in BUDGETS_MB:
                chunk = chunk_for(d, rows, b * 1e6)
                if chunk not in done:
                    if method == "vec":
                        fn = lambda: bump_grad_vmap(*a, h=H, chunk=chunk)[1].block_until_ready()  # noqa: E731
                        err = float(jnp.max(jnp.abs(fn() - g_loop)))
                        assert err < 1e-10, (d, chunk, err)
                    else:
                        fn = lambda: basket_greeks_fwd(*a, chunk=chunk)[1].block_until_ready()  # noqa: E731
                        fn()
                    done[chunk] = timed(fn)
                t = done[chunk]
                r = {**base, "method": method, "budget_mb": b, "chunk": chunk or rows, **t,
                     "x_price": t["ms"] / t_price["ms"]}
                sweep.append(r)
                print(f"   {method} budget {b:>3} MB -> chunk {r['chunk']:>4}: "
                      f"{r['x_price']:8.1f}x one pricing  [{t['q25_ms'] / t_price['ms']:.1f}, "
                      f"{t['q75_ms'] / t_price['ms']:.1f}]")
            best[method] = min((r for r in sweep if r["d"] == d and r["method"] == method),
                               key=lambda r: r["ms"])
            print(f"   best {method}: {best[method]['budget_mb']} MB (chunk {best[method]['chunk']})"
                  f", {best[method]['x_price']:.1f}x; loop is {t_loop['ms'] / t_price['ms']:.1f}x")

        # roofline: achieved throughput of each program at its best setting
        for name, t, c in (("price", t_price, c_price), ("rev", t_rev, c_rev),
                           ("loop", t_loop, {k: (2 * P + 1) * v for k, v in c_price.items()}),
                           ("vec", best["vec"], c_vec), ("fwd", best["fwd"], c_fwd)):
            s = t["ms"] / 1e3
            roof.append({"d": d, "method": name, "ms": t["ms"], "gflops": c["flops"] / s / 1e9,
                         "gbs": c["bytes"] / s / 1e9, "flops_per_byte": c["flops"] / c["bytes"],
                         "pct_gemm_peak": 100 * c["flops"] / s / 1e9 / gflops_peak,
                         "pct_triad_bw": 100 * c["bytes"] / s / 1e9 / gbs_peak,
                         "bytes_x_price": c["bytes"] / c_price["bytes"],
                         "flops_x_price": c["flops"] / c_price["flops"],
                         "machine_triad_gbs": gbs_peak, "machine_gemm_gflops": gflops_peak})

    print(f"\nRoofline (throughput from XLA flops and bytes accessed / median time; "
          f"triad {gbs_peak:.1f} GB/s, GEMM {gflops_peak:.1f} GFLOP/s):")
    print(f"{'d':>3} {'method':<6}{'GFLOP/s':>9}{'% GEMM':>8}{'GB/s':>8}{'% triad':>9}"
          f"{'flop/B':>8}{'bytes x':>9}{'flops x':>9}")
    for r in roof:
        print(f"{r['d']:>3} {r['method']:<6}{r['gflops']:>9.1f}{r['pct_gemm_peak']:>7.0f}%"
              f"{r['gbs']:>8.1f}{r['pct_triad_bw']:>8.0f}%{r['flops_per_byte']:>8.2f}"
              f"{r['bytes_x_price']:>9.2f}{r['flops_x_price']:>9.2f}")

    RESULTS.mkdir(exist_ok=True)
    for path, rows in ((RESULTS / "e6_chunk_sweep.csv", sweep), (RESULTS / "e6_roofline.csv", roof)):
        with path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
