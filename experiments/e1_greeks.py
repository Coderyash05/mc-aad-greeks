"""E1 (Greeks): autodiff delta, vega, rho vs Black-Scholes, plus the cost of AD.

Run:  python experiments/e1_greeks.py
Writes results/e1_greeks.csv, results/e1_timing.csv, results/e1_hardware.json.
"""
import csv
import json
from pathlib import Path

import jax

import mcgreeks  # noqa: F401
from mcgreeks.bench import cost, print_hardware, summary, time_runs
from mcgreeks.black_scholes import bs_greeks
from mcgreeks.greeks_ad import ad_greeks, ad_greeks_fwd, ad_greeks_se
from mcgreeks.greeks_fd import fd_greeks_loop, fd_greeks_vmap
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price

S0, R, N = 100.0, 0.05, 1_000_000
OUT = Path(__file__).resolve().parents[1] / "results" / "e1_greeks.csv"
TIMING_OUT = OUT.parent / "e1_timing.csv"
GREEKS = ("delta", "vega", "rho")


def validation():
    keys = iter(jax.random.split(jax.random.PRNGKey(123), 54))
    rows = []
    for kind in ("call", "put"):
        for sigma in (0.1, 0.2, 0.4):
            for T in (0.25, 1.0, 2.0):
                for K in (80.0, 100.0, 120.0):
                    est = ad_greeks_se(S0, sigma, R, T, K, normals(next(keys), N), kind=kind)
                    exact = bs_greeks(S0, K, R, sigma, T, kind=kind)
                    for g in GREEKS:
                        m, se = float(est[g][0]), float(est[g][1])
                        ex = float(exact[g])
                        # z is undefined when the per-path estimator is constant
                        # (se ~ 0): e.g. deep ITM call rho, see test_greeks_ad.py.
                        z = (m - ex) / se if se > 1e-10 else float("nan")
                        rows.append([kind, sigma, T, K, g, ex, m, se, z])

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "sigma", "T", "K", "greek", "bs", "ad", "se", "z_score"])
        w.writerows(rows)

    print("At-the-money, sigma = 0.2, T = 1:")
    print(f"{'kind':<6}{'greek':<7}{'BS':>10}{'AD':>10}{'SE':>9}{'z':>7}")
    for r in rows:
        if r[1] == 0.2 and r[2] == 1.0 and r[3] == 100.0:
            print(f"{r[0]:<6}{r[4]:<7}{r[5]:>10.4f}{r[6]:>10.4f}{r[7]:>9.4f}{r[8]:>7.2f}")

    print()
    for g in GREEKS:
        z = [r[8] for r in rows if r[4] == g and r[8] == r[8]]  # drop NaN
        n_const = sum(1 for r in rows if r[4] == g and r[8] != r[8])
        note = f"   ({n_const} zero-variance case(s) excluded)" if n_const else ""
        print(f"{g:<6} within 3 SE: {sum(abs(v) < 3 for v in z)}/{len(z)}   "
              f"max |z| = {max(map(abs, z)):.2f}   mean z = {sum(z) / len(z):+.2f}{note}")


def timing(budget_s=3.0):
    """Cost of price + 3 Greeks relative to price alone, five ways.

    Reverse mode: one forward + one backward sweep for all 3 inputs.
    Forward mode: one jvp per input (3 tangents, vmapped over one primal).
    Bumping: base + 6 CRN central-difference pricings, h = 1e-2, either as a
    Python loop of jit-compiled calls or vmapped in one compiled program.
    Expected ordering: with ONE scalar output, reverse mode (~3-4x, any number of
    inputs) beats forward mode (~1 + 1-2x per input) once there are >= 2 inputs.
    """
    hw = print_hardware()
    Z = normals(jax.random.PRNGKey(0), N)
    args = (S0, 0.2, R, 1.0, 100.0, Z)
    h = 1e-2
    methods = {
        "price": (lambda: mc_price(*args).block_until_ready(), mc_price, args),
        "forward AD": (lambda: ad_greeks_fwd(*args)["rho"].block_until_ready(),
                       ad_greeks_fwd, args),
        "reverse AD": (lambda: ad_greeks(*args)["rho"].block_until_ready(), ad_greeks, args),
        "bump, loop": (lambda: fd_greeks_loop(*args[:5], h, Z)["rho"].block_until_ready(),
                       None, None),
        "bump, vmap": (lambda: fd_greeks_vmap(*args[:5], h, Z)["rho"].block_until_ready(),
                       fd_greeks_vmap, args[:5] + (h, Z)),
    }

    # Same numbers, different evaluation order: check before timing anything.
    rev, fwd = ad_greeks(*args), ad_greeks_fwd(*args)
    loop, vec = fd_greeks_loop(*args[:5], h, Z), fd_greeks_vmap(*args[:5], h, Z)
    print(f"\nmax |forward - reverse| over price+3 Greeks: "
          f"{max(abs(float(fwd[g] - rev[g])) for g in fwd):.1e}")
    print(f"max |vmap bump - loop bump| over price+3 Greeks: "
          f"{max(abs(float(vec[g] - loop[g])) for g in vec):.1e}")

    rows = []
    for name, (fn, jitted, jargs) in methods.items():
        med, q25, q75, n = summary(time_runs(fn, min_reps=20, max_reps=200, budget_s=budget_s))
        c = cost(jitted, *jargs) if jitted is not None else None
        rows.append({"method": name, "median_ms": med * 1e3, "q25_ms": q25 * 1e3,
                     "q75_ms": q75 * 1e3, "runs": n,
                     "flops": c["flops"] if c else None,
                     "transcendentals": c["transcendentals"] if c else None})
    base = rows[0]
    # The loop is 7 calls of the compiled pricer: its op count is exactly 7 x one pricing.
    rows[3]["flops"] = 7 * base["flops"]
    rows[3]["transcendentals"] = 7 * base["transcendentals"]
    for r in rows:
        r["x_price"] = r["median_ms"] / base["median_ms"]
        r["flops_x_price"] = r["flops"] / base["flops"]
        r["transc_x_price"] = r["transcendentals"] / base["transcendentals"]

    print(f"\nTiming, N = {N:,} paths, price + delta, vega, rho "
          f"(median and IQR over repeated runs after compilation):")
    print(f"{'method':<13}{'median ms':>11}{'IQR ms':>18}{'runs':>6}{'x price':>9}"
          f"{'flops x':>9}{'exp/log x':>10}")
    for r in rows:
        print(f"{r['method']:<13}{r['median_ms']:>11.2f}   [{r['q25_ms']:6.2f}, {r['q75_ms']:6.2f}]"
              f"{r['runs']:>6}{r['x_price']:>9.2f}{r['flops_x_price']:>9.2f}"
              f"{r['transc_x_price']:>10.2f}")
    print("One output, 3 inputs: reverse mode is expected to beat forward mode "
          "(forward ~1 + 1-2x per input, reverse ~3-4x in total).\n"
          "AD evaluates each exp once (exp/log x = 1); bumping evaluates it 7 times.")

    with TIMING_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (TIMING_OUT.parent / "e1_hardware.json").write_text(json.dumps(hw, indent=2))


if __name__ == "__main__":
    validation()
    timing()
