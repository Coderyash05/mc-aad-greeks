"""E1 (Greeks): autodiff delta, vega, rho vs Black-Scholes, plus the cost of AD.

Run:  python experiments/e1_greeks.py
Writes results/e1_greeks.csv.
"""
import csv
import time
from pathlib import Path

import jax

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_greeks
from mcgreeks.greeks_ad import ad_greeks, ad_greeks_se
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price

S0, R, N = 100.0, 0.05, 1_000_000
OUT = Path(__file__).resolve().parents[1] / "results" / "e1_greeks.csv"
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


def timing(reps=50):
    """Cost of price + 3 Greeks (one reverse pass) relative to price alone."""
    Z = normals(jax.random.PRNGKey(0), N)
    args = (S0, 0.2, R, 1.0, 100.0, Z)
    mc_price(*args).block_until_ready()          # compile
    ad_greeks(*args)["delta"].block_until_ready()  # compile

    t0 = time.perf_counter()
    for _ in range(reps):
        mc_price(*args).block_until_ready()
    t_price = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        ad_greeks(*args)["delta"].block_until_ready()
    t_ad = (time.perf_counter() - t0) / reps

    # Bump-and-reprice for the same 3 Greeks: base + 2 bumps per parameter.
    h = 1e-2
    def bumped(a):
        S0_, sig_, r_, T_, K_, Z_ = a
        out = [mc_price(*a)]
        for d in ((h, 0, 0), (-h, 0, 0), (0, h, 0), (0, -h, 0), (0, 0, h), (0, 0, -h)):
            out.append(mc_price(S0_ + d[0], sig_ + d[1], r_ + d[2], T_, K_, Z_))
        return out
    bumped(args)[-1].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(reps):
        bumped(args)[-1].block_until_ready()
    t_fd = (time.perf_counter() - t0) / reps

    print(f"\nTiming, N = {N:,} paths (mean of {reps} runs, after compilation):")
    print(f"  price only                    {t_price * 1e3:7.2f} ms   (1.0x)")
    print(f"  price + 3 Greeks, autodiff    {t_ad * 1e3:7.2f} ms   ({t_ad / t_price:.1f}x)")
    print(f"  price + 3 Greeks, bumping     {t_fd * 1e3:7.2f} ms   ({t_fd / t_price:.1f}x)")


if __name__ == "__main__":
    validation()
    timing()
