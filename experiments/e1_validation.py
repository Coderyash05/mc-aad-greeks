"""E1: MC price vs Black-Scholes across strikes, maturities and vols.

Run:  python experiments/e1_validation.py
Writes results/e1_prices.csv and prints the table.
"""
import csv
import time
from pathlib import Path

import jax

import mcgreeks  # noqa: F401
from mcgreeks.black_scholes import bs_price
from mcgreeks.models import normals
from mcgreeks.pricer import mc_price_se

S0, R, N = 100.0, 0.05, 1_000_000
OUT = Path(__file__).resolve().parents[1] / "results" / "e1_prices.csv"


def main():
    # A fresh key per case, so the z-scores are independent. Reusing one Z for
    # every case makes all errors move together (common random numbers).
    keys = iter(jax.random.split(jax.random.PRNGKey(42), 54))
    mc_price_se(S0, 0.2, R, 1.0, 100.0, normals(next(keys), N))[0].block_until_ready()  # compile
    keys = iter(jax.random.split(jax.random.PRNGKey(42), 54))

    rows = []
    t0 = time.perf_counter()
    for kind in ("call", "put"):
        for sigma in (0.1, 0.2, 0.4):
            for T in (0.25, 1.0, 2.0):
                for K in (80.0, 100.0, 120.0):
                    p, se = mc_price_se(S0, sigma, R, T, K, normals(next(keys), N), kind=kind)
                    exact = float(bs_price(S0, K, R, sigma, T, kind=kind))
                    z = (float(p) - exact) / float(se)
                    rows.append([kind, sigma, T, K, exact, float(p), float(se), z])
    elapsed = time.perf_counter() - t0

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "sigma", "T", "K", "bs", "mc", "se", "z_score"])
        w.writerows(rows)

    print(f"{'kind':<5}{'sigma':>6}{'T':>6}{'K':>6}{'BS':>10}{'MC':>10}{'SE':>9}{'z':>7}")
    for r in rows:
        print(f"{r[0]:<5}{r[1]:>6}{r[2]:>6}{r[3]:>6.0f}{r[4]:>10.4f}{r[5]:>10.4f}{r[6]:>9.4f}{r[7]:>7.2f}")
    z = [r[7] for r in rows]
    within = sum(abs(v) < 3 for v in z)
    print(f"\n{len(rows)} cases, N={N:,} paths each, {elapsed:.2f}s total")
    print(f"within 3 SE: {within}/{len(rows)}   max |z| = {max(map(abs, z)):.2f}   "
          f"mean z = {sum(z)/len(z):+.2f}")


if __name__ == "__main__":
    main()
