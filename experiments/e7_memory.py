"""E7: memory of reverse-mode AD. One-shot vs chunked adjoint vs sqrt(M) checkpointing.

Question: is reverse mode's memory use (and the gap between its wall-clock and flop
ratios in E6) a property of adjoint AD, or an artefact of differentiating the mean
over all N paths in one pass? Giles & Glasserman (2006) run the adjoint path by
path, with small storage. mcgreeks.chunked does it B paths at a time (lax.scan
over batches), each batch drawing its own normals from fold_in(key, b), so TOTAL
memory (inputs + temporaries) is O(B).

Part A, basket call, d = 50: N in {1e4, 1e5, 1e6}; one-shot vs chunked B in {1e3, 1e4},
chunked with the Cholesky factor hoisted out of the batches (computed once, one vjp
back to rho); B = 1e3 also without hoisting, to show what the hoist saves.
Part B, arithmetic Asian call, N = 1e5, M in {12, 52, 252, 1000} dates: one-shot vs
chunked (B = 1e4) vs chunked with two-level sqrt(M) checkpointing (time blocks of
~sqrt(M) steps, each block jax.checkpoint-ed and drawing its own normals).

One-shot takes the full normals Z as an input, built from the SAME fold_in stream
as the B = 1e4 chunked run (keyed_normals / blocked_normals), so chunked == one-shot
is checked to 1e-10 on identical random numbers (B = 1e3 against its own stream).

Metrics:
  - total MB = input MB (Z; 0 for the keyed chunked programs, which take a key)
    + temp MB (compiled temporaries, compile().memory_analysis());
  - flops (primary): XLA's count on the lowered, pre-optimisation HLO
    (bench.cost_lowered). A loop body is counted once, so a chunked program's total
    is whole-program count + (n_batches - 1) x one batch, with the Asian time scans
    unrolled for counting. Each gradient is divided by the pricing that uses the
    same random numbers the same way: one-shot / one-shot price on Z, chunked /
    chunked price with the same in-batch RNG (so RNG work is in both).
Everything here is compile-only (memory_analysis, cost_analysis) plus one execution
per gradient for the correctness check, so the numbers are deterministic. Wall-clock
is measured separately, on whole pipelines, in e7_wallclock.py.

Run:  python experiments/e7_memory.py        (~1-2 minutes)
      python experiments/e7_memory.py --plot-only
Writes results/e7_memory.csv, results/e7_streams.csv, results/e7_hardware.json,
results/e7_memory.png. results/e7_memory_before_precomputedZ.csv is the previous
version (precomputed Z input, Cholesky per batch, per-step checkpoint).
"""
import csv
import json
import sys
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

import mcgreeks  # noqa: F401
from mcgreeks.asian import (asian_payoffs, asian_price, asian_price_blocked, blocked_normals,
                            sqrt_block)
from mcgreeks.basket import (basket_cholesky, basket_greeks, basket_price, basket_price_L,
                             basket_terminal)
from mcgreeks.bench import cost_lowered, print_hardware, temp_bytes
from mcgreeks.chunked import chunked_mean_keyed, chunked_value_and_grad_keyed, keyed_normals

R, T, K = 0.05, 1.0, 100.0
D, NS, BS = 50, [10_000, 100_000, 1_000_000], [10_000, 1_000]
N_ASIAN, MS, B_ASIAN = 100_000, [12, 52, 252, 1000], 10_000
MAX_TOTAL = 4e9
KEY = jax.random.PRNGKey(2027)
RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "e7_memory.csv"

# A method keeps one colour in every figure: one-shot reverse AD is the E6 aqua;
# the chunked variants are new methods and take palette slots 5-7 (never another
# method's colour), each with its own marker as secondary encoding.
# B = 1,000 is brown: the palette's green (#008300) was too close to the aqua, and
# red (#e34948) failed the normal-vision floor against the magenta (dataviz validator).
C_ONESHOT = "#1baf7a"
C_CHUNK_10K, C_CHUNK_1K, C_CKPT = "#e87ba4", "#9a5b13", "#4a3aa7"
INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def scan_flops(whole, body, n):
    """Total cost of a program whose lax.scan runs `body` n times (counted once)."""
    return {k: whole[k] + (n - 1) * body[k] for k in ("flops", "transcendentals")}


def measure(row, fn, jitted, args):
    """Memory only (compile, no execution). Wall-clock lives in e7_wallclock.py, which
    times whole pipelines fairly; `fn` is kept so call sites read the same."""
    row["temp_mb"] = temp_bytes(jitted, *args) / 1e6
    row["total_mb"] = row["input_mb"] + row["temp_mb"]
    assert row["total_mb"] * 1e6 <= MAX_TOTAL, row


def rel_diff(a, b):
    a, b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return max(float(jnp.max(jnp.abs(x - y) / jnp.abs(y))) for x, y in zip(a, b))


def ratios(row, ref):
    row["ref"] = ref["method"]
    row["flops_x"] = row["flops"] / ref["flops"]
    row["transc_x"] = row["transcendentals"] / ref["transcendentals"]


# ---- Part A: basket ---------------------------------------------------------------
def part_a():
    d = D
    S0, sigma = jnp.full(d, 100.0), jnp.linspace(0.15, 0.35, d)
    rho, w = jnp.full(d * (d - 1) // 2, 0.5), jnp.full(d, 1.0 / d)
    params = (S0, sigma, rho)

    def prepare(p):
        return p[0], p[1], basket_cholesky(p[2], d)

    rows = []
    for N in NS:
        base = {"part": "basket", "N": N, "M": ""}
        refs = {}   # one-shot reference per stream (B)
        for B in BS:
            if B > N:
                continue
            nb = N // B

            def batch_L(a, k, B=B):
                return basket_price_L(*a, R, T, K, w, jax.random.normal(k, (B, d)))

            def batch_rho(p, k, B=B):
                return basket_price(*p, R, T, K, w, jax.random.normal(k, (B, d)))

            Z = keyed_normals(KEY, nb, (B, d))
            a = (S0, sigma, rho, R, T, K, w, Z)
            refs[B] = basket_greeks(*a)
            ref_vg = (refs[B]["price"], (refs[B]["delta"], refs[B]["vega"], refs[B]["corr"]))
            if B == BS[0]:   # one-shot rows, on this stream
                inp = Z.nbytes / 1e6
                price = {**base, "method": "price, one-shot", "B": "", "input_mb": inp,
                         **cost_lowered(basket_price, *a)}
                measure(price, lambda: basket_price(*a).block_until_ready(), basket_price, a)
                one = {**base, "method": "one-shot", "B": "", "input_mb": inp,
                       **cost_lowered(basket_greeks, *a)}
                measure(one, lambda: basket_greeks(*a)["corr"].block_until_ready(),
                        basket_greeks, a)
                ratios(price, price)
                ratios(one, price)
                rows += [price, one]
            del Z, a

            aux = prepare(params)
            kb = jax.random.fold_in(KEY, 0)
            pr_f = chunked_mean_keyed(batch_L, nb)
            pr = {**base, "method": f"price, chunked B={B:,}", "B": B, "input_mb": 0.0,
                  **scan_flops(cost_lowered(pr_f, aux, KEY),
                               cost_lowered(jax.jit(batch_L), aux, kb), nb)}
            measure(pr, lambda: pr_f(aux, KEY).block_until_ready(), pr_f, (aux, KEY))
            ratios(pr, pr)
            rows.append(pr)
            variants = [(f"chunked B={B:,}", batch_L, prepare)]
            if B == 1_000:
                variants.append((f"chunked B={B:,}, Cholesky per batch", batch_rho, None))
            for name, fn, prep in variants:
                f = chunked_value_and_grad_keyed(fn, nb, prepare=prep)
                body_args = (aux if prep else params, kb)
                ch = {**base, "method": name, "B": B, "input_mb": 0.0,
                      **scan_flops(cost_lowered(f, params, KEY),
                                   cost_lowered(jax.jit(jax.value_and_grad(fn)), *body_args), nb)}
                ch["max_rel_diff_vs_oneshot"] = rel_diff(f(params, KEY), ref_vg)
                assert ch["max_rel_diff_vs_oneshot"] < 1e-10, ch
                measure(ch, lambda f=f: jax.block_until_ready(f(params, KEY)), f, (params, KEY))
                ratios(ch, pr)
                rows.append(ch)
    return rows


# ---- Part B: Asian -----------------------------------------------------------------
def part_b():
    params = (jnp.float64(100.0), jnp.float64(0.2))
    B, nb = B_ASIAN, N_ASIAN // B_ASIAN
    price_j = jax.jit(lambda p, z, u=1: asian_price(p[0], p[1], R, T, K, z, unroll=u),
                      static_argnums=2)
    vg_j = jax.jit(jax.value_and_grad(lambda p, z, u=1: asian_price(p[0], p[1], R, T, K, z,
                                                                    unroll=u)),
                   static_argnums=2)
    rows = []
    for M in MS:
        m = sqrt_block(M)
        base = {"part": "asian", "N": N_ASIAN, "M": M, "block": m}
        Z = jnp.concatenate([blocked_normals(jax.random.fold_in(KEY, b), B, M, m)
                             for b in range(nb)], axis=1)
        inp = Z.nbytes / 1e6
        price = {**base, "method": "price, one-shot", "B": "", "input_mb": inp,
                 **cost_lowered(price_j, params, Z, True)}
        measure(price, lambda: price_j(params, Z).block_until_ready(), price_j, (params, Z))
        one = {**base, "method": "one-shot", "B": "", "input_mb": inp,
               **cost_lowered(vg_j, params, Z, True)}
        measure(one, lambda: jax.block_until_ready(vg_j(params, Z)), vg_j, (params, Z))
        ref = vg_j(params, Z)
        ratios(price, price)
        ratios(one, price)
        rows += [price, one]
        del Z

        def batch(p, k, ckpt=False, unroll=1):
            return asian_price_blocked(p[0], p[1], R, T, K, k, B, M, m, ckpt, unroll)

        kb = jax.random.fold_in(KEY, 0)
        pr_f = chunked_mean_keyed(batch, nb)
        pr_body = jax.jit(lambda p, k: batch(p, k, unroll=True))
        pr_whole = chunked_mean_keyed(lambda p, k: batch(p, k, unroll=True), nb)
        pr = {**base, "method": "price, chunked", "B": B, "input_mb": 0.0,
              **scan_flops(cost_lowered(pr_whole, params, KEY),
                           cost_lowered(pr_body, params, kb), nb)}
        measure(pr, lambda: pr_f(params, KEY).block_until_ready(), pr_f, (params, KEY))
        ratios(pr, pr)
        rows.append(pr)
        for ckpt in (False, True):
            f = chunked_value_and_grad_keyed(lambda p, k, c=ckpt: batch(p, k, c), nb)
            whole = chunked_value_and_grad_keyed(lambda p, k, c=ckpt: batch(p, k, c, True), nb)
            body = jax.jit(jax.value_and_grad(lambda p, k, c=ckpt: batch(p, k, c, True)))
            ch = {**base, "method": "chunked + sqrt(M) checkpoint" if ckpt else "chunked",
                  "B": B, "input_mb": 0.0,
                  **scan_flops(cost_lowered(whole, params, KEY),
                               cost_lowered(body, params, kb), nb)}
            ch["max_rel_diff_vs_oneshot"] = rel_diff(f(params, KEY), ref)
            assert ch["max_rel_diff_vs_oneshot"] < 1e-10, ch
            measure(ch, lambda f=f: jax.block_until_ready(f(params, KEY)), f, (params, KEY))
            ratios(ch, pr)
            rows.append(ch)
    return rows


# ---- before / after: the estimates move with the random numbers, within MC error ----
def streams():
    """One-shot price +/- SE on the old precomputed Z (crc32 tags) vs the new fold_in
    stream. Different random numbers, so the estimates differ by Monte Carlo noise."""
    d = D
    S0, sigma = jnp.full(d, 100.0), jnp.linspace(0.15, 0.35, d)
    rho, w = jnp.full(d * (d - 1) // 2, 0.5), jnp.full(d, 1.0 / d)

    def old(tag, shape):
        return jax.random.normal(jax.random.PRNGKey(zlib.crc32(tag.encode())), shape,
                                 dtype=jnp.float64)

    def mse(x):
        return float(jnp.mean(x)), float(jnp.std(x, ddof=1) / jnp.sqrt(x.size))

    out = []
    for N in NS:
        for label, Z in (("before", old(f"e7-basket-{N}", (N, d))),
                         ("after", keyed_normals(KEY, N // BS[0], (BS[0], d)))):
            ST = basket_terminal(S0, sigma, rho, R, T, Z)
            m, se = mse(jnp.exp(-R * T) * jnp.maximum(ST @ w - K, 0.0))
            out.append({"part": "basket", "N": N, "M": "", "stream": label, "price": m, "se": se})
    for M in MS:
        m_ = sqrt_block(M)
        for label, Z in (("before", old(f"e7-asian-{M}", (M, N_ASIAN))),
                         ("after", jnp.concatenate(
                             [blocked_normals(jax.random.fold_in(KEY, b), B_ASIAN, M, m_)
                              for b in range(N_ASIAN // B_ASIAN)], axis=1))):
            m, se = mse(asian_payoffs(100.0, 0.2, R, T, K, Z))
            out.append({"part": "asian", "N": N_ASIAN, "M": M, "stream": label, "price": m,
                        "se": se})
    for a, b in zip(out[::2], out[1::2]):
        b["z_vs_before"] = (b["price"] - a["price"]) / (a["se"]**2 + b["se"]**2) ** 0.5
    return out


def report(rows, st):
    print(f"\n{'part':<7}{'N':>10}{'M':>6}  {'method':<38}{'input':>8}{'temp':>9}{'total MB':>10}"
          f"{'flops x':>9}{'exp x':>7}{'rel diff':>10}")
    for r in rows:
        rd = r.get("max_rel_diff_vs_oneshot")
        print(f"{r['part']:<7}{r['N']:>10,}{str(r['M']):>6}  {r['method']:<38}{r['input_mb']:>8.1f}"
              f"{r['temp_mb']:>9.1f}{r['total_mb']:>10.1f}{r['flops_x']:>9.2f}{r['transc_x']:>7.2f}"
              f"{'' if rd is None else f'{rd:.1e}':>10}")
    print("\nOne-shot price +/- SE, precomputed-Z stream (before) vs fold_in stream (after):")
    for a, b in zip(st[::2], st[1::2]):
        print(f"  {a['part']:<7} N={a['N']:>9,} M={str(a['M']):>4}: before {a['price']:.5f} +/- "
              f"{a['se']:.5f}   after {b['price']:.5f} +/- {b['se']:.5f}   z = {b['z_vs_before']:+.2f}")


def main():
    hw = print_hardware()
    rows = part_a() + part_b()
    st = streams()
    report(rows, st)
    RESULTS.mkdir(exist_ok=True)
    for path, data in ((OUT, rows), (RESULTS / "e7_streams.csv", st)):
        fields = list(dict.fromkeys(k for r in data for k in r))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    (RESULTS / "e7_hardware.json").write_text(json.dumps(hw, indent=2))
    plot(rows)


def plot(rows):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURF, sharey=True)
    for ax in (a1, a2):           # scales first: set_xscale resets tick locators
        ax.set_xscale("log")
        ax.set_yscale("log")

    def series(ax, part, method, xkey, **kw):
        pts = [r for r in rows if r["part"] == part and r["method"] == method]
        ax.plot([r[xkey] for r in pts], [r["total_mb"] for r in pts], **kw)

    line = {"lw": 2, "ms": 5}
    for ax, part, x in ((a1, "basket", "N"), (a2, "asian", "M")):
        series(ax, part, "price, one-shot", x, ls=":", color=MUTED, lw=1.5,
               label="One pricing, all N paths (reference)")
        series(ax, part, "one-shot", x, ls="-", marker="o", color=C_ONESHOT,
               label="Reverse AD, one-shot (all N paths)", **line)
    series(a1, "basket", "chunked B=10,000", "N", ls="-", marker="s", color=C_CHUNK_10K,
           label="Reverse AD, chunked, B = 10,000", **line)
    series(a1, "basket", "chunked B=1,000", "N", ls="-", marker="^", color=C_CHUNK_1K,
           label="Reverse AD, chunked, B = 1,000", **line)
    series(a2, "asian", "chunked", "M", ls="-", marker="s", color=C_CHUNK_10K,
           label=f"Reverse AD, chunked, B = {B_ASIAN:,}", **line)
    series(a2, "asian", "chunked + sqrt(M) checkpoint", "M", ls="-", marker="D", color=C_CKPT,
           label=f"Chunked, B = {B_ASIAN:,}, + sqrt(M)-block checkpoint", **line)

    a1.set_title(f"Basket call, d = {D}: memory vs paths N", loc="left", fontsize=11)
    a1.set_xlabel("Number of paths N")
    a1.xaxis.set_major_locator(FixedLocator(NS))
    a1.set_xticklabels([f"{n:,}" for n in NS])
    a2.set_title(f"Arithmetic Asian call, N = {N_ASIAN:,}: memory vs dates M", loc="left",
                 fontsize=11)
    a2.set_xlabel("Number of monitoring dates M (time steps)")
    a2.xaxis.set_major_locator(FixedLocator(MS))
    a2.set_xticklabels([f"{m:,}" for m in MS])

    yt = [1, 10, 100, 1000, 10000]
    for ax in (a1, a2):
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_facecolor(SURF)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    a1.yaxis.set_major_locator(FixedLocator(yt))
    a1.yaxis.set_minor_locator(NullLocator())
    a1.set_yticklabels([f"{t:,}" for t in yt])
    lo = min(r["total_mb"] for r in rows if r["total_mb"] > 0)
    a1.set_ylim(lo / 2, 6e4)      # headroom above the data for the legends
    a1.set_ylabel("Total memory, MB (input Z + compiled temporaries)")
    fig.suptitle("Reverse-mode memory: one-shot grows with what it stores; chunking caps it "
                 "in N, sqrt(M) checkpointing in M", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "e7_memory.png", dpi=160, facecolor=SURF)
    print(f"\nSaved {RESULTS / 'e7_memory.png'}")


def plot_from_csv():
    with OUT.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("N", "M"):
            r[k] = int(r[k]) if r[k] else ""
        r["total_mb"] = float(r["total_mb"])
    plot(rows)


if __name__ == "__main__":
    plot_from_csv() if "--plot-only" in sys.argv else main()
