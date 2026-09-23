"""Benchmark helpers shared by the experiments: timing, hardware, operation counts.

Timing protocol (used for every number in the README):
  1. one untimed call, so compilation and first-touch allocation are excluded;
  2. repeated timed calls, each ending in block_until_ready (JAX is asynchronous),
     with Python's garbage collector off, as the stdlib `timeit` does;
  3. enough repeats to fill ~`budget_s` seconds, at least `min_reps` (3 for the
     slow cases), at most `max_reps`;
  4. report the median and the interquartile range (IQR), which are robust to the
     occasional slow run caused by the OS scheduler.
"""
import gc
import math
import os
import platform
import time

import jax
import numpy as np


def time_runs(fn, min_reps=3, max_reps=100, budget_s=2.0, warm=True):
    """Seconds per call of fn(), which must block until its result is ready.

    warm=False skips the untimed first call; only use it if the caller has
    already run fn once (e.g. for a correctness check), so it is compiled.
    """
    if warm:
        fn()  # compile + warm-up, not timed
    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter()
        fn()
        first = time.perf_counter() - t0
        times = [first]
        reps = max(min_reps, min(max_reps, math.ceil(budget_s / max(first, 1e-9))))
        for _ in range(reps - 1):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        if gc_was_on:
            gc.enable()
    return np.array(times)


def summary(times):
    """(median, 25th percentile, 75th percentile, number of runs)."""
    q25, med, q75 = np.percentile(times, [25, 50, 75])
    return float(med), float(q25), float(q75), len(times)


def _cpu_name():
    """Marketing name of the CPU, where the OS exposes it (platform.processor()
    on Windows only gives the family/model/stepping string)."""
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def hardware():
    return {
        "processor": platform.processor() or "unknown",
        "cpu_name": _cpu_name(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }


def print_hardware():
    hw = hardware()
    print(f"Hardware: {hw['cpu_name'] or hw['processor']}  ({hw['processor']})")
    print(f"          os.cpu_count() = {hw['cpu_count']}, {hw['os']}")
    print(f"          Python {hw['python']}, JAX {hw['jax']}, backend = {hw['backend']} "
          f"({hw['device']})")
    return hw


def cost(jitted, *args, **kwargs):
    """XLA's static cost of one call: {'flops', 'transcendentals', 'bytes'}.

    From jitted.lower(...).compile().cost_analysis(). Caveats, measured on this
    JAX version: (a) a while/scan/lax.map loop body is counted ONCE regardless of
    trip count, so only loop-free programs give totals; (b) library custom calls
    (e.g. the LAPACK Cholesky) are not counted; (c) exp/log are 'transcendentals',
    not 'flops'; (d) 'bytes' is XLA's 'bytes accessed': reads + writes of every
    fused kernel's operands and results, i.e. modelled main-memory traffic, not a
    hardware counter. Returns None values if the backend provides no analysis.
    """
    ca = jitted.lower(*args, **kwargs).compile().cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else None
    if not ca:
        return {"flops": None, "transcendentals": None, "bytes": None}
    return {"flops": ca.get("flops"), "transcendentals": ca.get("transcendentals"),
            "bytes": ca.get("bytes accessed")}


def temp_bytes(jitted, *args, **kwargs):
    """Compiled temporary-buffer size (bytes) from memory_analysis(), or None."""
    ma = jitted.lower(*args, **kwargs).compile().memory_analysis()
    return getattr(ma, "temp_size_in_bytes", None) if ma is not None else None
