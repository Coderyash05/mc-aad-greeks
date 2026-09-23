"""Paired bootstrap: exact ratio for identical methods, coverage, reproducibility."""
import numpy as np

from mcgreeks.stats import paired_bootstrap, verdict


def test_identical_methods_give_ratio_exactly_one():
    e = np.random.default_rng(1).normal(0.0, 1.0, 500)
    res = paired_bootstrap({"a": e, "b": e.copy()}, seed=3, reference="a")
    b = res["methods"]["b"]
    assert b["ratio"] == 1.0 and b["ratio_ci"] == (1.0, 1.0)
    assert verdict(res, "b") == "not distinguishable from a"


def test_ci_covers_true_rmse_about_95_percent():
    """Errors ~ N(0, sigma^2): true RMSE = sigma. 200 independent experiments of
    R = 200 batches each; the 95% percentile CI should cover sigma ~95% of the time.
    Binomial(200, 0.95) has sd 3.1 covers, so 90-99% is a +/- 3 sd band."""
    sigma, rng = 0.37, np.random.default_rng(2)
    covered = 0
    for rep in range(200):
        e = rng.normal(0.0, sigma, 200)
        lo, hi = paired_bootstrap({"m": e}, n_boot=2000, seed=rep)["methods"]["m"]["ci"]
        covered += lo <= sigma <= hi
    assert 0.90 <= covered / 200 <= 0.99, covered


def test_reproducible_for_fixed_seed():
    rng = np.random.default_rng(4)
    errs = {"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1.2, 300)}
    r1, r2 = paired_bootstrap(errs, seed=11), paired_bootstrap(errs, seed=11)
    assert r1 == r2
    assert paired_bootstrap(errs, seed=12)["methods"]["b"]["ci"] != r1["methods"]["b"]["ci"]


def test_clearly_different_methods_are_distinguished():
    """Same batches, method b = 2 x method a's error: RMSE ratio exactly 2 in every
    replicate, so the CI is [2, 2] and a beats b."""
    e = np.random.default_rng(5).normal(0, 1, 200)
    res = paired_bootstrap({"a": e, "b": 2 * e}, seed=0)
    assert res["reference"] == "a"
    np.testing.assert_allclose(res["methods"]["b"]["ratio_ci"], (2.0, 2.0))
    assert verdict(res, "b") == "worse than a"


def test_pairing_tightens_the_ratio_interval():
    """Correlated errors (common random numbers): the paired ratio CI is much
    narrower than the one from resampling the two methods independently."""
    rng = np.random.default_rng(6)
    common = rng.normal(0, 1, 400)
    a, b = common + 0.1 * rng.normal(0, 1, 400), 1.05 * common + 0.1 * rng.normal(0, 1, 400)
    paired = paired_bootstrap({"a": a, "b": b}, seed=0, reference="a")["methods"]["b"]["ratio_ci"]
    ia = paired_bootstrap({"a": a}, seed=1)["methods"]["a"]["ci"]
    ib = paired_bootstrap({"b": b}, seed=2)["methods"]["b"]["ci"]
    unpaired_width = ib[1] / ia[0] - ib[0] / ia[1]
    assert paired[1] - paired[0] < 0.25 * unpaired_width
