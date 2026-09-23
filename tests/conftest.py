"""Collects the false-alarm ledger of tests/helpers.py and prints the family-wise
false-alarm probability of the run at the end of the pytest summary (docs/TESTING.md)."""
import math

import pytest

import helpers


@pytest.fixture(autouse=True)
def _current_test(request):
    helpers.CURRENT["test"] = request.node.nodeid
    yield
    helpers.CURRENT["test"] = None


def pytest_terminal_summary(terminalreporter):
    if not helpers.LEDGER:
        return
    n = sum(c for _, c, _ in helpers.LEDGER)
    expected = sum(c * p for _, c, p in helpers.LEDGER)
    fwer = 1.0 - math.exp(sum(c * math.log1p(-p) for _, c, p in helpers.LEDGER))
    by_file = {}
    for test, c, p in helpers.LEDGER:
        f = (test or "?").split("::")[0]
        a, b = by_file.get(f, (0, 0.0))
        by_file[f] = (a + c, b + c * p)
    tr = terminalreporter
    tr.write_sep("-", "statistical false-alarm budget (docs/TESTING.md)")
    tr.write_line(f"{n} statistical comparisons; expected false alarms {expected:.4f}; "
                  f"P(at least one) <= {fwer:.2%} if independent")
    for f, (c, e) in sorted(by_file.items(), key=lambda x: -x[1][1]):
        tr.write_line(f"  {f:<34} {c:>6} comparisons   expected false alarms {e:.5f}")
    tr.write_line(f"rare-event branches (structural checks instead of SE, < "
                  f"{helpers.RARE_THRESHOLD} minority paths): {len(helpers.RARE_LOG)}")
    for test, what in helpers.RARE_LOG:
        tr.write_line(f"  {test.split('::')[-1]}: {what}")
