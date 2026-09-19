"""The mesh rate-limit window a test spends is refunded before the next one.

``robot_mesh``'s sliding window is process-global: a consumed slot outlives
the test that took it, and a later test's accepted call then comes back
"rate limit exceeded" rather than doing the thing it asserts. Ten modules
used to reset the window in a fixture of their own; the session owns it now
(``tests/conftest.py::_mesh_rate_limit_history_is_left_empty``), and these two
cells - in this order, which is the order pytest runs them - are what says so.
"""

from __future__ import annotations

import strands_robots.tools.robot_mesh as rmt

ACTION = "tell"


def test_a_test_may_spend_the_whole_window() -> None:
    """Draining the window is legitimate - several tests do it deliberately."""
    limit, _window = rmt._RATE_LIMITS[ACTION]
    for _ in range(limit):
        assert rmt._rate_limit_check_and_record(ACTION) is None
    assert rmt._rate_limit_check(ACTION) is not None


def test_the_next_test_finds_the_window_empty() -> None:
    """The refund happened between the two cells, with no fixture in this file."""
    assert rmt._rate_limit_check(ACTION) is None
