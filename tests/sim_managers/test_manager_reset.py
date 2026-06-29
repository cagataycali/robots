"""Manager.reset propagates to stateful terms (e.g. feet_air_time, command)."""

from __future__ import annotations

import numpy as np

from strands_robots.sim_managers import RewardManager


def test_reset_clears_feet_air_time_history():
    mgr = RewardManager.from_config({"terms": [{"func": "feet_air_time", "weight": 1.0}]})
    label, term = next(iter(mgr))
    assert label == "feet_air_time"
    # set internal history then reset it
    term._prev_contact = np.array([True, True])
    mgr.reset()
    assert term._prev_contact is None
