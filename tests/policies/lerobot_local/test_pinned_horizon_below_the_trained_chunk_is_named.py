"""A re-query interval pinned below the trained chunk is named, not silent.

``_auto_detect_actions_per_step`` corrects the default ``actions_per_step=1``
up to the checkpoint's ``config.n_action_steps`` precisely because re-querying
more often than the model was trained to replay is out of distribution. A
caller-pinned interval is never overridden - but one strictly BELOW the trained
chunk lands in that same regime, truncating every chunk to its prefix and
starting every re-query from a state the checkpoint never replayed to, and it
used to pass with nothing logged.

RTC is the supported way to shorten the interval (it blends the unexecuted tail
of the previous chunk into the next one), so a caller who requested RTC is not
warned; there ``rtc_execution_horizon`` owns the interval and
``actions_per_step`` stays the trained chunk.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


class _Cfg:
    """Loaded-policy config stub. ``rtc_config`` present == flow-matching."""

    def __init__(self, *, n_action_steps=50, flow_matching=True, temporal_ensemble_coeff=None):
        self.n_obs_steps = 1
        self.n_action_steps = n_action_steps
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        if flow_matching:
            self.rtc_config = None  # every public flow-matching checkpoint ships None


class _Pol:
    def __init__(self, cfg):
        self.config = cfg


def _unloaded(**kw) -> LerobotLocalPolicy:
    from unittest.mock import patch

    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(pretrained_name_or_path="test/model", **kw)


# (pinned, trained, flow_matching, rtc_requested, expect_warning)
CASES = [
    (10, 50, True, None, True),  # the reactivity shortcut: truncates the chunk
    (5, 50, True, None, True),  # any value below the trained chunk
    (50, 50, True, None, False),  # exactly the trained chunk - as trained
    (100, 50, True, None, False),  # above it - the chunk drains before re-query
    (10, 50, True, True, False),  # RTC asked for: it owns the interval, seam blended
    (10, 100, False, None, True),  # ACT-style: warned, but RTC is not the remedy
]


@pytest.mark.parametrize("pinned,trained,flow,rtc,expect", CASES)
def test_only_a_horizon_below_the_trained_chunk_is_named(pinned, trained, flow, rtc, expect, caplog):
    kw = {"actions_per_step": pinned}
    if rtc is not None:
        kw["rtc_enabled"] = rtc
    pol = _unloaded(**kw)
    pol._policy = _Pol(_Cfg(n_action_steps=trained, flow_matching=flow))
    with caplog.at_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.policy"):
        pol._auto_detect_actions_per_step()
    named = [r.getMessage() for r in caplog.records if "actions_per_step" in r.getMessage()]
    assert bool(named) is expect, named
    # A pinned horizon is reported, never overridden.
    assert pol.actions_per_step == pinned
    if expect:
        text = named[0]
        assert str(trained) in text and str(pinned) in text
        # The remedy must be the one that applies to THIS policy family.
        assert ("rtc_enabled=True" in text) is flow, text


def test_the_default_is_still_corrected_up_to_the_trained_chunk(caplog):
    """The pinned-horizon notice must not disturb the default auto-adopt path."""
    pol = _unloaded()
    pol._policy = _Pol(_Cfg(n_action_steps=50))
    assert pol.actions_per_step == 1
    pol._auto_detect_actions_per_step()
    assert pol.actions_per_step == 50


def test_temporal_ensembling_still_overrides_rather_than_only_warning():
    """The ensembling branch returns first and keeps its override to 1."""
    pol = _unloaded(actions_per_step=10)
    pol._policy = _Pol(_Cfg(n_action_steps=100, flow_matching=False, temporal_ensemble_coeff=0.01))
    pol._auto_detect_actions_per_step()
    assert pol.actions_per_step == 1
