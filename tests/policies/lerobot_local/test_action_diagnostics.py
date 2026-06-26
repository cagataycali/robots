"""Diagnostics for the "policy runs but the robot does not move" failure mode.

A VLA (e.g. MolmoAct2 on SO-101) can step every control tick yet leave the arm
motionless when its action vector mis-matches the embodiment's actuator count
(unmatched actuators are zero-filled) or when it keeps emitting near-zero
actions (a broken obs/rename pipeline starves the model). These were silent.
This pins the surfaced diagnostics: a one-shot action-dim warning, a
consecutive near-zero-action warning, and an end-to-end MuJoCo check that a
correctly-mapped action actually moves the joints.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch  # real or conftest mock - both work

from strands_robots.policies.lerobot_local.embodiment import (
    ZeroActionMonitor,
    diagnose_action_dim,
    load_embodiment,
)
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# diagnose_action_dim (pure)


def test_diagnose_action_dim_match_is_silent():
    assert diagnose_action_dim(6, 6) is None
    assert diagnose_action_dim(6, 6, name="so101") is None


def test_diagnose_action_dim_fewer_values_flags_zero_fill():
    msg = diagnose_action_dim(4, 6, name="so101")
    assert msg is not None
    assert "zero-filled" in msg
    assert "so101" in msg
    # names the count of frozen actuators
    assert "2" in msg


def test_diagnose_action_dim_more_values_flags_dropped():
    msg = diagnose_action_dim(8, 6)
    assert msg is not None
    assert "dropped" in msg


# ZeroActionMonitor (pure)


def test_zero_action_monitor_fires_once_after_patience():
    mon = ZeroActionMonitor(threshold=1e-3, patience=5)
    fired = [mon.update(0.0) for _ in range(8)]
    hits = [m for m in fired if m]
    assert len(hits) == 1  # exactly one warning, not per-step spam
    assert "near-zero" in hits[0]
    # warning lands on the patience-th near-zero step (index 4), not before
    assert all(m is None for m in fired[:4])
    assert fired[4] is not None


def test_zero_action_monitor_above_threshold_rearms():
    mon = ZeroActionMonitor(threshold=1e-3, patience=3)
    assert all(mon.update(0.0) is None for _ in range(2))
    assert mon.update(1.0) is None  # a real action clears the streak
    assert all(mon.update(0.0) is None for _ in range(2))
    assert mon.update(0.0) is not None  # streak rebuilt -> warns again


def test_zero_action_monitor_reset_clears_state():
    mon = ZeroActionMonitor(threshold=1e-3, patience=2)
    assert mon.update(0.0) is None
    assert mon.update(0.0) is not None
    mon.reset()
    assert mon.update(0.0) is None  # streak restarted from zero


@pytest.mark.parametrize("bad", [{"threshold": -1.0}, {"patience": 0}])
def test_zero_action_monitor_rejects_bad_config(bad):
    with pytest.raises(ValueError):
        ZeroActionMonitor(**bad)


# Wiring into LerobotLocalPolicy._tensor_to_action_dicts


def test_tensor_to_action_dicts_warns_on_dim_mismatch(caplog):
    policy = LerobotLocalPolicy()
    policy.set_robot_state_keys(["1", "2", "3", "4", "5", "6"])
    with caplog.at_level(logging.WARNING):
        result = policy._tensor_to_action_dicts(torch.zeros(4))
    # unmatched actuators are still returned (zero-filled), but now flagged
    assert set(result[0].keys()) == {"1", "2", "3", "4", "5", "6"}
    assert result[0]["5"] == 0.0 and result[0]["6"] == 0.0
    assert any("action dim 4" in r.message and "zero-filled" in r.message for r in caplog.records)


def test_tensor_to_action_dicts_dim_warning_is_one_shot(caplog):
    policy = LerobotLocalPolicy()
    policy.set_robot_state_keys(["1", "2", "3", "4", "5", "6"])
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            policy._tensor_to_action_dicts(torch.zeros(4))
    dim_warnings = [r for r in caplog.records if "actuator count" in r.message]
    assert len(dim_warnings) == 1


def test_tensor_to_action_dicts_warns_on_persistent_near_zero(caplog):
    policy = LerobotLocalPolicy()
    policy.set_robot_state_keys(["1", "2", "3", "4", "5", "6"])
    with caplog.at_level(logging.WARNING):
        for _ in range(policy._zero_action_monitor.patience + 2):
            policy._tensor_to_action_dicts(torch.zeros(6))
    near_zero = [r for r in caplog.records if "near-zero actions" in r.message]
    assert len(near_zero) == 1


def test_reset_rearms_near_zero_warning(caplog):
    policy = LerobotLocalPolicy()
    policy.set_robot_state_keys(["1", "2", "3", "4", "5", "6"])
    with caplog.at_level(logging.WARNING):
        for _ in range(policy._zero_action_monitor.patience):
            policy._tensor_to_action_dicts(torch.zeros(6))
        policy.reset()
        caplog.clear()
        for _ in range(policy._zero_action_monitor.patience):
            policy._tensor_to_action_dicts(torch.zeros(6))
    assert any("near-zero actions" in r.message for r in caplog.records)


def test_nonzero_actions_never_warn(caplog):
    policy = LerobotLocalPolicy()
    policy.set_robot_state_keys(["1", "2", "3"])
    with caplog.at_level(logging.WARNING):
        for _ in range(20):
            policy._tensor_to_action_dicts(torch.tensor([0.3, -0.5, 0.9]))
    assert not [r for r in caplog.records if "near-zero" in r.message]


# End-to-end: a correctly-mapped action moves SO-101 joints in MuJoCo


def test_so101_mapped_action_produces_visible_motion():
    """A degree-space action mapped through the so101 embodiment must move the
    arm >5 deg on at least one joint within 20 steps (acceptance for the
    "not moving" report). Pins that the deg->rad / gripper RANGE_0_100 mapping
    yields real motion, not a saturated freeze.
    """
    pytest.importorskip("mujoco")
    from strands_robots import create_simulation

    sim = create_simulation("mujoco")
    try:
        sim.create_world()
        sim.add_robot("so101")
        emb = load_embodiment("so101")
        joints = emb.action_keys

        def read():
            obs = sim.get_observation("so101", skip_images=True)
            return {k: float(np.ravel(obs[k])[0]) for k in joints}

        before = read()
        # SO-arm checkpoint convention: arm joints in DEGREES, gripper 0..100.
        deg_action = [30.0, 30.0, 30.0, 30.0, 30.0, 50.0]
        rad_action = emb.model_action_to_sim(deg_action)
        action = {k: rad_action[i] for i, k in enumerate(joints)}
        for _ in range(20):
            sim.send_action(action, robot_name="so101", n_substeps=10)
        after = read()

        max_delta_deg = max(abs(after[k] - before[k]) for k in joints) * 180.0 / math.pi
        assert max_delta_deg > 5.0, f"arm barely moved ({max_delta_deg:.2f} deg)"
    finally:
        sim.cleanup()
