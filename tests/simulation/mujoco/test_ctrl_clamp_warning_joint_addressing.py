"""No-silent-clamp contract for joint-name-addressed actions.

The MuJoCo backend writes an action value to ``data.ctrl`` through two code
paths in ``_apply_action_by_name``:

* the caller keys by ACTUATOR name (direct lookup), or
* the caller keys by JOINT name, which is resolved to the actuator that drives
  that joint.

Both ultimately write to the same ``data.ctrl`` slot, so a value outside the
actuator's ``ctrlrange`` is silently clamped by MuJoCo inside ``mj_step`` in
either case - the commanded trajectory is NOT reproduced. The backend refuses
such a value by default (nothing written, no step, an error naming the key and
range) and, with ``clamp=True``, clamps it and warns once per ``(prefix, key)``
so a 50Hz control loop is not spammed. This module pins that both halves hold
regardless of which name the caller used, so the contract does not depend on
the addressing style.

``panda`` is used because its joint names (``joint1`` ...) differ from its
actuator names (``actuator1`` ...), so keying by joint name genuinely exercises
the joint-name resolution fallback rather than the direct-actuator branch.
"""

import logging

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

_CLAMP_LOGGER = "strands_robots.simulation.mujoco.rendering"


def _clamp_warnings(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if "ctrlrange" in r.getMessage()]


@pytest.fixture
def sim():
    s = Simulation()
    s.create_world()
    s.add_robot("panda")
    try:
        yield s
    finally:
        s.cleanup()


def test_out_of_range_joint_name_action_warns(sim, caplog):
    """An out-of-range value keyed by JOINT name surfaces the clamp warning.

    Regression: this path previously wrote ``data.ctrl`` verbatim with no
    warning, so a joint-addressed out-of-distribution command was clamped
    silently while the identical actuator-addressed command warned.
    """
    with caplog.at_level(logging.WARNING, logger=_CLAMP_LOGGER):
        result = sim.send_action({"joint1": 100.0}, clamp=True)

    assert result["status"] == "success"
    warnings = _clamp_warnings(caplog.records)
    assert len(warnings) == 1
    assert "joint1" in warnings[0]
    assert "clamp" in warnings[0]


def test_out_of_range_actuator_name_action_warns(sim, caplog):
    """The direct actuator-name branch warns too (the contract's other half)."""
    with caplog.at_level(logging.WARNING, logger=_CLAMP_LOGGER):
        result = sim.send_action({"actuator1": 100.0}, clamp=True)

    assert result["status"] == "success"
    assert len(_clamp_warnings(caplog.records)) == 1


def test_in_range_joint_name_action_does_not_warn(sim, caplog):
    """A value inside the actuator ctrlrange must not raise a false clamp warning."""
    with caplog.at_level(logging.WARNING, logger=_CLAMP_LOGGER):
        result = sim.send_action({"joint2": 0.5})

    assert result["status"] == "success"
    assert _clamp_warnings(caplog.records) == []


def test_repeated_out_of_range_joint_name_deduplicated(sim, caplog):
    """The clamp warning is emitted once per (prefix, key), not once per step."""
    with caplog.at_level(logging.WARNING, logger=_CLAMP_LOGGER):
        for _ in range(5):
            sim.send_action({"joint1": 100.0}, clamp=True)

    assert len(_clamp_warnings(caplog.records)) == 1


def test_out_of_range_joint_name_action_is_refused_by_default(sim):
    """Without ``clamp`` the same command is refused with the key, value and range."""
    before = sim._world._data.ctrl.copy()
    steps = sim._world.step_count
    result = sim.send_action({"joint1": 100.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "joint1=100" in text and "ctrlrange" in text and "clamp=True" in text
    assert result["content"][1]["json"]["out_of_range"][0]["key"] == "joint1"
    assert list(sim._world._data.ctrl) == list(before)
    assert sim._world.step_count == steps


def test_a_refused_action_rolls_back_the_in_range_keys_too(sim):
    """One bad value refuses the whole action: no half-applied vector, no step."""
    before = sim._world._data.ctrl.copy()
    result = sim.send_action({"joint2": 0.5, "joint1": 100.0})
    assert result["status"] == "error"
    assert list(sim._world._data.ctrl) == list(before)


def test_clamp_reports_the_applied_values(sim):
    result = sim.send_action({"joint1": 100.0}, clamp=True)
    assert result["status"] == "success"
    clamped = result["content"][1]["json"]["clamped"]
    assert clamped[0]["key"] == "joint1" and clamped[0]["applied"] == clamped[0]["ctrlrange"][1]
    assert "clamped" in result["content"][0]["text"]
