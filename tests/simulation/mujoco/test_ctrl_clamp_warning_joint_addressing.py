"""No-silent-clamp contract for joint-name-addressed actions.

The MuJoCo backend writes an action value to ``data.ctrl`` through two code
paths in ``_apply_action_by_name``:

* the caller keys by ACTUATOR name (direct lookup), or
* the caller keys by JOINT name, which is resolved to the actuator that drives
  that joint.

Both ultimately write to the same ``data.ctrl`` slot, so a value outside the
actuator's ``ctrlrange`` is silently clamped by MuJoCo inside ``mj_step`` in
either case - the commanded trajectory is NOT reproduced. The backend surfaces
this once per ``(prefix, key)`` via a clamp warning so a 50Hz control loop is
not spammed. This module pins that the warning fires regardless of which name
the caller used, so the contract does not depend on the addressing style.

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
        result = sim.send_action({"joint1": 100.0})

    assert result["status"] == "success"
    warnings = _clamp_warnings(caplog.records)
    assert len(warnings) == 1
    assert "joint1" in warnings[0]
    assert "clamp" in warnings[0]


def test_out_of_range_actuator_name_action_warns(sim, caplog):
    """The direct actuator-name branch warns too (the contract's other half)."""
    with caplog.at_level(logging.WARNING, logger=_CLAMP_LOGGER):
        result = sim.send_action({"actuator1": 100.0})

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
            sim.send_action({"joint1": 100.0})

    assert len(_clamp_warnings(caplog.records)) == 1
