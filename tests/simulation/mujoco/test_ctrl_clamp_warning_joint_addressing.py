"""No-silent-clamp contract for joint-name-addressed actions.

The MuJoCo backend writes an action value to ``data.ctrl`` through two code
paths in ``_apply_action_by_name``:

* the caller keys by ACTUATOR name (direct lookup), or
* the caller keys by JOINT name, which is resolved to the actuator that drives
  that joint.

Both ultimately write to the same ``data.ctrl`` slot, so a value outside the
actuator's ``ctrlrange`` is silently clamped by MuJoCo inside ``mj_step`` in
either case - the commanded trajectory is NOT reproduced. The backend logs
this once per ``(prefix, key)`` so a 50Hz control loop is not spammed, and
names it in the ``send_action`` envelope on every call. This module pins both,
whichever name the caller used.

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


def _clamped(result: dict) -> dict:
    """The ``clamped`` json block of a ``send_action`` envelope, or ``{}``."""
    blocks = [c["json"] for c in result["content"] if "json" in c]
    return blocks[0]["clamped"] if blocks else {}


@pytest.mark.parametrize(
    ("action", "clamped_key"),
    [
        ({"joint1": 100.0}, "joint1"),  # joint-name lookup reaches the actuator
        ({"actuator1": 100.0}, "actuator1"),  # direct actuator-name branch
        ({"joint2": 0.5}, None),  # in range: no false positive
    ],
)
def test_an_out_of_range_value_is_named_in_the_log_and_the_envelope(sim, caplog, action, clamped_key):
    """A value MuJoCo will not reproduce is named whichever name the caller used.

    The call still succeeds - the batch was written and the world stepped - but
    a caller reading only the returned envelope learns which key was held at a
    bound, and to what. Pre-fix the envelope said nothing and only the log knew.
    """
    with caplog.at_level(logging.WARNING, logger=_CLAMP_LOGGER):
        result = sim.send_action(action)

    assert result["status"] == "success"
    warnings = _clamp_warnings(caplog.records)
    if clamped_key is None:
        assert warnings == [] and _clamped(result) == {}
        assert len(result["content"]) == 1
        return
    assert len(warnings) == 1 and clamped_key in warnings[0]
    entry = _clamped(result)[clamped_key]
    lo, hi = entry["bounds"]
    assert entry["commanded"] == 100.0 and lo < hi < 100.0
    assert clamped_key in result["content"][0]["text"]


def test_a_repeated_out_of_range_value_logs_once_but_is_named_every_call(sim, caplog):
    """The log is de-duplicated per (prefix, key); the envelope is not.

    A 50 Hz loop that first hits a limit at tick 0 must still be told at tick 4.
    """
    with caplog.at_level(logging.WARNING, logger=_CLAMP_LOGGER):
        results = [sim.send_action({"joint1": 100.0}) for _ in range(5)]

    assert len(_clamp_warnings(caplog.records)) == 1
    assert all("joint1" in _clamped(r) for r in results)
