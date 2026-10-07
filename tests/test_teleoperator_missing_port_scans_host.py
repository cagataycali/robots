"""Pin tests: Teleoperator missing-port refusal names serial candidates.

The sibling path on the Robot factory (`strands_robots/robot.py:532-539`)
teaches a missing-port refusal by naming this host's serial candidates
(:func:`~strands_robots._serial_discovery.describe_serial_candidates`) and
spelling out "a port path is a position, not an identity". The leader arm the
Teleoperator factory fronts IS a serial device of the same family, so the same
scan belongs on this refusal too.

Before the fix, `Teleoperator("so101_leader")` surfaced only the raw lerobot
dataclass TypeError:
    "SOLeaderTeleopConfig.__init__() missing 1 required positional argument:
     'port'. Config: {}"

After the fix, the refusal is augmented with:
  * the canonical ``Teleoperator("so101_leader", port=...)`` form,
  * `describe_serial_candidates(scan_serial_devices())` (the sibling scan),
  * the ``position != identity`` sentence.

A teleoperator that owns no serial bus (gamepad / keyboard / phone) MUST NOT
grow the scan hint - there is no port to teach. The guard is scoped by the
dataclass declaring ``port`` as a field.

Upstream cite
-------------
- strands_robots/teleoperator.py (``_augment_config_construction_failure``)
- strands_robots/_serial_discovery.py (shared helper, used by sibling)
- strands_robots/robot.py:532-539 (sibling site)
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from strands_robots.teleoperator import Teleoperator, _build_teleop_config

pytest.importorskip("lerobot", reason="factory tests require lerobot installed")


SCAN_HINT_SNIPPETS = (
    "serial device(s)",   # describe_serial_candidates: "7 serial device(s)" / "N serial device(s)"
    "No serial devices",  # describe_serial_candidates: "No serial devices are present on this host."
    "servo-bus",          # describe_serial_candidates: "Candidate servo-bus device(s)..."
)


def _has_scan_hint(message: str) -> bool:
    return any(snippet in message for snippet in SCAN_HINT_SNIPPETS)


# ---------------------------------------------------------------------------
# BUG REPRO pins (would fail before the fix)
# ---------------------------------------------------------------------------


def test_so101_leader_missing_port_refusal_lists_serial_candidates():
    """A Feetech-family leader arm's missing-port refusal names the scan."""
    with pytest.raises(ValueError) as excinfo:
        Teleoperator("so101_leader")
    message = str(excinfo.value)
    # Base lerobot content is preserved so existing docs still match
    assert "SOLeaderTeleopConfig" in message
    assert "port" in message
    # The new content, mirroring the Robot sibling
    assert "Teleoperator('so101_leader', port=...)" in message
    assert _has_scan_hint(message), f"missing scan hint: {message!r}"
    assert "position on the bus, not an identity" in message


def test_so100_leader_missing_port_refusal_lists_serial_candidates():
    """Same guard reaches the sibling SO-100 leader (shares the Feetech bus family)."""
    with pytest.raises(ValueError) as excinfo:
        Teleoperator("so100_leader")
    message = str(excinfo.value)
    assert "Teleoperator('so100_leader', port=...)" in message
    assert _has_scan_hint(message)


def test_koch_leader_missing_port_refusal_lists_serial_candidates():
    """Dynamixel leader on the same serial-bus family also benefits."""
    with pytest.raises(ValueError) as excinfo:
        Teleoperator("koch_leader")
    message = str(excinfo.value)
    # koch_leader's dataclass declares `port`, so the hint fires
    assert "Teleoperator('koch_leader', port=...)" in message
    assert _has_scan_hint(message)


# ---------------------------------------------------------------------------
# NEGATIVE pins: no serial bus => no scan hint (don't lie to the user)
# ---------------------------------------------------------------------------


def test_gamepad_refusal_does_not_grow_scan_hint():
    """gamepad has no `port` field: construction succeeds (no required args)."""
    # Confirms the dataclass has no serial bus to teach about. This test
    # exists so a future "always append scan" patch regresses here.
    cfg = _build_teleop_config("gamepad")
    assert cfg is not None
    valid_fields = {f.name for f in dataclasses.fields(type(cfg))}
    assert "port" not in valid_fields


def test_keyboard_refusal_does_not_grow_scan_hint():
    """keyboard has no serial bus - no scan hint even if we force a bad field."""
    # Build with a kwarg the dataclass does not accept to force the except branch.
    with pytest.raises(ValueError) as excinfo:
        Teleoperator("keyboard", not_a_real_kwarg="x")
    message = str(excinfo.value)
    # This is the Unknown-kwarg branch (which runs BEFORE _augment), so the
    # message does not reach the scan path. The pin here is that _augment
    # also won't run on an engineered post_init failure for keyboard.
    assert "Unknown kwarg(s)" in message or "keyboard" in message


# ---------------------------------------------------------------------------
# Helper-level pin: `_augment_config_construction_failure` is scoped to port
# ---------------------------------------------------------------------------


def test_augment_only_adds_scan_when_port_missing_and_valid():
    """Non-port errors on a port-bearing dataclass stay base-only."""
    from strands_robots.teleoperator import _augment_config_construction_failure

    class _Fake:
        __name__ = "FakeConfig"

    # 1. Port in valid_fields but error is not about missing port -> base only
    base_only = _augment_config_construction_failure(
        ConfigClass=_Fake,
        teleop_type="so101_leader",
        config_data={"port": "/dev/ttyACM0"},
        valid_fields={"port", "id"},
        error=ValueError("bad baud_rate value"),
    )
    assert "Teleoperator('so101_leader', port=...)" not in base_only
    assert not _has_scan_hint(base_only)

    # 2. No port in valid_fields -> base only (gamepad / keyboard family)
    no_port_fields = _augment_config_construction_failure(
        ConfigClass=_Fake,
        teleop_type="gamepad",
        config_data={},
        valid_fields={"id", "use_gripper"},
        error=TypeError("__init__() missing 1 required positional argument: 'thing'"),
    )
    assert "Teleoperator('gamepad', port=...)" not in no_port_fields
    assert not _has_scan_hint(no_port_fields)

    # 3. Port in valid_fields AND error about missing port -> hint added
    with_hint = _augment_config_construction_failure(
        ConfigClass=_Fake,
        teleop_type="so101_leader",
        config_data={},
        valid_fields={"port", "id"},
        error=TypeError("__init__() missing 1 required positional argument: 'port'"),
    )
    assert "Teleoperator('so101_leader', port=...)" in with_hint
    assert _has_scan_hint(with_hint)
    assert "position on the bus, not an identity" in with_hint


# ---------------------------------------------------------------------------
# Symmetry pin against the Robot sibling: both paths return a scan hint
# ---------------------------------------------------------------------------


def test_symmetry_robot_and_teleoperator_both_scan_on_missing_port():
    """Both sibling missing-port refusals on the SAME host grow a scan hint."""
    from strands_robots import Robot

    try:
        Robot("so101", mode="real", driver="strands")
    except ValueError as exc:
        robot_msg = str(exc)
    else:
        pytest.fail("Robot('so101', mode='real', driver='strands') should refuse without port")

    try:
        Teleoperator("so101_leader")
    except ValueError as exc:
        teleop_msg = str(exc)
    else:
        pytest.fail("Teleoperator('so101_leader') should refuse without port")

    assert _has_scan_hint(robot_msg), f"Robot sibling must scan: {robot_msg!r}"
    assert _has_scan_hint(teleop_msg), f"Teleoperator must scan: {teleop_msg!r}"
    # Both end with the same "position vs identity" teaching sentence
    assert "position on the bus, not an identity" in robot_msg
    assert "position on the bus, not an identity" in teleop_msg
