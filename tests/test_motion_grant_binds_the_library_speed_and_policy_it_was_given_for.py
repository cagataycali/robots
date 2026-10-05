"""An operator's yes binds the pose library, the speed profile and the policy it was given for.

The grant key named the tool, action, port, instruction, calibration and a roster of motion
fields, but not ``pose_tool``'s ``robot_id`` (which pose library a ``pose_name`` is read from),
not ``smooth`` / ``step_delay`` (and ``steps`` only when it differed from 20), and not the policy
fields of a proxy ``execute`` / ``start``. A yes for ``wave`` from one library moved the arm to
another library's ``wave``; a yes for a slow interpolated move was spent by one full-speed write;
a yes for the ``mock`` policy ran a checkpoint. The operator's line was built from the same
roster, so it never showed those fields either.

Now each of them is keyed and shown. Both sides of the grant read the call through
``gated_view``, so a model call that relied on a default and the value ``pose_tool`` runs with
are one grant. ``pose_tool`` keys the calibration on the records it builds its controller from,
and spends the grant of a call it then refuses on its inputs. A guard over each gated tool's
full signature makes a new parameter either keyed or allowlisted with a reason.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots import _motion_grants
from strands_robots._motion_grants import (
    DIRECT_SERIAL_TOOLS,
    POLICY_FIELDS,
    UNKEYED_PARAMETERS,
    consume_grant,
    deposit_grant,
    gated_view,
    grant_key,
    unkeyed_parameters,
)
from strands_robots.dashboard.agent_hitl import _direct_serial_detail, motion_intent

pose_mod = importlib.import_module("strands_robots.tools.pose_tool")

PORT = "/dev/ttyFAKE0"
WAVE = {"shoulder_pan": 10.0, "elbow_flex": 5.0}


@pytest.fixture(autouse=True)
def _clean_store() -> Any:
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    yield
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()


# --- the key ------------------------------------------------------------------------------------


def _pose(action: str, **fields: Any) -> dict[str, Any]:
    return {"action": action, "port": PORT, **fields}


DIFFERENT_MOTIONS = {
    "another pose library": (
        ("pose_tool", _pose("load_pose", pose_name="wave", robot_id="lib_a")),
        ("pose_tool", _pose("load_pose", pose_name="wave", robot_id="lib_b")),
    ),
    "smooth against a single write": (
        ("pose_tool", _pose("move_multiple", positions=WAVE, smooth=True, step_delay=0.05)),
        ("pose_tool", _pose("move_multiple", positions=WAVE, smooth=False)),
    ),
    "a faster interpolation": (
        ("pose_tool", _pose("load_pose", pose_name="wave", step_delay=0.5)),
        ("pose_tool", _pose("load_pose", pose_name="wave", step_delay=0.001)),
    ),
    "fewer steps, spelled as the default": (
        ("pose_tool", _pose("reset_to_home", steps=20)),
        ("pose_tool", _pose("reset_to_home", steps=2)),
    ),
    "another policy provider": (
        ("arm_1", {"action": "execute", "instruction": "wave", "policy_provider": "mock"}),
        (
            "arm_1",
            {
                "action": "execute",
                "instruction": "wave",
                "policy_provider": "lerobot_local",
                "pretrained_name_or_path": "lerobot/smolvla_base",
            },
        ),
    ),
    "another policy server": (
        ("arm_1", {"action": "start", "instruction": "wave", "policy_host": "10.0.0.2", "policy_port": 5555}),
        ("arm_1", {"action": "start", "instruction": "wave", "policy_host": "10.0.0.9", "policy_port": 5555}),
    ),
}


@pytest.mark.parametrize("case", sorted(DIFFERENT_MOTIONS))
def test_a_yes_for_one_motion_is_not_spendable_by_another(case: str) -> None:
    (approved_tool, approved), (other_tool, other) = DIFFERENT_MOTIONS[case]
    assert grant_key(approved_tool, approved) != grant_key(other_tool, other)
    deposit_grant(approved_tool, approved)
    assert consume_grant(other_tool, other) is False
    assert consume_grant(approved_tool, approved) is True


def test_an_omitted_default_and_the_value_the_tool_uses_are_one_grant() -> None:
    model_call = _pose("load_pose", pose_name="wave")
    tool_call = _pose("load_pose", pose_name="wave", robot_id="so101_follower", smooth=True, steps=20, step_delay=0.05)
    assert grant_key("pose_tool", model_call) == grant_key("pose_tool", tool_call)


def test_a_speed_field_the_action_never_reads_does_not_split_the_grant() -> None:
    plain = _pose("move_motor", motor_name="shoulder_pan", position=10.0)
    assert grant_key("pose_tool", plain) == grant_key("pose_tool", {**plain, "steps": 3, "smooth": False})
    one_shot = _pose("load_pose", pose_name="wave", smooth=False)
    assert grant_key("pose_tool", one_shot) == grant_key("pose_tool", {**one_shot, "step_delay": 9.0})


@pytest.mark.parametrize("action", sorted(pose_mod.MOTION_ACTIONS))
@pytest.mark.parametrize("smooth", [True, False])
def test_the_view_keeps_the_speed_profile_exactly_when_the_tool_interpolates(action: str, smooth: bool) -> None:
    view = gated_view("pose_tool", _pose(action, smooth=smooth))
    assert ("steps" in view) is pose_mod._interpolates(action, smooth)
    assert ("step_delay" in view) is pose_mod._interpolates(action, smooth)


def test_the_view_fills_the_defaults_pose_tool_declares() -> None:
    declared = {
        name: param.default
        for name, param in inspect.signature(pose_mod.pose_tool.__wrapped__).parameters.items()
        if name in _motion_grants._TOOL_DEFAULTS["pose_tool"]
    }
    assert declared == _motion_grants._TOOL_DEFAULTS["pose_tool"]


# --- the operator's line ------------------------------------------------------------------------


def test_the_operator_line_names_the_library_and_the_speed_profile() -> None:
    line = _direct_serial_detail("pose_tool", "load_pose", _pose("load_pose", pose_name="wave", robot_id="lib_b"))
    assert "robot_id=lib_b" in line and "smooth=True" in line and "steps=20" in line and "step_delay=0.05" in line
    bare = _direct_serial_detail("pose_tool", "load_pose", _pose("load_pose", pose_name="wave"))
    assert "robot_id=so101_follower" in bare, "a default the model left out is still what the arm will use"


def test_the_operator_is_shown_the_policy_a_yes_runs() -> None:
    call = {
        "action": "execute",
        "instruction": "wave",
        "policy_provider": "lerobot_local",
        "pretrained_name_or_path": "lerobot/smolvla_base",
        "policy_port": 5555,
    }
    reason = motion_intent("arm_1", call, {}, extra_actions={"arm_1": frozenset({"execute"})})
    assert reason is not None
    assert reason["policy"] == (
        "policy_provider=lerobot_local policy_port=5555 pretrained_name_or_path=lerobot/smolvla_base"
    )


def test_every_policy_field_a_proxy_forwards_is_keyed() -> None:
    from strands_robots.dashboard import peer_tools

    assert set(peer_tools._POLICY_FIELDS) <= set(POLICY_FIELDS)
    for fields in (peer_tools._REAL_FIELDS["execute"], peer_tools._SIM_FIELDS["execute"]):
        assert {f for f in fields if f.startswith("policy_")} <= set(POLICY_FIELDS)


# --- the structural guard -----------------------------------------------------------------------


@pytest.mark.parametrize("tool_name", sorted(DIRECT_SERIAL_TOOLS))
def test_every_parameter_of_a_gated_tool_is_keyed_or_allowlisted_with_a_reason(tool_name: str) -> None:
    module = importlib.import_module(f"strands_robots.tools.{tool_name}")
    params = list(inspect.signature(getattr(module, tool_name).__wrapped__).parameters)
    assert unkeyed_parameters(tool_name, params) == []
    assert all(reason.strip() for reason in UNKEYED_PARAMETERS[tool_name].values())
    assert set(UNKEYED_PARAMETERS[tool_name]) <= set(params), "an allowlist entry for a parameter that is gone"


def test_the_guard_reports_a_parameter_nobody_keyed() -> None:
    def pose_tool(action: str, port: str, robot_id: str, acceleration: float) -> None: ...

    assert unkeyed_parameters("pose_tool", inspect.signature(pose_tool).parameters) == ["acceleration"]


# --- through the tool ---------------------------------------------------------------------------


class _FakeSerial:
    """A bus whose every motor sits at mid-travel and answers any read."""

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        self.is_open = True
        self.writes: list[bytes] = []
        self._last_id = 1

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))
        if len(data) > 2:
            self._last_id = data[2]

    def read(self, n: int = 1) -> bytes:
        body = [self._last_id, 4, 0, 2048 & 0xFF, 2048 >> 8]
        return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])

    def close(self) -> None:
        self.is_open = False


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[_FakeSerial]:
    ports: list[_FakeSerial] = []

    def _ctor(port: str, baudrate: int, timeout: float = 1.0) -> _FakeSerial:
        ports.append(_FakeSerial(port, baudrate, timeout))
        return ports[-1]

    monkeypatch.setattr(pose_mod.serial, "Serial", _ctor)
    monkeypatch.setattr(pose_mod.time, "sleep", lambda *_: None)
    for name in ("BYPASS_TOOL_CONSENT", pose_mod.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.chdir(tmp_path)
    pose_mod.PoseManager("lib_b").store_pose("wave", WAVE)
    return ports


def _declining() -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "n"
    return ctx


def test_a_yes_under_one_library_does_not_move_the_arm_to_another_librarys_pose(opened: list[_FakeSerial]) -> None:
    deposit_grant("pose_tool", _pose("load_pose", pose_name="wave", robot_id="lib_a"))
    res = pose_mod.pose_tool(
        action="load_pose", port=PORT, pose_name="wave", robot_id="lib_b", tool_context=_declining()
    )
    assert res["status"] == "error" and opened == []


def test_a_refused_call_spends_the_yes_it_was_given(opened: list[_FakeSerial]) -> None:
    approved = _pose("load_pose", pose_name="wave", robot_id="lib_a")
    deposit_grant("pose_tool", approved)
    res = pose_mod.pose_tool(action="load_pose", port=PORT, pose_name="wave", robot_id="lib_a")
    assert res["status"] == "error" and "not found" in res["content"][0]["text"]
    assert consume_grant("pose_tool", approved) is False, "the yes outlived the call it was given for"


def test_a_slow_moves_yes_does_not_cover_a_full_speed_write(opened: list[_FakeSerial]) -> None:
    deposit_grant("pose_tool", _pose("move_multiple", positions=WAVE))
    res = pose_mod.pose_tool(action="move_multiple", port=PORT, positions=WAVE, smooth=False, tool_context=_declining())
    assert res["status"] == "error" and opened == []


def test_the_dashboards_yes_for_a_call_on_defaults_is_spent_without_a_second_prompt(opened: list[_FakeSerial]) -> None:
    deposit_grant("pose_tool", _pose("load_pose", pose_name="wave", robot_id="lib_b"))
    ctx = _declining()
    res = pose_mod.pose_tool(action="load_pose", port=PORT, pose_name="wave", robot_id="lib_b", tool_context=ctx)
    assert res["status"] == "success", res
    ctx.interrupt.assert_not_called()
    assert opened and opened[0].writes


def test_the_grant_is_matched_against_the_calibration_the_controller_is_built_from(
    opened: list[_FakeSerial], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = {
        name: {"id": spec.motor_id, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for name, spec in pose_mod.SO_ARM_MOTORS.items()
    }
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(records), encoding="utf-8")
    call = _pose("move_motor", calibration=str(approved), motor_name="shoulder_pan", position=10.0)
    kwargs: dict[str, Any] = dict(call)

    deposit_grant("pose_tool", call)
    ctx = _declining()
    assert pose_mod.pose_tool(**kwargs, tool_context=ctx)["status"] == "success"
    ctx.interrupt.assert_not_called()
    opened.clear()

    # The file the operator approved is what sits on disk when the key is computed, but the
    # records the tool loaded, and builds its controller from, are another arm's.
    other = pose_mod.load_calibration(approved)
    other["shoulder_pan"] = pose_mod.MotorCalibration(**{**records["shoulder_pan"], "homing_offset": -900})
    monkeypatch.setattr(pose_mod, "load_calibration", lambda _path: dict(other))
    deposit_grant("pose_tool", call)
    ctx = _declining()
    res = pose_mod.pose_tool(**kwargs, tool_context=ctx)
    ctx.interrupt.assert_called_once()  # a fresh prompt, not the spent yes
    assert res["status"] == "error" and opened == [], res
