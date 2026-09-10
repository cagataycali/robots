"""``send_action`` refuses a value the joint cannot honor (F-004, CWE-1284).

Finiteness says a value CAN go on the wire; the per-joint envelope says the
joint can honor it.  Before this, ``{"left_knee": {"q": 30.0, "kp": 500.0}}``
or ``{"left_elbow": {"tau": 100000.0}}`` passed the finite check and went out
on ``rt/lowcmd`` as a valid, CRC'd frame - a joint target ten times past
mechanical travel, a stiffness that turns one radian of error into five times
the knee actuator's peak torque.  The docstring deferred magnitude to an
"arm-SDK client" that does not exist.

The envelope is the vendor's own: ``<limit lower= upper= effort= velocity=>``
per revolute joint in ``robots/g1_description/g1_29dof.urdf`` from
unitreerobotics/unitree_ros.  ``kp``/``kd`` have no URDF row and are ceilinged
at twice / five times the stiffest reference gain.  Out of envelope is
REFUSED - never clamped - with a reason naming joint, field, value and bounds,
and the frame is never built, so nothing reaches the publisher.

Both entry points are graded: the free builder (no driver needed) and the
``g1_send_action`` agent tool through a healthy wired driver, because the tool
passes the model's dict verbatim and the driver is the only place the bound
lives.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers.g1 import (
    _G1_DQ_MAX_RAD_S,
    _G1_JOINT_INDEX,
    _G1_JOINT_TRAVEL_RAD,
    _G1_KD_MAX,
    _G1_KP_MAX,
    _G1_TAU_MAX_NM,
    _SDK_KD,
    _SDK_KP,
    _build_lowcmd_from_action,
)


class _StubMotorCmd:
    def __init__(self) -> None:
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class _StubHgLowCmd:
    def __init__(self) -> None:
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [_StubMotorCmd() for _ in range(35)]
        self.crc = 0


class _StubCRC:
    def Crc(self, _cmd: Any) -> int:
        return 42


@pytest.fixture(autouse=True)
def _stub_unitree_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        name: types.ModuleType(name)
        for name in (
            "unitree_sdk2py",
            "unitree_sdk2py.idl",
            "unitree_sdk2py.idl.default",
            "unitree_sdk2py.idl.unitree_hg",
            "unitree_sdk2py.idl.unitree_hg.msg",
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            "unitree_sdk2py.utils",
            "unitree_sdk2py.utils.crc",
        )
    }
    modules["unitree_sdk2py.idl.default"].unitree_hg_msg_dds__LowCmd_ = _StubHgLowCmd  # type: ignore[attr-defined]
    modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"].LowCmd_ = _StubHgLowCmd  # type: ignore[attr-defined]
    modules["unitree_sdk2py.utils.crc"].CRC = _StubCRC  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# The tables themselves.
# ---------------------------------------------------------------------------


def test_every_named_joint_has_a_travel_torque_and_speed_row() -> None:
    """A joint added to the index without an envelope row would be ungated."""
    assert set(_G1_JOINT_TRAVEL_RAD) == set(_G1_JOINT_INDEX)
    assert set(_G1_TAU_MAX_NM) == set(_G1_JOINT_INDEX)
    assert set(_G1_DQ_MAX_RAD_S) == set(_G1_JOINT_INDEX)


def test_every_travel_row_is_ordered_and_contains_zero_or_is_close_to_it() -> None:
    """lower < upper, and the URDF's stated peaks are positive."""
    for joint, (lo, hi) in _G1_JOINT_TRAVEL_RAD.items():
        assert lo < hi, joint
        assert math.isfinite(lo) and math.isfinite(hi), joint
    assert all(v > 0 for v in _G1_TAU_MAX_NM.values())
    assert all(v > 0 for v in _G1_DQ_MAX_RAD_S.values())


def test_the_reference_gains_sit_inside_their_own_ceilings() -> None:
    """The defaults the builder falls back to must themselves pass the gate."""
    assert max(_SDK_KP) < _G1_KP_MAX
    assert max(_SDK_KD) < _G1_KD_MAX


def test_the_travel_rows_match_the_compiled_g1_asset() -> None:
    """The table is a transcription; the vendor's own model is the oracle.

    ``_G1_JOINT_TRAVEL_RAD`` is hand-copied from the URDF, and a hand-copied
    number drifts silently - a slipped decimal reads as a plausible bound.  The
    shipped ``unitree_g1`` asset carries the same 29 limits as MuJoCo joint
    ``range`` values, so it grades the copy without a second hand-maintained
    table.  ``allow_download=False`` declines to fetch: this confirms the rows
    where the asset is on disk and skips where it is not, rather than pulling an
    asset corpus into an unrelated run.
    """
    mujoco = pytest.importorskip("mujoco")
    from strands_robots.assets.manager import resolve_model_path

    path = resolve_model_path("unitree_g1", allow_download=False)
    if path is None or not Path(path).exists():
        pytest.skip("the unitree_g1 asset is not on disk, so the vendor model cannot grade the table")
    model = mujoco.MjModel.from_xml_path(str(path))
    asset_range = {}
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or ""
        asset_range[name.removesuffix("_joint")] = tuple(round(float(v), 5) for v in model.jnt_range[index])
    assert not set(_G1_JOINT_TRAVEL_RAD) - set(asset_range), "asset has no row for these joints"
    transcribed = {joint: (round(lo, 5), round(hi, 5)) for joint, (lo, hi) in _G1_JOINT_TRAVEL_RAD.items()}
    assert transcribed == {joint: asset_range[joint] for joint in transcribed}


def test_the_roll_joints_are_mirrored_left_to_right() -> None:
    """The URDF mirrors hip and shoulder roll; a copy-paste that lost it would let one side over-travel."""
    for joint in ("hip_roll", "shoulder_roll"):
        left_lo, left_hi = _G1_JOINT_TRAVEL_RAD[f"left_{joint}"]
        right_lo, right_hi = _G1_JOINT_TRAVEL_RAD[f"right_{joint}"]
        assert (right_lo, right_hi) == (-left_hi, -left_lo), joint


# ---------------------------------------------------------------------------
# The builder refuses, names the offence, and builds nothing.
# ---------------------------------------------------------------------------


def _assert_refusal_names(reason: str | None, joint: str, field: str, value: float) -> str:
    """Assert the refusal names the offence, and hand back the text it checked."""
    assert reason is not None
    assert joint in reason and f"{joint}.{field}" in reason
    assert repr(float(value)) in reason
    assert "refusing" in reason
    assert "envelope" in reason
    return reason


@pytest.mark.parametrize(
    ("joint", "q"),
    [
        pytest.param("left_knee", 30.0, id="knee-ten-times-past-travel"),
        pytest.param("left_knee", _G1_JOINT_TRAVEL_RAD["left_knee"][1] + 1e-3, id="knee-just-past-upper"),
        pytest.param("left_knee", _G1_JOINT_TRAVEL_RAD["left_knee"][0] - 1e-3, id="knee-just-past-lower"),
        pytest.param("right_hip_roll", 2.9, id="right-hip-roll-would-pass-on-the-left"),
        pytest.param("left_ankle_roll", -0.5, id="ankle-roll-past-lower"),
        pytest.param("waist_pitch", 1.0, id="waist-pitch-past-upper"),
    ],
)
def test_a_target_outside_joint_travel_is_refused_as_a_scalar(joint: str, q: float) -> None:
    cmd, reason = _build_lowcmd_from_action({joint: q}, mode_machine=1)
    assert cmd is None
    text = _assert_refusal_names(reason, joint, "q", q)
    lo, hi = _G1_JOINT_TRAVEL_RAD[joint]
    assert str(lo) in text and str(hi) in text, "the reason states both bounds"


def test_a_target_outside_joint_travel_is_refused_inside_a_per_joint_dict() -> None:
    cmd, reason = _build_lowcmd_from_action({"left_knee": {"q": 30.0, "kp": 60.0}}, mode_machine=1)
    assert cmd is None
    _assert_refusal_names(reason, "left_knee", "q", 30.0)


def test_a_stiffness_over_the_ceiling_is_refused() -> None:
    cmd, reason = _build_lowcmd_from_action({"left_knee": {"q": 0.5, "kp": 500.0}}, mode_machine=1)
    assert cmd is None
    text = _assert_refusal_names(reason, "left_knee", "kp", 500.0)
    assert str(_G1_KP_MAX) in text


def test_a_damping_over_the_ceiling_is_refused() -> None:
    cmd, reason = _build_lowcmd_from_action({"left_knee": {"q": 0.5, "kd": 50.0}}, mode_machine=1)
    assert cmd is None
    _assert_refusal_names(reason, "left_knee", "kd", 50.0)


@pytest.mark.parametrize("field", ["kp", "kd"])
def test_a_negative_gain_is_refused_not_read_as_softer(field: str) -> None:
    cmd, reason = _build_lowcmd_from_action({"left_elbow": {"q": 0.5, field: -1.0}}, mode_machine=1)
    assert cmd is None
    _assert_refusal_names(reason, "left_elbow", field, -1.0)


@pytest.mark.parametrize("tau", [100000.0, 26.0, -26.0])
def test_a_torque_past_the_actuator_peak_is_refused(tau: float) -> None:
    """The elbow actuator peaks at 25 N m; either sign past it is refused."""
    cmd, reason = _build_lowcmd_from_action({"left_elbow": {"q": 0.5, "tau": tau}}, mode_machine=1)
    assert cmd is None
    text = _assert_refusal_names(reason, "left_elbow", "tau", tau)
    assert str(_G1_TAU_MAX_NM["left_elbow"]) in text


def test_a_velocity_reference_past_the_actuator_peak_is_refused() -> None:
    cmd, reason = _build_lowcmd_from_action({"left_wrist_yaw": {"q": 0.0, "dq": 40.0}}, mode_machine=1)
    assert cmd is None
    _assert_refusal_names(reason, "left_wrist_yaw", "dq", 40.0)


def test_one_bad_joint_refuses_the_whole_action_and_touches_no_slot() -> None:
    """Refuse-not-clamp, and refuse-whole: the good joint is not sent without the bad one."""
    cmd, reason = _build_lowcmd_from_action(
        {"left_shoulder_pitch": 0.1, "left_knee": 30.0},
        mode_machine=1,
    )
    assert cmd is None
    assert reason is not None and "left_knee.q" in reason


def test_the_finite_check_still_answers_first_for_a_nan() -> None:
    """``nan`` compares false against every bound; the finite gate has to run first."""
    cmd, reason = _build_lowcmd_from_action({"left_knee": float("nan")}, mode_machine=1)
    assert cmd is None
    assert reason is not None and "finite" in reason


# ---------------------------------------------------------------------------
# Inside the envelope still passes - including at the exact bounds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joint", sorted(_G1_JOINT_INDEX))
def test_both_travel_bounds_are_inclusive_for_every_joint(joint: str) -> None:
    lo, hi = _G1_JOINT_TRAVEL_RAD[joint]
    for q in (lo, hi, 0.0 if lo <= 0.0 <= hi else lo):
        cmd, reason = _build_lowcmd_from_action({joint: q}, mode_machine=1)
        assert reason is None, reason
        assert cmd.motor_cmd[_G1_JOINT_INDEX[joint]].q == q


def test_a_full_per_joint_dict_at_the_ceilings_passes() -> None:
    action = {
        "left_knee": {
            "q": 0.5,
            "kp": _G1_KP_MAX,
            "kd": _G1_KD_MAX,
            "dq": _G1_DQ_MAX_RAD_S["left_knee"],
            "tau": -_G1_TAU_MAX_NM["left_knee"],
        }
    }
    cmd, reason = _build_lowcmd_from_action(action, mode_machine=1)
    assert reason is None, reason
    slot = cmd.motor_cmd[_G1_JOINT_INDEX["left_knee"]]
    assert (slot.kp, slot.kd, slot.dq, slot.tau) == (_G1_KP_MAX, _G1_KD_MAX, 20.0, -139.0)


def test_the_published_rl_gains_are_admitted() -> None:
    """unitree_rl_gym's g1 config reaches kp=150 on the hips/knees; a ceiling that refused it would be wrong."""
    cmd, reason = _build_lowcmd_from_action({"left_hip_pitch": {"q": -0.1, "kp": 150.0, "kd": 2.0}}, mode_machine=1)
    assert reason is None, reason


# ---------------------------------------------------------------------------
# Through the agent tool, on a healthy wired driver: nothing reaches the wire.
# ---------------------------------------------------------------------------


class _RecordingPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []

    def publish(self, topic: str, klass: Any, cmd: Any) -> str | None:
        self.calls.append((topic, klass, cmd))
        return None

    def close(self) -> None:
        pass


def _admit_every_scope(scope: str, *, refresh: bool = False) -> dict[str, Any] | None:
    """Stand in for the FSM and battery gates, which decide WHEN a frame may go.

    Carries the signature it replaces so the stand-in cannot drift from the
    method: this file's subject is WHICH values a frame may carry, and a gate
    refusal would answer first and hide the envelope.
    """
    return None


def _healthy_driver(publisher: _RecordingPublisher) -> Any:
    from strands_robots.drivers.g1 import G1Driver

    driver = G1Driver(tool_name="g1", port="1.2.3.4", battery_floor_pct=20.0)
    driver._connected = True
    driver._pubs = publisher  # type: ignore[assignment]
    driver._mode_machine = 1
    driver._battery = {"pct": 80.0}
    driver._check_motion_gates = _admit_every_scope  # type: ignore[method-assign]
    return driver


def test_the_agent_tool_returns_the_refusal_and_publishes_nothing() -> None:
    from strands_robots.tools.g1.g1_send_action import g1_send_action

    publisher = _RecordingPublisher()
    driver = _healthy_driver(publisher)

    result = g1_send_action(driver=driver, action={"left_knee": {"q": 30.0, "kp": 500.0}})

    assert result["status"] == "error"
    text = str(result["content"])
    assert "left_knee.q" in text and "envelope" in text
    assert publisher.calls == [], "an out-of-envelope frame was published"


def test_the_agent_tool_refuses_a_bare_torque_bomb_and_publishes_nothing() -> None:
    from strands_robots.tools.g1.g1_send_action import g1_send_action

    publisher = _RecordingPublisher()
    driver = _healthy_driver(publisher)

    result = g1_send_action(driver=driver, action={"left_elbow": {"q": 0.0, "tau": 100000.0}})

    assert result["status"] == "error"
    assert "left_elbow.tau" in str(result["content"])
    assert publisher.calls == []


def test_the_agent_tool_still_publishes_an_in_envelope_frame() -> None:
    from strands_robots.tools.g1.g1_send_action import g1_send_action

    publisher = _RecordingPublisher()
    driver = _healthy_driver(publisher)

    result = g1_send_action(driver=driver, action={"left_elbow": {"q": 0.5, "kp": 40.0, "kd": 1.0}})

    assert result["status"] == "success", result
    assert len(publisher.calls) == 1
    topic, _klass, cmd = publisher.calls[0]
    assert topic == "rt/lowcmd"
    assert cmd.motor_cmd[_G1_JOINT_INDEX["left_elbow"]].q == 0.5
