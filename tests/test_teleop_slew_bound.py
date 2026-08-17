"""The local teleop loop holds a leader frame to the same slew bound as the mesh.

``teleoperate(publish=True)`` drives a local follower and, from the same
``get_action()`` stream, every remote one. The mesh receive path bounds each
inbound frame's per-joint speed; the local path uses ``STRANDS_TELEOP_SLEW_ABS``
merge+apply loop applied its frames straight to ``send_action``. So one device
was judged by two different rules, and the follower physically next to the
operator was the unguarded one.

These tests pin that a frame no remote follower would accept is not applied to
a local one either, that a stream a physical leader arm can actually produce is
untouched, and that a refusal is visible rather than silently reported as a
clean run.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from strands_robots import teleop_mixin
from strands_robots.mesh import security
from tests.test_teleop import FakeHost, FakeTeleop

#: Loop rate every test drives, so the charged interval is floored at 1/50 s.
HZ = 50.0


class SteppingLeader:
    """Emits a scripted sequence of values for one joint, then holds the last."""

    name, id, is_connected = "leader", None, False

    def __init__(self, values: list[float], joint: str = "joint1") -> None:
        self.values = values
        self.joint = joint
        self.calls = 0

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        i = min(self.calls, len(self.values) - 1)
        self.calls += 1
        return {self.joint: self.values[i]}


def _drive(host: FakeHost, leader: object, ticks: int, hz: float = HZ) -> dict:
    host.attach_teleop(leader, name="leader")
    return host.teleoperate(block=True, hz=hz, duration=ticks / hz)


class TestAnOverSpeedFrameIsNotApplied:
    def test_a_full_scale_jump_never_reaches_send_action(self) -> None:
        # A leader that reads one full-scale value - an encoder glitch, a USB
        # re-enumerate - then returns to rest. 2.8 units in one 1/50 s tick is
        # 140 units/s, past the bound by more than 5x.
        host = FakeHost()
        _drive(host, SteppingLeader([0.0, 0.0, -2.8, 0.0, 0.0, 0.0]), ticks=12)

        applied = [a["joint1"] for a, _ in host.sent]
        assert applied, "premise: the loop must apply something"
        assert -2.8 not in applied, f"the glitched frame was applied: {applied}"

    def test_the_refusal_is_counted_and_reported_not_silent(self) -> None:
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, -2.8, 0.0, 0.0, 0.0]), ticks=12)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] >= 1
        # A refusal is not an error: nothing failed.
        assert telemetry["errors"] == 0
        assert "refused" in result["content"][0]["text"]

    def test_a_refused_frame_does_not_become_the_next_frame_baseline(self) -> None:
        # After refusing the jump, the loop must keep measuring from the last
        # value it actually applied, so the stream resumes on its own.
        host = FakeHost()
        _drive(host, SteppingLeader([0.0, 0.0, -2.8, 0.01, 0.02, 0.03]), ticks=14)

        applied = [a["joint1"] for a, _ in host.sent]
        assert 0.03 in applied, f"the stream never resumed: {applied}"


class TestTheBoundIsTheMeshBoundNotACopy:
    def test_the_loop_consults_the_mesh_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Single source of truth: the local path must call the mesh path's own
        # helper, so the two cannot drift to different rules for one device.
        seen: list[dict[str, float]] = []

        def spy(action, previous, now_mono, min_interval_s, max_slew=None):  # noqa: ANN001, ANN202
            seen.append(dict(action))
            return None

        monkeypatch.setattr(security, "input_frame_slew_violation", spy)
        host = FakeHost()
        _drive(host, FakeTeleop({"joint1": 0.1}), ticks=6)

        assert seen, "the local loop never consulted the shared slew helper"
        assert all("joint1" in frame for frame in seen)

    def test_the_operator_env_knob_widens_the_local_bound_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # STRANDS_TELEOP_SLEW_ABS is the documented way to widen the local
        # bound for a device whose driver units exceed the 500 units/s default.
        # (The mesh path uses its own STRANDS_MESH_INPUT_SLEW_ABS.)
        monkeypatch.setenv("STRANDS_TELEOP_SLEW_ABS", "10000")
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, -2.8, 0.0, 0.0, 0.0]), ticks=12)

        assert result["content"][1]["json"]["slew_rejected"] == 0
        assert -2.8 in [a["joint1"] for a, _ in host.sent]


class TestAPhysicalLeaderIsUntouched:
    def test_a_stream_at_servo_speed_is_applied_in_full(self) -> None:
        # The bound is a speed above what a leader arm's own servos produce, so
        # a real stream must pass unchanged. 0.1 units per 1/50 s tick is
        # 5 units/s, under a fifth of the bound.
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), ticks=12)

        assert result["status"] == "success"
        assert result["content"][1]["json"]["slew_rejected"] == 0
        applied = [a["joint1"] for a, _ in host.sent]
        assert 0.5 in applied, f"a servo-speed stream was throttled: {applied}"

    def test_a_clean_session_text_names_no_refusal(self) -> None:
        host = FakeHost()
        result = _drive(host, FakeTeleop({"joint1": 0.1}), ticks=8)

        assert "refused" not in result["content"][0]["text"]

    def test_the_very_first_frame_is_always_applied(self) -> None:
        # There is no baseline to measure the first frame against, so it cannot
        # be refused however far from the follower's pose it reaches.
        host = FakeHost()
        _drive(host, SteppingLeader([-2.8]), ticks=4)

        assert host.sent, "the first frame was refused with no baseline to judge it"
        assert host.sent[0][0]["joint1"] == -2.8


class TestRefusalsAreVisibleInTheSessionStatus:
    def test_a_wholly_refused_session_does_not_report_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A device whose units the bound does not expect has every frame
        # refused. 0 frames / 0 errors used to derive "success": a silent
        # no-op, which is what this derivation exists to refuse.
        monkeypatch.setenv("STRANDS_TELEOP_SLEW_ABS", "0.0001")
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), ticks=12)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] >= 1
        assert result["status"] != "success", telemetry

    def test_a_partially_refused_session_is_degraded(self) -> None:
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, -2.8, 0.0, 0.0, 0.0]), ticks=12)

        assert result["status"] == "degraded"
        assert result["content"][1]["json"]["frames"] > 0

    def test_the_live_status_surface_reports_refusals(self) -> None:
        host = FakeHost()
        _drive(host, SteppingLeader([0.0, 0.0, -2.8, 0.0, 0.0, 0.0]), ticks=12)

        live = host.get_teleoperate_status()
        assert live["content"][1]["json"]["slew_rejected"] >= 1
        assert "slew_rejected" in live["content"][0]["text"]

    def test_the_baseline_does_not_carry_across_sessions(self) -> None:
        # A new session starts with no baseline, so its first frame is applied
        # rather than measured against wherever the previous session ended.
        host = FakeHost()
        _drive(host, SteppingLeader([2.0]), ticks=4)
        host.stop_teleoperate()
        host.detach_teleop()

        host.sent.clear()
        result = _drive(host, SteppingLeader([-2.0]), ticks=4)

        assert host.sent[0][0]["joint1"] == -2.0
        assert result["content"][1]["json"]["slew_rejected"] == 0


class TestTheMixinStaysLight:
    """The shared bound must not drag a layer this module may not depend on.

    ``strands_robots.teleop_mixin`` must not depend on
    ``strands_robots.simulation`` - that separation is why the shared numeric
    domains live in :mod:`strands_robots.utils` (see
    :func:`strands_robots.utils.positive_finite_number_error`). Importing
    ``strands_robots.mesh.security`` executes the ``mesh`` package, which does
    reach ``strands_robots.simulation``, so the slew helper has to be imported
    inside the loop. Hoisting it to module scope would still pass every test
    above while quietly inverting the layering, so it is pinned here.
    """

    @staticmethod
    def _module_scope_imports() -> set[str]:
        source = inspect.getsource(teleop_mixin)
        tree = ast.parse(source)
        names: set[str] = set()
        for node in tree.body:  # module scope only, not nested in a def
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module)
        return names

    def test_the_mesh_package_is_not_imported_at_module_scope(self) -> None:
        offenders = sorted(
            name
            for name in self._module_scope_imports()
            if name.startswith(("strands_robots.mesh", "strands_robots.simulation"))
        )
        assert not offenders, (
            f"{offenders} imported at module scope: importing the mesh package reaches "
            f"strands_robots.simulation, which this module must not depend on. Import the "
            f"slew helper inside _teleop_loop instead."
        )

    def test_the_slew_helper_is_imported_inside_the_loop(self) -> None:
        # The converse of the test above: proves the bound is reached lazily
        # rather than not reached at all.
        source = inspect.getsource(teleop_mixin.TeleopMixin._teleop_loop)
        assert "from strands_robots.mesh.security import" in source
        assert "input_frame_slew_violation" in source


class TestDefaultBoundAccommodatesDriverUnits:
    """The default local slew bound (500 units/s) must accommodate degree-valued
    and range-0-100 devices at their shipped defaults without env-var tuning.

    These are the streams that `robot.attach_teleop("so101_leader", port=...).
    teleoperate()` produces: joints in degrees, gripper in 0-100 range.
    """

    def test_a_90_degree_sweep_over_1s_is_not_refused(self) -> None:
        # A calm 90-degree arm sweep at 50 Hz: each tick moves 1.8 degrees,
        # producing 90 deg/s peak speed. The 500 units/s default must accept it.
        positions = [i * 1.8 for i in range(50)]  # 0.0, 1.8, ... 88.2
        host = FakeHost()
        result = _drive(host, SteppingLeader(positions), ticks=50)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] == 0, (
            f"a 90 deg/s degree-valued stream was refused at the default bound: {telemetry}"
        )

    def test_a_half_second_gripper_close_is_not_refused(self) -> None:
        # Gripper in range-0-100: close from 0 to 100 in 0.5 s at 50 Hz is
        # 25 ticks of 4 units each = 200 units/s peak. Must be accepted.
        positions = [i * 4.0 for i in range(25)]  # 0, 4, 8, ... 96
        host = FakeHost()
        result = _drive(host, SteppingLeader(positions), ticks=25)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] == 0, (
            f"a 200 units/s gripper close was refused at the default bound: {telemetry}"
        )

    def test_sts3215_no_load_max_in_degrees_is_not_refused(self) -> None:
        # STS3215 no-load max is 6.5 rad/s = ~372 deg/s. At 50 Hz that is
        # 7.44 deg/tick. Must be accepted at the default bound.
        positions = [i * 7.44 for i in range(20)]
        host = FakeHost()
        result = _drive(host, SteppingLeader(positions), ticks=20)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] == 0, (
            f"a 372 deg/s stream (STS3215 max) was refused at the default bound: {telemetry}"
        )

    def test_a_2000_units_per_second_glitch_is_still_refused(self) -> None:
        # An encoder glitch that jumps 40 units in one tick at 50 Hz =
        # 2000 units/s. This MUST still be caught even at the wider default.
        host = FakeHost()
        result = _drive(host, SteppingLeader([0.0, 0.0, 40.0, 0.0, 0.0, 0.0]), ticks=12)

        telemetry = result["content"][1]["json"]
        assert telemetry["slew_rejected"] >= 1, (
            f"a 2000 units/s glitch was NOT refused at the default bound: {telemetry}"
        )
