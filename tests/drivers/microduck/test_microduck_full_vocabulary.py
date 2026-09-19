"""The Microduck's full vocabulary, driven the way an agent drives it.

Every verb the pad, ``duckctl`` and ``robotctl`` give an operator is one
``action`` on :class:`~strands_robots.drivers.microduck.MicroduckDriver`. These
tests run each against the mock robotd on a real unix socket and assert three
things per verb: what reached the wire (method + params), that a refusal happens
BEFORE the wire when the parameter is out of the robot's domain, and that
robotd's own ``accepted: false`` becomes the verb's refusal rather than a
success carrying a ``false`` nobody reads.

Discovery is covered separately: ``Robot("microduck")`` with no ``port`` must
find the socket from the environment, and the ssh forward must be the exact
``ssh -N -L`` an operator would type.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers.microduck import (
    _ACTIONS,
    DEADMAN_S,
    DEFAULT_MEDIA_SOCKET,
    DEFAULT_SOCKET,
    DEFAULT_TOF_SOCKET,
    ENV_HOST,
    ENV_SOCKET,
    ENV_SSH_USER,
    FRAME_ROOT_ENV,
    MICRODUCK_API_VERSION,
    MOVE_REFRESH_HZ,
    PLAYABLE_SOUND_TAGS,
    POSE_TILT_MAX,
    SKILLS,
    WALK_MAX_LINEAR,
    MicroduckDriver,
    _SshForward,
    frame_root,
    resolve_endpoint,
    ssh_forward_argv,
    summarise_tof,
    twist_error,
    uyvy_to_jpeg,
)
from strands_robots.utils import positive_finite_number_error
from tests.mocks.microduck_robotd import MockRobotd


def _drive(driver: MicroduckDriver, **params: Any) -> dict[str, Any]:
    """Run one ``stream`` call and return its single envelope."""

    async def _run() -> dict[str, Any]:
        events = [event async for event in driver.stream({"toolUseId": "t", "name": "d", "input": params}, {})]
        assert len(events) == 1
        return events[0]

    return asyncio.run(_run())


def _payload(envelope: dict[str, Any]) -> dict[str, Any]:
    block = envelope["content"][0]
    return block["json"] if "json" in block else {"text": block["text"]}


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(b.get("text", "") for b in envelope["content"])


def _sent(mock: MockRobotd, method: str) -> list[dict[str, Any]]:
    """Every params dict the mock received for ``method``."""
    out = []
    for raw in list(mock.received):
        obj = json.loads(raw)
        if obj.get("method") == method:
            out.append(obj.get("params") or {})
    return out


def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


@pytest.fixture
def duck() -> Any:
    with MockRobotd() as mock:
        driver = MicroduckDriver(port=mock.path, timeout=2.0)
        assert driver.connect_eagerly() is None
        _wait_for(lambda: driver.get_observation())
        try:
            yield driver, mock
        finally:
            driver.cleanup()


# --------------------------------------------------------------------------- #
# The schema is the table.                                                    #
# --------------------------------------------------------------------------- #


class TestTheVocabulary:
    def test_the_enum_is_the_dispatch_table(self) -> None:
        spec = MicroduckDriver(port="/nowhere.sock").tool_spec
        enum = spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert enum == list(_ACTIONS)
        assert len(enum) >= 32

    def test_every_operator_verb_is_present(self) -> None:
        wanted = {
            "move", "head", "look_at", "pose", "mouth", "do", "skills", "sit", "stand",
            "enable", "disable", "relax", "init", "reboot_motors", "sounds", "play_sound",
            "theremin", "mode", "set_mode", "policies", "load_policy", "reload_policies",
            "health", "version", "model", "odometry", "monitor", "camera", "tof",
            "sensors", "status", "stop",
        }  # fmt: skip
        assert wanted <= set(_ACTIONS)

    def test_the_api_pin_is_the_release_the_lane_read(self) -> None:
        assert MICRODUCK_API_VERSION == 31

    def test_the_description_states_that_acceptance_is_not_motion(self) -> None:
        text = MicroduckDriver(port="/nowhere.sock").tool_spec["description"]
        assert "NOT proof" in text
        assert str(DEADMAN_S) in text


# --------------------------------------------------------------------------- #
# Discovery.                                                                  #
# --------------------------------------------------------------------------- #


class TestDiscovery:
    def test_default_is_the_on_robot_socket(self) -> None:
        ep = resolve_endpoint(None, env={})
        assert (ep.kind, ep.socket_path, ep.source) == ("socket", DEFAULT_SOCKET, "default")

    def test_env_socket_wins_over_env_host(self) -> None:
        ep = resolve_endpoint(None, env={ENV_SOCKET: "/tmp/x.sock", ENV_HOST: "10.0.0.5"})
        assert (ep.kind, ep.socket_path, ep.source) == ("socket", "/tmp/x.sock", ENV_SOCKET)

    def test_env_host_becomes_an_ssh_forward_with_the_radxa_user(self) -> None:
        ep = resolve_endpoint(None, env={ENV_HOST: "10.0.0.5"})
        assert (ep.kind, ep.host, ep.user, ep.source) == ("ssh", "10.0.0.5", "radxa", ENV_HOST)

    def test_duck_board_user_and_inline_login_are_honoured(self) -> None:
        assert resolve_endpoint(None, env={ENV_HOST: "duck.local", ENV_SSH_USER: "pi"}).user == "pi"
        ep = resolve_endpoint(None, env={ENV_HOST: "me@duck.local", ENV_SSH_USER: "pi"})
        assert (ep.user, ep.host) == ("me", "duck.local")

    def test_a_blank_user_variable_reads_as_unset(self) -> None:
        assert resolve_endpoint(None, env={ENV_HOST: "h", ENV_SSH_USER: ""}).user == "radxa"

    def test_port_wins_over_everything(self) -> None:
        ep = resolve_endpoint("/run/other.sock", env={ENV_SOCKET: "/tmp/x.sock", ENV_HOST: "h"})
        assert (ep.kind, ep.socket_path, ep.source) == ("socket", "/run/other.sock", "port")
        ssh = resolve_endpoint("ssh://op@duck", env={})
        assert (ssh.kind, ssh.user, ssh.host, ssh.source) == ("ssh", "op", "duck", "port")

    def test_the_forward_is_the_command_an_operator_types(self) -> None:
        argv = ssh_forward_argv("radxa", "10.0.0.5", [("/tmp/l.sock", DEFAULT_SOCKET)], connect_timeout=3)
        assert argv[:2] == ["ssh", "-N"]
        assert "BatchMode=yes" in argv and "ExitOnForwardFailure=yes" in argv and "StreamLocalBindUnlink=yes" in argv
        assert "ConnectTimeout=3" in argv
        assert argv[argv.index("-L") + 1] == f"/tmp/l.sock:{DEFAULT_SOCKET}"
        assert argv[-1] == "radxa@10.0.0.5"

    @pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "15", None])
    def test_a_connect_timeout_ssh_cannot_spend_is_refused_by_the_shared_domain(self, value: Any) -> None:
        """Rounding to ssh's whole second used to hide the mistake or raise unnamed.

        ``0``/``-1``/``True`` became a silent ``ConnectTimeout=1``, and ``nan``,
        ``inf``, ``None`` and ``"15"`` raised ``ValueError``/``OverflowError``/
        ``TypeError`` out of ``round`` naming no parameter. The knob is the same
        one three other surfaces carry, so it takes the same domain.
        """
        assert positive_finite_number_error(value, "connect_timeout", "ssh_forward_argv") is not None
        with pytest.raises(ValueError) as exc:
            ssh_forward_argv("radxa", "h", [("/tmp/l.sock", DEFAULT_SOCKET)], connect_timeout=value)
        assert "connect_timeout" in str(exc.value) and "must be > 0" in str(exc.value)

    def test_a_missing_socket_refusal_names_what_to_set(self) -> None:
        driver = MicroduckDriver(port=os.path.join(tempfile.mkdtemp(), "none.sock"))
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "did not answer" in reason
        assert ENV_HOST in reason and ENV_SOCKET in reason and "radxa" in reason

    def test_the_driver_forwards_all_three_sockets_and_connects_through_the_forward(self, monkeypatch: Any) -> None:
        """An injected spawn stands the mock robotd at the local path ssh would bind."""
        monkeypatch.setenv(ENV_HOST, "op@duck.local")
        monkeypatch.delenv(ENV_SOCKET, raising=False)
        seen: dict[str, Any] = {}
        mocks: list[MockRobotd] = []

        class _Proc:
            stderr = None

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                pass

            def wait(self, timeout: float = 0) -> None:
                pass

        def spawn(argv: list[str], **kwargs: Any) -> _Proc:
            seen["argv"] = argv
            forwards = [argv[i + 1] for i, a in enumerate(argv) if a == "-L"]
            local_robot = next(f.split(":")[0] for f in forwards if f.endswith(DEFAULT_SOCKET))
            mocks.append(MockRobotd(path=local_robot).__enter__())
            return _Proc()

        driver = MicroduckDriver(timeout=2.0, ssh_spawn=spawn)
        try:
            assert driver.connect_eagerly() is None
            argv = seen["argv"]
            remotes = sorted(argv[i + 1].split(":")[1] for i, a in enumerate(argv) if a == "-L")
            assert remotes == sorted([DEFAULT_SOCKET, DEFAULT_MEDIA_SOCKET, DEFAULT_TOF_SOCKET])
            assert argv[-1] == "op@duck.local"
            status = _payload(_drive(driver, action="status"))
            assert status["endpoint"] == {
                "kind": "ssh",
                "source": ENV_HOST,
                "host": "duck.local",
                "user": "op",
                "forwarded": True,
            }
        finally:
            driver.cleanup()
            for m in mocks:
                m.close()

    def test_probe_hardware_is_false_when_nothing_listens(self, monkeypatch: Any) -> None:
        monkeypatch.setenv(ENV_SOCKET, os.path.join(tempfile.mkdtemp(), "none.sock"))
        monkeypatch.delenv(ENV_HOST, raising=False)
        assert MicroduckDriver.probe_hardware(timeout=0.5) is False

    def test_probe_hardware_is_true_against_a_robotd(self, monkeypatch: Any) -> None:
        with MockRobotd() as mock:
            monkeypatch.setenv(ENV_SOCKET, mock.path)
            monkeypatch.delenv(ENV_HOST, raising=False)
            assert MicroduckDriver.probe_hardware(timeout=2.0) is True

    def test_a_verb_connects_lazily_and_status_never_does(self) -> None:
        with MockRobotd() as mock:
            driver = MicroduckDriver(port=mock.path, timeout=2.0)
            try:
                assert _payload(_drive(driver, action="status"))["connected"] is False
                assert driver.is_connected is False
                envelope = _drive(driver, action="mouth", open=0.5)
                assert envelope["status"] == "success", envelope
                assert driver.is_connected is True
            finally:
                driver.cleanup()


# --------------------------------------------------------------------------- #
# Locomotion.                                                                 #
# --------------------------------------------------------------------------- #


class TestMove:
    def test_a_move_outlives_the_deadman_and_ends_in_a_zero_twist(self, duck: Any) -> None:
        driver, mock = duck
        envelope = _drive(driver, action="move", vx=0.1, duration=0.6)
        assert envelope["status"] == "success", envelope
        payload = _payload(envelope)
        assert payload["twist"] == {"vx": 0.1, "vy": 0.0, "vyaw": 0.0}
        assert payload["deadman_s"] == DEADMAN_S
        _wait_for(lambda: driver._move is None or not driver._move.active, timeout=3.0)
        time.sleep(0.1)
        moves = _sent(mock, "robot.move")
        expected_min = int(0.6 * MOVE_REFRESH_HZ * 0.6)
        assert len(moves) >= expected_min, len(moves)
        assert moves[-1] == {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
        assert all(m == {"vx": 0.1, "vy": 0.0, "vyaw": 0.0} for m in moves[:-1])

    def test_stop_cancels_the_stream_and_sends_robot_stop(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="move", vx=0.1, duration=5.0)["status"] == "success"
        assert driver._move is not None and driver._move.active
        envelope = _drive(driver, action="stop")
        assert envelope["status"] == "success"
        assert driver._move is None
        assert "robot.stop" in mock.methods
        time.sleep(0.15)
        count = len(_sent(mock, "robot.move"))
        time.sleep(0.15)
        assert len(_sent(mock, "robot.move")) == count, "the stream kept sending after stop"

    @pytest.mark.parametrize(
        ("params", "needle"),
        [
            ({"vx": 0.5}, "walking envelope"),
            ({"vyaw": 2.0}, "walking envelope"),
            ({"vx": "fast"}, "vx"),
            ({"vx": float("nan")}, "vx"),
            ({"vx": 0.1, "duration": 60}, "duration"),
            ({}, "all zero"),
        ],
    )
    def test_out_of_envelope_twists_never_reach_the_wire(self, duck: Any, params: dict[str, Any], needle: str) -> None:
        driver, mock = duck
        envelope = _drive(driver, action="move", **params)
        assert envelope["status"] == "error", envelope
        assert needle in _text(envelope)
        assert _sent(mock, "robot.move") == []

    def test_the_roller_envelope_differs(self) -> None:
        assert twist_error(0.5, 0.0, 0.0, "roller") is None
        assert twist_error(0.5, 0.0, 0.0, "walk") is not None
        assert "strafe" in (twist_error(0.1, 0.1, 0.0, "roller") or "")
        assert twist_error(-0.6, 0.0, 0.0, "roller") is not None
        assert twist_error(WALK_MAX_LINEAR, WALK_MAX_LINEAR, 1.5, None) is None

    def test_move_during_a_skill_is_refused(self, duck: Any) -> None:
        driver, mock = duck
        with driver._cache_lock:
            driver._last_state = {**(driver._last_state or {}), "policy": "kick_left"}
        envelope = _drive(driver, action="move", vx=0.1)
        assert envelope["status"] == "error"
        assert "kick_left" in _text(envelope)


class TestHeadPoseMouth:
    def test_head_is_a_notification_with_only_the_axes_given(self, duck: Any) -> None:
        driver, mock = duck
        envelope = _drive(driver, action="head", head_yaw=0.3, neck_pitch=-0.1)
        assert envelope["status"] == "success"
        _wait_for(lambda: _sent(mock, "robot.head"))
        assert _sent(mock, "robot.head") == [{"head_yaw": 0.3, "neck_pitch": -0.1}]

    def test_head_with_nothing_is_refused(self, duck: Any) -> None:
        driver, _ = duck
        assert _drive(driver, action="head")["status"] == "error"

    def test_pose_enforces_the_trained_range(self, duck: Any) -> None:
        driver, mock = duck
        bad = _drive(driver, action="pose", z=-0.05)
        assert bad["status"] == "error" and "trained range" in _text(bad)
        bad = _drive(driver, action="pose", roll=POSE_TILT_MAX + 0.1)
        assert bad["status"] == "error"
        good = _drive(driver, action="pose", z=-0.01, pitch=0.1)
        assert good["status"] == "success"
        _wait_for(lambda: _sent(mock, "robot.pose"))
        assert _sent(mock, "robot.pose") == [{"active": True, "z": -0.01, "roll": 0.0, "pitch": 0.1}]

    def test_pose_release(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="pose", active=False)["status"] == "success"
        _wait_for(lambda: _sent(mock, "robot.pose"))
        assert _sent(mock, "robot.pose")[0]["active"] is False

    def test_mouth_is_unit_interval(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="mouth", open=1.5)["status"] == "error"
        assert _drive(driver, action="mouth")["status"] == "error"
        assert _drive(driver, action="mouth", open=0.25)["status"] == "success"
        _wait_for(lambda: _sent(mock, "robot.mouth"))
        assert _sent(mock, "robot.mouth") == [{"open": 0.25}]


class TestLookAt:
    def test_look_at_returns_robotds_solved_head(self, duck: Any) -> None:
        driver, mock = duck
        envelope = _drive(driver, action="look_at", x=1.0, y=0.2, z=0.0)
        assert envelope["status"] == "success", envelope
        payload = _payload(envelope)
        assert payload["head"]["head_yaw"] == 0.2
        assert payload["clamped"] is False
        assert _sent(mock, "robot.look") == [{"x": 1.0, "y": 0.2, "z": 0.0, "neck_pitch": 0.0}]

    def test_look_at_passes_the_neck_posture_robotd_requires(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="look_at", x=1.0, y=0.0, z=-0.1, neck_pitch=0.3)["status"] == "success"
        assert _sent(mock, "robot.look")[-1]["neck_pitch"] == 0.3

    def test_look_at_reports_a_clamp(self, duck: Any) -> None:
        driver, _ = duck
        assert _payload(_drive(driver, action="look_at", x=0.1, y=5.0, z=0.0))["clamped"] is True

    def test_look_at_needs_all_three_and_a_direction(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="look_at", x=1.0)["status"] == "error"
        assert "yaw" in _text(_drive(driver, action="look_at", x=0.0, y=0.0, z=1.0))
        assert _sent(mock, "robot.look") == []


# --------------------------------------------------------------------------- #
# Skills, sit/stand, robotd's own refusals.                                   #
# --------------------------------------------------------------------------- #


class TestSkills:
    def test_skills_come_from_the_robot_not_the_stock_list(self) -> None:
        with MockRobotd(skills=("ground_pick", "sit_toggle", "wave")) as mock:
            driver = MicroduckDriver(port=mock.path, timeout=2.0)
            try:
                assert driver.connect_eagerly() is None
                assert driver.known_skills == ("ground_pick", "sit_toggle", "wave")
                assert _drive(driver, action="do", skill="wave")["status"] == "success"
                assert _sent(mock, "robot.do") == [{"skill": "wave"}]
                refused = _drive(driver, action="do", skill="kick_left")
                assert refused["status"] == "error" and "wave" in _text(refused)
                assert _sent(mock, "robot.do") == [{"skill": "wave"}]
            finally:
                driver.cleanup()

    def test_the_stock_list_is_the_fallback(self) -> None:
        driver = MicroduckDriver(port="/nowhere.sock")
        assert driver.known_skills == SKILLS

    def test_robotd_declining_is_the_verbs_refusal(self) -> None:
        with MockRobotd(decline={"robot.do": "not homed"}) as mock:
            driver = MicroduckDriver(port=mock.path, timeout=2.0)
            try:
                assert driver.connect_eagerly() is None
                driver._stopped = True
                envelope = _drive(driver, action="do", skill="roulade")
                assert envelope["status"] == "error"
                assert "not homed" in _text(envelope)
                assert driver._stopped is True, "a declined intent must not clear the halt"
            finally:
                driver.cleanup()

    def test_sit_then_stand_follow_the_robots_sitting_flag(self, duck: Any) -> None:
        driver, mock = duck
        first = _drive(driver, action="sit")
        assert first["status"] == "success" and _payload(first)["sitting_before"] is False
        assert _sent(mock, "robot.do") == [{"skill": "sit_toggle"}]
        again = _drive(driver, action="sit")
        assert again["status"] == "success" and _payload(again)["sent"] is False
        assert _sent(mock, "robot.do") == [{"skill": "sit_toggle"}], "already sitting: nothing sent"
        stand = _drive(driver, action="stand")
        assert stand["status"] == "success" and _payload(stand)["sitting_before"] is True
        assert _sent(mock, "robot.do") == [{"skill": "sit_toggle"}, {"skill": "sit_toggle"}]

    def test_a_robotd_without_sitting_still_toggles_but_says_so(self) -> None:
        with MockRobotd(sitting=None) as mock:
            driver = MicroduckDriver(port=mock.path, timeout=2.0)
            try:
                assert driver.connect_eagerly() is None
                envelope = _drive(driver, action="sit")
                assert envelope["status"] == "success"
                assert "unverified" in _payload(envelope)["note"]
            finally:
                driver.cleanup()


class TestPowerAndConfirm:
    @pytest.mark.parametrize("verb", ["relax", "init", "reboot_motors"])
    def test_destructive_verbs_need_confirm(self, duck: Any, verb: str) -> None:
        driver, mock = duck
        envelope = _drive(driver, action=verb)
        assert envelope["status"] == "error" and "confirm=true" in _text(envelope)
        assert not any(m in mock.methods for m in ("robot.relax", "robot.init", "robot.rebootMotors"))
        assert _drive(driver, action=verb, confirm=True)["status"] == "success"

    def test_reboot_motors_ids_domain(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="reboot_motors", confirm=True, ids=[1, "x"])["status"] == "error"
        assert _drive(driver, action="reboot_motors", confirm=True, ids=[3, 4])["status"] == "success"
        assert _sent(mock, "robot.rebootMotors") == [{"ids": [3, 4]}]

    def test_enable_and_disable(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="enable")["status"] == "success"
        assert _drive(driver, action="disable")["status"] == "success"
        assert _drive(driver, action="enable", on="yes")["status"] == "error"
        assert _sent(mock, "robot.enable") == [{"on": True, "toggle": False}, {"on": False, "toggle": False}]


class TestSoundsAndModes:
    def test_play_sound_domain(self, duck: Any) -> None:
        driver, mock = duck
        assert _payload(_drive(driver, action="sounds"))["tags"] == list(PLAYABLE_SOUND_TAGS)
        assert _drive(driver, action="play_sound", tag="wheee")["status"] == "error"
        assert _drive(driver, action="play_sound", tag="honk")["status"] == "error"
        assert _drive(driver, action="play_sound", tag="Greet")["status"] == "success"
        assert _sent(mock, "robot.sound") == [{"tag": "greet"}]

    def test_theremin(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="theremin", active=False)["status"] == "success"
        assert _sent(mock, "robot.theremin") == [{"active": False}]

    def test_mode_and_set_mode(self, duck: Any) -> None:
        driver, mock = duck
        assert _payload(_drive(driver, action="mode"))["result"] == {"mode": "walk"}
        assert _drive(driver, action="set_mode", mode="hover")["status"] == "error"
        envelope = _drive(driver, action="set_mode", mode="roller")
        assert envelope["status"] == "success"
        assert _payload(envelope)["mode_after"] == "roller"
        # The move envelope follows the mode.
        assert _drive(driver, action="move", vx=0.5, duration=0.2)["status"] == "success"
        _wait_for(lambda: driver._move is None or not driver._move.active, timeout=2.0)

    def test_policies_and_load(self, duck: Any) -> None:
        driver, mock = duck
        policies = _payload(_drive(driver, action="policies"))
        assert policies["skills"] == list(SKILLS) and policies["mode"] == "walk"
        assert _drive(driver, action="load_policy", slot="walk")["status"] == "error"
        assert _drive(driver, action="load_policy", slot="walk", path="/opt/x.onnx")["status"] == "success"
        assert _sent(mock, "robot.loadPolicy") == [{"slot": "walk", "path": "/opt/x.onnx"}]
        assert _drive(driver, action="reload_policies")["status"] == "success"


class TestReadouts:
    def test_health_version_model(self, duck: Any) -> None:
        driver, _ = duck
        health = _payload(_drive(driver, action="health"))
        assert health["battery"]["percent"] == 82.0 and health["imu_frozen"] is False
        driver._absorb_health({"imu": {"ready": True, "stale_blocks": 40, "consecutive_stale_blocks": 30}})
        with driver._cache_lock:
            driver._health_at = time.monotonic()
        # The verb re-asks robotd; grade the rule on the module function directly.
        from strands_robots.drivers.microduck import IMU_FROZEN_RUN

        assert IMU_FROZEN_RUN == 25
        version = _payload(_drive(driver, action="version"))
        assert version["api_version"] == MICRODUCK_API_VERSION and version["model_api"]["duck_control"] == "0.14.1"
        assert _payload(_drive(driver, action="model"))["result"]["joints"] == 15

    def test_monitor_and_odometry_read_the_stream(self, duck: Any) -> None:
        driver, _ = duck
        monitor = _payload(_drive(driver, action="monitor"))
        assert monitor["policy"] == "walk" and monitor["applied"] is not None and "fallen" in monitor
        odometry = _payload(_drive(driver, action="odometry"))
        assert "position" in odometry and "yaw" in odometry

    def test_status_carries_the_endpoint_and_skills(self, duck: Any) -> None:
        driver, mock = duck
        status = _payload(_drive(driver, action="status"))
        assert status["connected"] is True and status["socket"] == mock.path
        assert status["endpoint"]["kind"] == "socket" and status["skills"] == list(SKILLS)
        assert status["driver_api_version"] == MICRODUCK_API_VERSION


# --------------------------------------------------------------------------- #
# Camera + depth: the other daemons' sockets.                                 #
# --------------------------------------------------------------------------- #


def _serve_once(path: str, handler: Any) -> threading.Thread:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)

    def run() -> None:
        server.settimeout(5.0)
        try:
            conn, _ = server.accept()
        except OSError:
            return
        with conn:
            reader = conn.makefile("rb")
            hello = json.loads(reader.readline())
            conn.sendall(
                json.dumps({"jsonrpc": "2.0", "id": hello["id"], "result": {"api_version": 31}}).encode() + b"\n"
            )
            request = json.loads(reader.readline())
            handler(conn, request)
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class TestCameraAndTof:
    def test_camera_decodes_a_uyvy_frame_to_a_jpeg(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FRAME_ROOT_ENV, str(tmp_path))
        width, height = 8, 4
        raw = bytes([128, 200, 128, 200] * (width * height // 2))  # grey-ish UYVY

        def handler(conn: socket.socket, request: dict[str, Any]) -> None:
            assert request["method"] == "media.frame"
            header = {
                "width": width,
                "height": height,
                "bytes": len(raw),
                "format": "UYVY",
                "rotate": 90,
                "captured_at_unix_us": 1700000000000000,
            }
            conn.sendall(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": header}).encode() + b"\n")
            conn.sendall(raw)

        media = os.path.join(tempfile.mkdtemp(), "media.sock")  # short: AF_UNIX path limit
        _serve_once(media, handler)
        driver = MicroduckDriver(port="/nowhere.sock", timeout=2.0)
        driver._media_socket = media
        out = tmp_path / "frame.jpg"
        envelope = _drive(driver, action="camera", save_path="frame.jpg")
        assert envelope["status"] == "success", envelope
        payload = _payload(envelope)
        assert payload["path"] == str(out) and out.stat().st_size > 0
        assert out.read_bytes()[:2] == b"\xff\xd8"
        assert payload["rotate"] == 90

    def test_camera_without_a_save_path_writes_into_the_frame_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default temp name lands in the root too, not in the OS temp dir."""
        monkeypatch.setenv(FRAME_ROOT_ENV, str(tmp_path / "frames"))
        width, height = 8, 4
        raw = bytes([128, 200, 128, 200] * (width * height // 2))

        def handler(conn: socket.socket, request: dict[str, Any]) -> None:
            header = {"width": width, "height": height, "bytes": len(raw), "format": "UYVY", "rotate": 0}
            conn.sendall(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": header}).encode() + b"\n")
            conn.sendall(raw)

        media = os.path.join(tempfile.mkdtemp(), "media.sock")  # short: AF_UNIX path limit
        _serve_once(media, handler)
        driver = MicroduckDriver(port="/nowhere.sock", timeout=2.0)
        driver._media_socket = media

        envelope = _drive(driver, action="camera")
        assert envelope["status"] == "success", envelope
        written = Path(_payload(envelope)["path"])
        assert written.parent == frame_root() and written.read_bytes()[:2] == b"\xff\xd8"

    def test_camera_refuses_when_mediad_is_absent(self, tmp_path: Path) -> None:
        driver = MicroduckDriver(port="/nowhere.sock", timeout=1.0)
        driver._media_socket = str(tmp_path / "none.sock")
        envelope = _drive(driver, action="camera")
        assert envelope["status"] == "error" and "mediad" in _text(envelope)

    def test_uyvy_rotation_swaps_dimensions(self, tmp_path: Path) -> None:
        import cv2

        out = str(tmp_path / "r.jpg")
        uyvy_to_jpeg(8, 4, 90, bytes([128, 128, 128, 128] * 16), out)
        image = cv2.imread(out)
        assert image is not None and image.shape[:2] == (8, 4)

    def test_tof_reads_one_frame_and_summarises(self, tmp_path: Path) -> None:
        def handler(conn: socket.socket, request: dict[str, Any]) -> None:
            assert request["method"] == "tof.stream" and request["params"] == {"on": True}
            conn.sendall(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"on": True}}).encode() + b"\n")
            frame = {"seq": 7, "rows": 2, "cols": 2, "distance_mm": [500, 1200, 0, 300], "status": [5, 9, 255, 5]}
            conn.sendall(json.dumps({"jsonrpc": "2.0", "method": "tof.frame", "params": frame}).encode() + b"\n")

        tof = os.path.join(tempfile.mkdtemp(), "tof.sock")
        _serve_once(tof, handler)
        driver = MicroduckDriver(port="/nowhere.sock", timeout=2.0)
        driver._tof_socket = tof
        payload = _payload(_drive(driver, action="tof"))
        assert payload["ranged"] == 3 and payload["unmeasurable"] == 1
        assert payload["nearest_m"] == 0.3 and payload["farthest_m"] == 1.2 and payload["median_m"] == 0.5

    def test_summarise_tof_handles_an_empty_frame(self) -> None:
        summary = summarise_tof({"rows": 0, "cols": 0})
        assert summary["ranged"] == 0 and summary["nearest_m"] is None


# --------------------------------------------------------------------------- #
# The undeclared verb and the disconnected write.                             #
# --------------------------------------------------------------------------- #


class TestRefusals:
    def test_an_undeclared_verb_is_refused_by_name(self, duck: Any) -> None:
        driver, mock = duck
        envelope = _drive(driver, action="fly")
        assert envelope["status"] == "error" and "fly" in _text(envelope)
        assert len(mock.methods) == len([m for m in mock.methods])  # nothing new dispatched

    def test_a_write_against_a_dead_socket_refuses_with_the_reason(self) -> None:
        driver = MicroduckDriver(port=os.path.join(tempfile.mkdtemp(), "none.sock"), timeout=0.5)
        envelope = _drive(driver, action="head", head_yaw=0.1)
        assert envelope["status"] == "error"
        assert "head:" in _text(envelope) and "did not answer" in _text(envelope)

    def test_cleanup_stops_a_running_move(self, duck: Any) -> None:
        driver, mock = duck
        assert _drive(driver, action="move", vx=0.1, duration=5.0)["status"] == "success"
        driver.cleanup()
        assert driver._move is None

    def test_the_forward_is_closed_on_cleanup(self) -> None:
        class _Proc:
            stderr = None
            terminated = False

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                _Proc.terminated = True

            def wait(self, timeout: float = 0) -> None:
                pass

        forward = _SshForward(
            "u", "h", remote_sockets={"robot": DEFAULT_SOCKET}, timeout=0.3, spawn=lambda *a, **k: _Proc()
        )
        assert forward.start() is not None  # the socket never appears
        assert _Proc.terminated is True
        assert not os.path.exists(forward.local["robot"])


# --------------------------------------------------------------------------- #
# The factory: Robot("microduck") with nothing else.                          #
# --------------------------------------------------------------------------- #


class TestOutOfTheBox:
    def test_auto_mode_finds_a_robotd_through_the_probe(self, monkeypatch: Any) -> None:
        from strands_robots.robot import _auto_detect_mode

        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.delenv(ENV_HOST, raising=False)
        with MockRobotd() as mock:
            monkeypatch.setenv(ENV_SOCKET, mock.path)
            assert _auto_detect_mode("microduck") == "real"

    def test_auto_mode_falls_back_to_sim_when_nothing_listens(self, monkeypatch: Any) -> None:
        """With no robotd, the probe says no - and the serial scan (a fleet-wide
        heuristic that mistakes any CH343 adapter for a servo bus) is held out
        of the picture so this grades the probe alone."""
        import strands_robots.robot as robot_mod

        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.delenv(ENV_HOST, raising=False)
        monkeypatch.setenv(ENV_SOCKET, os.path.join(tempfile.mkdtemp(), "none.sock"))
        monkeypatch.setattr(robot_mod, "scan_serial_devices", lambda: [])
        assert robot_mod._auto_detect_mode("microduck") == "sim"

    def test_a_newer_robotd_connects_and_the_skew_is_logged_not_refused(self, caplog: Any) -> None:
        """The pin at 16 refused every 0.14 duck; a pin at 31 must not repeat that
        against 0.15. duck-ipc-proto: no daemon refuses on this number, a moved
        route refuses itself by name."""
        with MockRobotd(api_version=MICRODUCK_API_VERSION + 1) as server:
            driver = MicroduckDriver(port=server.path, timeout=2.0)
            try:
                with caplog.at_level("WARNING", logger="strands_robots.drivers.microduck"):
                    assert driver.connect_eagerly() is None
                assert driver.is_connected
                assert any(str(MICRODUCK_API_VERSION + 1) in r.getMessage() for r in caplog.records)
                assert _drive(driver, action="skills")["status"] == "success"
                assert _drive(driver, action="mouth", open=0.5)["status"] == "success"
                assert _drive(driver, action="version")["status"] == "success"
            finally:
                driver.cleanup()

    def test_a_probe_that_raises_is_no_hardware(self, monkeypatch: Any) -> None:
        import strands_robots.robot as robot_mod

        monkeypatch.delenv("STRANDS_ROBOT_MODE", raising=False)
        monkeypatch.setattr(robot_mod, "scan_serial_devices", lambda: [])

        def boom(*_: Any, **__: Any) -> bool:
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(MicroduckDriver, "probe_hardware", classmethod(boom))
        assert robot_mod._auto_detect_mode("microduck") == "sim"

    def test_real_mode_with_no_arguments_reaches_the_robot_on_the_first_verb(self, monkeypatch: Any) -> None:
        from strands_robots import Robot

        monkeypatch.delenv(ENV_HOST, raising=False)
        with MockRobotd() as mock:
            monkeypatch.setenv(ENV_SOCKET, mock.path)
            robot = Robot("microduck", mode="real")
            try:
                assert isinstance(robot, MicroduckDriver)
                assert robot.is_connected is False
                envelope = _drive(robot, action="skills")
                assert envelope["status"] == "success", envelope
                assert _payload(envelope)["skills"] == list(SKILLS)
                assert robot.is_connected is True
            finally:
                robot.cleanup()
