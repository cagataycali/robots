"""``FoxgloveBridge`` publishes on the telemetry seam and stays read only unless asked.

These tests use the real ``foxglove`` message encoders (so the bytes Foxglove
reads are the bytes graded) but never open a socket: ``start_server`` is
replaced by a stand-in that only remembers what it was asked to advertise, and
every message is read back from the MCAP the bridge wrote. Skipped without the
``[foxglove]`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("foxglove")

from strands_robots.foxglove import FoxgloveBridge, FoxgloveOptions, TelemetryFanout, mcap_info  # noqa: E402
from strands_robots.foxglove import bridge as bridge_mod  # noqa: E402
from strands_robots.foxglove.services import FOXGLOVE_COMMAND_ALLOW_ENV, SET_JOINT_POSITIONS  # noqa: E402
from tests.foxglove._engine_double import TelemetryEngine  # noqa: E402


class _FakeServer:
    """What ``foxglove.start_server`` hands back, minus the socket."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.port = kwargs["port"] or 43210
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> list[_FakeServer]:
    import foxglove

    started: list[_FakeServer] = []

    def _start(**kwargs: Any) -> _FakeServer:
        server = _FakeServer(**kwargs)
        started.append(server)
        return server

    monkeypatch.setattr(foxglove, "start_server", _start)
    monkeypatch.delenv(FOXGLOVE_COMMAND_ALLOW_ENV, raising=False)
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    return started


def _options(tmp_path: Path, **overrides: Any) -> FoxgloveOptions:
    return FoxgloveOptions(host="127.0.0.1", port=0, mcap=tmp_path / "run.mcap", **overrides)


def _messages(path: Path, topic: str) -> list[Any]:
    from mcap.reader import make_reader

    with path.open("rb") as handle:
        return [message for _, channel, message in make_reader(handle).iter_messages() if channel.topic == topic]


class TestReadOnlyByDefault:
    def test_no_capability_is_advertised_without_services(self, fake_server: list[_FakeServer], tmp_path: Path) -> None:
        bridge = FoxgloveBridge(_options(tmp_path), name="probe")
        try:
            assert bridge.capabilities == []
            assert fake_server[0].kwargs["capabilities"] is None
            assert fake_server[0].kwargs["services"] is None
        finally:
            bridge.shutdown()

    def test_services_asked_for_without_a_sink_stay_off(self, fake_server: list[_FakeServer], tmp_path: Path) -> None:
        bridge = FoxgloveBridge(_options(tmp_path, services=True), name="probe", command_sink=None)
        try:
            assert bridge.capabilities == []
        finally:
            bridge.shutdown()

    def test_client_publish_and_parameters_are_never_advertised(
        self, fake_server: list[_FakeServer], tmp_path: Path
    ) -> None:
        bridge = FoxgloveBridge(_options(tmp_path, services=True), name="probe", command_sink=lambda r, p: {})
        try:
            names = [c.name for c in fake_server[0].kwargs["capabilities"]]
            assert names == ["Services"]
        finally:
            bridge.shutdown()


class TestUrls:
    def test_url_and_deep_link_name_the_bound_port(self, fake_server: list[_FakeServer], tmp_path: Path) -> None:
        bridge = FoxgloveBridge(_options(tmp_path), name="probe")
        try:
            assert bridge.url == "ws://127.0.0.1:43210"
            assert bridge.link == "foxglove://open?ds=foxglove-websocket&ds.url=ws://127.0.0.1:43210"
        finally:
            bridge.shutdown()

    def test_a_busy_port_moves_to_the_next_free_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import foxglove

        tried: list[int] = []

        def _start(**kwargs: Any) -> _FakeServer:
            tried.append(kwargs["port"])
            if kwargs["port"] < 8767:
                raise RuntimeError("FoxgloveError: Failed to bind port: Address already in use (os error 48)")
            return _FakeServer(**kwargs)

        monkeypatch.setattr(foxglove, "start_server", _start)
        bridge = FoxgloveBridge(FoxgloveOptions(port=8765), name="probe")
        try:
            assert tried == [8765, 8766, 8767]
            assert bridge.port == 8767
        finally:
            bridge.shutdown()

    def test_no_free_port_is_refused_with_the_remedy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import foxglove

        def _start(**kwargs: Any) -> _FakeServer:
            raise RuntimeError("Failed to bind port: Address already in use")

        monkeypatch.setattr(foxglove, "start_server", _start)
        with pytest.raises(RuntimeError, match=r"no free port in 8765-8780 .* pass foxglove='host:port' or ':0'"):
            FoxgloveBridge(FoxgloveOptions(port=8765, mcap=tmp_path / "run.mcap"), name="probe")

    def test_a_bind_failure_that_is_not_a_busy_port_is_raised_as_is(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import foxglove

        def _start(**kwargs: Any) -> _FakeServer:
            raise RuntimeError("Failed to bind port: Permission denied")

        monkeypatch.setattr(foxglove, "start_server", _start)
        with pytest.raises(RuntimeError, match=r"Permission denied"):
            FoxgloveBridge(FoxgloveOptions(host="0.0.0.0", port=80), name="probe")


class TestPublishing:
    def test_joint_states_and_images_land_under_the_robot_namespace(
        self, fake_server: list[_FakeServer], tmp_path: Path
    ) -> None:
        bridge = FoxgloveBridge(_options(tmp_path), name="probe", camera_hz=0)
        bridge.publish_joint_states("so101", ["1", "2"], [0.5, -0.25])
        bridge.publish_image("so101", "front", np.zeros((8, 8, 3), dtype=np.uint8))
        bridge.shutdown()
        info = mcap_info(tmp_path / "run.mcap")
        assert info["channels"]["/so101/joint_states"]["messages"] == 1
        assert info["channels"]["/so101/joint_states"]["schema"] == "foxglove.JointStates"
        assert info["channels"]["/so101/camera/front"]["messages"] == 1
        assert bridge.frames_sent == 1

    def test_an_array_frame_is_jpeg_and_bytes_pass_through(
        self, fake_server: list[_FakeServer], tmp_path: Path
    ) -> None:
        from foxglove.messages import CompressedImage

        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        bridge = FoxgloveBridge(_options(tmp_path), name="probe", camera_hz=0)
        bridge.publish_image("so101", "a", np.full((4, 4, 3), 200, dtype=np.uint8))
        bridge.publish_image("so101", "b", png)
        bridge.shutdown()
        (array_msg,) = _messages(tmp_path / "run.mcap", "/so101/camera/a")
        (bytes_msg,) = _messages(tmp_path / "run.mcap", "/so101/camera/b")
        assert b"jpeg" in array_msg.data
        assert array_msg.data.find(b"\xff\xd8") >= 0
        assert b"png" in bytes_msg.data
        assert png in bytes_msg.data
        assert CompressedImage is not None

    @pytest.mark.parametrize(
        "frame", [np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 4), dtype=np.uint8), b"not-an-image", 3.5]
    )
    def test_a_value_that_is_not_an_image_is_dropped_not_raised(
        self, fake_server: list[_FakeServer], tmp_path: Path, frame: Any
    ) -> None:
        bridge = FoxgloveBridge(_options(tmp_path), name="probe", camera_hz=0)
        bridge.publish_image("so101", "x", frame)
        bridge.shutdown()
        assert "/so101/camera/x" not in mcap_info(tmp_path / "run.mcap")["channels"]

    def test_the_camera_rate_limit_drops_frames_and_says_when_one_is_due(
        self, fake_server: list[_FakeServer], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = {"t": 100.0}
        monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: clock["t"])
        bridge = FoxgloveBridge(_options(tmp_path), name="probe", camera_hz=10.0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        assert bridge.wants_images() is True
        bridge.publish_image("so101", "cam", frame)
        assert bridge.wants_images() is False
        bridge.publish_image("so101", "cam", frame)  # same instant: dropped
        clock["t"] += 0.05
        bridge.publish_image("so101", "cam", frame)  # 50 ms later: still inside the 100 ms period
        clock["t"] += 0.06
        assert bridge.wants_images() is True
        bridge.publish_image("so101", "cam", frame)
        bridge.shutdown()
        assert bridge.frames_sent == 2

    def test_the_state_rate_limit_is_per_robot(
        self, fake_server: list[_FakeServer], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 5.0)
        bridge = FoxgloveBridge(_options(tmp_path), name="probe", state_hz=50.0)
        for _ in range(3):
            bridge.publish_joint_states("left", ["1"], [0.0])
            bridge.publish_joint_states("right", ["1"], [0.0])
        bridge.shutdown()
        channels = mcap_info(tmp_path / "run.mcap")["channels"]
        assert (channels["/left/joint_states"]["messages"], channels["/right/joint_states"]["messages"]) == (1, 1)

    def test_log_and_events_have_their_own_channels(self, fake_server: list[_FakeServer], tmp_path: Path) -> None:
        bridge = FoxgloveBridge(_options(tmp_path), name="probe")
        bridge.log("warning", "gate said no", name="gate")
        bridge.event({"event": "gate", "decision": "refused"})
        bridge.shutdown()
        channels = mcap_info(tmp_path / "run.mcap")["channels"]
        assert channels["/strands/log"]["schema"] == "foxglove.Log"
        assert channels["/strands/events"] == {"schema": "strands.Event", "encoding": "json", "messages": 2}
        events = [json.loads(m.data) for m in _messages(tmp_path / "run.mcap", "/strands/events")]
        assert events[0]["event"] == "session"
        assert events[1]["decision"] == "refused"

    def test_publishing_after_shutdown_is_a_no_op(self, fake_server: list[_FakeServer], tmp_path: Path) -> None:
        bridge = FoxgloveBridge(_options(tmp_path), name="probe")
        bridge.shutdown()
        bridge.publish_joint_states("so101", ["1"], [0.0])
        bridge.publish_image("so101", "cam", np.zeros((4, 4, 3), dtype=np.uint8))
        bridge.shutdown()
        assert fake_server[0].stopped is True
        assert "/so101/joint_states" not in mcap_info(tmp_path / "run.mcap")["channels"]


class TestTheSeam:
    """The bridge plugs into ``SimEngine._init_ros_bridge`` and the hardware ``Robot`` unchanged."""

    def test_the_sim_engine_publishes_through_it_and_reports_the_url(
        self, fake_server: list[_FakeServer], tmp_path: Path
    ) -> None:
        obs = {"shoulder_pan": 0.5, "elbow": -0.25, "front": np.zeros((4, 4, 3), dtype=np.uint8)}
        engine = TelemetryEngine(obs, foxglove=":0", foxglove_mcap=tmp_path / "run.mcap")
        try:
            assert engine.foxglove_url == "ws://127.0.0.1:43210"
            assert engine.foxglove_link is not None and engine.foxglove_link.endswith(engine.foxglove_url)
            engine._publish_ros_telemetry()
        finally:
            engine._shutdown_ros_bridge()
        channels = mcap_info(tmp_path / "run.mcap")["channels"]
        assert channels["/so101/joint_states"]["messages"] == 1
        assert channels["/so101/camera/front"]["messages"] == 1
        assert engine.foxglove_url is None

    def test_the_engine_skips_the_render_when_no_frame_is_due(
        self, fake_server: list[_FakeServer], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bridge_mod.time, "monotonic", lambda: 9.0)
        renders: list[bool] = []
        obs = {"elbow": 0.0, "front": np.zeros((4, 4, 3), dtype=np.uint8)}

        class _Engine(TelemetryEngine):
            def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
                renders.append(not skip_images)
                return super().get_observation(robot_name, skip_images=skip_images)

        engine = _Engine(obs, foxglove=":0", foxglove_mcap=tmp_path / "run.mcap")
        try:
            engine._publish_ros_telemetry()
            engine._publish_ros_telemetry()
        finally:
            engine._shutdown_ros_bridge()
        assert renders == [True, False], "the second step is inside the camera period, so no image is rendered"

    def test_a_ros_bridge_and_a_foxglove_bridge_share_the_slot(
        self, fake_server: list[_FakeServer], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import strands_robots.simulation.base as base_mod

        class _Ros:
            def __init__(self) -> None:
                self.joint_calls = 0
                self.image_calls = 0
                self.down = False

            def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
                self.joint_calls += 1

            def publish_image(self, robot: str, key: str, frame: Any) -> None:
                self.image_calls += 1

            def shutdown(self) -> None:
                self.down = True

        ros = _Ros()
        monkeypatch.setattr("strands_robots.simulation.ros_bridge.SimRosBridge", lambda domain_id: ros, raising=False)
        obs = {"elbow": 0.0, "front": np.zeros((4, 4, 3), dtype=np.uint8)}
        engine = TelemetryEngine(obs, ros2_bridge=True, foxglove=":0", foxglove_mcap=tmp_path / "run.mcap")
        assert isinstance(engine._ros_bridge, TelemetryFanout)
        engine._publish_ros_telemetry()
        engine._publish_ros_telemetry()
        engine._shutdown_ros_bridge()
        assert (ros.joint_calls, ros.image_calls, ros.down) == (2, 2, True), "the ROS side keeps a frame per step"
        assert mcap_info(tmp_path / "run.mcap")["channels"]["/so101/camera/front"]["messages"] == 1
        assert base_mod is not None

    def test_the_hardware_robot_publishes_through_it(self, fake_server: list[_FakeServer], tmp_path: Path) -> None:
        from strands_robots.foxglove.options import resolve_foxglove_options
        from strands_robots.hardware_robot import Robot

        robot = Robot.__new__(Robot)
        robot.tool_name_str = "arm"
        robot.robot = type("Device", (), {"name": "so101"})()
        robot._init_ros_bridge()
        robot._init_foxglove_bridge(
            resolve_foxglove_options(":0", foxglove_mcap=tmp_path / "run.mcap", context="Robot")
        )
        try:
            assert robot.foxglove_url == "ws://127.0.0.1:43210"
            robot._publish_ros_telemetry({"shoulder.pos": 1.0, "front": np.zeros((4, 4, 3), dtype=np.uint8)})
        finally:
            robot._shutdown_ros_bridge()
        channels = mcap_info(tmp_path / "run.mcap")["channels"]
        assert channels["/so101/joint_states"]["messages"] == 1
        assert channels["/so101/camera/front"]["messages"] == 1
        assert robot.foxglove_url is None


class TestServices:
    def _bridge(self, tmp_path: Path, sink: Any) -> FoxgloveBridge:
        return FoxgloveBridge(_options(tmp_path, services=True), name="probe", command_sink=sink)

    def test_a_call_without_pre_approval_is_refused_with_the_gate_sentence(
        self, fake_server: list[_FakeServer], tmp_path: Path
    ) -> None:
        calls: list[Any] = []
        bridge = self._bridge(tmp_path, lambda robot, positions: calls.append((robot, positions)))
        (service,) = fake_server[0].kwargs["services"]
        request = type("Request", (), {"payload": json.dumps({"robot": "so101", "positions": {"1": 0.3}}).encode()})()
        with pytest.raises(PermissionError) as refused:
            service.handler(request)
        bridge.shutdown()
        text = str(refused.value)
        assert text.startswith(f"'{SET_JOINT_POSITIONS}' moves the robot from a Foxglove panel")
        assert "No tool_context available for operator approval" in text
        assert f"{FOXGLOVE_COMMAND_ALLOW_ENV}={SET_JOINT_POSITIONS}" in text
        assert calls == []
        events = [json.loads(m.data) for m in _messages(tmp_path / "run.mcap", "/strands/events")]
        assert {"service": SET_JOINT_POSITIONS, "decision": "refused"}.items() <= events[-1].items()

    def test_a_pre_approved_call_reaches_the_sink_and_answers(
        self, fake_server: list[_FakeServer], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(FOXGLOVE_COMMAND_ALLOW_ENV, SET_JOINT_POSITIONS)
        calls: list[Any] = []

        def _sink(robot: str | None, positions: dict[str, float]) -> dict[str, Any]:
            calls.append((robot, positions))
            return {"status": "success", "content": [{"text": "Set 1/1 joint positions"}]}

        bridge = self._bridge(tmp_path, _sink)
        (service,) = fake_server[0].kwargs["services"]
        request = type("Request", (), {"payload": json.dumps({"positions": {"1": 0.3}}).encode()})()
        answer = json.loads(service.handler(request))
        bridge.shutdown()
        assert calls == [(None, {"1": 0.3})]
        assert answer == {"status": "success", "text": "Set 1/1 joint positions"}

    @pytest.mark.parametrize(
        "payload, sentence",
        [
            (b"{", "request is not JSON"),
            (b"{}", "with at least one joint"),
            (b'{"positions": {"1": "0.3"}}', "positions[1] must be a number"),
            (b'{"positions": {"1": 0.3}, "robot": 7}', "robot must be a string"),
        ],
    )
    def test_a_malformed_request_is_refused_before_the_gate(
        self, fake_server: list[_FakeServer], tmp_path: Path, payload: bytes, sentence: str
    ) -> None:
        bridge = self._bridge(tmp_path, lambda robot, positions: {})
        (service,) = fake_server[0].kwargs["services"]
        with pytest.raises(ValueError, match=sentence.replace("[", r"\[").replace("]", r"\]")):
            service.handler(type("Request", (), {"payload": payload})())
        bridge.shutdown()

    def test_the_sim_engine_routes_the_call_to_set_joint_positions(
        self, fake_server: list[_FakeServer], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(FOXGLOVE_COMMAND_ALLOW_ENV, "*")
        engine = TelemetryEngine({"elbow": 0.0}, foxglove=":0", foxglove_services=True)
        (service,) = fake_server[0].kwargs["services"]
        service.handler(type("Request", (), {"payload": b'{"robot": "so101", "positions": {"elbow": 0.4}}'})())
        engine._shutdown_ros_bridge()
        assert engine.last_command == ("so101", {"elbow": 0.4}, True)
