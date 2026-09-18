"""Camera capture belongs on the native driver, including its JSON tool surface."""

from unittest.mock import AsyncMock

import pytest

from strands_robots.device_connect import reachy_transport
from strands_robots.drivers.reachy import ReachyDriver


def test_camera_is_a_json_callable_driver_action() -> None:
    schema = ReachyDriver().tool_spec["inputSchema"]["json"]
    assert "camera" in schema["properties"]["action"]["enum"]
    assert "driver" not in schema["properties"]
    assert schema["properties"]["save_path"]["type"] == "string"


def test_camera_requires_connection_before_loading_media() -> None:
    result = ReachyDriver().capture_frame()
    assert result["status"] == "error"
    assert "not connected" in result["content"][0]["text"]


def test_camera_dispatch_reaches_the_bound_driver(connected, monkeypatch: pytest.MonkeyPatch) -> None:
    # Through the ``connected`` fixture's stubbed daemon, because the verb
    # connects lazily before dispatching: left to discover one, this passed only
    # on a host that happens to have a Mini on its network.
    from strands import Agent

    seen = []

    def capture(save_path=""):
        seen.append(save_path)
        return {"status": "success", "content": [{"json": {"path": save_path, "width": 16, "height": 8}}]}

    monkeypatch.setattr(connected, "capture_frame", capture)
    agent = Agent(tools=[connected], callback_handler=None)
    result = agent.tool.reachy_mini(action="camera", save_path="desk.jpg")
    assert result["status"] == "success"
    assert seen == ["desk.jpg"]


@pytest.fixture
def connected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(reachy_transport, "api", lambda *a, **k: {"wireless_version": True})
    socket = AsyncMock()

    async def start(self, on_joints, on_imu):
        self._ws = socket

    monkeypatch.setattr(reachy_transport.WebSocketLink, "start", start)
    driver = ReachyDriver(port="reachy-a.local")
    assert driver.connect_eagerly() is None
    try:
        yield driver
    finally:
        driver.cleanup()


def _jpeg():
    import io

    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (16, 8), color="blue").save(output, format="JPEG")
    return output.getvalue()


def test_native_capture_validates_saves_and_dispatches_without_a_handle(connected, monkeypatch, tmp_path):
    from strands import Agent

    from strands_robots.drivers import reachy_media

    calls = []
    data = _jpeg()

    def capture(host, port):
        calls.append((host, port))
        return data

    monkeypatch.setattr(reachy_media, "_capture_jpeg", capture)
    out = tmp_path / "frame.jpg"
    agent = Agent(tools=[connected], callback_handler=None)
    result = agent.tool.reachy_mini(action="camera", save_path=str(out))
    assert result["status"] == "success"
    assert result["content"][0]["json"]["width"] == 16
    assert result["content"][0]["json"]["source"] == "daemon-webrtc"
    assert out.read_bytes() == data
    assert out.stat().st_mode & 0o777 == 0o600
    assert calls == [("reachy-a.local", 8443)]


def test_camera_never_overwrites_a_file_or_symlink(connected, monkeypatch, tmp_path):
    from strands_robots.drivers import reachy_media

    monkeypatch.setattr(reachy_media, "_capture_jpeg", lambda *a: _jpeg())
    out = tmp_path / "keep.jpg"
    out.write_bytes(b"owner data")
    link = tmp_path / "link.jpg"
    link.symlink_to(out)
    for path in (out, link):
        assert connected.capture_frame(str(path))["status"] == "error"
    assert out.read_bytes() == b"owner data"
    assert link.is_symlink()


@pytest.mark.parametrize(
    "failure", [TimeoutError("frame timeout"), RuntimeError("stream failed"), ImportError("gi absent")]
)
def test_camera_returns_backend_failures_and_writes_nothing(connected, monkeypatch, tmp_path, failure):
    from strands_robots.drivers import reachy_media

    def fail(*args):
        raise failure

    monkeypatch.setattr(reachy_media, "_capture_jpeg", fail)
    out = tmp_path / "absent.jpg"
    result = connected.capture_frame(str(out))
    assert result["status"] == "error"
    assert str(failure) in result["content"][0]["text"]
    assert not out.exists()


@pytest.mark.parametrize("value", [None, 4, {}, []])
def test_camera_path_domain_is_checked_before_backend(connected, value):
    assert connected.capture_frame(value)["status"] == "error"


@pytest.mark.parametrize("variable,value", [("REACHY_DAEMON_TLS", "true"), ("REACHY_DAEMON_TOKEN", "test-token")])
def test_camera_does_not_downgrade_configured_authentication(connected, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    result = connected.capture_frame()
    assert result["status"] == "error"
    assert "authenticated/TLS" in result["content"][0]["text"]
    assert value not in result["content"][0]["text"]


def test_camera_rejects_undecodable_frame_without_writing(connected, monkeypatch, tmp_path):
    from strands_robots.drivers import reachy_media

    monkeypatch.setattr(reachy_media, "_capture_jpeg", lambda *args: b"\xff\xd8not-a-jpeg")
    out = tmp_path / "bad.jpg"
    assert connected.capture_frame(str(out))["status"] == "error"
    assert not out.exists()


@pytest.mark.parametrize("port", [0, 65536, True, float("nan")])
def test_media_port_is_validated_at_construction(port):
    with pytest.raises(ValueError, match="media_port"):
        ReachyDriver(media_port=port)
