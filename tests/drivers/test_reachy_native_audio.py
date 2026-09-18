"""Native microphone capture is a bounded JSON-callable read, never playback."""

import pytest

from strands_robots.drivers.reachy import ReachyDriver


def test_audio_is_declared_without_a_live_handle_parameter():
    schema = ReachyDriver().tool_spec["inputSchema"]["json"]
    assert "record_audio" in schema["properties"]["action"]["enum"]
    assert "driver" not in schema["properties"]
    # ``duration`` is shared with the motion verbs (up to 10 s); the 5 s
    # microphone ceiling is the accessor's own, graded below.
    assert schema["properties"]["duration"]["minimum"] == 0.1
    assert schema["properties"]["duration"]["maximum"] == 10


def test_a_recording_longer_than_five_seconds_is_refused_by_the_accessor():
    driver = ReachyDriver()
    driver._connected = True  # past the connection gate, so the ceiling itself is graded
    result = driver.record_audio(duration=6.0)
    assert result["status"] == "error"
    assert "between 0.1 and 5 seconds" in result["content"][0]["text"]


def test_audio_requires_a_connection():
    result = ReachyDriver().record_audio(duration=0.5)
    assert result["status"] == "error"
    assert "not connected" in result["content"][0]["text"]


@pytest.mark.parametrize("duration", [True, None, "1", 0, -1, 5.01, float("nan"), float("inf"), [], {}])
def test_audio_refuses_unbounded_or_malformed_duration(duration):
    driver = ReachyDriver()
    driver._connected = True  # No network or link is created.
    result = driver.record_audio(duration=duration)
    assert result["status"] == "error"
    assert "duration" in result["content"][0]["text"]


def test_audio_agent_dispatch_writes_exact_private_wav(monkeypatch, tmp_path):
    import wave

    from strands import Agent

    from strands_robots.drivers import reachy_media

    driver = ReachyDriver(port="reachy-a.local")
    driver._connected = True
    calls = []
    pcm = b"\x01\x00" * 8000

    def capture(host, port, duration):
        calls.append((host, port, duration))
        return pcm, {"duration_basis": "decoded_pcm_samples", "transport_loss_verified": False}

    monkeypatch.setattr(reachy_media, "_capture_pcm", capture)
    path = tmp_path / "mic.wav"
    agent = Agent(tools=[driver], callback_handler=None)
    result = agent.tool.reachy_mini(action="record_audio", duration=0.5, save_path=str(path))
    assert result["status"] == "success"
    meta = result["content"][0]["json"]
    assert meta["duration"] == 0.5
    assert meta["frames"] == 8000
    assert calls == [("reachy-a.local", 8443, 0.5)]
    assert path.stat().st_mode & 0o777 == 0o600
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, 8000)
        assert wav.readframes(8000) == pcm
    # Another recording may not overwrite this one.
    assert driver.record_audio(0.5, str(path))["status"] == "error"


@pytest.mark.parametrize("failure", [TimeoutError("missing samples"), RuntimeError("gap"), ImportError("gi absent")])
def test_audio_backend_failure_saves_nothing(monkeypatch, tmp_path, failure):
    from strands_robots.drivers import reachy_media

    def fail(*args):
        raise failure

    monkeypatch.setattr(reachy_media, "_capture_pcm", fail)
    driver = ReachyDriver()
    driver._connected = True
    path = tmp_path / "mic.wav"
    result = driver.record_audio(0.5, str(path))
    assert result["status"] == "error"
    assert str(failure) in result["content"][0]["text"]
    assert not path.exists()


def test_short_recording_is_not_reported_as_requested_duration(monkeypatch, tmp_path):
    from strands_robots.drivers import reachy_media

    monkeypatch.setattr(reachy_media, "_capture_pcm", lambda *a: (b"\x00\x00" * 10, {}))
    driver = ReachyDriver()
    driver._connected = True
    path = tmp_path / "mic.wav"
    assert driver.record_audio(0.5, str(path))["status"] == "error"
    assert not path.exists()


@pytest.mark.parametrize("variable", ["REACHY_DAEMON_TOKEN", "REACHY_DAEMON_TLS"])
def test_audio_does_not_bypass_daemon_security(monkeypatch, variable):
    monkeypatch.setenv(variable, "true")
    driver = ReachyDriver()
    driver._connected = True
    result = driver.record_audio()
    assert result["status"] == "error"
    assert "authenticated/TLS" in result["content"][0]["text"]
