"""Native media requests teardown on every outcome and verifies it before returning data."""

from unittest.mock import MagicMock

import pytest

from strands_robots.drivers import reachy_media


@pytest.fixture
def media(monkeypatch):
    gst = MagicMock()
    gst.SECOND = 1_000_000_000
    pipe = gst.parse_launch.return_value
    pipe.set_state.return_value = gst.StateChangeReturn.SUCCESS
    pipe.get_state.return_value = (gst.StateChangeReturn.SUCCESS, gst.State.NULL, gst.State.VOID_PENDING)
    pipe.get_bus.return_value.pop_filtered.return_value = None
    sink = MagicMock()
    source = MagicMock()
    pipe.get_by_name.side_effect = lambda name: sink if name == "frame" else source
    buffer = sink.try_pull_sample.return_value.get_buffer.return_value
    buffer.get_size.return_value = 4
    buffer.extract_dup.return_value = b"jpeg"
    monkeypatch.setattr(reachy_media, "_load_gst", lambda: gst)
    monkeypatch.setattr(reachy_media, "_producer_id", lambda *args: "producer-id")
    return gst, pipe, sink, source


def test_capture_uses_properties_not_pipeline_interpolation_and_closes(media):
    gst, pipe, sink, source = media
    assert reachy_media._capture_jpeg("reachy-a.local", 8443) == b"jpeg"
    pipeline_text = gst.parse_launch.call_args.args[0]
    assert "reachy-a.local" not in pipeline_text
    assert "enable-control-data-channel=false" in pipeline_text
    assert "enable-data-channel-navigation=false" in pipeline_text
    assert "fakesink" in pipeline_text  # no speaker output
    source.get_property.return_value.set_property.assert_any_call("uri", "ws://reachy-a.local:8443")
    pipe.set_state.assert_any_call(gst.State.NULL)
    assert sink.try_pull_sample.call_args.args[0] <= 100_000_000


@pytest.mark.parametrize("fault", ["start", "stream", "timeout", "empty", "oversized", "property"])
def test_capture_failure_still_closes_its_pipeline(media, fault):
    gst, pipe, sink, source = media
    timeout = 10
    if fault == "start":
        pipe.set_state.side_effect = lambda state: gst.StateChangeReturn.FAILURE if state == gst.State.PLAYING else None
    elif fault == "stream":
        message = MagicMock()
        message.parse_error.return_value = (type("Error", (), {"message": "broken stream"})(), "private debug")
        pipe.get_bus.return_value.pop_filtered.return_value = message
    elif fault == "timeout":
        timeout = 0
    elif fault in ("empty", "oversized"):
        sink.try_pull_sample.return_value.get_buffer.return_value.get_size.return_value = (
            0 if fault == "empty" else 20_000_000
        )
    elif fault == "property":
        source.get_property.side_effect = RuntimeError("signaller absent")
    with pytest.raises((RuntimeError, TimeoutError)):
        reachy_media._capture_jpeg("reachy-a.local", 8443, timeout)
    pipe.set_state.assert_any_call(gst.State.NULL)


def test_failed_pipeline_cleanup_cannot_report_capture_success(media):
    gst, pipe, _, _ = media
    pipe.get_state.return_value = (gst.StateChangeReturn.FAILURE, gst.State.PLAYING, None)
    with pytest.raises(RuntimeError, match="cleanup"):
        reachy_media._capture_jpeg("reachy-a.local", 8443)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"type": "list", "producers": None},
        {"type": "list", "producers": []},
        {"type": "list", "producers": [{"id": "x", "meta": {"name": "other"}}]},
        {"type": "list", "producers": [{"id": "x", "meta": {"name": "reachymini"}}] * 2},
        {"type": "list", "producers": [{"id": None, "meta": {"name": "reachymini"}}]},
    ],
)
def test_producer_discovery_never_guesses_a_robot(monkeypatch, response):
    import json
    import time

    client = MagicMock()
    socket = client.connect.return_value.__enter__.return_value
    socket.recv.side_effect = [json.dumps({"type": "welcome"}), json.dumps(response)]
    monkeypatch.setattr(reachy_media, "require_optional", lambda *a, **k: client)
    with pytest.raises(RuntimeError):
        reachy_media._producer_id("ws://reachy-a.local:8443", time.monotonic() + 10)
    client.connect.return_value.__exit__.assert_called_once()


def test_producer_discovery_bounds_every_wait_and_closes(monkeypatch):
    import json
    import time

    client = MagicMock()
    socket = client.connect.return_value.__enter__.return_value
    socket.recv.side_effect = [
        '{"type":"welcome"}',
        json.dumps({"type": "list", "producers": [{"id": "robot-id", "meta": {"name": "reachymini"}}]}),
    ]
    monkeypatch.setattr(reachy_media, "require_optional", lambda *a, **k: client)
    assert reachy_media._producer_id("ws://reachy-a.local:8443", time.monotonic() + 10) == "robot-id"
    assert 0 < client.connect.call_args.kwargs["open_timeout"] <= 10
    assert client.connect.call_args.kwargs["close_timeout"] == 1
    assert all(0 < call.kwargs["timeout"] <= 10 for call in socket.recv.call_args_list)
    client.connect.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("fault", ["request_failed", "wait_failed", "wait_async", "pending_transition"])
def test_null_state_alone_does_not_prove_successful_cleanup(media, fault):
    gst, pipe, _, _ = media
    if fault == "request_failed":
        pipe.set_state.side_effect = lambda state: (
            gst.StateChangeReturn.FAILURE if state == gst.State.NULL else gst.StateChangeReturn.SUCCESS
        )
    elif fault == "wait_failed":
        pipe.get_state.return_value = (gst.StateChangeReturn.FAILURE, gst.State.NULL, gst.State.VOID_PENDING)
    elif fault == "wait_async":
        pipe.get_state.return_value = (gst.StateChangeReturn.ASYNC, gst.State.NULL, gst.State.VOID_PENDING)
    else:
        pipe.get_state.return_value = (gst.StateChangeReturn.SUCCESS, gst.State.NULL, gst.State.PLAYING)
    with pytest.raises(RuntimeError, match="cleanup"):
        reachy_media._capture_jpeg("reachy-a.local", 8443)
