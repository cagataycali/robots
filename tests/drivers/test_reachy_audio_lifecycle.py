"""Native microphone recording preserves sample duration and refuses discontinuities."""

import itertools
from unittest.mock import MagicMock

import pytest

from strands_robots.drivers import reachy_media


@pytest.fixture
def audio(monkeypatch):
    monkeypatch.setattr(reachy_media, "_AUDIO_PREROLL_FRAMES", 1600)
    gst = MagicMock()
    gst.SECOND = 1_000_000_000
    gst.CLOCK_TIME_NONE = 2**64 - 1
    gst.BufferFlags.DISCONT = 64
    gst.BufferFlags.RESYNC = 128
    gst.BufferFlags.CORRUPTED = 256
    gst.BufferFlags.GAP = 2048
    pipe = gst.parse_launch.return_value
    pipe.set_state.return_value = gst.StateChangeReturn.SUCCESS
    pipe.get_state.return_value = (gst.StateChangeReturn.SUCCESS, gst.State.NULL, gst.State.VOID_PENDING)
    pipe.get_bus.return_value.pop_filtered.return_value = None
    sink = MagicMock()
    source = MagicMock()
    pipe.get_by_name.side_effect = lambda name: sink if name == "audio" else source

    def sample(pts):
        item = MagicMock()
        item.get_caps.return_value.get_structure.return_value.get_value.side_effect = {
            "format": "S16LE",
            "rate": 16000,
            "channels": 1,
        }.get
        buffer = item.get_buffer.return_value
        buffer.get_flags.return_value = 0
        buffer.pts = pts
        buffer.get_size.return_value = 1280  # 40ms mono
        buffer.extract_dup.side_effect = lambda offset, size: b"\x01\x00" * (size // 2)
        return item

    packets = [sample(i * 40_000_000) for i in range(6)]
    sink.try_pull_sample.side_effect = packets
    monkeypatch.setattr(reachy_media, "_load_gst", lambda: gst)
    monkeypatch.setattr(reachy_media, "_producer_id", lambda *a: "producer-id")
    return gst, pipe, sink, packets


def test_audio_trims_final_packet_to_exact_sample_count_and_closes(audio):
    gst, pipe, _, _ = audio
    assert reachy_media._capture_pcm("reachy-a.local", 8443, 0.1)[0] == b"\x01\x00" * 1600
    text = gst.parse_launch.call_args.args[0]
    assert "rate=16000,channels=1" in text
    assert "fakesink" in text  # video is discarded
    assert "enable-control-data-channel=false" in text
    pipe.set_state.assert_any_call(gst.State.NULL)


@pytest.mark.parametrize(
    "fault",
    [
        "gap",
        "discont",
        "corrupted",
        "timestamp",
        "format",
        "empty",
        "oversized",
        "odd",
        "short",
        "start",
        "stream",
        "timeout",
    ],
)
def test_audio_refuses_bad_packets_or_pipeline_and_closes(audio, monkeypatch, fault):
    gst, pipe, sink, packets = audio
    buffer = packets[3].get_buffer.return_value
    if fault in ("gap", "discont", "corrupted"):
        packets[4].get_buffer.return_value.get_flags.return_value = {
            "gap": gst.BufferFlags.GAP,
            "discont": gst.BufferFlags.DISCONT,
            "corrupted": gst.BufferFlags.CORRUPTED,
        }[fault]
    elif fault == "timestamp":
        buffer.pts = gst.CLOCK_TIME_NONE
    elif fault == "format":
        packets[0].get_caps.return_value.get_structure.return_value.get_value.side_effect = lambda *a: None
    elif fault in ("empty", "oversized", "odd"):
        buffer.get_size.return_value = {"empty": 0, "oversized": 100000, "odd": 17}[fault]
    elif fault == "short":
        buffer.extract_dup.side_effect = lambda *a: b""
    elif fault == "start":
        pipe.set_state.return_value = gst.StateChangeReturn.FAILURE
    elif fault == "stream":
        msg = MagicMock()
        msg.parse_error.return_value = (type("Error", (), {"message": "stream failed"})(), "debug")
        pipe.get_bus.return_value.pop_filtered.return_value = msg
    elif fault == "timeout":
        # ``reachy_media.time`` is the stdlib module, so this clock answers
        # every thread in the process, not just the capture under test. It must
        # therefore never run out: an exhausting iterator raised StopIteration
        # inside an unrelated asyncio loop thread.
        clock = itertools.chain([0.0], itertools.repeat(20.0))
        monkeypatch.setattr(reachy_media.time, "monotonic", lambda: next(clock))
    with pytest.raises((RuntimeError, TimeoutError)):
        reachy_media._capture_pcm("reachy-a.local", 8443, 0.1)
    pipe.set_state.assert_any_call(gst.State.NULL)


def test_audio_cannot_report_success_when_cleanup_fails(audio):
    gst, pipe, _, _ = audio
    pipe.get_state.return_value = (gst.StateChangeReturn.FAILURE, gst.State.PLAYING, None)
    with pytest.raises(RuntimeError, match="cleanup"):
        reachy_media._capture_pcm("reachy-a.local", 8443, 0.1)


def test_audio_discards_unstable_startup_but_never_pads_it(audio):
    gst, pipe, sink, packets = audio
    # The first interval overlaps while the remote receive clock settles.
    # Discard startup samples, without filling/padding clock corrections.
    packets[0].get_buffer.return_value.pts = 10_000_000
    extra = MagicMock()
    extra.get_caps.return_value = packets[0].get_caps.return_value
    buffer = extra.get_buffer.return_value
    buffer.pts = 240_000_000
    buffer.get_size.return_value = 1280
    buffer.extract_dup.side_effect = lambda offset, size: b"\x01\x00" * (size // 2)
    sink.try_pull_sample.side_effect = packets + [extra]
    assert reachy_media._capture_pcm("reachy-a.local", 8443, 0.1)[0] == b"\x01\x00" * 1600
    for packet in packets[:3]:
        packet.get_buffer.return_value.extract_dup.assert_not_called()
    pipe.set_state.assert_any_call(gst.State.NULL)


@pytest.mark.parametrize("delta_ns", [-231230, 231230, -20000000, 20000000])
def test_receive_clock_adjustment_is_reported_not_misread_as_lost_samples(audio, delta_ns):
    gst, pipe, sink, packets = audio
    packets[4].get_buffer.return_value.pts += delta_ns
    pcm, quality = reachy_media._capture_pcm("reachy-a.local", 8443, 0.1)
    assert pcm == b"\x01\x00" * 1600
    assert quality["duration_basis"] == "decoded_pcm_samples"
    assert quality["timestamp_adjustments"] == 2
    assert quality["max_timestamp_step_ms"] == abs(delta_ns) / 1e6
    assert quality["transport_loss_verified"] is False
    pipe.set_state.assert_any_call(gst.State.NULL)


def test_clock_resync_flag_is_not_misread_as_decoder_discontinuity(audio):
    gst, pipe, sink, packets = audio
    packets[4].get_buffer.return_value.get_flags.return_value = gst.BufferFlags.RESYNC
    pcm, quality = reachy_media._capture_pcm("reachy-a.local", 8443, 0.1)
    assert len(pcm) == 3200
    assert quality["clock_resyncs"] == 1


@pytest.mark.parametrize("fault", ["request_failed", "wait_failed", "wait_async", "pending_transition"])
def test_null_state_alone_does_not_prove_successful_cleanup(audio, fault):
    gst, pipe, _, _ = audio
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
        reachy_media._capture_pcm("reachy-a.local", 8443, 0.1)
