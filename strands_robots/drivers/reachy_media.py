"""Bounded receive-only camera and microphone sessions on the daemon's native GStreamer LAN service.

No SDK, dashboard proxy, media acquire/release, outbound track, or motor command.
GStreamer/PyGObject are optional system dependencies; importing this module does
not load either. Captured data is returned only after teardown is confirmed;
a failed teardown may leave the receiver active. Native calls are not preempted.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from strands_robots.utils import require_optional

_GST_INSTALL = "Install PyGObject and GStreamer with the rswebrtc, JPEG, audio and video-conversion plugins."
_MAX_JPEG_BYTES = 16 * 1024 * 1024
_AUDIO_PREROLL_FRAMES = 16000


def _load_gst() -> Any:
    gi: Any = require_optional("gi", system_install=_GST_INSTALL, purpose="native Reachy media capture")
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gst: Any = require_optional("gi.repository.Gst", system_install=_GST_INSTALL)
    require_optional("gi.repository.GstApp", system_install=_GST_INSTALL)
    gst.init(None)
    return gst


def _producer_id(uri: str, deadline: float) -> str:
    client: Any = require_optional("websockets.sync.client", pip_install="websockets>=17.0")
    with client.connect(uri, open_timeout=max(0.01, deadline - time.monotonic()), close_timeout=1) as ws:
        welcome = json.loads(ws.recv(timeout=max(0.01, deadline - time.monotonic())))
        if not isinstance(welcome, dict) or welcome.get("type") != "welcome":
            raise RuntimeError("Reachy media signaling did not send a welcome")
        ws.send(json.dumps({"type": "list"}))
        message = json.loads(ws.recv(timeout=max(0.01, deadline - time.monotonic())))
    if not isinstance(message, dict) or message.get("type") != "list":
        raise RuntimeError("Reachy media signaling did not return a producer list")
    producers = message.get("producers")
    if not isinstance(producers, list):
        raise RuntimeError("Reachy media producer list is malformed")
    matches = [
        p.get("id")
        for p in producers
        if isinstance(p, dict) and isinstance(p.get("meta"), dict) and p["meta"].get("name") == "reachymini"
    ]
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise RuntimeError("Reachy media needs exactly one named reachymini producer; none or ambiguous")
    return matches[0]


def _stop_pipeline(gst: Any, pipeline: Any) -> None:
    """Request teardown and confirm completion before returning captured data.

    Args:
        gst: The loaded GStreamer module.
        pipeline: This capture's privately owned receiver pipeline.

    Raises:
        RuntimeError: Teardown failed or completion could not be confirmed.
            The receiver may remain active; the caller must not save data.

    The state verification waits at most three seconds. This is not a hard
    deadline on native calls, nor a guarantee of release after an error.
    PyGObject owns reference management; manually unrefing here is unsafe.
    """
    requested = pipeline.set_state(gst.State.NULL)
    changed, current, pending = pipeline.get_state(3 * gst.SECOND)
    if (
        requested == gst.StateChangeReturn.FAILURE
        or changed != gst.StateChangeReturn.SUCCESS
        or current != gst.State.NULL
        or pending != gst.State.VOID_PENDING
    ):
        raise RuntimeError(
            "Reachy media cleanup could not be confirmed; receiver may remain active and output was not saved"
        )


def _capture_jpeg(host: str, port: int, timeout: float = 10.0) -> bytes:
    gst = _load_gst()
    deadline = time.monotonic() + timeout
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    uri = f"ws://{authority}:{port}"
    producer = _producer_id(uri, deadline)
    # Caller strings are properties, never interpolated into pipeline syntax.
    pipeline = gst.parse_launch(
        "webrtcsrc name=source enable-control-data-channel=false enable-data-channel-navigation=false "
        "source. ! video/x-raw ! queue ! videoconvert ! jpegenc ! "
        "appsink name=frame drop=true max-buffers=1 sync=false "
        "source. ! audio/x-raw ! queue ! fakesink sync=false"
    )
    try:
        signaller = pipeline.get_by_name("source").get_property("signaller")
        signaller.set_property("uri", uri)
        signaller.set_property("producer-peer-id", producer)
        sink = pipeline.get_by_name("frame")
        bus = pipeline.get_bus()
        if pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Reachy camera pipeline could not start")
        while time.monotonic() < deadline:
            message = bus.pop_filtered(gst.MessageType.ERROR)
            if message is not None:
                error, _debug = message.parse_error()
                raise RuntimeError(f"Reachy camera stream failed: {error.message}")
            sample = sink.try_pull_sample(int(min(0.1, max(0, deadline - time.monotonic())) * gst.SECOND))
            if sample is not None:
                buffer = sample.get_buffer()
                if buffer is None or not 0 < buffer.get_size() <= _MAX_JPEG_BYTES:
                    raise RuntimeError("Reachy camera returned an empty or oversized frame")
                return bytes(buffer.extract_dup(0, buffer.get_size()))
        raise TimeoutError("Reachy camera did not deliver a frame within the capture timeout")
    finally:
        _stop_pipeline(gst, pipeline)


def _save_jpeg(data: bytes, save_path: str) -> dict[str, Any]:
    image_module: Any = require_optional("PIL.Image", pip_install="Pillow")
    with image_module.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.width * image.height > 16_000_000:
            raise ValueError("Reachy camera returned an unsupported image format or size")
        image.load()  # A JPEG marker alone does not prove decodability.
        width, height = image.size
    path = _save_bytes(data, save_path, "reachy-camera-", ".jpg")
    return {"path": str(path), "width": width, "height": height, "bytes": len(data), "source": "daemon-webrtc"}


def _save_bytes(data: bytes, save_path: str, prefix: str, suffix: str) -> Path:
    if save_path:
        path = Path(save_path).expanduser()
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
            output.write(data)
    else:
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as output:
            output.write(data)
            path = Path(output.name)
    return path


def _capture_pcm(host: str, port: int, duration: float) -> tuple[bytes, dict[str, Any]]:
    """Receive normalized PCM and report receiver-clock adjustments separately.

    GStreamer rtpjitterbuffer applies calculate_skew()/apply_offset() to PTS;
    PTS spacing is not a sample-loss oracle. Refuse decoder-reported damage,
    but retain clock diagnostics rather than inserting/dropping PCM to align it.
    Opus/network concealment may be invisible here: no lossless claim is made.
    """
    gst = _load_gst()
    deadline = time.monotonic() + 10.0 + duration
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    uri = f"ws://{authority}:{port}"
    producer = _producer_id(uri, deadline)
    pipeline = gst.parse_launch(
        "webrtcsrc name=source enable-control-data-channel=false enable-data-channel-navigation=false "
        "source. ! video/x-raw ! queue ! fakesink sync=false "
        "source. ! audio/x-raw ! queue ! audioconvert ! audioresample ! "
        "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1 ! "
        "appsink name=audio drop=false max-buffers=64 sync=false"
    )
    remaining = round(duration * 16000) * 2
    chunks = bytearray()
    next_pts = None
    preroll_frames = 0
    quality: dict[str, Any] = {
        "duration_basis": "decoded_pcm_samples",
        "timestamp_adjustments": 0,
        "max_timestamp_step_ms": 0.0,
        "clock_resyncs": 0,
        "transport_loss_verified": False,
    }
    try:
        signaller = pipeline.get_by_name("source").get_property("signaller")
        signaller.set_property("uri", uri)
        signaller.set_property("producer-peer-id", producer)
        sink = pipeline.get_by_name("audio")
        bus = pipeline.get_bus()
        if pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Reachy audio pipeline could not start")
        while time.monotonic() < deadline:
            message = bus.pop_filtered(gst.MessageType.ERROR)
            if message is not None:
                error, _debug = message.parse_error()
                raise RuntimeError(f"Reachy audio stream failed: {error.message}")
            sample = sink.try_pull_sample(int(min(0.1, max(0, deadline - time.monotonic())) * gst.SECOND))
            if sample is None:
                continue
            caps = sample.get_caps().get_structure(0)
            if (caps.get_value("format"), caps.get_value("rate"), caps.get_value("channels")) != ("S16LE", 16000, 1):
                raise RuntimeError("Reachy audio negotiated an unexpected PCM format")
            buffer = sample.get_buffer()
            if buffer is None or not 0 < buffer.get_size() <= 65536 or buffer.get_size() % 2:
                raise RuntimeError("Reachy audio returned an empty, unaligned or oversized buffer")
            if buffer.pts == gst.CLOCK_TIME_NONE:
                raise RuntimeError("Reachy audio returned a packet without a timestamp")
            if preroll_frames < _AUDIO_PREROLL_FRAMES:
                # Discard startup clock acquisition, within the original budget.
                preroll_frames += buffer.get_size() // 2
                continue
            flags = buffer.get_flags()
            if flags & (gst.BufferFlags.DISCONT | gst.BufferFlags.GAP | gst.BufferFlags.CORRUPTED):
                raise RuntimeError(
                    "Reachy audio decoder reported discontinuous, gap or corrupted samples; recording was not saved"
                )
            if flags & gst.BufferFlags.RESYNC:
                quality["clock_resyncs"] += 1
            if next_pts is not None:
                step = abs(buffer.pts - next_pts)
                if step > gst.SECOND / 16000:
                    quality["timestamp_adjustments"] += 1
                quality["max_timestamp_step_ms"] = max(quality["max_timestamp_step_ms"], step / 1_000_000)
            next_pts = buffer.pts + buffer.get_size() // 2 * gst.SECOND // 16000
            size = min(remaining, buffer.get_size())
            chunk = bytes(buffer.extract_dup(0, size))
            if len(chunk) != size:
                raise RuntimeError("Reachy audio packet extraction was incomplete")
            chunks.extend(chunk)
            remaining -= size
            if not remaining:
                return bytes(chunks), quality
        raise TimeoutError("Reachy microphone did not deliver the requested duration within the timeout")
    finally:
        _stop_pipeline(gst, pipeline)


def _save_wav(pcm: bytes, save_path: str) -> dict[str, Any]:
    """Write normalized mono PCM as a new private WAV, reporting sample-derived duration."""
    if not pcm or len(pcm) % 2 or len(pcm) > 160000:
        raise ValueError("Reachy audio PCM must be nonempty, sample-aligned and at most five seconds")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    data = output.getvalue()
    path = _save_bytes(data, save_path, "reachy-audio-", ".wav")
    return {
        "path": str(path),
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 2,
        "frames": len(pcm) // 2,
        "duration": len(pcm) / 32000,
        "bytes": len(data),
        "source": "daemon-webrtc",
    }
