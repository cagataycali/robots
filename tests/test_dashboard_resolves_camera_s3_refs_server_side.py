"""The dashboard resolves a camera S3 reference server side; the browser never sees the URL.

Over AWS IoT Core a robot with a bucket publishes ``strands/<peer>/camera/<cam>/ref``
(:mod:`~strands_robots.mesh.iot.camera_offload`): a presigned GET URL in place of the
frame. The dashboard bridge used to ignore that payload (it only knew the inline
``data`` field), so an IoT robot's cameras never reached a card.

The reference is fetched by the bridge process, off the transport callback, one
fetch in flight per camera, with a time and size budget, and only from an
``https`` URL on an ``amazonaws.com`` host. The resulting JPEG is filed under the
wire peer like an inline frame, tagged ``via: "s3"``, and the presigned URL and
S3 URI stay out of the peer entry, the frame store and every emitted event: the
browser's session cookie is not a grant to read the fleet's bucket.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import mesh_bridge
from strands_robots.dashboard.mesh_bridge import MeshBridge

ROBOT = "dashiot-so101"
URL = "https://fleet-frames.s3.us-west-2.amazonaws.com/dashiot-so101/front/1.jpg?X-Amz-Signature=abc"
JPEG = b"\xff\xd8\xff\xe0resolved"


class _Immediate:
    """An executor stand-in that runs the job on the caller's thread."""

    def submit(self, fn: Any, *args: Any) -> Any:
        fut: Any = mock.MagicMock()
        fut.result.return_value = fn(*args)
        return fut

    def shutdown(self, wait: bool = True) -> None:
        return None


def _sample(key: str, payload: Any) -> Any:
    sample = mock.MagicMock(spec=["payload", "key_expr"])
    sample.payload.to_bytes.return_value = json.dumps(payload).encode()
    sample.key_expr = key
    return sample


def _ref(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "peer_id": ROBOT,
        "cam": "front",
        "t": time.time() - 0.25,
        "shape": [480, 640, 3],
        "encoding": "jpeg",
        "s3_uri": "s3://fleet-frames/dashiot-so101/front/1.jpg",
        "presigned_url": URL,
        "expires_at": time.time() + 60,
    }
    body.update(over)
    return body


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> MeshBridge:
    b = MeshBridge(peer_id="dash")
    b._running = True
    b._ref_pool = _Immediate()  # type: ignore[assignment]
    b._on_presence(
        _sample(f"strands/{ROBOT}/presence", {"robot_id": ROBOT, "robot_type": "robot", "timestamp": time.time()})
    )
    return b


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    def fake(url: str, *, timeout: float, max_bytes: int) -> bytes:
        urls.append(url)
        assert timeout <= 5.0 and max_bytes >= 1024
        return JPEG

    monkeypatch.setattr(mesh_bridge, "fetch_camera_ref", fake)
    return urls


def test_a_ref_becomes_a_frame_under_the_wire_peer_tagged_s3(bridge: MeshBridge, fetched: list[str]) -> None:
    events: list[dict[str, Any]] = []
    bridge._emit = events.append  # type: ignore[method-assign,assignment]
    bridge._on_camera(_sample(f"strands/{ROBOT}/camera/front/ref", _ref()))
    assert fetched == [URL]
    frame = bridge.latest_frame(ROBOT, "front")
    assert frame is not None and frame["jpeg"] == JPEG
    assert frame["via"] == "s3" and frame["displayable"] is True
    assert frame["shape"] == [480, 640, 3]
    meta = bridge.peers[ROBOT]["cameras"]["front"]
    assert meta["via"] == "s3"
    assert 100 <= meta["latency_ms"] <= 5000
    assert [e["type"] for e in events if e["type"] == "camera_meta"] == ["camera_meta"]


def test_the_presigned_url_and_s3_uri_never_leave_the_bridge(bridge: MeshBridge, fetched: list[str]) -> None:
    events: list[dict[str, Any]] = []
    bridge._emit = events.append  # type: ignore[method-assign,assignment]
    bridge._on_camera(_sample(f"strands/{ROBOT}/camera/front/ref", _ref()))
    everything = json.dumps({"peer": bridge.peers[ROBOT], "events": events, "snapshot": bridge.snapshot()}, default=str)
    assert "X-Amz-Signature" not in everything and "presigned_url" not in everything and "s3://" not in everything
    frame = bridge.latest_frame(ROBOT, "front")
    assert frame is not None and "presigned_url" not in frame and "s3_uri" not in frame


@pytest.mark.parametrize(
    "url",
    [
        "http://fleet-frames.s3.us-west-2.amazonaws.com/x.jpg",
        "https://evil.example.com/x.jpg",
        "https://fleet-frames.s3.us-west-2.amazonaws.com.evil.example/x.jpg",
        "file:///etc/passwd",
        "https://169.254.169.254/latest/meta-data/",
        "",
        42,
    ],
)
def test_only_an_https_amazonaws_url_is_fetched(bridge: MeshBridge, fetched: list[str], url: Any) -> None:
    bridge._on_camera(_sample(f"strands/{ROBOT}/camera/front/ref", _ref(presigned_url=url)))
    assert fetched == []
    assert bridge.latest_frame(ROBOT, "front") is None
    assert "cameras" not in bridge.peers[ROBOT]


def test_a_ref_from_another_peer_cannot_replace_the_victims_tile(bridge: MeshBridge, fetched: list[str]) -> None:
    attacker = "dashiot-evil"
    bridge._on_presence(
        _sample(f"strands/{attacker}/presence", {"robot_id": attacker, "robot_type": "robot", "timestamp": time.time()})
    )
    bridge._on_camera(_sample(f"strands/{attacker}/camera/front/ref", _ref(peer_id=ROBOT)))
    assert fetched == []
    assert bridge.latest_frame(ROBOT, "front") is None


def test_a_ref_for_a_peer_that_never_announced_is_not_fetched(bridge: MeshBridge, fetched: list[str]) -> None:
    ghost = "dashiot-ghost"
    bridge._on_camera(_sample(f"strands/{ghost}/camera/front/ref", _ref(peer_id=ghost)))
    assert fetched == []
    assert ghost not in bridge.peers


def test_one_fetch_in_flight_per_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    b = MeshBridge(peer_id="dash")
    b._running = True
    b._on_presence(
        _sample(f"strands/{ROBOT}/presence", {"robot_id": ROBOT, "robot_type": "robot", "timestamp": time.time()})
    )
    submitted: list[Any] = []

    class _Parked:
        def submit(self, fn: Any, *args: Any) -> Any:
            submitted.append((fn, args))
            return mock.MagicMock()

    b._ref_pool = _Parked()  # type: ignore[assignment]
    monkeypatch.setattr(mesh_bridge, "fetch_camera_ref", lambda url, *, timeout, max_bytes: JPEG)
    b._on_camera(_sample(f"strands/{ROBOT}/camera/front/ref", _ref()))
    b._on_camera(_sample(f"strands/{ROBOT}/camera/front/ref", _ref()))
    b._on_camera(_sample(f"strands/{ROBOT}/camera/wrist/ref", _ref(cam="wrist")))
    assert len(submitted) == 2, "a second ref for a camera whose fetch is still running is dropped"
    fn, args = submitted[0]
    fn(*args)
    b._on_camera(_sample(f"strands/{ROBOT}/camera/front/ref", _ref()))
    assert len(submitted) == 3, "the slot is free again once the fetch finished"


def test_a_failed_or_oversized_fetch_leaves_the_last_good_frame(
    bridge: MeshBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = base64.b64encode(JPEG).decode()
    bridge._on_camera(_sample(f"strands/{ROBOT}/camera/front", {"cam": "front", "data": good}))
    assert bridge.latest_frame(ROBOT, "front")["via"] == "inline"  # type: ignore[index]

    def boom(url: str, *, timeout: float, max_bytes: int) -> bytes:
        raise mesh_bridge.CameraRefError("body exceeds the frame cap")

    monkeypatch.setattr(mesh_bridge, "fetch_camera_ref", boom)
    bridge._on_camera(_sample(f"strands/{ROBOT}/camera/front/ref", _ref()))
    frame = bridge.latest_frame(ROBOT, "front")
    assert frame is not None and frame["jpeg"] == JPEG and frame["via"] == "inline"
    assert bridge.peers[ROBOT]["cameras"]["front"]["error"] == "body exceeds the frame cap"


def test_fetch_camera_ref_enforces_the_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self, n: int = -1) -> bytes:
            out, self._body = self._body[:n], self._body[n:]
            return out

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(mesh_bridge, "_urlopen", lambda url, timeout: _Resp(b"x" * 2048))
    assert mesh_bridge.fetch_camera_ref(URL, timeout=1.0, max_bytes=4096) == b"x" * 2048
    with pytest.raises(mesh_bridge.CameraRefError, match="1024"):
        mesh_bridge.fetch_camera_ref(URL, timeout=1.0, max_bytes=1024)


def test_stopping_the_bridge_closes_the_resolver_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    b = MeshBridge(peer_id="dash")
    b._running = True
    b.stop()
    with pytest.raises(RuntimeError):
        b._ref_pool.submit(lambda: None)
