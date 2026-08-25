"""A LiDAR point cloud gets from a robot to a viewer without displacing the summary chip.

Three properties are pinned here, and the first is a regression.

**The summary survives.** ``strands/*/lidar/**`` is one subscription serving several leaves, and the
bridge used to derive the document name as ``"state" if key.endswith("/state") else "summary"``. That
default is a catch-all: the first sub-topic added under ``lidar/`` was filed as the summary, so a
cloud publish landed on top of the scalar chip an operator reads and reported the result under the
summary's name. Measured on the pre-fix bridge, one ``lidar/cloud`` message left
``lidar["summary"]["count"]`` and ``["bbox"]`` reading ``None`` while ``/ws/mesh`` announced a
healthy ``kind="summary"`` frame. Nothing failed; the chip simply told the operator something the
sensor never said.

**The budget is a downsample, not a crop.** A sweep arrives in scan order, so keeping the first
``max_points`` of it keeps one wedge. These pin the stride, and pin the coverage that stride buys.

**Geometry does not ride the snapshot socket.** ``/ws/mesh`` is fanned out to every viewer, so the
points go to :attr:`MeshBridge.clouds` and only their counts are announced -- the same split
:meth:`MeshBridge._on_camera` makes for pixels.
"""

from __future__ import annotations

import base64
import json
import threading
import types
from typing import Any

import pytest

from strands_robots.dashboard.mesh_bridge import MeshBridge
from strands_robots.mesh.sensors import SensorLoopsMixin
from strands_robots.mesh.session import LIDAR_CLOUD_MAX_POINTS

np = pytest.importorskip("numpy")

BYTES_PER_POINT = 16


class _Sample:
    """The two attributes ``MeshBridge`` reads off a zenoh sample."""

    def __init__(self, key: str, payload: dict[str, Any]) -> None:
        self.key_expr = key
        raw = json.dumps(payload).encode()
        self.payload = types.SimpleNamespace(to_bytes=lambda: raw)


class _Publisher(SensorLoopsMixin):
    """The mixin with the host ``Mesh`` collaborators it declares, and nothing else."""

    peer_id = "g1"

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self._running = True
        self._stop_event = threading.Event()
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        self.sent.append((key, payload))


def _sweep(n: int = 24000) -> Any:
    """A modelled MID-360 sweep: scan-ordered, so azimuth walks 0..360 down the rows."""
    az = np.linspace(0.0, 360.0, n, endpoint=False)
    el = 7.0 * np.sin(np.deg2rad(az) * 11.0)
    r = 6.0 + 2.0 * np.sin(np.deg2rad(az) * 3.0)
    return np.stack(
        [
            r * np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
            r * np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(el)),
            r * np.sin(np.deg2rad(el)),
            np.linspace(0.0, 1.0, n),
        ],
        axis=1,
    ).astype(np.float32)


def _sectors(points: Any) -> int:
    """Ten-degree azimuth sectors these points touch, out of 36."""
    az = np.rad2deg(np.arctan2(points[:, 1], points[:, 0])) % 360.0
    return len(np.unique((az // 10).astype(int)))


def _published(cloud: Any) -> dict[str, Any]:
    """The payload the publisher builds for *cloud*.

    Narrowed here rather than at each use: ``_read_lidar_cloud`` returns ``None`` for a provider it
    cannot read, and every fixture below hands it one it can. A ``None`` reaching a caller is a
    broken fixture, and this says so once instead of failing as an unsubscriptable value later.
    """
    payload = _Publisher(types.SimpleNamespace(_lidar_cloud=cloud))._read_lidar_cloud()
    assert payload is not None, "this fixture cloud is meant to publish"
    return payload


def _served(bridge: MeshBridge, peer_id: str = "g1") -> dict[str, Any]:
    """The cloud ``/ws/lidar`` would send for *peer_id*."""
    cloud = bridge.latest_cloud(peer_id)
    assert cloud is not None, f"the bridge kept no cloud for {peer_id}"
    return cloud


SUMMARY = {"peer_id": "g1", "t": 1.0, "count": 9000, "bbox": [[-3, -3, 0], [3, 3, 2]], "range_max": 40.0}
STATE = {"peer_id": "g1", "t": 1.0, "mode": "scanning", "temperature_c": 41.0}


@pytest.fixture
def bridge() -> tuple[MeshBridge, list[dict[str, Any]]]:
    b = MeshBridge()
    emitted: list[dict[str, Any]] = []
    b._emit = emitted.append  # type: ignore[assignment]
    return b, emitted


class TestTheSummaryChipSurvivesACloud:
    """The regression: a cloud must not be filed as the document it is not."""

    def test_a_cloud_does_not_overwrite_the_summary(self, bridge) -> None:
        b, _ = bridge
        b._on_lidar(_Sample("strands/g1/lidar/summary", SUMMARY))
        cloud = _published(_sweep(64))
        assert cloud is not None
        b._on_lidar(_Sample("strands/g1/lidar/cloud", cloud))

        summary = b.peers["g1"]["lidar"]["summary"]
        assert summary["count"] == 9000, "the operator's point count came from the summary topic"
        assert summary["bbox"] == [[-3, -3, 0], [3, 3, 2]], "and so did the bounding box"
        assert "data" not in summary, "no base64 blob was filed as a scalar summary"

    def test_a_cloud_is_not_announced_as_a_summary(self, bridge) -> None:
        b, emitted = bridge
        cloud = _published(_sweep(64))
        b._on_lidar(_Sample("strands/g1/lidar/cloud", cloud))
        assert [(e["type"], e.get("kind")) for e in emitted] == [("lidar_cloud", None)], (
            "a cloud announced as kind=summary is the failure this pins: the frame looks healthy"
        )

    def test_a_leaf_with_no_reader_is_dropped_rather_than_misfiled(self, bridge) -> None:
        b, emitted = bridge
        b._on_lidar(_Sample("strands/g1/lidar/summary", SUMMARY))
        b._on_lidar(_Sample("strands/g1/lidar/motor_rpm", {"peer_id": "g1", "t": 2.0, "rpm": 600}))

        assert b.peers["g1"]["lidar"]["summary"]["count"] == 9000, "the summary is untouched"
        assert "rpm" not in b.peers["g1"]["lidar"]["summary"], "the unknown leaf was not filed as one"
        assert [e["type"] for e in emitted] == ["lidar"], "and nothing was announced for it"


class TestTheSummaryAndStateStillWork:
    """Controls: the two documents that predate the cloud are unchanged."""

    def test_the_two_documents_are_kept_apart(self, bridge) -> None:
        b, emitted = bridge
        b._on_lidar(_Sample("strands/g1/lidar/summary", SUMMARY))
        b._on_lidar(_Sample("strands/g1/lidar/state", STATE))
        lidar = b.peers["g1"]["lidar"]
        assert lidar["summary"]["count"] == 9000
        assert lidar["state"]["mode"] == "scanning"
        assert [(e["type"], e["kind"]) for e in emitted] == [("lidar", "summary"), ("lidar", "state")]

    def test_a_sample_naming_no_peer_is_ignored(self, bridge) -> None:
        b, emitted = bridge
        b._on_lidar(_Sample("strands/g1/lidar/summary", {"t": 1.0, "count": 5}))
        assert b.peers == {} and emitted == []


class TestThePublisherHonoursItsBudget:
    def test_a_sweep_is_downsampled_to_the_cap(self) -> None:
        payload = _published(_sweep(24000))
        assert payload is not None
        assert payload["n"] == LIDAR_CLOUD_MAX_POINTS
        assert payload["raw_count"] == 24000, "what the sensor produced is reported alongside"
        assert payload["stride"] == 6
        assert len(base64.b64decode(payload["data"])) == LIDAR_CLOUD_MAX_POINTS * BYTES_PER_POINT

    def test_the_downsample_keeps_the_whole_sweep_not_a_wedge(self) -> None:
        sweep = _sweep(24000)
        payload = _published(sweep)
        kept = np.frombuffer(base64.b64decode(payload["data"]), dtype="<f4").reshape(-1, 4)

        assert _sectors(kept) == 36, "a stride keeps every azimuth sector"
        assert _sectors(sweep[:LIDAR_CLOUD_MAX_POINTS]) == 6, (
            "truncating to the same budget keeps 6 of 36 -- a 60-degree wedge. This is the premise: "
            "without it, the sector count above would pass for either implementation"
        )

    def test_a_cloud_under_the_cap_is_published_whole(self) -> None:
        payload = _published(_sweep(900))
        assert (payload["n"], payload["raw_count"], payload["stride"]) == (900, 900, 1)

    def test_a_position_only_cloud_gets_a_zero_intensity_column(self) -> None:
        payload = _published(_sweep(10)[:, :3])
        back = np.frombuffer(base64.b64decode(payload["data"]), dtype="<f4").reshape(-1, 4)
        assert back.shape == (10, 4)
        assert np.array_equal(back[:, 3], np.zeros(10, dtype=np.float32)), (
            "a fabricated intensity would render as detail the sensor never reported"
        )

    def test_non_returns_are_dropped_so_a_reduction_over_the_buffer_stays_finite(self) -> None:
        sweep = _sweep(100)
        sweep[7] = [np.nan, np.nan, np.nan, np.nan]
        sweep[9] = [np.inf, 0.0, 0.0, 0.0]
        payload = _published(sweep)

        back = np.frombuffer(base64.b64decode(payload["data"]), dtype="<f4").reshape(-1, 4)
        assert payload["n"] == 98, "two non-returns left the cloud"
        assert payload["raw_count"] == 100, "and the raw count still reports what arrived"
        assert bool(np.isfinite(back).all())

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(None, id="absent"),
            pytest.param([], id="empty"),
            pytest.param([[1.0, 2.0]], id="two-columns"),
            pytest.param([1.0, 2.0, 3.0], id="one-dimensional"),
            pytest.param("a cloud", id="a-string"),
            pytest.param([[float("nan")] * 4], id="no-finite-row"),
        ],
    )
    def test_a_provider_that_offers_no_usable_cloud_publishes_nothing(self, value: Any) -> None:
        pub = _Publisher(types.SimpleNamespace(_lidar_cloud=value))
        assert pub._read_lidar_cloud() is None

    def test_the_loop_is_silent_when_the_robot_has_no_lidar(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_LIDAR_CLOUD_HZ", "1000")
        pub = _Publisher(types.SimpleNamespace())
        pub._running = False  # one pass of the paced generator, then stop
        pub._lidar_cloud_loop()
        assert pub.sent == []

    def test_a_non_positive_rate_disables_the_loop(self, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_LIDAR_CLOUD_HZ", "0")
        pub = _Publisher(types.SimpleNamespace(_lidar_cloud=_sweep(10)))
        pub._lidar_cloud_loop()
        assert pub.sent == [], "an operator opt-out must not cost a single publish"

    def test_the_publish_key_is_the_documented_topic(self) -> None:
        pub = _Publisher(types.SimpleNamespace(_lidar_cloud=_sweep(10)))
        pub.publish(f"strands/{pub.peer_id}/lidar/cloud", _published(_sweep(10)))
        assert pub.sent[0][0] == "strands/g1/lidar/cloud"


class TestGeometryStaysOffTheSnapshotSocket:
    def test_the_announcement_carries_counts_and_no_points(self, bridge) -> None:
        b, emitted = bridge
        cloud = _published(_sweep(24000))
        b._on_lidar(_Sample("strands/g1/lidar/cloud", cloud))

        (frame,) = emitted
        assert frame["type"] == "lidar_cloud"
        assert frame["data"]["n"] == LIDAR_CLOUD_MAX_POINTS
        assert frame["data"]["bytes"] == LIDAR_CLOUD_MAX_POINTS * BYTES_PER_POINT
        assert frame["data"]["raw_count"] == 24000
        # The whole point of the split: this frame is JSON-serialised once per viewer.
        assert "data" not in frame["data"] and "xyzi" not in frame["data"]
        assert len(json.dumps(frame)) < 500, "the announcement is small enough to fan out"

    def test_the_peer_snapshot_carries_no_geometry(self, bridge) -> None:
        b, _ = bridge
        cloud = _published(_sweep(24000))
        b._on_lidar(_Sample("strands/g1/lidar/cloud", cloud))
        assert "lidar" not in b.peers["g1"], "a snapshot is re-encoded per viewer; points must not be in it"
        assert b.peers["g1"]["last_seen"] > 0, "the peer is still marked alive by the cloud"

    def test_latest_cloud_is_the_newest_one(self, bridge) -> None:
        b, _ = bridge
        first = _published(_sweep(64))
        second = _published(_sweep(128))
        b._on_lidar(_Sample("strands/g1/lidar/cloud", first))
        b._on_lidar(_Sample("strands/g1/lidar/cloud", second))
        assert _served(b)["n"] == 128
        assert b.latest_cloud("nobody") is None

    @pytest.mark.parametrize(
        "encoded",
        [
            pytest.param("not base64!!", id="not-base64"),
            pytest.param(base64.b64encode(b"\x00" * 20).decode(), id="not-whole-points"),
            # Garbage injected into an OTHERWISE VALID encoding. This one matters because the
            # length check cannot see it: lenient base64 silently discards the non-alphabet
            # characters, so it decodes to exactly 16 bytes -- one whole point, assembled from
            # characters that shifted position. Only validate=True refuses it.
            pytest.param(
                base64.b64encode(b"\x01" * 16).decode()[:4] + "!!" + base64.b64encode(b"\x01" * 16).decode()[4:],
                id="corrupt-but-a-whole-point-long",
            ),
        ],
    )
    def test_a_payload_that_is_not_a_cloud_is_refused(self, bridge, encoded: str) -> None:
        b, emitted = bridge
        b._on_lidar(_Sample("strands/g1/lidar/cloud", {"peer_id": "g1", "t": 1.0, "data": encoded}))
        assert b.latest_cloud("g1") is None, "a buffer that cannot be a cloud is not kept as one"
        assert emitted == [], "and nothing announces it"


class TestTheRoundTripIsExact:
    def test_the_points_a_viewer_receives_are_the_points_the_sensor_reported(self, bridge) -> None:
        b, _ = bridge
        sweep = _sweep(24000)
        payload = _published(sweep)
        b._on_lidar(_Sample("strands/g1/lidar/cloud", payload))

        # What /ws/lidar sends is exactly these bytes.
        served = _served(b)["xyzi"]
        received = np.frombuffer(served, dtype="<f4").reshape(-1, 4)
        expected = sweep[:: payload["stride"]][:LIDAR_CLOUD_MAX_POINTS]

        assert np.array_equal(received, expected), (
            "base64 over a JSON envelope is lossless; a mismatch here means a coordinate was "
            "reinterpreted somewhere between the sensor and the socket"
        )

    def test_the_wire_is_smaller_than_the_per_point_json_it_replaces(self) -> None:
        sweep = _sweep(4000)
        payload = _published(sweep)
        compact = len(json.dumps(payload))
        per_point = len(
            json.dumps(
                {
                    "peer_id": "g1",
                    "t": 1.0,
                    "points": [[round(float(v), 4) for v in row] for row in sweep],
                }
            )
        )
        assert compact < per_point, f"compact {compact} is not smaller than per-point {per_point}"
        assert per_point / compact > 1.5, (
            "the encoding is chosen for this margin; if it narrows, the reason for base64 over a "
            f"readable spelling has weakened (measured {per_point / compact:.2f}x)"
        )
