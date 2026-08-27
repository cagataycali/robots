"""A sensor payload built from real readings has to survive the JSON encoder.

Every extended sensor topic is published as JSON:
:func:`strands_robots.mesh.session._put_zenoh_directly` encodes the payload
before it reaches the wire. A payload the encoder refuses is not a transient
failure that the next tick retries - it fails identically forever, so the topic
never publishes at all, which is why
:func:`strands_robots.mesh.session._report_unencodable_payload` reports it at
ERROR rather than absorbing it.

The values a sensor stack actually reports are numpy: a lidar summary's bounding
box is whatever ``ndarray.min(axis=0)`` returned, an IMU's orientation a
``float32``, a device state code an ``int64``. None of those are JSON values, and
:class:`~strands_robots.mesh.sensors.SensorLoopsMixin` merges the robot's
provider dictionaries into the outgoing payload verbatim - so a robot whose
readings are numpy had every tick of those topics dropped.

Two paths in the same class already coerced for exactly this reason
(``_read_imu``'s inner-observation branch calls ``tolist()``, ``_read_pose``'s
SE(3) branch calls ``float()``), and so does
:meth:`~strands_robots.mesh.core.Mesh.publish_step`. These tests hold every
reader to that rule, and pin the boundary: a value that is not a reading is left
untouched so the transport still names it.
"""

from __future__ import annotations

import ast
import json
import pathlib
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.mesh import sensors as mesh_sensors
from strands_robots.mesh.sensors import _JSONABLE_MAX_DEPTH, SensorLoopsMixin, _jsonable


class _Host(SensorLoopsMixin):
    """Minimal SensorLoopsMixin host that records published payloads."""

    def __init__(self, robot: Any, peer_id: str = "neon") -> None:
        self.robot = robot
        self.peer_id = peer_id
        self._running = True
        self._stop_event = threading.Event()
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        self.published.append((key, payload))


class _Robot:
    """Bare attribute bag standing in for a robot exposing sensor providers."""


def _host(**robot_attrs: Any) -> _Host:
    robot = _Robot()
    for name, value in robot_attrs.items():
        setattr(robot, name, value)
    return _Host(robot)


def _encode(payload: Any) -> Any:
    """Round-trip *payload* through the encoder the wire uses."""
    return json.loads(json.dumps(payload))


# One row per provider attribute, so every reader and every branch that merges a
# provider dictionary is driven with the numpy a real sensor stack reports.
# ``expected`` is read out of the decoded payload, which is what a subscriber
# sees - the assertion is on the published record, not on an internal.
_ROWS: tuple[tuple[str, dict[str, Any], str, dict[str, Any]], ...] = (
    (
        "lidar-summary",
        {"_lidar_summary": {"count": np.int64(200), "bbox": np.array([-1.5, -2.5, 0.0]), "range": np.float32(7.25)}},
        "_read_lidar_summary",
        {"count": 200, "bbox": [-1.5, -2.5, 0.0], "range": 7.25},
    ),
    (
        "lidar-state",
        {"_lidar_state": {"freq": np.float32(10.0), "code": np.int64(3)}},
        "_read_lidar_state",
        {"freq": 10.0, "code": 3},
    ),
    (
        "imu",
        {"_imu": {"rpy": np.array([0.25, -0.5, 0.75]), "gyro": np.float32(0.125)}},
        "_read_imu",
        {"rpy": [0.25, -0.5, 0.75], "gyro": 0.125},
    ),
    (
        "odom",
        {"_odom": {"x": np.int64(1), "y": np.float32(-2.0)}},
        "_read_odom",
        {"x": 1, "y": -2.0, "frame": "odom"},
    ),
    (
        "pose-provider",
        {"_pose": {"x": np.float32(3.5), "quat": np.array([1.0, 0.0, 0.0, 0.0])}},
        "_read_pose",
        {"x": 3.5, "quat": [1.0, 0.0, 0.0, 0.0], "source": "provider"},
    ),
    (
        "pose-slam",
        {"_slam_pose": {"x": np.float32(-4.0)}},
        "_read_pose",
        {"x": -4.0, "source": "slam"},
    ),
    (
        "pose-odom",
        {"_odom_pose": {"y": np.float32(6.5)}},
        "_read_pose",
        {"y": 6.5, "source": "odom"},
    ),
    (
        "health-battery",
        {"_battery": {"pct": np.float32(87.5)}},
        "_read_health",
        {"battery_pct": 87.5},
    ),
    (
        "health-temps",
        {"_temps": {"cpu": np.float32(41.5)}},
        "_read_health",
        {"temps": {"cpu": 41.5}},
    ),
    (
        "map-info",
        {"_map_info": {"resolution": np.float32(0.05), "size": np.array([512, 512])}},
        "_read_map_info",
        {"resolution": 0.05, "size": [512, 512]},
    ),
)

_IDS = tuple(row[0] for row in _ROWS)


class TestEveryReadingReachesTheWire:
    """A provider dictionary of numpy readings must encode, with its values intact."""

    @pytest.mark.parametrize(("label", "attrs", "reader", "expected"), _ROWS, ids=_IDS)
    def test_payload_encodes(self, label: str, attrs: dict[str, Any], reader: str, expected: dict[str, Any]) -> None:
        payload = getattr(_host(**attrs), reader)()
        assert payload is not None, f"{reader} read nothing for {label}"
        # The encoder is the gate: this is the call that drops the topic forever.
        decoded = _encode(payload)
        for key, want in expected.items():
            assert decoded[key] == pytest.approx(want), f"{label}: {key} arrived as {decoded[key]!r}"

    def test_a_mixed_width_payload_encodes_and_keeps_both_values(self) -> None:
        """One payload, one width already accepted and one refused.

        The coercion has to repair the refused value without disturbing the one
        that already encoded.
        """
        payload = _host(_lidar_summary={"wide": np.float64(1.5), "narrow": np.float32(2.5)})._read_lidar_summary()
        assert payload is not None
        decoded = _encode(payload)
        assert decoded["wide"] == pytest.approx(1.5)
        assert decoded["narrow"] == pytest.approx(2.5)

    def test_hands_payload_encodes(self) -> None:
        """``_read_hands`` returns a mapping of per-hand records, so it nests."""
        hands = _host(_hands={"left": {"q": np.array([0.1, 0.2]), "force": np.float32(3.5)}})._read_hands()
        assert hands is not None
        decoded = _encode(hands)
        assert decoded["left"]["q"] == pytest.approx([0.1, 0.2])
        assert decoded["left"]["force"] == pytest.approx(3.5)
        assert decoded["left"]["hand"] == "left"

    def test_a_container_of_readings_is_coerced_throughout(self) -> None:
        """A reading can sit inside a list, or be a mapping key.

        A per-beam list and a channel-indexed mapping are both ordinary shapes
        for a lidar summary, and the encoder refuses a numpy value wherever it
        sits: ``json.dumps`` rejects a ``float32`` key with "keys must be str,
        int, float, bool or None" just as it rejects one as a value.
        """
        provider = {
            "per_beam": [np.float32(1.0), np.int64(2)],
            # A key inside a NESTED mapping: the top-level keys are coerced by
            # _coerce_record, so this is what grades the recursive helper's own
            # key handling.
            "nested": {"inner": (np.float32(3.0),), np.int64(3): "beam-3"},
            np.int64(7): "channel-7",
        }
        payload = _host(_lidar_summary=provider)._read_lidar_summary()
        assert payload is not None
        decoded = _encode(payload)
        assert decoded["per_beam"] == pytest.approx([1.0, 2.0])
        assert decoded["nested"]["inner"] == pytest.approx([3.0])
        assert decoded["nested"]["3"] == "beam-3"
        assert decoded["7"] == "channel-7"

    def test_those_container_shapes_are_refused_raw(self) -> None:
        """Premise for the case above: each shape is unencodable before coercion."""
        shapes: tuple[dict[Any, Any], ...] = (
            {"per_beam": [np.float32(1.0)]},
            {"nested": {"inner": (np.float32(3.0),)}},
            {"nested": {np.int64(3): "beam-3"}},
            {np.int64(7): "x"},
        )
        for raw in shapes:
            with pytest.raises(TypeError):
                json.dumps(raw)

    @pytest.mark.parametrize(("label", "attrs", "reader", "expected"), _ROWS, ids=_IDS)
    def test_the_raw_reading_is_what_the_encoder_refuses(
        self, label: str, attrs: dict[str, Any], reader: str, expected: dict[str, Any]
    ) -> None:
        """Premise: each row's provider value is unencodable before any coercion.

        Without this the table above would pass on a payload that never needed
        coercing, and the suite would grade nothing.
        """
        (provider_value,) = attrs.values()
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(provider_value)


class TestAPlainPayloadIsUnchanged:
    """Control: a provider already reporting JSON values keeps today's payload."""

    def test_plain_values_survive_verbatim(self) -> None:
        provider = {"count": 200, "bbox": [-1.5, -2.5, 0.0], "label": "front"}
        payload = _host(_lidar_summary=provider)._read_lidar_summary()
        assert payload is not None
        assert {k: v for k, v in payload.items() if k in provider} == provider

    def test_the_encoder_accepts_one_numpy_width_and_refuses_another(self) -> None:
        """Premise: the defect turned on the numpy WIDTH, which is why it hid.

        ``np.float64`` subclasses Python's ``float`` and ``np.str_`` subclasses
        ``str``, so a payload built from those always encoded; ``np.float32`` and
        ``np.int64`` subclass nothing and were refused. A robot on a float64
        pipeline therefore published fine, and the same code reading a float32
        lidar - what a point cloud cast to ``float32`` yields - dropped every
        tick.
        """
        assert json.dumps({"already-fine": np.float64(1.5)}) == '{"already-fine": 1.5}'
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps({"dropped": np.float32(1.5)})

    def test_absent_providers_still_read_nothing(self) -> None:
        host = _host()
        assert host._read_lidar_summary() is None
        assert host._read_lidar_state() is None
        assert host._read_odom() is None
        assert host._read_pose() is None
        assert host._read_map_info() is None
        assert host._read_hands() is None

    def test_frame_and_source_defaults_are_still_applied(self) -> None:
        odom = _host(_odom={"x": np.float32(1.0)})._read_odom()
        pose = _host(_slam_pose={"x": np.float32(1.0)})._read_pose()
        assert odom is not None and pose is not None
        assert odom["frame"] == "odom"
        assert pose["frame"] == "map"

    def test_the_se3_pose_branch_is_untouched(self) -> None:
        """The SE(3) branch already coerced with ``float()``; it must still decode."""
        mat = np.eye(4)
        mat[0, 3], mat[1, 3], mat[2, 3] = 1.0, 2.0, 3.0
        decoded = _encode(_host(_pose=mat)._read_pose())
        assert (decoded["x"], decoded["y"], decoded["z"]) == pytest.approx((1.0, 2.0, 3.0))
        assert len(decoded["quat"]) == 4


class TestTheCoercionDoesNotReachIntoRobotState:
    """The payload is repaired; the robot's own readings are left as they are."""

    def test_the_robot_s_own_provider_mapping_is_not_edited(self) -> None:
        """Coercion rebuilds nested values; it must not reach into robot state.

        ``_read_health`` stores the provider's ``_temps`` mapping in the payload
        by reference, so coercing the payload in place must replace that entry
        rather than rewrite the mapping the robot is still using.
        """
        temps = {"cpu": np.float32(41.5)}
        host = _host(_temps=temps)
        payload = host._read_health()
        assert payload is not None
        assert _encode(payload)["temps"]["cpu"] == pytest.approx(41.5)
        assert type(temps["cpu"]) is np.float32, "the robot's own reading was rewritten"
        assert temps is not payload["temps"]


class TestAValueThatIsNotAReadingIsLeftForTheTransport:
    """Over-reach boundary: coercion repairs readings, it does not launder payloads."""

    def test_an_opaque_object_is_preserved_by_identity(self) -> None:
        sentinel = object()
        payload = _host(_map_info={"sensor": sentinel})._read_map_info()
        assert payload is not None
        assert payload["sensor"] is sentinel
        # Still unencodable, so the transport reports it by name rather than a
        # substituted value being published as though it were a reading.
        with pytest.raises(TypeError):
            json.dumps(payload)

    def test_a_string_is_not_taken_apart_into_characters(self) -> None:
        payload = _host(_lidar_state={"status": "ok", "code": np.int64(0)})._read_lidar_state()
        assert payload is not None
        assert payload["status"] == "ok"

    def test_bytes_are_not_guessed_at(self) -> None:
        blob = b"\x00\x01"
        payload = _host(_map_info={"blob": blob})._read_map_info()
        assert payload is not None
        assert payload["blob"] is blob

    def test_a_payload_deeper_than_the_bound_is_handed_over_unchanged(self) -> None:
        """Past the depth bound the value goes to the encoder, which reports it.

        A sensor record is shallow, so a deeper structure is a pathological
        payload rather than a reading. Absorbing it here would replace the
        transport's ERROR report with silence from the reader's own handler.
        """
        deep: dict[str, Any] = {"leaf": np.float32(1.0)}
        for _ in range(_JSONABLE_MAX_DEPTH + 2):
            deep = {"next": deep}
        coerced = _jsonable(deep)
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(coerced)


class TestTheCoercionIsSingleSourced:
    """Every reader routes its payload through the one helper, by construction.

    Derived from the module rather than listed, so a reader added later is held
    to the rule the moment it lands instead of inheriting an exemption.
    """

    @staticmethod
    def _reader_defs() -> list[ast.FunctionDef]:
        source = pathlib.Path(mesh_sensors.__file__).read_text(encoding="utf-8")
        return [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_read_")
        ]

    def test_every_payload_return_is_coerced(self) -> None:
        """Each payload is handed to the coercion before the reader returns it."""
        offenders: list[str] = []
        for fn in self._reader_defs():
            coerced: dict[str, int] = {}
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_coerce_record"
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                ):
                    name = node.args[0].id
                    coerced[name] = min(coerced.get(name, node.lineno), node.lineno)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    continue
                returned = node.value.id if isinstance(node.value, ast.Name) else None
                if returned is None and isinstance(node.value, ast.IfExp) and isinstance(node.value.body, ast.Name):
                    returned = node.value.body.id
                rendered = ast.unparse(node.value)
                if returned is None:
                    offenders.append(f"{fn.name} line {node.lineno}: unrecognised payload return {rendered}")
                elif returned not in coerced:
                    offenders.append(f"{fn.name} line {node.lineno}: return {rendered} is never coerced")
                elif coerced[returned] > node.lineno:
                    offenders.append(f"{fn.name} line {node.lineno}: return {rendered} precedes its coercion")
        assert not offenders, "sensor payloads reach the wire uncoerced:\n" + "\n".join(offenders)

    def test_the_scan_reaches_every_reader(self) -> None:
        """Non-vacuity: the rule above is graded over the readers that exist."""
        found = {fn.name for fn in self._reader_defs()}
        assert {
            "_read_pose",
            "_read_health",
            "_read_imu",
            "_read_odom",
            "_read_lidar_summary",
            "_read_lidar_state",
            "_read_hands",
            "_read_map_info",
        } <= found, f"reader discovery found only {sorted(found)}"
