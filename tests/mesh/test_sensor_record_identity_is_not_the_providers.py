"""A sensor record's identity keys are the publisher's, not the provider's.

Every ``SensorLoopsMixin._read_*`` reader seeds a record with the keys this
process decided, merges the robot's provider mapping over it, and publishes the
result to a topic it builds from those same keys. Merged last, a provider
mapping carrying one of the seeded names replaced the local reading, so a record
published to ``strands/{peer_id}/...`` could name a different peer inside it and
a hand record published under one hand's name could name another.

The precedence is not a new rule. ``docs/mesh.md`` already states it for the
presence payload -- the locally decided keys "win a name collision" with what a
peer reports, because "the ``peer_id`` a peer is filed under is the one its topic
and certificate bind - not a field inside the payload" -- and
:meth:`strands_robots.mesh.session.PeerInfo.to_dict` implements it by spreading
the peer's own payload first. These cells hold the sensor readers to the same
precedence, and pin the two boundaries deliberately left alone: ``t`` is a stamp
rather than a locally computed duration, and the ``source``/``frame`` labels are
seeded with :meth:`dict.setdefault` precisely so a provider may name its own.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
import time
from typing import Any

import pytest

from strands_robots.mesh.sensors import SensorLoopsMixin
from strands_robots.mesh.session import PeerInfo

LOCAL_PEER = "alice"
OTHER_PEER = "bob"


class _Host(SensorLoopsMixin):
    """Minimal SensorLoopsMixin host that records what it published."""

    def __init__(self, robot: Any, peer_id: str = LOCAL_PEER) -> None:
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


# Each row: (id, provider attribute, the reader, a reading the provider supplies).
# The provider mapping additionally claims ``peer_id`` in every row.
_PROVIDER_ROWS: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
    ("pose-provider", "_pose", "_read_pose", {"x": 1.5}),
    ("pose-slam", "_slam_pose", "_read_pose", {"x": 2.5}),
    ("pose-odom", "_odom_pose", "_read_pose", {"x": 3.5}),
    ("imu", "_imu", "_read_imu", {"rpy": [0.1, 0.2, 0.3]}),
    ("odom", "_odom", "_read_odom", {"vx": 0.25}),
    ("lidar-summary", "_lidar_summary", "_read_lidar_summary", {"count": 1234}),
    ("lidar-state", "_lidar_state", "_read_lidar_state", {"freq": 10.0}),
    ("map-info", "_map_info", "_read_map_info", {"width": 64}),
)
_ROW_IDS = tuple(row[0] for row in _PROVIDER_ROWS)


def _read_with_hostile_provider(attr: str, reader: str, reading: dict[str, Any]) -> dict[str, Any]:
    """Drive one reader with a provider mapping that also claims ``peer_id``."""
    host = _host(**{attr: {**reading, "peer_id": OTHER_PEER}})
    record = getattr(host, reader)()
    assert record is not None, f"{reader} declined to read {attr}"
    return record


def _merging_readers() -> dict[str, tuple[str, ...]]:
    """Readers that merge a provider mapping wholesale -> the vars they merge into.

    Derived from the shipped class rather than listed, so a reader added later is
    held to the same precedence instead of inheriting an exemption by omission.
    A wholesale merge is ``<record>.update(<name>)``: the whole provider mapping,
    under whatever names it carries. ``_read_health`` is deliberately absent --
    it lifts named fields (``battery.get("pct")``) and namespaces the rest under
    its own key, so no provider name reaches the top level of its record.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(SensorLoopsMixin)))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    found: dict[str, tuple[str, ...]] = {}
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("_read_"):
            continue
        merged = [
            ast.unparse(node.func.value)
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
        ]
        if merged:
            found[fn.name] = tuple(sorted(set(merged)))
    return found


def _stamp_calls(method_name: str) -> int:
    """How many times *method_name* re-asserts the locally decided keys."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(SensorLoopsMixin, method_name))))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_stamp_local_keys"
    )


# The precedence this file holds the readers to ------------------------------


class TestThePresencePathAlreadyResolvesThisCollision:
    """The rule is the presence path's, measured rather than quoted."""

    def test_a_presence_payload_cannot_rename_the_peer_it_is_filed_under(self) -> None:
        peer = PeerInfo(peer_id=LOCAL_PEER, caps={"peer_id": OTHER_PEER, "tool_name": "arm"})
        record = peer.to_dict()
        assert record["peer_id"] == LOCAL_PEER
        # The peer's own capabilities still come through; only the collision moves.
        assert record["tool_name"] == "arm"


# Regression -----------------------------------------------------------------


class TestAProviderCannotRenameThePeerAReadingCameFrom:
    """Nine merge sites, each publishing to a topic built from ``self.peer_id``."""

    @pytest.mark.parametrize(("_id", "attr", "reader", "reading"), _PROVIDER_ROWS, ids=_ROW_IDS)
    def test_the_record_names_the_publishing_peer(
        self, _id: str, attr: str, reader: str, reading: dict[str, Any]
    ) -> None:
        record = _read_with_hostile_provider(attr, reader, reading)
        assert record["peer_id"] == LOCAL_PEER

    def test_a_hand_record_names_the_hand_it_is_published_under(self) -> None:
        host = _host(_hands={"left": {"pos": 1.0, "peer_id": OTHER_PEER, "hand": "RIGHT"}})
        hands = host._read_hands()
        assert hands is not None
        record = hands["left"]
        assert record["peer_id"] == LOCAL_PEER
        assert record["hand"] == "left"
        assert record["pos"] == 1.0

    def test_the_published_topic_and_the_record_agree_on_the_hand(self) -> None:
        # The loop publishes to strands/{peer}/hand/{name}/state, so a record
        # naming another hand contradicts the key it arrived under.
        host = _host(_hands={"left": {"peer_id": OTHER_PEER, "hand": "RIGHT"}})
        hands = host._read_hands()
        assert hands is not None
        for name, record in hands.items():
            topic = f"strands/{host.peer_id}/hand/{name}/state"
            assert topic == f"strands/{record['peer_id']}/hand/{record['hand']}/state"


class TestReAssertingTheIdentityDoesNotCostTheReading:
    """Over-reach guard: the merge must still deliver what it was read for."""

    @pytest.mark.parametrize(("_id", "attr", "reader", "reading"), _PROVIDER_ROWS, ids=_ROW_IDS)
    def test_the_providers_own_reading_still_reaches_the_record(
        self, _id: str, attr: str, reader: str, reading: dict[str, Any]
    ) -> None:
        record = _read_with_hostile_provider(attr, reader, reading)
        for key, value in reading.items():
            assert record[key] == value


# Structure: the rule is derived, so a reader added later inherits it ---------


class TestEveryWholesaleMergeReAssertsTheLocalKeys:
    """A reader that merges a provider mapping must re-assert what it decided."""

    def test_every_merging_reader_re_asserts_them_once_per_merge(self) -> None:
        for name, merged_into in sorted(_merging_readers().items()):
            assert _stamp_calls(name) == len(
                [
                    node
                    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(getattr(SensorLoopsMixin, name)))))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                ]
            ), f"{name} merges into {merged_into} more often than it re-asserts the local keys"

    def test_the_derivation_finds_the_readers_that_merge(self) -> None:
        # Non-vacuity: the rule above is empty unless the scan finds the mergers.
        assert set(_merging_readers()) == {
            "_read_pose",
            "_read_imu",
            "_read_odom",
            "_read_lidar_summary",
            "_read_lidar_state",
            "_read_hands",
            "_read_map_info",
        }

    def test_the_health_reader_is_not_a_wholesale_merge(self) -> None:
        # Its exemption is structural, not a listed exception: it lifts named
        # fields, so no provider key reaches the top level of its record.
        assert "_read_health" not in _merging_readers()


# Boundaries deliberately left alone (these hold before the change too) ------


class TestAStampStaysTheReadingsOwn:
    """``t`` is not re-asserted: a decode-time stamp is truer than publish time."""

    @pytest.mark.parametrize(("_id", "attr", "reader", "reading"), _PROVIDER_ROWS, ids=_ROW_IDS)
    def test_a_provider_may_stamp_its_own_reading(
        self, _id: str, attr: str, reader: str, reading: dict[str, Any]
    ) -> None:
        stamped = time.time() - 30.0
        host = _host(**{attr: {**reading, "t": stamped}})
        record = getattr(host, reader)()
        assert record is not None
        assert record["t"] == stamped

    def test_a_reader_stamps_the_record_when_the_provider_does_not(self) -> None:
        before = time.time()
        record = _host(_lidar_summary={"count": 7})._read_lidar_summary()
        assert record is not None
        assert before <= record["t"] <= time.time()


class TestAProviderStillNamesItsOwnFrameAndSource:
    """``source``/``frame`` are seeded with setdefault, so the provider wins."""

    def test_a_pose_provider_names_its_own_frame_and_source(self) -> None:
        record = _host(_pose={"x": 1.0, "frame": "base_link", "source": "vio"})._read_pose()
        assert record is not None
        assert (record["frame"], record["source"]) == ("base_link", "vio")

    def test_an_odom_provider_names_its_own_frame(self) -> None:
        record = _host(_odom={"vx": 1.0, "frame": "wheel_odom"})._read_odom()
        assert record is not None
        assert record["frame"] == "wheel_odom"

    def test_the_seeded_labels_are_still_applied_when_absent(self) -> None:
        record = _host(_slam_pose={"x": 1.0})._read_pose()
        assert record is not None
        assert (record["frame"], record["source"]) == ("map", "slam")


class TestTheHealthReaderIsUnchanged:
    """It never merged wholesale, so nothing about it moves."""

    def test_a_battery_mapping_cannot_rename_the_peer_or_restamp_the_record(self) -> None:
        before = time.time()
        record = _host(_battery={"pct": 55.0, "peer_id": OTHER_PEER, "t": 0.0})._read_health()
        assert record is not None
        assert record["peer_id"] == LOCAL_PEER
        assert before <= record["t"] <= time.time()
        assert record["battery_pct"] == 55.0

    def test_a_temps_mapping_stays_under_its_own_key(self) -> None:
        record = _host(_temps={"cpu": 41.0, "peer_id": OTHER_PEER})._read_health()
        assert record is not None
        assert record["peer_id"] == LOCAL_PEER
        assert record["temps"] == {"cpu": 41.0, "peer_id": OTHER_PEER}


class TestTheInnerRobotObservationPathIsUnaffected:
    """The IMU fallback builds its own fields rather than merging a mapping."""

    def test_an_inner_robot_imu_reading_still_names_the_publishing_peer(self) -> None:
        class _Inner:
            is_connected = True

            def get_observation(self) -> dict[str, Any]:
                return {"imu_rpy": [0.1, 0.2, 0.3], "gyroscope": [1.0, 2.0, 3.0]}

        record = _host(robot=_Inner())._read_imu()
        assert record is not None
        assert record["peer_id"] == LOCAL_PEER
        assert record["rpy"] == [0.1, 0.2, 0.3]
        assert record["gyro"] == [1.0, 2.0, 3.0]
