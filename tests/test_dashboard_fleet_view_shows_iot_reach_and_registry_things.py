"""The fleet view says which path a peer's presence took, and lists provisioned IoT Things nobody has heard.

A dashboard on the ``bridge`` backend hears the same fleet on two legs: Zenoh on the
LAN and AWS IoT Core across the network edge. Until this change the fleet grid could
not say which leg carried a card, and a robot provisioned as a Thing but not yet
booted (or gone quiet) was simply absent. Now each peer record remembers the legs
that carried its presence inside the TTL (``legs``), ``snapshot()`` derives a
``reach`` chip (``lan`` / ``iot`` / ``both``) from them, and a read-only registry
view (:mod:`strands_robots.mesh.iot.registry`) feeds grey cards for Things without a
peer. Two rules are pinned here because they are the security posture: the leg comes
from the sample type the transport constructed, never from the body, and a Thing in
the registry is never written into the bridge's peer table.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import routes_mesh
from strands_robots.dashboard.mesh_bridge import PEER_TTL_S, MeshBridge, peer_reach
from strands_robots.mesh.iot import registry
from strands_robots.mesh.transport.base import SAMPLE_LEGS, sample_leg
from strands_robots.mesh.transport.iot_transport import _MqttSample


def _zenoh_sample(key: str, payload: dict[str, Any]) -> Any:
    sample = mock.MagicMock(spec=["payload", "key_expr"])
    sample.payload.to_bytes.return_value = json.dumps(payload).encode()
    sample.key_expr = key
    return sample


def _mqtt_sample(key: str, payload: dict[str, Any]) -> _MqttSample:
    return _MqttSample(key, json.dumps(payload).encode())


def _presence(robot_id: str) -> dict[str, Any]:
    return {"robot_id": robot_id, "robot_type": "robot", "timestamp": time.time()}


@pytest.fixture
def bridge() -> MeshBridge:
    b = MeshBridge(peer_id="dash")
    b._running = True
    return b


# --- which leg carried a sample -----------------------------------------------------------------


def test_the_two_legs_are_lan_and_iot() -> None:
    assert SAMPLE_LEGS == ("lan", "iot")


def test_an_mqtt_sample_reads_as_the_iot_leg() -> None:
    assert sample_leg(_mqtt_sample("strands/arm/presence", {})) == "iot"


def test_a_zenoh_shaped_sample_reads_as_the_lan_leg() -> None:
    assert sample_leg(_zenoh_sample("strands/arm/presence", {})) == "lan"


def test_a_body_claiming_a_leg_does_not_change_the_leg() -> None:
    # The leg is a property of the transport's sample class, not of the payload.
    sample = _zenoh_sample("strands/arm/presence", {"leg": "iot"})
    assert sample_leg(sample) == "lan"


def test_an_object_declaring_an_unknown_leg_reads_as_lan() -> None:
    class Odd:
        leg = "satellite"
        key_expr = "strands/arm/presence"

    assert sample_leg(Odd()) == "lan"


def test_a_leg_set_on_the_instance_is_not_read() -> None:
    # Only the class declares its leg; an attribute smuggled onto one object is ignored.
    sample = _zenoh_sample("strands/arm/presence", {})
    sample.leg = "iot"
    assert sample_leg(sample) == "lan"


# --- reach from the legs -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legs", "expected"),
    [
        ({"lan": 1000.0}, "lan"),
        ({"iot": 1000.0}, "iot"),
        ({"lan": 1000.0, "iot": 999.0}, "both"),
        ({"lan": 1000.0 - PEER_TTL_S - 1, "iot": 1000.0}, "iot"),
        ({"lan": 1000.0 - PEER_TTL_S - 1}, None),
        ({}, None),
        (None, None),
        ({"lan": True}, None),
        ({"lan": "1000"}, None),
        ({"satellite": 1000.0}, None),
    ],
)
def test_reach_is_the_set_of_legs_that_spoke_inside_the_ttl(legs: Any, expected: str | None) -> None:
    assert peer_reach(legs, now=1000.0) == expected


def test_reach_honours_a_caller_supplied_window() -> None:
    assert peer_reach({"lan": 990.0}, now=1000.0, ttl_s=5.0) is None
    assert peer_reach({"lan": 990.0}, now=1000.0, ttl_s=20.0) == "lan"


# --- the bridge records legs and the snapshot carries reach --------------------------------------


def test_presence_over_zenoh_marks_the_peer_lan(bridge: MeshBridge) -> None:
    bridge._on_presence(_zenoh_sample("strands/arm-1/presence", _presence("arm-1")))
    peer = bridge.snapshot()["peers"]["arm-1"]
    assert peer["reach"] == "lan"
    assert set(peer["legs"]) == {"lan"}


def test_presence_over_iot_marks_the_peer_iot(bridge: MeshBridge) -> None:
    bridge._on_presence(_mqtt_sample("strands/dm-so101-01/presence", _presence("dm-so101-01")))
    assert bridge.snapshot()["peers"]["dm-so101-01"]["reach"] == "iot"


def test_presence_on_both_legs_marks_the_peer_both(bridge: MeshBridge) -> None:
    bridge._on_presence(_zenoh_sample("strands/arm-1/presence", _presence("arm-1")))
    bridge._on_presence(_mqtt_sample("strands/arm-1/presence", _presence("arm-1")))
    assert bridge.snapshot()["peers"]["arm-1"]["reach"] == "both"


def test_reach_is_a_second_axis_next_to_origin(bridge: MeshBridge) -> None:
    # ``origin`` keeps saying who started the process; ``reach`` says which path its heartbeat took.
    bridge._on_presence(_mqtt_sample("strands/arm-1/presence", _presence("arm-1")))
    peer = bridge.snapshot()["peers"]["arm-1"]
    assert (peer["origin"], peer["reach"]) == ("external", "iot")


def test_a_leg_that_fell_silent_drops_out_of_reach(bridge: MeshBridge) -> None:
    bridge._on_presence(_zenoh_sample("strands/arm-1/presence", _presence("arm-1")))
    bridge._on_presence(_mqtt_sample("strands/arm-1/presence", _presence("arm-1")))
    with bridge._peers_lock:
        bridge.peers["arm-1"]["legs"]["lan"] = time.time() - PEER_TTL_S - 5
    assert bridge.snapshot()["peers"]["arm-1"]["reach"] == "iot"


def test_a_dropped_presence_records_no_leg(bridge: MeshBridge) -> None:
    # The body names another peer: the sample is dropped and no peer, and no leg, is recorded.
    bridge._on_presence(_mqtt_sample("strands/evil/presence", _presence("victim")))
    assert "victim" not in bridge.peers
    assert "legs" not in bridge.peers.get("evil", {})


# --- the registry view ---------------------------------------------------------------------------


class _Boto3Error(Exception):
    pass


class _ClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


class _NoCredentials(Exception):
    pass


class _NoRegion(Exception):
    pass


class _FakeIot:
    def __init__(
        self, pages: list[dict[str, Any]], *, indexing: str = "OFF", raise_list: Exception | None = None
    ) -> None:
        self._pages = pages
        self._indexing = indexing
        self._raise = raise_list
        self.calls: list[str] = []
        self.meta = mock.Mock(region_name="us-west-2")

    def list_things(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("list_things")
        if self._raise is not None:
            raise self._raise
        token = kwargs.get("nextToken")
        index = int(token) if token else 0
        page = dict(self._pages[index])
        if index + 1 < len(self._pages):
            page["nextToken"] = str(index + 1)
        return page

    def get_indexing_configuration(self) -> dict[str, Any]:
        self.calls.append("get_indexing_configuration")
        return {"thingIndexingConfiguration": {"thingConnectivityIndexingMode": self._indexing}}

    def search_index(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("search_index")
        return {
            "things": [
                {"thingName": "dm-so101-01", "connectivity": {"connected": True, "timestamp": 1_700_000_000_000}},
                {"thingName": "dm-so101-02", "connectivity": {"connected": False, "timestamp": 1_600_000_000_000}},
            ]
        }


def _install_fake_boto3(
    monkeypatch: pytest.MonkeyPatch, iot: _FakeIot | None, *, client_raises: Exception | None = None
) -> None:
    import sys
    import types

    boto3 = types.ModuleType("boto3")

    def client(name: str, region_name: str | None = None) -> Any:
        assert name == "iot"
        if client_raises is not None:
            raise client_raises
        return iot

    setattr(boto3, "client", client)
    botocore = types.ModuleType("botocore")
    exceptions = types.ModuleType("botocore.exceptions")
    setattr(exceptions, "BotoCoreError", _Boto3Error)
    setattr(exceptions, "ClientError", _ClientError)
    setattr(exceptions, "NoCredentialsError", _NoCredentials)
    setattr(exceptions, "NoRegionError", _NoRegion)
    setattr(botocore, "exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions)


def _things(*names: str) -> dict[str, Any]:
    return {
        "things": [
            {"thingName": n, "thingTypeName": "arm", "attributes": {"strands-mesh-role": "robot"}} for n in names
        ]
    }


def test_the_registry_lists_every_thing_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(registry.DISABLE_ENV, raising=False)
    iot = _FakeIot([_things("dm-so101-01", "dm-so101-02")])
    _install_fake_boto3(monkeypatch, iot)
    view = registry.list_things()
    assert view.status == "ok"
    assert [t.thing_name for t in view.things] == ["dm-so101-01", "dm-so101-02"]
    assert view.things[0].attributes == {"strands-mesh-role": "robot"}
    assert view.indexed is False
    assert view.things[0].connectivity is None and view.things[0].last_seen is None
    # Only reads: nothing beyond list_things and the indexing-configuration probe was called.
    assert set(iot.calls) <= {"list_things", "get_indexing_configuration"}


def test_the_registry_pages_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(registry.DISABLE_ENV, raising=False)
    iot = _FakeIot([_things("a", "b"), _things("c", "d"), _things("e")])
    _install_fake_boto3(monkeypatch, iot)
    view = registry.list_things(max_things=3)
    assert [t.thing_name for t in view.things] == ["a", "b", "c"]


def test_a_prefix_keeps_one_naming_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(registry.DISABLE_ENV, raising=False)
    _install_fake_boto3(monkeypatch, _FakeIot([_things("dm-so101-01", "pentest-robot-a")]))
    view = registry.list_things(prefix="dm-")
    assert [t.thing_name for t in view.things] == ["dm-so101-01"]


def test_connectivity_comes_from_the_fleet_index_when_it_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(registry.DISABLE_ENV, raising=False)
    _install_fake_boto3(monkeypatch, _FakeIot([_things("dm-so101-01", "dm-so101-02")], indexing="STATUS"))
    view = registry.list_things()
    assert view.indexed is True
    by_name = {t.thing_name: t for t in view.things}
    assert by_name["dm-so101-01"].connectivity == "connected"
    assert by_name["dm-so101-01"].last_seen == 1_700_000_000.0
    assert by_name["dm-so101-02"].connectivity == "disconnected"


@pytest.mark.parametrize(
    ("client_raises", "list_raises", "expected"),
    [
        (None, _NoCredentials(), "no-credentials"),
        (_NoRegion(), None, "no-credentials"),
        (None, _ClientError("AccessDeniedException"), "denied"),
        (None, _ClientError("ExpiredTokenException"), "denied"),
        (None, _ClientError("ThrottlingException"), "error"),
        (None, _Boto3Error("endpoint"), "error"),
    ],
)
def test_every_failure_is_a_status_word_not_an_exception(
    monkeypatch: pytest.MonkeyPatch, client_raises: Exception | None, list_raises: Exception | None, expected: str
) -> None:
    monkeypatch.delenv(registry.DISABLE_ENV, raising=False)
    _install_fake_boto3(monkeypatch, _FakeIot([_things("a")], raise_list=list_raises), client_raises=client_raises)
    view = registry.list_things()
    assert view.status == expected
    assert view.things == ()
    assert view.detail


def test_the_status_words_are_the_documented_set() -> None:
    assert set(registry.STATUSES) == {"ok", "off", "no-boto3", "no-credentials", "denied", "error"}


def test_boto3_absent_is_a_status_word(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.delenv(registry.DISABLE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "boto3", None)
    view = registry.list_things()
    assert view.status == "no-boto3"
    assert "mesh-iot" in view.detail


def test_the_env_switch_turns_the_read_off_before_boto3_is_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setenv(registry.DISABLE_ENV, "0")
    monkeypatch.setitem(sys.modules, "boto3", None)
    view = registry.list_things()
    assert view.status == "off"
    assert registry.DISABLE_ENV in view.detail


def test_the_view_serialises_for_the_route() -> None:
    thing = registry.RegistryThing(thing_name="a", attributes={"k": "v"})
    view = registry.RegistryView(status="ok", things=(thing,), region="us-west-2")
    out = view.as_dict()
    assert out["count"] == 1 and out["things"][0]["thing_name"] == "a"
    assert set(out) == {"status", "detail", "region", "indexed", "things", "count"}


# --- the route merges the registry with what the bridge heard ------------------------------------


def _cached_view(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> list[str]:
    calls: list[str] = []

    def fake_list_things(*args: Any, **kwargs: Any) -> registry.RegistryView:
        calls.append("list_things")
        return registry.RegistryView(status="ok", things=tuple(registry.RegistryThing(thing_name=n) for n in names))

    monkeypatch.setattr(registry, "list_things", fake_list_things)
    monkeypatch.setattr(
        routes_mesh, "_IOT_REGISTRY_CACHE", type(routes_mesh._IOT_REGISTRY_CACHE)(routes_mesh.IOT_REGISTRY_TTL_S)
    )
    return calls


def test_a_heard_thing_is_marked_live_and_keeps_the_bridge_stamp(
    monkeypatch: pytest.MonkeyPatch, bridge: MeshBridge
) -> None:
    _cached_view(monkeypatch, ["dm-so101-01", "dm-so101-02"])
    bridge._on_presence(_mqtt_sample("strands/dm-so101-01/presence", _presence("dm-so101-01")))
    view = routes_mesh.iot_registry_view(bridge)
    rows = {t["thing_name"]: t for t in view["things"]}
    assert rows["dm-so101-01"]["peer_live"] is True
    assert rows["dm-so101-01"]["heard_by_bridge"] is True
    assert rows["dm-so101-01"]["last_seen"] == pytest.approx(time.time(), abs=5)
    assert rows["dm-so101-02"]["peer_live"] is False
    assert rows["dm-so101-02"]["heard_by_bridge"] is False
    assert rows["dm-so101-02"]["last_seen"] is None


def test_the_registry_never_writes_into_the_peer_table(monkeypatch: pytest.MonkeyPatch, bridge: MeshBridge) -> None:
    _cached_view(monkeypatch, ["dm-so101-02"])
    routes_mesh.iot_registry_view(bridge)
    assert bridge.peers == {}
    assert bridge.snapshot()["peers"] == {}


def test_the_aws_read_is_cached_across_polls(monkeypatch: pytest.MonkeyPatch, bridge: MeshBridge) -> None:
    calls = _cached_view(monkeypatch, ["dm-so101-02"])
    routes_mesh.iot_registry_view(bridge)
    routes_mesh.iot_registry_view(bridge)
    routes_mesh.iot_registry_view(bridge)
    assert calls == ["list_things"]


def test_the_cache_ttl_is_thirty_seconds() -> None:
    assert routes_mesh.IOT_REGISTRY_TTL_S == 30.0
