"""The dashboard files every mesh sample under the peer that published it, not the peer the body names.

The bridge subscribes to ``strands/*/presence``, ``strands/*/state``, ``strands/*/camera/**``
and the sensor topics with one wildcard each, so every peer's samples land in one callback.
Until this change each callback took the peer identity out of the JSON body (``robot_id`` or
``peer_id``) and stored the whole document under it. The ``*`` segment of the key expression is
the one part of a sample that mTLS and the ACL bind to the publisher, so a peer allowed to
publish only on its own topics could still put another robot's name in the body and rewrite that
robot's presence, joints or camera tile. Presence is the record ``peer_is_physical`` reads, so a
body saying ``robot_type: sim`` for a real arm was enough to skip the motion consent gate.

Now the key expression is the authority: a body that names a different peer is dropped and
logged, a sample with no usable key is dropped (fail closed), ``hw`` stays sticky for the life
of a peer record, presence carries the same freshness check the SDK's own ``_on_presence``
applies, and telemetry for a peer that never announced itself does not mint a fleet entry.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import agent_motion
from strands_robots.dashboard.mesh_bridge import MeshBridge, wire_peer_id

VICTIM = "arm-real-1"
ATTACKER = "evil-1"


def _sample(key: str | None, payload: Any) -> Any:
    """A zenoh sample: *payload* as JSON on key expression *key* (``None`` = no key at all)."""
    sample = mock.MagicMock(spec=["payload", "key_expr"] if key is not None else ["payload"])
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    sample.payload.to_bytes.return_value = body
    if key is not None:
        sample.key_expr = key
    return sample


def _presence(robot_id: str, **fields: Any) -> dict[str, Any]:
    return {"robot_id": robot_id, "robot_type": "robot", "timestamp": time.time(), **fields}


@pytest.fixture
def bridge() -> MeshBridge:
    b = MeshBridge(peer_id="dash")
    b._running = True
    return b


# --- the helper -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("strands/arm-real-1/presence", "arm-real-1"),
        ("strands/so101-sim-7487__so101/state", "so101-sim-7487__so101"),
        ("strands/g1.left/camera/front", "g1.left"),
        ("strands/g1/lidar/state", "g1"),
        ("strands/safety/estop", None),
        ("strands/broadcast", None),
        ("strands//state", None),
        ("other/arm/presence", None),
        ("strands/*/presence", None),
        ("strands/arm/", None),
        ("", None),
    ],
)
def test_the_wire_peer_is_the_second_segment_of_a_peer_topic(key: str, expected: str | None) -> None:
    assert wire_peer_id(_sample(key, {})) == expected


def test_a_sample_without_a_key_expression_has_no_wire_peer() -> None:
    assert wire_peer_id(_sample(None, {})) is None
    assert wire_peer_id(mock.MagicMock()) is None  # a Mock repr is not a key


# --- presence (f001) --------------------------------------------------------------------------


def test_presence_is_filed_under_the_publishing_peer(bridge: MeshBridge) -> None:
    bridge._on_presence(_sample(f"strands/{VICTIM}/presence", _presence(VICTIM, hw="so101_follower")))
    assert VICTIM in bridge.peers
    assert bridge.peers[VICTIM]["presence"]["hw"] == "so101_follower"


def test_a_presence_body_naming_another_peer_is_dropped_and_logged(bridge: MeshBridge) -> None:
    bridge._on_presence(_sample(f"strands/{VICTIM}/presence", _presence(VICTIM, hw="so101_follower")))
    forged = _presence(VICTIM, robot_type="sim")
    bridge._on_presence(_sample(f"strands/{ATTACKER}/presence", forged))
    assert bridge.peers[VICTIM]["presence"]["hw"] == "so101_follower"
    assert bridge.peers[VICTIM]["presence"]["robot_type"] == "robot"
    assert ATTACKER not in bridge.peers, "the forged body must not mint a peer under either name"
    events = [e for e in bridge.activity_log() if e["action"] == "identity_mismatch"]
    assert events, "an impersonation attempt must show up in the activity trail"
    assert events[0]["target"] == ATTACKER
    assert events[0]["detail"]["claimed"] == VICTIM
    assert events[0]["detail"]["topic"] == "presence"


def test_presence_without_a_usable_key_is_dropped_not_trusted(bridge: MeshBridge) -> None:
    bridge._on_presence(_sample(None, _presence(VICTIM, robot_type="sim")))
    bridge._on_presence(_sample("strands/safety/estop", _presence(VICTIM, robot_type="sim")))
    assert bridge.peers == {}


def test_hardware_evidence_is_sticky_across_presence_updates(bridge: MeshBridge) -> None:
    key = f"strands/{VICTIM}/presence"
    bridge._on_presence(_sample(key, _presence(VICTIM, hw="so101_follower")))
    bridge._on_presence(_sample(key, _presence(VICTIM, robot_type="sim")))
    presence = bridge.peers[VICTIM]["presence"]
    assert presence["hw"] == "so101_follower"
    assert presence["robot_type"] == "sim", "other fields still follow the latest heartbeat"
    physical, why = agent_motion.peer_is_physical(bridge.peers[VICTIM])
    assert physical is True
    assert "so101_follower" in why


@pytest.mark.parametrize("stamp", [None, "now", float("nan"), float("inf"), True])
def test_presence_needs_a_finite_numeric_timestamp(bridge: MeshBridge, stamp: Any) -> None:
    body = _presence(VICTIM)
    body["timestamp"] = stamp
    if stamp is None:
        del body["timestamp"]
    bridge._on_presence(_sample(f"strands/{VICTIM}/presence", body))
    assert bridge.peers == {}


def test_a_replayed_or_future_presence_is_dropped(bridge: MeshBridge, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "30")
    monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", "5")
    key = f"strands/{VICTIM}/presence"
    bridge._on_presence(_sample(key, _presence(VICTIM, timestamp=time.time() - 300)))
    bridge._on_presence(_sample(key, _presence(VICTIM, timestamp=time.time() + 60)))
    assert bridge.peers == {}
    bridge._on_presence(_sample(key, _presence(VICTIM)))
    assert VICTIM in bridge.peers


def test_the_dashboards_own_presence_is_still_ignored(bridge: MeshBridge) -> None:
    bridge._on_presence(_sample("strands/dash/presence", _presence("dash")))
    assert bridge.peers == {}


# --- the motion gate corroborates a sim claim (f001) ------------------------------------------


def test_a_bare_sim_claim_from_the_wire_gates_as_physical(bridge: MeshBridge) -> None:
    bridge._on_presence(_sample(f"strands/{VICTIM}/presence", _presence(VICTIM, robot_type="sim")))
    physical, why = agent_motion.peer_is_physical(bridge.peers[VICTIM])
    assert physical is True
    assert "did not launch" in why


def test_a_sim_this_dashboard_launched_is_a_sim(bridge: MeshBridge) -> None:
    host = "so101-sim-7487"
    bridge.managed_children = lambda: [{"peer_id": host, "mode": "sim", "alive": True}]
    for pid in (host, f"{host}__so101"):
        bridge._on_presence(_sample(f"strands/{pid}/presence", _presence(pid, robot_type="sim")))
        physical, why = agent_motion.peer_is_physical(bridge.peers[pid])
        assert physical is False, (pid, why)
        assert "launched" in why


def test_a_managed_real_child_that_claims_sim_is_still_physical(bridge: MeshBridge) -> None:
    bridge.managed_children = lambda: [{"peer_id": VICTIM, "mode": "real", "alive": True}]
    bridge._on_presence(_sample(f"strands/{VICTIM}/presence", _presence(VICTIM, robot_type="sim")))
    assert agent_motion.peer_is_physical(bridge.peers[VICTIM])[0] is True


def test_a_sim_this_dashboard_launched_that_reports_hardware_is_physical(bridge: MeshBridge) -> None:
    bridge.managed_children = lambda: [{"peer_id": VICTIM, "mode": "sim", "alive": True}]
    bridge._on_presence(_sample(f"strands/{VICTIM}/presence", _presence(VICTIM, robot_type="sim", hw="so101")))
    assert agent_motion.peer_is_physical(bridge.peers[VICTIM])[0] is True


def test_a_peer_dict_that_never_met_the_bridge_reads_as_before() -> None:
    """Callers that build peer dicts themselves (tests, the HITL hook's snapshot) keep the old contract."""
    assert agent_motion.peer_is_physical({"presence": {"robot_type": "sim"}})[0] is False
    assert agent_motion.peer_is_physical({"presence": {"hw": "so101"}})[0] is True
    assert agent_motion.peer_is_physical(None)[0] is True


# --- telemetry (f008) --------------------------------------------------------------------------


def _announce(bridge: MeshBridge, *peers: str) -> None:
    for pid in peers:
        bridge._on_presence(_sample(f"strands/{pid}/presence", _presence(pid, hw="so101_follower")))


def test_state_from_another_peer_cannot_overwrite_the_victims_joints(bridge: MeshBridge) -> None:
    _announce(bridge, VICTIM, ATTACKER)
    bridge._on_state(_sample(f"strands/{VICTIM}/state", {"peer_id": VICTIM, "joints": {"j1": 0.1}}))
    bridge._on_state(_sample(f"strands/{ATTACKER}/state", {"peer_id": VICTIM, "joints": {"j1": 9.9}}))
    assert bridge.peers[VICTIM]["state"]["joints"] == {"j1": 0.1}
    assert "state" not in bridge.peers[ATTACKER]


def test_stream_from_another_peer_is_dropped(bridge: MeshBridge) -> None:
    _announce(bridge, VICTIM, ATTACKER)
    bridge._on_stream(_sample(f"strands/{ATTACKER}/stream", {"peer_id": VICTIM, "positions": [1, 2]}))
    assert "stream" not in bridge.peers[VICTIM]


def test_a_camera_frame_from_another_peer_cannot_replace_the_victims_tile(bridge: MeshBridge) -> None:
    import base64

    _announce(bridge, VICTIM, ATTACKER)
    genuine = base64.b64encode(b"\xff\xd8genuine").decode()
    forged = base64.b64encode(b"\xff\xd8forged").decode()
    bridge._on_camera(_sample(f"strands/{VICTIM}/camera/front", {"peer_id": VICTIM, "cam": "front", "data": genuine}))
    bridge._on_camera(_sample(f"strands/{ATTACKER}/camera/front", {"peer_id": VICTIM, "cam": "front", "data": forged}))
    frame = bridge.latest_frame(VICTIM, "front")
    assert frame is not None and frame["jpeg"] == b"\xff\xd8genuine"
    assert bridge.latest_frame(ATTACKER, "front") is None
    assert "cameras" not in bridge.peers[ATTACKER]


def test_a_camera_frame_without_a_body_peer_id_is_filed_under_the_wire_peer(bridge: MeshBridge) -> None:
    import base64

    _announce(bridge, VICTIM)
    data = base64.b64encode(b"\xff\xd8ok").decode()
    bridge._on_camera(_sample(f"strands/{VICTIM}/camera/wrist", {"cam": "wrist", "data": data}))
    assert bridge.latest_frame(VICTIM, "wrist") is not None
    assert "wrist" in bridge.peers[VICTIM]["cameras"]


@pytest.mark.parametrize("topic", ["pose", "health", "imu", "odom"])
def test_sensor_samples_from_another_peer_are_dropped(bridge: MeshBridge, topic: str) -> None:
    _announce(bridge, VICTIM, ATTACKER)
    handler = getattr(bridge, f"_on_{topic}")
    handler(_sample(f"strands/{ATTACKER}/{topic}", {"peer_id": VICTIM, "value": 1}))
    assert topic not in bridge.peers[VICTIM]
    handler(_sample(f"strands/{VICTIM}/{topic}", {"peer_id": VICTIM, "value": 2}))
    assert bridge.peers[VICTIM][topic]["value"] == 2


def test_lidar_from_another_peer_is_dropped(bridge: MeshBridge) -> None:
    _announce(bridge, VICTIM, ATTACKER)
    bridge._on_lidar(_sample(f"strands/{ATTACKER}/lidar/state", {"peer_id": VICTIM, "rate": 10}))
    assert "lidar" not in bridge.peers[VICTIM]
    bridge._on_lidar(_sample(f"strands/{VICTIM}/lidar/summary", {"peer_id": VICTIM, "points": 3}))
    assert bridge.peers[VICTIM]["lidar"]["summary"]["points"] == 3


def test_telemetry_alone_does_not_mint_a_peer(bridge: MeshBridge) -> None:
    ghost = "ghost-9"
    bridge._on_state(_sample(f"strands/{ghost}/state", {"peer_id": ghost, "joints": {"j1": 0.0}}))
    bridge._on_pose(_sample(f"strands/{ghost}/pose", {"peer_id": ghost}))
    assert ghost not in bridge.peers
    dropped = [e for e in bridge.activity_log() if e["action"] == "unannounced_peer"]
    assert dropped and dropped[0]["target"] == ghost


def test_every_wildcard_handler_reads_the_key_expression() -> None:
    """The guard against the next inline ``data.get("peer_id")`` identity read."""
    import ast
    import inspect

    import strands_robots.dashboard.mesh_bridge as mod

    tree = ast.parse(inspect.getsource(mod))
    handlers = {"_on_presence", "_on_state", "_on_stream", "_on_camera", "_sensor_sample", "_on_lidar"}
    seen: dict[str, bool] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in handlers:
            calls = {
                getattr(n.func, "attr", getattr(n.func, "id", "")) for n in ast.walk(node) if isinstance(n, ast.Call)
            }
            seen[node.name] = "_attributed" in calls
    assert seen == dict.fromkeys(handlers, True), seen
