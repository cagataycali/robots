"""Device shadows (MAPPING 1.1–1.5) on the in-process broker: manifest once, state bounded, desired → confirm_act."""

from __future__ import annotations

import json
import time
import uuid

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh.iot import ShadowMirror, shadow_topic
from strands_robots.mesh.record import record
from strands_robots.mesh.serve import ARGS
from strands_robots.mesh.transports.iot_link import FakeIotBroker, FakeLink
from strands_robots.robot.safety import RequestRefused


@pytest.fixture
def lab():
    broker = FakeIotBroker.named(f"sh-{uuid.uuid4().hex[:8]}")
    arm = Robot("so101", sim=Simulation())
    rec = record(arm, "arm", {"role": "arm"}, "lab", tuple(ARGS))
    device, operator = FakeLink(broker, "arm"), FakeLink(broker, "op")
    seen: dict[str, list[dict]] = {}

    def keep(t, raw):
        seen.setdefault(t, []).append(json.loads(raw) if raw else {})

    operator.subscribe("$aws/things/arm/shadow/name/+/update", keep)
    operator.subscribe("$aws/things/arm/shadow/name/+/delete", keep)
    yield arm, rec, device, operator, seen
    device.close()
    operator.close()
    arm.sim.close()


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while not pred() and time.monotonic() < end:
        time.sleep(0.01)
    assert pred()


def test_manifest_shadow_is_the_record_once_and_is_deleted_at_stop(lab):
    arm, rec, device, _op, seen = lab
    mirror = ShadowMirror(device, arm, rec, shadow_hz=1.0).start()
    topic = shadow_topic("arm", "manifest", "update")
    _wait(lambda: topic in seen)
    assert seen[topic][0]["state"]["reported"] == json.loads(json.dumps(rec))  # tuples → lists
    assert len(json.dumps(seen[topic][0])) <= 8192
    mirror.stop()
    _wait(lambda: shadow_topic("arm", "manifest", "delete") in seen)
    assert mirror.updates["manifest"] == 1


def test_state_is_reported_at_shadow_hz_not_at_the_stream_rate(lab):
    arm, rec, device, _op, seen = lab
    mirror = ShadowMirror(device, arm, rec, shadow_hz=1.0, lockout=lambda: {"engaged": False}).start()
    time.sleep(2.2)
    mirror.stop()
    n = mirror.updates["state"]
    assert 2 <= n <= 3, n  # 1 Hz for 2.2 s, while the manifest's state stream runs at 10 Hz
    doc = seen[shadow_topic("arm", "state", "update")][-1]["state"]["reported"]
    assert "shoulder_pan.pos" in doc and doc["lockout"] == {"engaged": False}


def test_shadow_hz_above_one_is_refused():
    with pytest.raises(ValueError, match="20 req/s"):
        ShadowMirror(None, None, {"id": "x", "manifest": {}}, shadow_hz=30)


def test_desired_goes_through_confirm_act_and_a_refusal_travels_verbatim(lab):
    arm, rec, device, op, seen = lab
    mirror = ShadowMirror(device, arm, rec, shadow_hz=1.0, stale_s=None).start()
    before = float(arm.observe(images=False).joints.positions[0])
    delta = {"state": {"shoulder_pan.pos": 0.3}, "metadata": {}, "version": 7, "timestamp": int(time.time())}
    op.publish(shadow_topic("arm", "control", "update/delta"), json.dumps(delta).encode())
    _wait(lambda: mirror.applied)
    assert mirror.applied == [{"shoulder_pan": 0.3}]
    assert float(arm.observe(images=False).joints.positions[0]) != before
    _wait(lambda: shadow_topic("arm", "control", "update") in seen)
    rep = seen[shadow_topic("arm", "control", "update")][-1]["state"]
    assert rep["reported"]["shoulder_pan.pos"] == 0.3 and rep["reported"]["refused"] is None

    with pytest.raises(RequestRefused) as exc:
        arm.confirm_act({"elbow_flex": 3.1})
    out = mirror.apply({"elbow_flex.pos": 3.1})
    assert out["refused"]["elbow_flex.pos"] == str(exc.value)  # the safety chain's own message
    control = seen[shadow_topic("arm", "control", "update")]
    _wait(lambda: len(control) >= 2)
    rep = control[-1]["state"]
    assert rep["desired"] == {"elbow_flex.pos": None}  # the bad target is cleared from desired
    mirror.stop()


def test_estop_as_a_shadow_field_is_refused_and_unknown_controls_too(lab):
    arm, rec, device, _op, _seen = lab
    mirror = ShadowMirror(device, arm, rec, shadow_hz=1.0, stale_s=None).start()
    out = mirror.apply({"estop": True, "resume": "me", "nonsuch.pos": 1.0, "wrist_roll.pos": "fast"})
    assert out["refused"]["estop"] == "estop is an rpc, not a shadow field"
    assert out["refused"]["resume"] == "resume is an rpc, not a shadow field"
    assert "no control 'nonsuch.pos'" in out["refused"]["nonsuch.pos"]
    assert "is not a number" in out["refused"]["wrist_roll.pos"]
    assert mirror.applied == [] and not arm.estopped
    mirror.stop()


def test_a_stale_delta_is_dropped_not_applied(lab):
    arm, rec, device, _op, _seen = lab
    now = [1_000.0]
    mirror = ShadowMirror(device, arm, rec, shadow_hz=1.0, stale_s=2.0, clock=lambda: now[0]).start()
    stamps = {"desired": {"shoulder_pan.pos": {"timestamp": 990}}}  # 10 s old
    out = mirror.apply({"shoulder_pan.pos": 0.2}, stamps)
    assert "stale_s=2" in out["refused"]["shoulder_pan.pos"] and mirror.applied == []
    stamps = {"desired": {"shoulder_pan.pos": {"timestamp": 999.9}}}  # 100 ms old
    out = mirror.apply({"shoulder_pan.pos": 0.2}, stamps)
    assert out["refused"] is None and mirror.applied == [{"shoulder_pan": 0.2}]
    mirror.stop()


def test_an_aws_integer_second_stamp_is_never_stale_at_the_default_bound(lab):
    """AWS stamps ``metadata.<key>.timestamp`` in whole seconds, so a target written 121–312 ms ago (the measured
    desired→delta latency) carries a stamp up to 1,121–1,312 ms old. At the default bound none of them is stale,
    whatever the sub-second phase; the mesh watchdog (500 ms) plays no part."""
    arm, rec, device, _op, _seen = lab
    now = [1_000_000.0]
    mirror = ShadowMirror(device, arm, rec, shadow_hz=1.0, clock=lambda: now[0]).start()
    assert mirror.stale_s == ShadowMirror(device, arm, rec).stale_s == 2.0
    refused = 0
    for latency in (0.121, 0.312):
        for phase in range(1000):
            now[0] = 1_000_000.0 + phase / 1000.0
            written = now[0] - latency
            stamps = {"desired": {"shoulder_pan.pos": {"timestamp": int(written)}}}  # AWS truncates
            out = mirror.apply({"shoulder_pan.pos": 0.1}, stamps)
            refused += out["refused"] is not None
    assert refused == 0 and len(mirror.applied) == 2000
    # a stamp that is old even by the youngest write it can stand for IS stale
    now[0] = 1_000_010.0
    out = mirror.apply({"shoulder_pan.pos": 0.1}, {"desired": {"shoulder_pan.pos": {"timestamp": 1_000_006}}})
    assert "stale" in out["refused"]["shoulder_pan.pos"]
    out = mirror.apply({"shoulder_pan.pos": 0.1}, {"desired": {"shoulder_pan.pos": {"timestamp": 1_000_008}}})
    assert out["refused"] is None  # 1_000_008 means [8, 9): the youngest write is 1 s old
    mirror.stop()


def test_a_real_shaped_delta_with_an_integer_stamp_is_applied_over_the_broker(lab):
    arm, rec, device, op, _seen = lab
    mirror = ShadowMirror(device, arm, rec, shadow_hz=1.0).start()  # the default stale_s
    now = time.time()
    delta = {
        "state": {"shoulder_pan.pos": 0.25},
        "metadata": {"shoulder_pan.pos": {"timestamp": int(now - 0.3)}},
        "version": 12,
        "timestamp": int(now),
    }
    op.publish(shadow_topic("arm", "control", "update/delta"), json.dumps(delta).encode())
    _wait(lambda: mirror.applied)
    assert mirror.applied == [{"shoulder_pan": 0.25}]
    mirror.stop()


def test_a_stale_bound_under_the_stamp_resolution_is_refused_by_name(lab):
    arm, rec, device, _op, _seen = lab
    with pytest.raises(ValueError, match="stale_s=0.5 is under the 1 s resolution"):
        ShadowMirror(device, arm, rec, stale_s=0.5)
    assert ShadowMirror(device, arm, rec, stale_s=1.0).stale_s == 1.0
    assert ShadowMirror(device, arm, rec, stale_s=None).stale_s is None
