"""``Mesh.mount(robot, iot=IotOptions(...))``: shadows appear on the IoT transport, nothing else changes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh
from strands_robots.mesh.iot import IotOptions
from strands_robots.mesh.transports.iot_link import FakeIotBroker, FakeLink


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while not pred() and time.monotonic() < end:
        time.sleep(0.01)
    assert pred()


def test_mount_with_iot_options_mirrors_shadows_and_a_shadow_target_moves_the_arm_through_confirm_act():
    name = f"m-{uuid.uuid4().hex[:8]}"
    url, tenant = f"iot://fake?broker={name}", "lab"
    broker = FakeIotBroker.named(name)
    dash = FakeLink(broker, "dashboard")
    seen: dict[str, list] = {}
    dash.subscribe(
        "$aws/things/arm/shadow/name/+/update", lambda t, raw: seen.setdefault(t, []).append(json.loads(raw))
    )
    arm = Robot("so101", sim=Simulation())
    device = Mesh(url, tenant=tenant)
    dev = device.mount(arm, device_id="arm", watchdog_ms=None, iot=IotOptions(shadow_hz=1.0))
    assert dev.id == "arm"
    manifest = "$aws/things/arm/shadow/name/manifest/update"
    _wait(lambda: manifest in seen)
    assert seen[manifest][0]["state"]["reported"]["device_id"] == "arm"  # the compact document

    controller = Mesh(url, tenant=tenant)  # the mesh itself is unchanged: discover + invoke still work
    assert controller.discover("arm", min_count=1, timeout=3.0)
    controller.invoke("arm", "act", action={"shoulder_pan": 0.1})

    calls = []
    real = arm.confirm_act
    arm.confirm_act = lambda action, **kw: (calls.append(action), real(action, **kw))[1]
    delta = {"state": {"shoulder_pan.pos": 0.25}, "metadata": {}, "version": 3, "timestamp": int(time.time())}
    dash.publish("$aws/things/arm/shadow/name/control/update/delta", json.dumps(delta).encode())
    _wait(lambda: calls)
    assert calls == [{"shoulder_pan": 0.25}]
    _wait(lambda: "$aws/things/arm/shadow/name/state/update" in seen)
    assert seen["$aws/things/arm/shadow/name/state/update"][-1]["state"]["reported"]["lockout"]["engaged"] is False

    device.unmount("arm")
    time.sleep(0.2)
    n = len(seen.get("$aws/things/arm/shadow/name/state/update", []))
    time.sleep(1.2)
    assert len(seen["$aws/things/arm/shadow/name/state/update"]) == n  # reporting stopped with the device
    controller.close()
    device.close()
    dash.close()
    arm.sim.close()


def test_iot_options_without_an_iot_transport_is_refused_and_a_plain_mount_is_untouched():
    arm = Robot("so101", sim=Simulation())
    mesh = Mesh(f"mem://{uuid.uuid4().hex[:8]}", tenant="lab")
    with pytest.raises(ValueError, match="no transport of this Mesh is iot://"):
        mesh.mount(arm, device_id="arm", watchdog_ms=None, iot={"shadows": True})
    assert mesh.mount(arm, device_id="arm", watchdog_ms=None).id == "arm"
    mesh.close()
    arm.sim.close()


def test_importing_the_mesh_loads_no_aws_sdk():
    code = "import json, sys, strands_robots.mesh.iot; print(json.dumps(list(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert not {"boto3", "botocore", "awsiot", "awscrt", "cryptography"} & {m.split(".")[0] for m in json.loads(out)}


def test_the_state_shadow_reports_a_local_estop_without_waiting_for_an_rpc():
    """``robot.estop()`` from a button or a fault handler latches locally; the ``state`` shadow is the surface an
    operator who is not on the mesh reads, so it must say ``engaged: true`` by itself, as the ``lockout`` rpc does."""
    name = f"m-{uuid.uuid4().hex[:8]}"
    broker = FakeIotBroker.named(name)
    dash = FakeLink(broker, "dashboard")
    states: list[dict] = []
    dash.subscribe("$aws/things/arm/shadow/name/state/update", lambda t, raw: states.append(json.loads(raw)))
    arm = Robot("so101", sim=Simulation())
    device = Mesh(f"iot://fake?broker={name}", tenant="lab")
    device.mount(arm, device_id="arm", watchdog_ms=None, iot=IotOptions(shadow_hz=1.0))
    try:
        _wait(lambda: states)
        assert states[-1]["state"]["reported"]["lockout"]["engaged"] is False
        arm.estop()  # local: no rpc, no broadcast
        n = len(states)
        _wait(lambda: len(states) > n, timeout=3.0)
        lock = states[-1]["state"]["reported"]["lockout"]
        assert lock["engaged"] is True and lock["epoch"], lock
        arm.resume(approved_by="alice")
        n = len(states)
        _wait(lambda: len(states) > n, timeout=3.0)
        assert states[-1]["state"]["reported"]["lockout"]["engaged"] is False
    finally:
        device.close()
        dash.close()
        arm.sim.close()
