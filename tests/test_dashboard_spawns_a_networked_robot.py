"""A networked robot (G1, Go2, UR, Spot...) is added from the Devices sheet by its address.

``port`` is polymorphic across drivers: a servo bus takes a ``/dev`` path on this machine, a
networked robot takes where it is on the network. The dashboard held every robot to the serial
rule, so ``unitree_g1`` was listed as real-capable and could not be spawned: an IP was refused
as "not a path under /dev/", and the sheet offered only a serial picker and a lerobot
calibration id. :func:`~strands_robots.drivers.port_kind` is the fact the sheet and the
refusals now branch on.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from strands_robots.dashboard import device_manager, routes_mesh
from strands_robots.dashboard.device_manager import DeviceManager, validate_port
from strands_robots.drivers import port_kind

APP_JS = Path(device_manager.__file__).parent / "static" / "app.js"


@pytest.mark.parametrize(
    ("robot", "kind"),
    [("so101", "serial"), ("koch", "serial"), ("unitree_g1", "address"), ("ur5e", "address"), ("spot", "address")],
)
def test_port_kind_says_what_port_names_for_each_driver(robot: str, kind: str) -> None:
    assert port_kind(robot) == kind


@pytest.mark.parametrize(
    ("port", "kind", "ok"),
    [
        ("/dev/ttyACM0", "serial", True),
        ("192.168.123.161", "serial", False),
        ("192.168.123.161", "address", True),
        ("ur5e.local:30004", "address", True),
        ("[fe80::1]", "address", True),
        ("radio://0/80/2M", "address", True),
        ("-oProxyCommand=x", "address", False),
        ("host/../etc", "address", False),
        ("10.0.0.1 ; rm", "address", False),
        ("/dev/../etc/passwd", "address", False),  # a /dev path is a bus, whatever the robot
    ],
)
def test_a_port_is_judged_by_the_shape_its_driver_reads(port: str, kind: str, ok: bool) -> None:
    assert (validate_port(port, kind) is None) is ok


class _Proc:
    pid = 4242
    stdout = None

    def poll(self) -> int | None:
        return None


@pytest.fixture
def dm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[DeviceManager, list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    probed: list[str] = []

    def popen(argv: list[str], **_: Any) -> _Proc:
        calls.append(json.loads(argv[-1]))
        return _Proc()

    def bus_holders(port: str, **_: Any) -> list[Any]:
        probed.append(port)
        return []

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(device_manager, "_drain", lambda *a, **k: None)
    monkeypatch.setattr(device_manager.bus_claim, "bus_holders", bus_holders)
    monkeypatch.setattr(device_manager.camera_liveness, "stamp_device_names", lambda cams, roster: cams)
    monkeypatch.setattr(DeviceManager, "_roster_for_stamp", lambda self: [])
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.delenv("STRANDS_MESH_LOCAL_DEV", raising=False)
    return DeviceManager(profiles_path=str(tmp_path / "profiles.json")), calls, probed


@pytest.mark.parametrize(
    ("robot", "kwargs", "spawned"),
    [
        ("unitree_g1", {"port": "192.168.123.161", "network_interface": "enp3s0"}, True),
        ("unitree_g1", {}, True),  # the G1 driver binds a NIC; its address is optional
        ("ur5e", {"port": "192.168.1.10"}, True),
        ("so101", {}, False),  # a servo bus still needs its port
        ("so101", {"port": "/dev/ttyACM0", "network_interface": "eth0"}, False),
        ("ur5e", {"port": "192.168.1.10", "network_interface": "eth0"}, False),  # UR takes no NIC
        ("unitree_g1", {"network_interface": "-eth0"}, False),
    ],
)
def test_spawn_takes_a_networked_robot_by_address(
    dm: tuple[DeviceManager, list[dict[str, Any]], list[str]], robot: str, kwargs: dict[str, Any], spawned: bool
) -> None:
    manager, calls, probed = dm
    out = manager.spawn(robot, "real", peer_id="peer", remember=False, **kwargs)
    assert ("error" not in out) is spawned, out
    if spawned:
        assert calls[0]["port"] == kwargs.get("port")
        assert calls[0].get("network_interface") == kwargs.get("network_interface")
        assert probed == [], "an address is not a serial bus: lsof must not be asked about it"


def test_the_spawn_form_is_told_which_transport_each_robot_uses() -> None:
    rows = {r["name"]: r for r in asyncio.run(routes_mesh.registry({}))["robots"]}
    assert rows["unitree_g1"]["real_transport"] == {"port_kind": "address", "network_interface": True}
    assert rows["so101"]["real_transport"] == {"port_kind": "serial", "network_interface": False}
    # A native driver the entry does not declare still builds it for real.
    assert rows["ur5e"]["real_transport"] == {"port_kind": "address", "network_interface": False}
    app_js = APP_JS.read_text(encoding="utf-8")
    assert "real_transport" in app_js and "network_interface" in app_js, "static/app.js was not rebuilt"
