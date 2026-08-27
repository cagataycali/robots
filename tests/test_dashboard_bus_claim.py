"""A bus another PROCESS holds must refuse the spawn (BUGS.md Q84).

The incident these tests pin: 185 parentless processes held both SO-101 buses for 10 hours while
DeviceManager.robots was empty, so every in-process check passed and the only symptom was an arm
missing from the fleet. The rule under test is therefore not "is this peer_id taken" but "does
anything on this machine already own this UART".
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard import bus_claim
from strands_robots.dashboard.device_manager import DeviceManager

PORT = "/dev/cu.usbmodem5AB0181806"


def test_a_free_bus_is_no_conflict():
    assert bus_claim.bus_conflict(PORT, [], {}) is None


def test_our_own_child_is_named_and_the_advice_is_despawn():
    msg = bus_claim.bus_conflict(PORT, [4242], {4242: "so101-arm-1"})
    assert "so101-arm-1" in msg and "pid 4242" in msg
    assert "Despawn" in msg
    # Our own child must NOT send the operator to the reaper: that script only kills PARENTLESS
    # holders, so it would do nothing and the advice would be a dead end.
    assert "reap_orphan_buses" not in msg


def test_a_stranger_names_the_pid_and_the_cure():
    msg = bus_claim.bus_conflict(PORT, [99001, 99002], {})
    assert "2 process(es)" in msg and "99001" in msg and "99002" in msg
    assert "reap_orphan_buses.sh" in msg and "Q84" in msg
    # The physical reason, so the refusal reads as a fact about serial buses rather than as policy.
    assert "half-duplex" in msg


def test_a_stranger_wins_even_when_one_holder_is_ours():
    msg = bus_claim.bus_conflict(PORT, [4242, 99001], {4242: "so101-arm-1"})
    assert "reap_orphan_buses.sh" in msg  # the stranger decides the advice
    assert "so101-arm-1" in msg  # but our own child is still disclosed


def test_the_tty_sibling_of_a_cu_device_is_checked_too():
    # A holder of /dev/tty.usbmodemX blocks /dev/cu.usbmodemX -- same UART, two device files. Checking
    # only the path we were handed is how a probe reports a free bus that is not free.
    assert bus_claim.sibling_devices(PORT) == [PORT, "/dev/tty.usbmodem5AB0181806"]


def test_no_lsof_means_no_refusal(monkeypatch):
    # Absence of evidence must not become evidence: a machine without lsof would otherwise be unable
    # to spawn anything at all.
    monkeypatch.setattr(bus_claim.shutil, "which", lambda _n: None)
    monkeypatch.setattr(bus_claim.os.path, "exists", lambda _p: False)
    assert bus_claim.bus_holders(PORT) == []


def test_spawn_refuses_before_starting_a_process(tmp_path, monkeypatch):
    """The whole point: the refusal happens instead of a Popen, not after one."""
    mgr = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    monkeypatch.setattr(bus_claim, "bus_holders", lambda _port, **_k: [99001])

    started: list = []
    import strands_robots.dashboard.device_manager as dm

    monkeypatch.setattr(dm.subprocess, "Popen", lambda *a, **k: started.append(a) or pytest.fail("spawned anyway"))

    out = mgr.spawn("so101", mode="real", port=PORT, peer_id="so101-arm-1")
    assert "error" in out and "99001" in out["error"]
    assert started == []


def test_a_sim_spawn_never_consults_the_bus(tmp_path, monkeypatch):
    # A sim robot has no UART; probing lsof for it would spend 8s of timeout budget per spawn and could
    # refuse a simulation because a real arm happens to be busy.
    asked: list[str] = []
    monkeypatch.setattr(bus_claim, "bus_holders", lambda port, **_k: asked.append(port) or [])
    mgr = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    import strands_robots.dashboard.device_manager as dm

    class _P:
        pid = 4321
        stdout = None

        def poll(self):
            return None

    monkeypatch.setattr(dm.subprocess, "Popen", lambda *a, **k: _P())
    monkeypatch.setattr(dm, "_drain", lambda *a, **k: None)
    mgr.spawn("so101", mode="sim", peer_id="sim-a")
    assert asked == []
