"""broadcast("estop") latches every matching device; resume goes through the device's resume authority."""

from __future__ import annotations

import time

import pytest

from strands_robots import Robot
from strands_robots.mesh import Mesh, RequestRefused, authority


@pytest.fixture
def lab(mesh_pair):
    a, b = mesh_pair
    arms = [Robot("so101", id=f"arm{i}") for i in range(3)]
    for i, arm in enumerate(arms):
        a.mount(arm, device_id=f"arm{i}", labels={"bench": "left" if i < 2 else "right"})
    assert len(b.discover("device(*)", min_count=3, timeout=3.0)) == 3
    yield arms, a, b
    a.close()  # unmount before the robots go
    for arm in arms:
        arm.close()


def test_broadcast_estop_reaches_every_device_and_reports_acks(lab):
    arms, _, b = lab
    t0 = time.perf_counter()
    acked = b.broadcast("estop", selector="device(*)")
    assert time.perf_counter() - t0 < 1.0
    assert sorted(acked) == ["arm0", "arm1", "arm2"]
    assert all(arm.estopped for arm in arms)


def test_broadcast_estop_honours_the_selector(lab):
    arms, _, b = lab
    assert sorted(b.broadcast("estop", selector="device(bench:left)")) == ["arm0", "arm1"]
    assert [arm.estopped for arm in arms] == [True, True, False]


def test_an_undiscovered_peer_still_latches_from_the_tenant_estop(lab):
    arms, a, _ = lab
    late = Mesh(a.transports[0], tenant=a.tenant, own_transports=False)
    late.broadcast("estop", selector="device(*)", deadline=0.0)
    deadline = time.monotonic() + 2.0
    while not all(arm.estopped for arm in arms) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert all(arm.estopped for arm in arms)
    late.close()


def test_one_device_whose_stop_raises_does_not_keep_the_tenant_estop_from_the_rest(lab, monkeypatch, caplog):
    arms, _, b = lab

    def bus_in_a_bad_state():
        raise OSError("the bus did not answer")

    monkeypatch.setattr(arms[0].body, "estop", bus_in_a_bad_state)
    b.broadcast("estop", selector="device(*)", deadline=0.0)  # the tenant key only: no rpc fan-out
    deadline = time.monotonic() + 2.0
    while not all(arm.estopped for arm in arms) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [arm.estopped for arm in arms] == [True, True, True]
    assert any("arm0" in r.getMessage() and "OSError" in r.getMessage() for r in caplog.records)
    with pytest.raises(RequestRefused):  # the raising stop still locked resume to its authority
        b.invoke("arm0", "resume", approved_by="someone-else")


def test_resume_by_the_operator_who_stopped(lab):
    arms, _, b = lab
    b.broadcast("estop", selector="arm0")
    b.invoke("arm0", "resume", approved_by=b.operator)
    assert not arms[0].estopped


def test_resume_from_a_non_authority_is_refused(lab):
    arms, a, b = lab
    b.broadcast("estop", selector="arm0")
    with pytest.raises(RequestRefused, match="not a resume authority"):
        b.invoke("arm0", "resume", approved_by="mallory")
    assert arms[0].estopped
    assert b.broadcast("resume", selector="arm0", approved_by="mallory") == []
    assert arms[0].estopped


def test_a_named_resume_authority_may_resume(mesh_pair):
    a, b = mesh_pair
    a.resume_authority = authority.Authority(names=("safety-lead",))
    arm = Robot("so101")
    a.mount(arm, device_id="arm")
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.broadcast("estop", selector="arm")
    assert b.broadcast("resume", selector="arm", approved_by="safety-lead") == ["arm"]
    assert not arm.estopped
    a.close()  # unmount before the robot goes
    arm.close()


def test_with_keys_only_a_signed_fresh_assertion_for_this_lockout_resumes(mesh_pair):
    a, b = mesh_pair
    private, public = authority.keygen()
    a.resume_authority = authority.Authority(public_keys=(public,))
    arm = Robot("so101")
    a.mount(arm, device_id="arm")
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.broadcast("estop", selector="arm")
    with pytest.raises(RequestRefused, match="signed"):
        b.invoke("arm", "resume", approved_by=b.operator)
    b.resume_key, _ = authority.keygen()
    with pytest.raises(RequestRefused, match="signature"):
        b.invoke("arm", "resume", approved_by="op")
    assert arm.estopped
    b.resume_key = private
    b.invoke("arm", "resume", approved_by="op")
    assert not arm.estopped
    assert b.invoke("arm", "lockout")["engaged"] is False


def test_the_mesh_kill_switch(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH", "false")
    with pytest.raises(RuntimeError, match="STRANDS_MESH"):
        Mesh("mem://killed")


def test_claimed_names_are_not_trusted_on_a_real_network():
    from strands_robots.mesh.authority import Authority, Lockout
    from strands_robots.mesh.serve import is_local

    assert is_local("mem://x") and is_local("zenoh://?listen=tcp/127.0.0.1:7447")
    assert not is_local("zenoh://") and not is_local("zenoh://?connect=tcp/10.0.0.5:7447")
    assert not is_local("zenoh://?listen=tcp/127.0.0.1:7447&connect=tcp/10.0.0.5:7447")
    assert not is_local("zenoh://?listen=tcp/0.0.0.0:7447")
    assert not is_local("zenoh://?connect=tcp/127.0.0.1:7447")
    assert is_local("iot://fake?broker=b") and not is_local("iot://fake-west-2?cert=c&endpoint=e")
    stop = Lockout(True, "e1", "alice")
    with pytest.raises(RequestRefused, match="needs operator keys"):
        Authority().check("arm", stop, "alice", None, local=False)
    assert Authority(trust_names=True).check("arm", stop, "alice", None, local=False) == "alice"
    assert Authority().check("arm", stop, "alice", None, local=True) == "alice"


def test_the_refusal_names_the_ways_out():
    from strands_robots.mesh.authority import Authority, Lockout

    with pytest.raises(RequestRefused) as err:
        Authority().check("arm", Lockout(True, "e1", None), "me", None, local=True)
    msg = str(err.value)
    assert "by=<operator>" in msg and "Authority(names=" in msg and "public_keys" in msg


def test_a_transport_whose_publish_raises_does_not_stop_the_estop_on_the_others(tenant):
    from strands_robots.mesh.transports import open_transport

    down, good = open_transport(f"mem://down-{tenant}"), f"mem://good-{tenant}"
    device, controller = Mesh(good, tenant=tenant), Mesh([down, good], tenant=tenant)

    def dead(*_a, **_k):
        raise RuntimeError("session closed")

    down.publish = dead  # type: ignore[method-assign]
    arm = Robot("so101")
    device.mount(arm, device_id="arm0")
    assert controller.discover("arm0", min_count=1, timeout=3.0)
    assert controller.broadcast("estop", selector="arm0") == ["arm0"]
    assert arm.estopped
    device.close()
    controller.close()
    arm.close()


def _three(tenant, **device_kw):
    """A device mesh and two operators, alice and mallory, on one in-process bus."""
    url = f"mem://ops-{tenant}"
    dev = Mesh(url, tenant=tenant, operator="device", **device_kw)
    alice = Mesh(url, tenant=tenant, operator="alice")
    mallory = Mesh(url, tenant=tenant, operator="mallory")
    return dev, alice, mallory


def test_the_first_stopper_keeps_the_stop_and_the_device_mints_its_epoch(tenant):
    dev, alice, mallory = _three(tenant)
    arm = Robot("so101")
    dev.mount(arm, device_id="arm", watchdog_ms=None)
    assert alice.discover("arm", min_count=1, timeout=3.0) and mallory.discover("arm", min_count=1, timeout=3.0)
    alice.invoke("arm", "estop", epoch="deadbeef")
    first = alice.invoke("arm", "lockout")
    assert first["by"] == "alice" and first["epoch"] != "deadbeef"
    assert abs(first["at"] - time.time()) < 5.0, "wall clock, so an audit can place it"
    mallory.invoke("arm", "estop", epoch="cafe")
    assert alice.invoke("arm", "lockout") == first
    with pytest.raises(RequestRefused, match="not a resume authority"):
        mallory.invoke("arm", "resume", approved_by="mallory")
    alice.invoke("arm", "resume", approved_by="alice")
    assert not arm.estopped
    arm.estop()  # pressed on the device: no mesh stopper may claim it
    mallory.invoke("arm", "estop")
    with pytest.raises(RequestRefused, match="not a resume authority"):
        mallory.invoke("arm", "resume", approved_by="mallory")
    for m in (alice, mallory, dev):
        m.close()
    arm.close()


def test_a_captured_signed_resume_cannot_clear_a_stop_after_the_device_restarts(tenant, monkeypatch):
    private, public = authority.keygen()
    captured = []
    sign = authority.sign
    monkeypatch.setattr(authority, "sign", lambda *a, **k: captured.append(sign(*a, **k)) or captured[-1])
    dev, alice, mallory = _three(tenant, resume_authority=authority.Authority(public_keys=(public,)))
    alice.resume_key = private
    arm = Robot("so101")
    dev.mount(arm, device_id="arm", watchdog_ms=None)
    assert alice.discover("arm", min_count=1, timeout=3.0) and mallory.discover("arm", min_count=1, timeout=3.0)
    mallory.invoke("arm", "estop", epoch="deadbeef")
    alice.invoke("arm", "resume", approved_by="alice")
    assert not arm.estopped
    dev.close()  # the device restarts: a fresh process remembers no nonce
    dev = Mesh(f"mem://ops-{tenant}", tenant=tenant, resume_authority=authority.Authority(public_keys=(public,)))
    dev.mount(arm, device_id="arm", watchdog_ms=None)
    assert mallory.discover("arm", min_count=1, timeout=3.0)
    mallory.invoke("arm", "estop", epoch=captured[-1]["epoch"])
    with pytest.raises(RequestRefused, match="not for this device's current stop"):
        mallory.invoke("arm", "resume", approved_by="mallory", assertion=captured[-1])
    assert arm.estopped
    for m in (alice, mallory, dev):
        m.close()
    arm.close()


def test_an_operator_key_can_be_trusted_and_revoked_and_used_nonces_age_out():
    private, public = authority.keygen()
    auth = authority.Authority(public_keys=(authority.keygen()[1],), max_age=60.0)
    auth.trust(public)
    lock = authority.Lockout(engaged=True, epoch="e1")
    approver = auth.check("arm", lock, "op", authority.sign(private, "e1", ["arm"], "op"))
    assert approver == f"op key={authority.fingerprint(public)}", "the record names the key, not only the typed name"
    auth._seen = {n: t - 100.0 for n, t in auth._seen.items()}  # older than max_age + skew
    auth.check("arm", lock, "op", authority.sign(private, "e1", ["arm"], "op"))
    assert len(auth._seen) == 1, "nonces older than max_age are pruned"
    auth.distrust(public)
    with pytest.raises(RequestRefused, match="no trusted operator key"):
        auth.check("arm", lock, "op", authority.sign(private, "e1", ["arm"], "op"))
