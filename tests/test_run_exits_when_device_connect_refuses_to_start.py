"""``Robot(...).run()`` exits 1 when Device Connect refuses an unauthenticated transport.

The refusal (no TLS, no ``DEVICE_CONNECT_ALLOW_INSECURE``) is not a broker
outage: waiting never produces credentials, and the opt-in is read once at
start. Before this fix the foreground loop caught it as a generic bring-up
failure, printed "Ctrl+C to stop" and slept forever - the README's first command
ended in a hung terminal with the remedy scrolled off above it. The broker path
is unchanged and still parks.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from strands_robots import robot as robot_module
from strands_robots.device_connect._authz import DeviceConnectRefused


def _run_with_bring_up_raising(monkeypatch, capsys, error: BaseException, *, instance=None):
    if instance is None:
        instance = types.SimpleNamespace(_peer_id="arm-1", _peer_type="sim", mesh=None)
    module = types.ModuleType("strands_robots.device_connect")
    module.init_device_connect_sync = lambda *a, **k: (_ for _ in ()).throw(error)
    monkeypatch.setitem(sys.modules, "strands_robots.device_connect", module)
    # A regression back into the park loop must fail, not hang the suite.
    monkeypatch.setattr("time.sleep", lambda _s: (_ for _ in ()).throw(AssertionError("parked in the sleep loop")))
    exits: list[int] = []

    class _Exited(Exception):
        pass

    def _exit(code):
        exits.append(code)
        raise _Exited()

    monkeypatch.setattr(os, "_exit", _exit)
    with pytest.raises(_Exited):
        robot_module._run_device_connect_foreground(instance)
    return exits, capsys.readouterr().out


def test_a_refusal_exits_one_instead_of_parking(monkeypatch, capsys):
    exits, out = _run_with_bring_up_raising(monkeypatch, capsys, DeviceConnectRefused("no TLS configured"))
    assert exits == [1]
    assert "arm-1 is NOT online: Device Connect refused to start" in out
    assert "Ctrl+C" not in out, out


def test_the_refusal_releases_the_instance_first(monkeypatch, capsys):
    released: list[str] = []
    instance = types.SimpleNamespace(
        _peer_id="arm-1", _peer_type="robot", mesh=None, cleanup=lambda: released.append("yes")
    )
    exits, _ = _run_with_bring_up_raising(monkeypatch, capsys, DeviceConnectRefused("no TLS"), instance=instance)
    assert released == ["yes"]
    assert exits == [1]


def test_the_refusal_is_a_runtime_error_for_callers_that_catch_the_broad_class():
    assert issubclass(DeviceConnectRefused, RuntimeError)


def test_init_raises_the_named_refusal(monkeypatch):
    """The real bring-up raises the subclass, not a bare ``RuntimeError``."""
    pytest.importorskip("device_connect_edge")
    from strands_robots.device_connect import init_device_connect_sync

    for name in ("DEVICE_CONNECT_ALLOW_INSECURE", "ZENOH_CONNECT", "MESSAGING_URLS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MESSAGING_BACKEND", "zenoh")
    instance = types.SimpleNamespace(_peer_id="arm-1")
    with pytest.raises(DeviceConnectRefused, match="refused to start"):
        init_device_connect_sync(instance, peer_id="arm-1", peer_type="sim")
