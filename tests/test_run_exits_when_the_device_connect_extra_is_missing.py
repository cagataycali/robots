"""``Robot(...).run()`` without the ``[device-connect]`` extra exits 1.

Measured on macOS with the ``dev`` venv (no ``device-connect`` extra) at
1276a3285: ``python -c 'from strands_robots import Robot; Robot("so100").run()'``
printed the install remedy and "Ctrl+C to stop", then slept forever. The README
path therefore ended in a hung terminal, and the shell saw no exit code at all.
Parking is defensible for a broker outage - the docstring says so and that
branch is untouched - but no amount of waiting installs a package into the
running interpreter.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from strands_robots import robot as robot_module


def _run_without_the_extra(monkeypatch, capsys, *, instance=None):
    """Drive the foreground loop with the integration module unimportable.

    ``time.sleep`` raises so a regression back into the park loop fails loudly
    instead of hanging the suite; ``os._exit`` is captured to read the code.
    """
    if instance is None:
        instance = types.SimpleNamespace(_peer_id="arm-1", _peer_type="sim", mesh=None)
    monkeypatch.setitem(sys.modules, "strands_robots.device_connect", None)
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


def test_the_process_exits_one_instead_of_parking(monkeypatch, capsys):
    exits, out = _run_without_the_extra(monkeypatch, capsys)
    assert exits == [1]
    assert "arm-1 is NOT online" in out
    assert "Ctrl+C" not in out, out


def test_the_operator_still_reads_what_was_lost(monkeypatch, capsys):
    """The exit path keeps the transport line the park path had."""

    class _Mesh:
        def stop(self):
            pass

    instance = types.SimpleNamespace(_peer_id="arm-1", _peer_type="sim", mesh=_Mesh())
    _, out = _run_without_the_extra(monkeypatch, capsys, instance=instance)
    assert "The built-in mesh was stopped for it" in out
    assert "serves no transport" in out


def test_the_instance_is_released_before_exiting(monkeypatch, capsys):
    """On hardware this is the path to the driver's disconnect()."""
    released: list[str] = []
    instance = types.SimpleNamespace(
        _peer_id="arm-1", _peer_type="robot", mesh=None, cleanup=lambda: released.append("yes")
    )
    exits, _ = _run_without_the_extra(monkeypatch, capsys, instance=instance)
    assert released == ["yes"]
    assert exits == [1]


def test_a_broker_failure_still_parks(monkeypatch, capsys):
    """The unchanged half: a RuntimeError from bring-up is possibly transient."""
    module = types.ModuleType("strands_robots.device_connect")
    module.init_device_connect_sync = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no broker"))
    monkeypatch.setitem(sys.modules, "strands_robots.device_connect", module)
    monkeypatch.setattr("time.sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt()))
    exits: list[int] = []

    class _Exited(Exception):
        pass

    monkeypatch.setattr(os, "_exit", lambda code: (exits.append(code), (_ for _ in ()).throw(_Exited()))[1])
    instance = types.SimpleNamespace(_peer_id="arm-1", _peer_type="sim", mesh=None)
    with pytest.raises(_Exited):
        robot_module._run_device_connect_foreground(instance)
    assert exits == [0]
    assert "Ctrl+C to stop" in capsys.readouterr().out
