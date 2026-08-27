"""Q2: ``mode`` was free text, so a typo became a label instead of a refusal.

The child spawner branches on ``if cfg["mode"] == "real"`` and takes the sim path
for everything else. So an unvalidated mode never failed:

* ``mode="quantum"`` spawned a working SIM peer that showed "quantum" in the
  fleet grid -- nothing anywhere said no.
* ``mode="Real"`` missed the comparison by one capital letter (the SDK lowercases
  modes, the dashboard did not), so the operator got a SIMULATION on a card
  labelled Real. That direction matters: believing hardware is moving when it is
  not, or the reverse.

An unknown robot name had the same shape -- it reached ``Popen`` and raised inside
a child whose pid had already been reported as success.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

import strands_robots.dashboard.device_manager as dm
from strands_robots.dashboard.device_manager import DeviceManager, validate_spawn


class FakeProc:
    _next_pid = 9000

    def __init__(self) -> None:
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.stdout = None

    def poll(self):
        return None


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(
        dm,
        "subprocess",
        types.SimpleNamespace(Popen=lambda *a, **k: FakeProc(), PIPE=-1, STDOUT=-2),
    )
    d = DeviceManager.__new__(DeviceManager)
    d.robots = {}
    d._lock = threading.Lock()
    # A mode=real spawn remembers the port in the profile store; keep that off disk.
    d.profiles = types.SimpleNamespace(save=lambda *a, **k: None)
    return d


# --------------------------------------------------------------------------
# validate_spawn: the refusal
# --------------------------------------------------------------------------


def test_a_good_pair_passes_through_canonicalised():
    assert validate_spawn("so101", "sim") == ("so101", "sim")


def test_the_mode_is_normalised_for_case_and_whitespace():
    """'Real' used to miss `== "real"` and silently produce a simulation."""
    assert validate_spawn("so101", "Real") == ("so101", "real")
    assert validate_spawn("so101", " SIM ") == ("so101", "sim")
    assert validate_spawn("so101", "REAL") == ("so101", "real")


def test_the_robot_name_is_canonicalised_the_way_the_sdk_does_it():
    assert validate_spawn("SO-101", "sim") == ("so101", "sim")


def test_an_invented_mode_is_refused_and_names_the_valid_ones():
    out = validate_spawn("so101", "quantum")
    assert isinstance(out, dict)
    assert "sim, real" in out["error"] and "quantum" in out["error"]


def test_auto_is_refused_with_the_reason_it_cannot_be_honoured_here():
    """The SDK resolves auto inside the child; a card cannot be labelled from that."""
    out = validate_spawn("so101", "auto")
    assert isinstance(out, dict)
    assert "auto" in out["error"] and "label" in out["error"]


@pytest.mark.parametrize("mode", [None, 1, True, [], {"mode": "sim"}])
def test_a_non_string_mode_is_refused_without_leaking_a_traceback(mode):
    out = validate_spawn("so101", mode)
    assert isinstance(out, dict) and "mode must be one of" in out["error"]


def test_an_unknown_robot_name_is_refused_with_the_sdk_s_own_words():
    out = validate_spawn("so1010", "sim")
    assert isinstance(out, dict)
    assert "so1010" in out["error"]
    # The registry is the vocabulary, not a second list maintained here.
    assert "list_robots" in out["error"] or "registered name" in out["error"]


@pytest.mark.parametrize("name", ["", "   ", None, 7])
def test_a_missing_robot_name_is_refused(name):
    out = validate_spawn(name, "sim")
    assert isinstance(out, dict) and "error" in out


def test_validation_never_fails_because_the_sdk_could_not_be_asked(monkeypatch):
    """If the validator itself breaks, a spawn must still be possible."""
    import strands_robots.robot as robot_mod

    def boom(*a, **k):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(robot_mod, "resolve_name", boom)
    assert validate_spawn("so101", "sim") == ("so101", "sim")


# --------------------------------------------------------------------------
# spawn(): no process is created for a refused request
# --------------------------------------------------------------------------


def test_an_invented_mode_creates_no_process_and_no_card(manager):
    out = manager.spawn("so101", "quantum")
    assert "error" in out and "pid" not in out
    assert manager.robots == {}  # nothing to show in the fleet grid


def test_an_unknown_robot_creates_no_process(manager):
    out = manager.spawn("so1010", "sim")
    assert "error" in out and "pid" not in out
    assert manager.robots == {}


def test_a_real_spawn_without_a_port_is_still_refused(manager):
    """The pre-existing guard must survive the new validation in front of it."""
    out = manager.spawn("so101", "real")
    assert out["error"] == "port required for mode=real"
    assert manager.robots == {}


def test_a_valid_spawn_still_starts_and_is_tracked(manager):
    out = manager.spawn("so101", "sim")
    assert out["mode"] == "sim" and out["pid"]
    assert list(manager.robots) == [out["peer_id"]]


def test_the_card_is_labelled_with_the_normalised_mode(manager):
    """'Real' must not label a card while the peer is something else."""
    out = manager.spawn("so101", "Real", port="/dev/tty.fake")
    assert out["mode"] == "real"
    assert manager.robots[out["peer_id"]].mode == "real"


def test_the_generated_peer_id_uses_the_canonical_name_and_mode(manager):
    out = manager.spawn("SO-101", "sim")
    assert out["peer_id"].startswith("so101-sim-")


def test_two_spawns_in_one_second_do_not_collide(manager, monkeypatch):
    """spawn() now mints its default id through the same unique-id path."""
    monkeypatch.setattr(
        dm,
        "time",
        types.SimpleNamespace(
            time=lambda: 1_787_140_000.0,
            monotonic=time.monotonic,
            sleep=time.sleep,
            strftime=time.strftime,
        ),
    )
    a = manager.spawn("so101", "sim")
    b = manager.spawn("so101", "sim")
    assert a["peer_id"] != b["peer_id"]
    assert len(manager.robots) == 2
