"""The signed safety rail asks the mesh kill switch before it opens a session.

``STRANDS_MESH=false`` is a hard kill switch, and
:func:`strands_robots.mesh.core.mesh_kill_switch_engaged` exists so that EVERY
path which can open a session asks the same question -- its docstring names the
incident (BUGS.md Q32) where a direct ``Mesh(...)`` construction bypassed the
inline test and turned a unit-test run into a live ``gateway-*`` peer with six
subscriber callback threads.

``MeshBridge.start()`` honours it. ``MeshBridge._safety_mesh()`` is the second
site in the same file that constructs and starts a ``Mesh``, and it did not:
with the switch set, the first e-stop still put a ``<peer>-safety`` gateway peer
on the live fleet. That is the ghost-peer case the switch was added to close,
arriving on the one path where an operator is least likely to be watching the
peer list, by way of the action they are least likely to want to debug
afterwards.

These pin, for the bridge's construction sites:

* the switch is asked BEFORE a Mesh is constructed -- not after, because
  constructing one is what joins the fleet;
* ``signed_estop`` says the rail was switched OFF rather than "unavailable", so
  the operator is pointed at the switch they set and not at a fault;
* with the switch clear, the rail still starts -- the guard refuses a session,
  it does not disable the feature;
* a future third construction site is graded from the source, since a new site
  added without the predicate is exactly how this defect arrived.

Parametrized over ``_mesh_switch.NEGATIVE`` directly, per that module's own
note: the vocabulary has one owner and a test that restates it would pass while
the product and the switch disagreed.

Run with --no-cov.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from strands_robots._mesh_switch import NEGATIVE
from strands_robots.dashboard.mesh_bridge import MeshBridge

#: Every kill spelling, plus the normalisation ``mesh_env_request`` documents
#: (case and surrounding whitespace), derived from the owner rather than typed.
KILLED = (*NEGATIVE, *(v.upper() for v in NEGATIVE), *(f"  {v}  " for v in NEGATIVE))

#: Values that leave the switch clear: an opt-in, and "said nothing".
ALLOWED = ("true", "1", "")


class RecordingMesh:
    """Stands in for Mesh and records that it was constructed/started at all.

    Construction is recorded separately from ``start()`` because constructing a
    Mesh is already asking for one: a guard placed after the constructor must
    not pass this.
    """

    events: list[tuple[str, str | None]] = []
    alive = True

    def __init__(self, robot: Any, peer_id: str | None = None, peer_type: str = "robot") -> None:
        self.peer_id = peer_id
        RecordingMesh.events.append(("constructed", peer_id))

    def start(self) -> None:
        RecordingMesh.events.append(("started", self.peer_id))

    def emergency_stop(self) -> list[dict[str, Any]]:
        RecordingMesh.events.append(("emergency_stop", self.peer_id))
        return []


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[RecordingMesh]:
    """Patch the Mesh class the bridge imports, and reset the event log."""
    import strands_robots.mesh.core as core

    RecordingMesh.events = []
    monkeypatch.setattr(core, "Mesh", RecordingMesh)
    return RecordingMesh


@pytest.mark.parametrize("value", KILLED)
def test_the_kill_switch_refuses_the_safety_rail_a_session(value, recorder, monkeypatch):
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    assert bridge._safety_mesh() is None
    assert recorder.events == [], f"STRANDS_MESH={value!r} still opened a session: {recorder.events}"


@pytest.mark.parametrize("value", KILLED)
def test_an_estop_under_the_kill_switch_creates_no_ghost_peer(value, recorder, monkeypatch):
    """The regression: the FIRST e-stop was what put the ghost peer on the fleet."""
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    out = bridge.signed_estop()

    assert out["signed"] is False
    assert recorder.events == [], f"e-stop opened a session under STRANDS_MESH={value!r}"
    # Switched off, not broken - the operator is pointed at the switch.
    assert "STRANDS_MESH" in out["error"], out["error"]
    assert "unavailable" not in out["error"], out["error"]


def test_a_broken_rail_still_reports_unavailable_not_switched_off(recorder, monkeypatch):
    """The two answers stay distinguishable: this one really is a fault."""
    monkeypatch.setenv("STRANDS_MESH", "true")
    bridge = MeshBridge(peer_id="dash")

    with mock.patch.object(MeshBridge, "_safety_mesh", return_value=None):
        out = bridge.signed_estop()

    assert out == {"signed": False, "error": "safety mesh unavailable"}


@pytest.mark.parametrize("value", ALLOWED)
def test_the_guard_refuses_a_session_it_does_not_disable_the_rail(value, recorder, monkeypatch):
    """With the switch clear the rail starts, so the guard is a gate not a mute."""
    monkeypatch.setenv("STRANDS_MESH", value)
    bridge = MeshBridge(peer_id="dash")

    m = bridge._safety_mesh()

    assert m is not None
    assert recorder.events == [("constructed", "dash-safety"), ("started", "dash-safety")]


def test_every_mesh_construction_site_in_the_bridge_asks_the_predicate():
    """Derived from the source, so a NEW site cannot skip the gate silently.

    Grading behaviour alone would leave a third construction site untested
    until someone wrote a case for it, and the absence of that case is how
    ``_safety_mesh`` stayed ungated while ``start`` was gated.
    """
    import ast
    import inspect

    import strands_robots.dashboard.mesh_bridge as bridge_mod

    tree = ast.parse(inspect.getsource(bridge_mod))
    opens_session: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        called = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if "Mesh" in called:  # constructing one is what joins the fleet
            opens_session[node.name] = "mesh_kill_switch_engaged" in called

    assert opens_session, "no Mesh construction site found - has the class been renamed?"
    ungated = sorted(name for name, gated in opens_session.items() if not gated)
    assert not ungated, (
        f"these open a mesh session without asking mesh_kill_switch_engaged(): {ungated}. "
        "STRANDS_MESH=false must be honoured at every construction site."
    )
