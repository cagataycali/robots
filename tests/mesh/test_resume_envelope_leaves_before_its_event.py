"""Regression for GH #4171: a remote resume clears only the issuing process.

Every peer's Zenoh session carries an ingress ``downsampling`` rule on
``**/safety/**`` (:func:`strands_robots.mesh._zenoh_config.downsampling_block`,
2 Hz by default). Zenoh keeps ONE timestamp per rule per link, not one per
concrete key: two ``safety/**`` messages from the same peer inside one period
lose the second one before any subscriber callback runs, whatever its key.

:meth:`strands_robots.mesh.core.Mesh.resume` used to publish its
``resume_ok`` safety event first and the fleet-wide ``strands/safety/resume``
envelope a few microseconds later. The event took the rule's slot, the envelope
was dropped at every receiver's ingress (and at the hub's), and nothing was
logged anywhere: the issuer cleared, the operator and every other robot stayed
locked. Reproduced with three processes on a private hub (operator e-stops, RPC
``resume`` to one robot); skipping the event alone made the fleet clear.

The contract pinned here: on the wire, the envelope that clears the fleet is the
FIRST ``safety/**`` message a resume emits, exactly as :meth:`Mesh.emergency_stop`
already publishes its ``strands/safety/estop`` envelope before its event.
"""

from __future__ import annotations

from strands_robots.mesh import core

_CODE = "operator-secret-1234567890"


def _issuer(monkeypatch, tmp_path) -> tuple[core.Mesh, list[str]]:
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    mesh = core.Mesh(robot=object(), peer_id="issuer")
    # ``publish_safety_event`` is a no-op on a Mesh that is not running; the
    # real method is what this test is about, so mark the mesh running instead
    # of mocking the event half away.
    mesh._running = True
    # No native publisher available: both halves go through the module ``put``.
    monkeypatch.setattr(mesh, "_safety_publisher_for", lambda key: None)
    wire: list[str] = []
    monkeypatch.setattr(core, "put", lambda key, payload: wire.append(key))
    return mesh, wire


def _safety_keys(wire: list[str]) -> list[str]:
    return [key for key in wire if "/safety/" in key]


def test_resume_envelope_is_the_first_safety_message_of_a_resume(monkeypatch, tmp_path):
    mesh, wire = _issuer(monkeypatch, tmp_path)
    mesh._estop_lockout.set()
    mesh._last_estop_ts = core.time.time()
    mesh._last_estop_mono = core.time.monotonic()

    assert mesh.resume(_CODE) == {"status": "ok"}

    safety = _safety_keys(wire)
    assert safety[0] == "strands/safety/resume", safety
    # The event still goes out, after the envelope that clears the fleet.
    assert "strands/issuer/safety/event" in safety[1:], safety


def test_estop_envelope_is_the_first_safety_message_of_an_estop(monkeypatch, tmp_path):
    """The order the resume now follows is the one the e-stop already had."""
    mesh, wire = _issuer(monkeypatch, tmp_path)
    monkeypatch.setattr(mesh, "broadcast", lambda *a, **k: [])

    mesh.emergency_stop()

    safety = _safety_keys(wire)
    assert safety[0] == "strands/safety/estop", safety
    assert "strands/issuer/safety/event" in safety[1:], safety
