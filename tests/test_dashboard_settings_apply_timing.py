"""Q51: a settings screen's "when does this take effect" must survive reading the code.

Auditing the timing claims one field at a time. mesh.camera_hz said "needs a mesh restart",
and a mesh restart cannot deliver it: the rate is resolved inside Mesh._resolve_camera_hz()
when a ROBOT starts its camera loop, in the robot's own process. The dashboard has no robot
and publishes no frames, so re-pointing its session changes nothing whatsoever — the operator
who lowered the rate to save bandwidth, clicked the restart button and saw "mesh re-pointed"
had been told the job was done by both the field and the result line.
"""

from __future__ import annotations

import inspect

from strands_robots.dashboard import config_api, settings


def _apply(patch):
    return config_api.apply(patch)


def test_camera_hz_asks_for_a_respawn_not_a_mesh_restart(monkeypatch):
    monkeypatch.setattr(settings, "update_strict", lambda patch: (["mesh.camera_hz"], []))
    res = _apply({"mesh": {"camera_hz": 12}})
    assert res["respawn_required"] == ["mesh.camera_hz"]
    assert res["restart_required"] == []
    # And it is not claimed as applied: nothing running changed its rate.
    assert "mesh.camera_hz" not in res["applied"]


def test_endpoints_still_ask_for_a_mesh_restart(monkeypatch):
    """The keys the dashboard's OWN session reads keep their claim — over-correcting here would
    hide a real restart requirement."""
    monkeypatch.setattr(settings, "update_strict", lambda patch: (["mesh.port"], []))
    res = _apply({"mesh": {"port": 7448}})
    assert res["restart_required"] == ["mesh.port"]
    assert res["respawn_required"] == []


def test_a_live_key_needs_neither(monkeypatch):
    monkeypatch.setattr(settings, "update_strict", lambda patch: (["voice.provider"], []))
    res = _apply({"voice": {"provider": "openai"}})
    assert res["applied"] == ["voice.provider"]
    assert res["restart_required"] == res["respawn_required"] == []


def test_the_claim_matches_where_the_value_is_actually_read():
    """The evidence for this fix, pinned: camera_hz is resolved per-ROBOT at Mesh.start().

    If a later change makes the dashboard's own session read the rate, this test fails and the
    timing claim must be re-derived rather than inherited.
    """
    from strands_robots.mesh import core

    src = inspect.getsource(core.Mesh.start)
    assert "_resolve_camera_hz" in src, "the rate is read at robot start"
    # ... and the camera loop only runs for a mesh that HAS a robot.
    assert "self.robot" in src


def test_respawn_and_restart_key_sets_are_disjoint():
    assert not (config_api._RESTART_KEYS & config_api._RESPAWN_KEYS)
