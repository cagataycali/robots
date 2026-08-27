"""Q51: a settings screen's "when does this take effect" must survive reading the code.

Auditing the timing claims one field at a time. mesh.camera_hz said "needs a mesh restart",
and a mesh restart cannot deliver it: the rate is resolved inside Mesh._resolve_camera_hz()
when a ROBOT starts its camera loop, in the robot's own process. The dashboard has no robot
and publishes no frames, so re-pointing its session changes nothing whatsoever -- the operator
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
    """The keys the dashboard's OWN session reads keep their claim -- over-correcting here would
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


# --- Q52: the startup-only key, and the agent claim that turned out TRUE -----------------


def test_cors_origins_is_reported_as_startup_not_applied(monkeypatch):
    monkeypatch.setattr(settings, "update_strict", lambda patch: (["security.cors_origins"], []))
    res = _apply({"security": {"cors_origins": ["https://lab.example"]}})
    assert res["startup_required"] == ["security.cors_origins"]
    assert res["applied"] == []
    assert res["restart_required"] == res["respawn_required"] == []


def test_cors_has_two_readers_with_different_lifetimes():
    """The evidence for that wording, pinned.

    create_app() bakes the origin list into CORSMiddleware (browser header, startup-only);
    TokenAuthMiddleware re-reads settings per request (the write/websocket gate). So removing an
    origin tightens immediately while adding one needs a restart -- the safe asymmetry, and the
    reason the field cannot simply say "applies immediately".
    """
    import inspect

    from strands_robots.dashboard import server

    app_src = inspect.getsource(server.create_app)
    assert "cors_origins" in app_src and "CORSMiddleware" in app_src
    gate_src = inspect.getsource(server.TokenAuthMiddleware._cross_origin_refused)
    assert 'settings.get("security", "cors_origins"' in gate_src, "the gate must read live"


def test_the_agent_keys_really_do_apply_on_the_next_turn(monkeypatch):
    """Checked with the same method and found HONEST -- recorded so nobody re-audits it blind.

    reset_agent() drops the cached agent; the next get_agent() calls _build_agent(), which reads
    settings.load()["agent"] then. Nothing captures the model id earlier.
    """
    import inspect

    from strands_robots.dashboard import agent_bridge

    assert "settings.load()" in inspect.getsource(agent_bridge._build_agent)
    assert "_build_agent()" in inspect.getsource(agent_bridge.get_agent)
    assert "_agent = None" in inspect.getsource(agent_bridge.reset_agent)

    calls: list[bool] = []
    monkeypatch.setattr(settings, "update_strict", lambda patch: (["agent.model_id"], []))
    monkeypatch.setattr(
        "strands_robots.dashboard.agent_bridge.reset_agent",
        lambda clear_history=False: calls.append(True),
    )
    res = _apply({"agent": {"model_id": "anthropic.claude"}})
    assert calls == [True], "a model change must drop the cached agent"
    assert res["agent_reset"] is True
    assert res["applied"] == ["agent.model_id"]
