"""Settings surface of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_settings_apply_timing.py, test_dashboard_settings_nonfinite.py, test_dashboard_settings_typed_values.py, test_dashboard_settings_unknown_keys.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import inspect
import json

import pytest

from strands_robots.dashboard import config_api, settings
from strands_robots.dashboard.settings import CoercionError, _coerce, _coerce_strict

# ============================================================================
# from tests/test_dashboard_settings_apply_timing.py
# Q51: a settings screen's "when does this take effect" must survive reading the code.
# ============================================================================


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


# ============================================================================
# from tests/test_dashboard_settings_nonfinite.py
# Q14/Q15: a config write can never brick the config screen.
# ============================================================================


def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings._cache = None
    yield
    settings._cache = None


def _file_text():
    return settings.SETTINGS_FILE.read_text() if settings.SETTINGS_FILE.exists() else ""


# --- strict path (the API) ---------------------------------------------------


@pytest.mark.parametrize("bad", ["NaN", "nan", "inf", "-inf", float("nan"), float("inf")])
def test_nonfinite_temperature_is_an_error_and_never_stored(bad):
    changed, errors = settings.update_strict({"agent": {"temperature": bad}})
    assert changed == []
    assert len(errors) == 1 and "agent.temperature" in errors[0]
    assert "NaN" not in _file_text() and "Infinity" not in _file_text()


def test_out_of_range_values_are_errors():
    for section, key, value in [
        ("agent", "temperature", 3.5),
        ("mesh", "camera_hz", 0),
        ("mesh", "camera_hz", -5),
        ("mesh", "camera_hz", 1e9),
        ("mesh", "port", 0),
        ("mesh", "port", 70000),
        ("agent", "max_tokens", 0),
        ("mesh", "camera_hz", "banana"),
    ]:
        changed, errors = settings.update_strict({section: {key: value}})
        assert changed == [], (key, value)
        assert errors and f"{section}.{key}" in errors[0], (key, value, errors)


def test_valid_keys_in_a_mixed_patch_still_apply():
    changed, errors = settings.update_strict({"agent": {"temperature": "NaN", "max_tokens": 512}})
    assert changed == ["agent.max_tokens"]
    assert len(errors) == 1 and "temperature" in errors[0]
    assert settings.get("agent", "max_tokens") == 512


def test_valid_values_round_trip():
    changed, errors = settings.update_strict({"agent": {"temperature": 0.7}})
    assert changed == ["agent.temperature"] and errors == []
    assert settings.get("agent", "temperature") == 0.7
    json.loads(_file_text())  # strictly valid JSON


def test_clearing_with_empty_string_still_works():
    settings.update_strict({"agent": {"temperature": 0.7}})
    changed, errors = settings.update_strict({"agent": {"temperature": ""}})
    assert changed == ["agent.temperature"] and errors == []
    assert settings.get("agent", "temperature") is None


# --- lenient path (env/CLI) keeps degrading, never raises ---------------------


def test_lenient_update_never_raises_and_stores_none():
    changed = settings.update({"agent": {"temperature": "NaN"}})
    assert changed in ([], ["agent.temperature"])
    assert "NaN" not in _file_text()


# --- belt and braces -----------------------------------------------------------


def test_write_file_refuses_nonfinite_outright():
    with pytest.raises(ValueError):
        settings._write_file({"agent": {"temperature": float("nan")}})


def test_poisoned_file_heals_to_defaults():
    settings.SETTINGS_FILE.write_text('{"agent": {"temperature": NaN}}')
    settings._cache = None
    assert settings._read_file() == {}  # treated as corrupt
    assert settings.get("agent", "temperature") in (None, 0.7)  # env default
    # and the next valid write replaces the poison with strict JSON
    settings.update_strict({"agent": {"temperature": 1.0}})
    json.loads(_file_text())


# ============================================================================
# from tests/test_dashboard_settings_typed_values.py
# Q15 remainder: a wrong-TYPED settings value is reported, never transmuted.
# ============================================================================


class TestListKeysRefuseScalars:
    @pytest.mark.parametrize("bad", [5, 1.5, True, {"a": 1}])
    def test_cors_origins_refuses_a_non_list_on_the_strict_path(self, bad: object) -> None:
        with pytest.raises(CoercionError, match="cors_origins"):
            _coerce_strict("security", "cors_origins", bad)

    def test_cors_origins_still_takes_a_comma_string_and_a_list(self) -> None:
        assert _coerce_strict("security", "cors_origins", "http://a, http://b") == ["http://a", "http://b"]
        assert _coerce_strict("security", "cors_origins", ["http://a"]) == ["http://a"]
        assert _coerce_strict("security", "cors_origins", None) == []

    def test_the_lenient_path_degrades_to_the_keys_own_shape(self) -> None:
        # env/CLI writes must not kill startup, but a list key degrading to a
        # SCALAR poisons every consumer that iterates it.
        assert _coerce("security", "cors_origins", 5, strict=False) == []


class TestBooleansRefuseTypos:
    @pytest.mark.parametrize("bad", ["banana", "yess", "enabled", 2])
    def test_a_spelling_that_is_neither_true_nor_false_is_reported(self, bad: object) -> None:
        with pytest.raises(CoercionError, match="boolean"):
            _coerce_strict("runtime", "trust_remote_code", bad)

    @pytest.mark.parametrize(
        ("spelled", "expected"),
        [(True, True), (False, False), ("true", True), ("ON", True), ("0", False), ("no", False), ("", False)],
    )
    def test_every_honest_spelling_still_works(self, spelled: object, expected: bool) -> None:
        assert _coerce_strict("runtime", "trust_remote_code", spelled) is expected


class TestStringKeysRefuseContainers:
    @pytest.mark.parametrize("bad", [{"a": 1}, ["x"], ("y",), {"z"}])
    def test_a_container_is_never_repred_into_a_string(self, bad: object) -> None:
        # str({'a': 1}) as auth_token = the operator locked out by a value
        # nobody can retype.
        with pytest.raises(CoercionError, match="auth_token"):
            _coerce_strict("security", "auth_token", bad)

    def test_numbers_are_still_stringified(self) -> None:
        # Ids get typed unquoted; that is not a bug worth a 422.
        assert _coerce_strict("agent", "model_id", 42) == "42"

    def test_a_plain_string_passes_untouched(self) -> None:
        assert _coerce_strict("security", "auth_token", "s3cret") == "s3cret"


class TestTheErrorReachesTheCaller:
    def test_update_strict_reports_the_refusal_and_stores_nothing(self, tmp_path, monkeypatch) -> None:
        from strands_robots.dashboard import settings as mod

        monkeypatch.setattr(mod, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(mod, "_cache", None)
        changed, errors = mod.update_strict({"security": {"cors_origins": 5}})
        assert changed == []
        assert errors and "cors_origins" in errors[0]
        assert not (tmp_path / "settings.json").exists(), "a refused value must not be persisted"


# ============================================================================
# from tests/test_dashboard_settings_unknown_keys.py
# A setting name the backend does not know must be SAID, not swallowed.
# ============================================================================


def test_unknown_key_inside_a_known_section_is_named() -> None:
    assert settings.unknown_keys({"agent": {"model_id": "x", "modle_id": "typo"}}) == ["agent.modle_id"]


def test_unknown_section_is_named_once_not_per_key() -> None:
    # A whole section that moved should read as one problem, not five.
    assert settings.unknown_keys({"camera": {"fps": 30, "width": 640}}) == ["camera.*"]


def test_a_fully_known_patch_reports_nothing() -> None:
    assert settings.unknown_keys({"agent": {"model_id": "x"}, "mesh": {"connect": ""}}) == []
    assert settings.unknown_keys({}) == []
    assert settings.unknown_keys(None) == []  # type: ignore[arg-type]


def test_names_are_sorted_and_deduped_by_shape() -> None:
    out = settings.unknown_keys({"zzz": {"a": 1}, "agent": {"nope": 1, "alsono": 2}})
    assert out == ["agent.alsono", "agent.nope", "zzz.*"]


def test_junk_shapes_cannot_raise() -> None:
    # A section whose value is not a dict is the caller's problem elsewhere; this
    # helper must not crash the whole apply over it.
    for patch in ({"agent": "not a dict"}, {"agent": None}, {"": {}}, {"x": 5}):  # type: ignore[var-annotated]
        assert isinstance(settings.unknown_keys(patch), list)  # type: ignore[arg-type]


def test_reporting_is_not_enforcement() -> None:
    # The extra name must NOT block the valid keys in the same patch, and must
    # not be stored either.
    changed, errors = settings.update_strict({"agent": {"temperature": 0.4, "nope": 1}})
    assert "agent.temperature" in changed
    assert errors == []
    assert "nope" not in settings.load()["agent"]
    settings.update_strict({"agent": {"temperature": 0.7}})


def test_apply_surfaces_ignored_names(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.load(refresh=True)
    out = config_api.apply({"agent": {"nope": 1}, "camera": {"fps": 30}})
    assert out["ignored"] == ["agent.nope", "camera.*"]
    # ...and the body's own vocabulary is never accused of being a setting.
    out2 = config_api.apply({"env": {"FOO": "bar"}, "agent": {"temperature": 0.5}})
    assert out2["ignored"] == []


def test_ignored_is_always_present_even_with_an_empty_body(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.load(refresh=True)
    out = config_api.apply({})
    assert out["ignored"] == []
    assert out["applied"] == []
