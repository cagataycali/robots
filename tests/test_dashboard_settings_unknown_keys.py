"""A setting name the backend does not know must be SAID, not swallowed.

settings._update() ``continue``s past an unknown section and an unknown key, so
such a name is neither stored nor reported: no `changed` entry, no coercion
error. The settings drawer builds its status line from exactly those two lists,
so a patch full of unrecognised names - a frontend field renamed on one side
only, a typo, a section that moved - produced the same reassuring "nothing
changed" as re-saving values that were already correct.
"""

from strands_robots.dashboard import config_api, settings


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
