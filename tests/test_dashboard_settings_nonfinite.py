"""Q14/Q15: a config write can never brick the config screen.

NaN/inf floats used to pass float() coercion, json.dumps then emitted bare
NaN into settings.json - valid to Python, poison to every browser's
JSON.parse, and the file reloads on each request: the Config screen died
until a shell edited a chmod-600 file. Strict coercion now rejects
non-finite and out-of-range values WITH an error message, _write_file
refuses to serialize them at all, and a previously poisoned file heals to
defaults instead of being served forever.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.dashboard import settings


@pytest.fixture(autouse=True)
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
    changed, errors = settings.update_strict(
        {"agent": {"temperature": "NaN", "max_tokens": 512}}
    )
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
