"""Q13: the config env upsert is configuration -> code execution unless caged.

.env is read by every process the dashboard spawns. PATH=/tmp/evil was
accepted (hijacking python/ffmpeg for every child), and a newline in a
VALUE wrote a second variable on its own line - defeating any key
allow-list by itself. Both halves are now refused, at apply() AND inside
upsert_env_file() so no future caller can skip the check.
"""

from __future__ import annotations

import os

import pytest

from strands_robots.dashboard import config_api


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config_api, "ENV_FILE", tmp_path / ".env")
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    # apply() deliberately exports what it writes into os.environ, so an
    # un-snapshotted test here leaks STRANDS_*/HF_* into every LATER test in
    # the session (it poisoned tests/test_dashboard_ws_chat_frames.py once).
    before = dict(os.environ)
    yield
    dsettings._cache = None
    for key in set(os.environ) - set(before):
        del os.environ[key]
    os.environ.update(before)


def _env_text() -> str:
    return config_api.ENV_FILE.read_text() if config_api.ENV_FILE.exists() else ""


# --- key allow-list ------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "PATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HOME",
        "SHELL",
        "ZENOH_CONNECT",
        "NODE_OPTIONS",
    ],
)
def test_hostile_keys_are_refused_and_reported(key):
    out = config_api.apply({"env": {key: "/tmp/evil"}})
    assert out["env_written"] == []
    assert out["errors"] and "not dashboard-managed" in out["errors"][0]
    assert key not in _env_text()


@pytest.mark.parametrize(
    "key",
    [
        "STRANDS_MESH_LOCAL_DEV",
        "DASHBOARD_AUTH_TOKEN",
        "VOICE_MODEL",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "AWS_REGION",
        "AWS_PROFILE",
    ],
)
def test_dashboard_owned_keys_still_work(key, monkeypatch):
    monkeypatch.delenv(key, raising=False)
    out = config_api.apply({"env": {key: "value1"}})
    assert out["env_written"] == [key], out["errors"]
    assert f"{key}=value1" in _env_text()


# --- value injection ------------------------------------------------------------


def test_newline_in_value_cannot_smuggle_a_second_variable():
    out = config_api.apply({"env": {"HF_TOKEN": "a\nZENOH_CONNECT=tcp/evil:7447"}})
    assert out["env_written"] == []
    assert out["errors"] and "control characters" in out["errors"][0]
    assert "ZENOH_CONNECT" not in _env_text()


@pytest.mark.parametrize("value", ["a\rb", "a\x00b", "a\tb"])
def test_other_control_characters_are_refused(value):
    out = config_api.apply({"env": {"HF_TOKEN": value}})
    assert out["env_written"] == [] and out["errors"]


def test_value_length_is_capped():
    out = config_api.apply({"env": {"HF_TOKEN": "x" * 5000}})
    assert out["env_written"] == []
    assert out["errors"] and "exceeds" in out["errors"][0]


def test_mixed_batch_writes_only_the_allowed_entries():
    out = config_api.apply({"env": {"PATH": "/tmp/evil", "STRANDS_MESH_MULTICAST": "true"}})
    assert out["env_written"] == ["STRANDS_MESH_MULTICAST"]
    assert len(out["errors"]) == 1 and "PATH" in out["errors"][0]
    text = _env_text()
    assert "STRANDS_MESH_MULTICAST=true" in text and "PATH" not in text


# --- defense in depth: the writer itself refuses --------------------------------


def test_upsert_env_file_raises_on_disallowed_key():
    with pytest.raises(ValueError, match="not dashboard-managed"):
        config_api.upsert_env_file({"PATH": "/tmp/evil"})
    assert _env_text() == ""


def test_upsert_env_file_raises_on_newline_value():
    with pytest.raises(ValueError, match="control characters"):
        config_api.upsert_env_file({"HF_TOKEN": "a\nEVIL=1"})
    assert _env_text() == ""


def test_masked_values_are_still_skipped_not_errors():
    config_api.apply({"env": {"HF_TOKEN": "realtoken123"}})
    out = config_api.apply({"env": {"HF_TOKEN": "rea••••••23"}})
    assert out["skipped_masked"] == ["HF_TOKEN"] and not out["errors"]
    assert "HF_TOKEN=realtoken123" in _env_text()
