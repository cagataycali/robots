"""Env allowlist/file-load/removal of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_env_allowlist.py, test_dashboard_env_file_load.py, test_dashboard_env_removal.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import os

import pytest

from strands_robots.dashboard import config_api

# ============================================================================
# from tests/test_dashboard_env_allowlist.py
# Q13: the config env upsert is configuration -> code execution unless caged.
# ============================================================================


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


# ============================================================================
# from tests/test_dashboard_env_file_load.py
# Q50: the Env tab wrote a file nothing ever read.
# ============================================================================


def test_a_file_key_the_process_lacks_is_exported():
    to_set, shadowed = config_api.bootstrap_env({"HF_TOKEN": "hf_abc"}, {})
    assert to_set == {"HF_TOKEN": "hf_abc"}
    assert shadowed == []


def test_the_launch_environment_wins_and_is_reported():
    """`HF_TOKEN=other ./restart_dashboard.sh` is a deliberate statement about THIS run;
    a file written weeks ago must not overrule it -- but the operator must be told."""
    to_set, shadowed = config_api.bootstrap_env({"HF_TOKEN": "from_file"}, {"HF_TOKEN": "from_shell"})
    assert to_set == {}
    assert shadowed == ["HF_TOKEN"]


def test_an_identical_value_is_not_a_conflict():
    to_set, shadowed = config_api.bootstrap_env({"A": "1"}, {"A": "1"})
    assert (to_set, shadowed) == ({}, [])


def test_empty_string_in_the_environment_is_still_a_decision():
    """Exporting A= deliberately blanks a var; treating "" as absent would resurrect the file
    value and undo that."""
    to_set, shadowed = config_api.bootstrap_env({"A": "1"}, {"A": ""})
    assert to_set == {}
    assert shadowed == ["A"]


def test_load_env_file_actually_exports(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_zzz\nSTRANDS_ROBOTS_VIDEO_ROOT=/data/vids\n")
    monkeypatch.setattr(config_api, "ENV_FILE", env)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("STRANDS_ROBOTS_VIDEO_ROOT", "/somewhere/else")

    exported, shadowed = config_api.load_env_file()

    import os

    assert os.environ["HF_TOKEN"] == "hf_zzz"
    assert exported == ["HF_TOKEN"]
    # The shell's video root stands, and the file's is reported as ignored rather than applied.
    assert os.environ["STRANDS_ROBOTS_VIDEO_ROOT"] == "/somewhere/else"
    assert shadowed == ["STRANDS_ROBOTS_VIDEO_ROOT"]


def test_env_view_shows_what_the_process_uses(tmp_path, monkeypatch):
    """A row must never display a value nothing is acting on."""
    env = tmp_path / ".env"
    env.write_text("STRANDS_ROBOTS_VIDEO_ROOT=/data/vids\n")
    monkeypatch.setattr(config_api, "ENV_FILE", env)
    monkeypatch.setenv("STRANDS_ROBOTS_VIDEO_ROOT", "/live/root")

    row = next(r for r in config_api.env_view() if r["key"] == "STRANDS_ROBOTS_VIDEO_ROOT")
    assert row["value"] == "/live/root", "the file's value is the one nothing is using"
    assert row["shadowed"] is True
    assert row["in_file"] is True


def test_env_view_secret_stays_masked_when_shadowed(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setattr(config_api, "ENV_FILE", env)
    monkeypatch.setenv("HF_TOKEN", "hf_from_shell_value")

    row = next(r for r in config_api.env_view() if r["key"] == "HF_TOKEN")
    assert "hf_from_shell_value" not in row["value"]
    assert "•" in row["value"]
    assert row["shadowed"] is True


def test_env_view_marks_nothing_when_they_agree(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AWS_PROFILE=lab\n")
    monkeypatch.setattr(config_api, "ENV_FILE", env)
    monkeypatch.setenv("AWS_PROFILE", "lab")

    row = next(r for r in config_api.env_view() if r["key"] == "AWS_PROFILE")
    assert row["shadowed"] is False


# ============================================================================
# from tests/test_dashboard_env_removal.py
# Q75: removing an env var is a REMOVAL, not "set it to empty".
# ============================================================================


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    monkeypatch.setattr(config_api, "ENV_FILE", f)
    return f


# --- the pure split ---------------------------------------------------------


def test_null_means_remove_and_empty_string_still_means_empty():
    updates, deletions = config_api.split_env_patch({"STRANDS_A": "1", "STRANDS_B": None, "STRANDS_C": ""})
    assert updates == {"STRANDS_A": "1", "STRANDS_C": ""}
    assert deletions == ["STRANDS_B"]


def test_split_is_pure_and_keeps_empty_string_distinct_from_absent():
    # The distinction IS the bug: if these two collapsed, "remove" would be unexpressible again.
    _, only_empty = config_api.split_env_patch({"STRANDS_X": ""})
    assert only_empty == []
    _, only_null = config_api.split_env_patch({"STRANDS_X": None})
    assert only_null == ["STRANDS_X"]


def test_split_strips_key_whitespace_like_the_write_path():
    updates, deletions = config_api.split_env_patch({" STRANDS_A ": "1", " STRANDS_B ": None})
    assert updates == {"STRANDS_A": "1"}
    assert deletions == ["STRANDS_B"]


# --- the file operation -----------------------------------------------------


def test_delete_removes_the_line_and_keeps_everything_else(env_file):
    env_file.write_text("# a hand-written comment\nSTRANDS_KEEP=1\n\nSTRANDS_GONE=secret\nSTRANDS_ALSO_KEEP=2\n")
    removed = config_api.delete_env_keys(["STRANDS_GONE"])
    assert removed == ["STRANDS_GONE"]
    text = env_file.read_text()
    assert "STRANDS_GONE" not in text
    # Comments and blank lines survive: this file is edited by hand too.
    assert "# a hand-written comment" in text
    assert "STRANDS_KEEP=1" in text and "STRANDS_ALSO_KEEP=2" in text
    assert text.endswith("\n")


def test_delete_refuses_keys_it_could_not_have_written(env_file):
    # Same allowlist as the write path: a caller who cannot SET a variable must not be able to
    # delete one either -- otherwise this is a hole for removing someone else's PATH or HOME.
    env_file.write_text("PATH=/usr/bin\nSTRANDS_OK=1\n")
    assert config_api.delete_env_keys(["PATH"]) == []
    assert "PATH=/usr/bin" in env_file.read_text()
    assert config_api.delete_env_keys(["STRANDS_OK"]) == ["STRANDS_OK"]


def test_deleting_an_absent_key_is_silent_and_touches_nothing(env_file):
    env_file.write_text("STRANDS_A=1\n")
    before = env_file.read_text()
    assert config_api.delete_env_keys(["STRANDS_NOPE"]) == []
    assert env_file.read_text() == before


def test_delete_with_no_file_does_not_create_one(env_file):
    assert not env_file.exists()
    assert config_api.delete_env_keys(["STRANDS_A"]) == []
    assert not env_file.exists()


# --- through apply() --------------------------------------------------------


def test_apply_reports_removal_separately_from_writes(env_file, monkeypatch):
    env_file.write_text("STRANDS_GONE=1\n")
    monkeypatch.setitem(os.environ, "STRANDS_GONE", "1")
    r = config_api.apply({"env": {"STRANDS_GONE": None, "STRANDS_NEW": "2"}})
    assert r["env_removed"] == ["STRANDS_GONE"]
    assert "STRANDS_NEW" in r["env_written"]
    # The live process must not become the one place the deleted variable is still in effect.
    assert "STRANDS_GONE" not in os.environ
    assert "STRANDS_GONE" not in env_file.read_text()


def test_apply_still_writes_an_explicit_empty_string(env_file):
    # Empty remains a legitimate VALUE -- some tools want KEY= to mean "configured, blank".
    config_api.apply({"env": {"STRANDS_BLANK": ""}})
    assert "STRANDS_BLANK=" in env_file.read_text()


def test_apply_refuses_to_delete_an_unmanaged_key(env_file):
    env_file.write_text("HOME=/Users/nobody\n")
    r = config_api.apply({"env": {"HOME": None}})
    assert r["env_removed"] == []
    assert any("not dashboard-managed" in e for e in r.get("errors", []))
    assert "HOME=/Users/nobody" in env_file.read_text()


def test_a_masked_value_is_still_never_written_back(env_file):
    env_file.write_text("STRANDS_SECRET=real\n")
    r = config_api.apply({"env": {"STRANDS_SECRET": "re\u2026al"}})
    assert r["skipped_masked"] == ["STRANDS_SECRET"]
    assert "STRANDS_SECRET=real" in env_file.read_text()
