"""Q75: removing an env var is a REMOVAL, not "set it to empty".

The Env tab had no delete button, so the only gesture available for "take this key out" was clearing
the field — which wrote ``KEY=`` and exported ``""`` into the live process. Set-and-empty is not
absent: ``os.getenv("X", default)`` returns ``""``, an empty token authenticates as an empty token
instead of falling back to anonymous, and the operator who did it believes the variable is gone.
"""
from __future__ import annotations

import os

import pytest

from strands_robots.dashboard import config_api


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    monkeypatch.setattr(config_api, "ENV_FILE", f)
    return f


# --- the pure split ---------------------------------------------------------

def test_null_means_remove_and_empty_string_still_means_empty():
    updates, deletions = config_api.split_env_patch(
        {"STRANDS_A": "1", "STRANDS_B": None, "STRANDS_C": ""}
    )
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
    env_file.write_text(
        "# a hand-written comment\n"
        "STRANDS_KEEP=1\n"
        "\n"
        "STRANDS_GONE=secret\n"
        "STRANDS_ALSO_KEEP=2\n"
    )
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
    # delete one either — otherwise this is a hole for removing someone else's PATH or HOME.
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
    # Empty remains a legitimate VALUE — some tools want KEY= to mean "configured, blank".
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
