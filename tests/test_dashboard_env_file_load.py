"""Q50: the Env tab wrote a file nothing ever read.

Saving a key exported it into os.environ AND wrote .env, so it worked for the life of that
process — and vanished on the next start, while the settings screen kept listing it as set
because the screen reads the FILE. That is the worst shape a settings screen can have: it
agrees with you about a credential the process does not have.
"""

from __future__ import annotations

from strands_robots.dashboard import config_api


def test_a_file_key_the_process_lacks_is_exported():
    to_set, shadowed = config_api.bootstrap_env({"HF_TOKEN": "hf_abc"}, {})
    assert to_set == {"HF_TOKEN": "hf_abc"}
    assert shadowed == []


def test_the_launch_environment_wins_and_is_reported():
    """`HF_TOKEN=other ./restart_dashboard.sh` is a deliberate statement about THIS run;
    a file written weeks ago must not overrule it — but the operator must be told."""
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
