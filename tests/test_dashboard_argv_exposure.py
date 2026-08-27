"""The dashboard notices its own credential sitting in argv (and stays quiet otherwise)."""

from strands_robots.dashboard.argv_exposure import argv_token_notice, token_flag_in_argv

BASE = ["-m", "strands_robots", "dashboard", "--port", "8090"]


def test_the_live_command_line_is_flagged() -> None:
    # Exactly what `pgrep -fl` showed for pid 2519 on cagatay's Mac.
    argv = BASE + ["--auth-token", "kDD6toTMVDwOXYn51XfDI0vNnKGC4tSM", "--force"]
    assert token_flag_in_argv(argv) == "--auth-token"
    notice = argv_token_notice(argv)
    assert notice is not None
    assert notice["kind"] == "token_in_argv"
    assert "ps" in notice["text"]
    assert "--auth-token-file" in notice["remedy"]
    # The notice must never carry the secret it is complaining about.
    assert "kDD6toTMVDwOXYn51XfDI0vNnKGC4tSM" not in str(notice)


def test_equals_form_is_the_same_exposure() -> None:
    assert token_flag_in_argv(BASE + ["--auth-token=sekrit"]) == "--auth-token"


def test_the_remedy_is_not_flagged_as_the_problem() -> None:
    # --auth-token-file starts with the same characters. Flagging it would teach the operator to
    # ignore this warning, which costs more than the warning is worth.
    assert token_flag_in_argv(BASE + ["--auth-token-file", "/tmp/tok"]) is None
    assert argv_token_notice(BASE + ["--auth-token-file", "/tmp/tok"]) is None


def test_quiet_when_there_is_nothing_to_say() -> None:
    assert argv_token_notice(BASE) is None
    assert argv_token_notice([]) is None
    assert argv_token_notice(None) is None
    # A token from settings or DASHBOARD_AUTH_TOKEN is not in argv and not this warning's business.
    assert argv_token_notice(BASE + ["--local-dev"]) is None


def test_a_valueless_flag_is_not_an_exposure() -> None:
    # argparse would have refused that start; inventing a warning about it is noise.
    assert token_flag_in_argv(BASE + ["--auth-token"]) is None
    assert token_flag_in_argv(BASE + ["--auth-token", "--force"]) is None


def test_the_notice_reaches_the_config_document(monkeypatch) -> None:
    """It must ride the AUTHENTICATED endpoint, and vanish when the posture is clean."""
    from strands_robots.dashboard import config_api

    monkeypatch.setattr("sys.argv", ["-m", "strands_robots", "dashboard", "--auth-token", "secret-value", "--force"])
    doc = config_api.snapshot()
    assert doc["security"]["notice"]["kind"] == "token_in_argv"
    assert "secret-value" not in str(doc["security"])

    monkeypatch.setattr("sys.argv", ["-m", "strands_robots", "dashboard", "--auth-token-file", "/tmp/tok"])
    # ABSENT, not null: a screen that renders on presence shows nothing at all.
    assert "notice" not in config_api.snapshot()["security"]
