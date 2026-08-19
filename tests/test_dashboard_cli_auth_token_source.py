"""Where the dashboard's bearer token came FROM is itself a security fact.

JOURNEYS #15 (the last open row of the end-to-end journey audit): a token passed
as ``--auth-token`` is readable by every local user for the whole life of the
process — this machine's own audit lifted one out of ``ps eww``. The file form
exists and ``restart_dashboard.sh`` uses it, but nothing on the code side ever
*said* so at runtime: startup printed ``auth: bearer token required``, which
reads like the security question is settled.

These tests pin the three things that make the difference honest:

* the argv form works AND is warned about, with the fix named;
* the file form works and is NOT warned about (a warning that fires either way
  teaches the operator to ignore it);
* a missing or empty token file REFUSES to start rather than starting open —
  the operator asked for auth and silence would hand them the opposite.

The token VALUE must never be printed by any of it, which is asserted
separately: a warning that echoes the secret into the logfile it is warning
about would defeat itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from strands_robots.dashboard import cli

TOKEN = "s3cret-token-value-not-to-be-printed"


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the settings module at a temp file, never the operator's own."""
    from strands_robots.dashboard import settings

    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.load(refresh=True)
    yield settings
    settings.load(refresh=True)


@pytest.fixture
def stub_server(monkeypatch: pytest.MonkeyPatch):
    """Let ``cli.main()`` reach the serving call without building anything real.

    ``main()`` imports uvicorn / MeshBridge / create_app INSIDE the function, so
    the patches must land on the source modules rather than on ``cli``. Nothing
    real is constructed on purpose: building the app opens a mesh session, and a
    test suite that joins the live fleet is its own incident (BUGS.md Q30/Q32).

    Returns the list that records whether the server was reached, so a test can
    assert on "started" as well as on what was printed.
    """
    import uvicorn

    from strands_robots.dashboard import mesh_bridge, server

    started: list[dict] = []
    monkeypatch.setattr(server, "create_app", lambda *a, **k: object())
    monkeypatch.setattr(mesh_bridge, "MeshBridge", lambda *a, **k: object())
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(k))
    return started


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["dashboard", *argv])
    cli.main()


class TestTheArgvFormIsWarnedAbout:
    def test_a_token_on_the_command_line_says_so_and_names_the_fix(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        isolated_settings, stub_server,
    ) -> None:
        _main(monkeypatch, "--auth-token", TOKEN, "--port", "8099")
        assert stub_server, "the CLI must reach the server after printing its banner"
        out = capsys.readouterr().out

        assert "auth: bearer token required" in out, "the token must still take effect"
        assert "command line" in out and "ps" in out, (
            f"no warning that the token is in ps output: {out!r}"
        )
        assert "--auth-token-file" in out, "a warning without the fix is just anxiety"

    def test_the_warning_does_not_print_the_secret_it_warns_about(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        isolated_settings, stub_server,
    ) -> None:
        _main(monkeypatch, "--auth-token", TOKEN, "--port", "8099")
        out = capsys.readouterr().out

        assert TOKEN not in out, (
            "the startup banner echoed the token into stdout - which is usually "
            "redirected to a logfile, so the warning would create a second copy "
            "of the leak it is warning about"
        )


class TestTheFileFormIsTheQuietPath:
    def test_a_token_file_applies_the_token_with_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        tmp_path: Path, isolated_settings, stub_server,
    ) -> None:
        token_file = tmp_path / "token.txt"
        token_file.write_text(f"{TOKEN}\n")

        _main(monkeypatch, "--auth-token-file", str(token_file), "--port", "8099")
        out = capsys.readouterr().out

        assert "auth: bearer token required" in out
        assert "command line" not in out, (
            f"the file form was warned about as if it were argv: {out!r}. A warning "
            f"that fires on the SAFE path too teaches the operator to ignore it."
        )
        assert isolated_settings.load(refresh=True)["security"]["auth_token"] == TOKEN

    def test_surrounding_whitespace_and_extra_lines_are_stripped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_settings, stub_server
    ) -> None:
        # A hand-made token file usually ends with a newline, and a copy-paste
        # can leave a trailing blank line. A token that differs from the one the
        # operator pasted by an invisible character fails as "wrong password".
        token_file = tmp_path / "token.txt"
        token_file.write_text(f"  {TOKEN}  \n\n")

        _main(monkeypatch, "--auth-token-file", str(token_file), "--port", "8099")

        assert isolated_settings.load(refresh=True)["security"]["auth_token"] == TOKEN


class TestAnUnusableTokenFileRefusesToStart:
    @pytest.mark.parametrize(
        ("name", "contents"),
        [("missing.txt", None), ("empty.txt", ""), ("blank.txt", "   \n\n")],
    )
    def test_it_exits_instead_of_starting_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_settings,
        capsys: pytest.CaptureFixture[str], stub_server, name: str, contents: str | None,
    ) -> None:
        path = tmp_path / name
        if contents is not None:
            path.write_text(contents)
        started = stub_server

        with pytest.raises(SystemExit) as exc:
            _main(monkeypatch, "--auth-token-file", str(path))

        assert exc.value.code != 0
        assert not started, (
            "the dashboard started with NO auth after being asked for a token file "
            "it could not read - the operator's intent was authentication, and "
            "starting open is the opposite of the refusal they can see"
        )
        assert "auth-token-file" in capsys.readouterr().err
