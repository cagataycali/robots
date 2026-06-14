"""Tests for the ``python -m strands_robots <command>`` entry point.

Covers the top-level CLI dispatcher in ``strands_robots/__main__.py``:
argv handling, the ``doctor`` sub-command hand-off, unknown-command and
no-command error paths, and their exit codes. The dispatcher is pure
argv plumbing, so every branch is exercised by monkeypatching ``sys.argv``
and stubbing the ``doctor`` entry point - no real diagnostics run.
"""

from __future__ import annotations

import pytest

from strands_robots.__main__ import main


class TestMainDispatch:
    """main() - command routing and argv normalisation."""

    def test_no_command_prints_usage_and_exits_1(self, monkeypatch, capsys) -> None:
        """Bare ``python -m strands_robots`` should print usage and exit 1."""
        monkeypatch.setattr("sys.argv", ["strands_robots"])

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Usage: python -m strands_robots <command>" in out
        assert "doctor" in out

    def test_unknown_command_prints_error_and_exits_1(self, monkeypatch, capsys) -> None:
        """An unrecognised command should report it and exit 1."""
        monkeypatch.setattr("sys.argv", ["strands_robots", "bogus"])

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Unknown command: bogus" in out
        assert "Available commands: doctor" in out

    def test_doctor_command_dispatches_to_doctor_main(self, monkeypatch) -> None:
        """``doctor`` should call doctor.main() exactly once."""
        calls: list[bool] = []
        monkeypatch.setattr(
            "strands_robots.doctor.main",
            lambda: calls.append(True),
        )
        monkeypatch.setattr("sys.argv", ["strands_robots", "doctor"])

        main()

        assert calls == [True]

    def test_doctor_command_strips_subcommand_from_argv(self, monkeypatch) -> None:
        """The dispatched command name must be removed so sub-parsers see clean args.

        After dispatch, argv[0] is preserved and the ``doctor`` token is gone,
        with any trailing flags forwarded to the sub-command intact.
        """
        seen_argv: list[list[str]] = []

        def fake_doctor_main() -> None:
            import sys

            seen_argv.append(list(sys.argv))

        monkeypatch.setattr("strands_robots.doctor.main", fake_doctor_main)
        monkeypatch.setattr("sys.argv", ["strands_robots", "doctor", "--verbose", "extra"])

        main()

        assert seen_argv == [["strands_robots", "--verbose", "extra"]]
