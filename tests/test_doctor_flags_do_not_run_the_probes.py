"""``strands-robots doctor --help`` / ``--list`` / a typo must not run the checks.

``main()`` used to have no parser: every argument was ignored and the full
check ran, so ``--help`` printed the doctor report and exited with the doctor's
status. Now the parser answers first, and only a bare invocation probes.
"""

from __future__ import annotations

import pytest

from strands_robots import doctor


@pytest.fixture
def probes_that_must_not_run(monkeypatch):
    calls: list[str] = []

    def _tripwire():
        calls.append("ran")
        raise AssertionError("a probe ran")

    monkeypatch.setattr(doctor, "check_python_version", _tripwire)
    monkeypatch.setattr(doctor, "CHECKS", (("Python", "check_python_version"),))
    return calls


class TestFlagsAnswerBeforeAnyProbe:
    def test_help_exits_zero_with_usage_and_runs_nothing(self, probes_that_must_not_run, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            doctor.main(["--help"])
        assert exc.value.code == 0
        assert "usage: strands-robots doctor" in capsys.readouterr().out
        assert probes_that_must_not_run == []

    def test_list_prints_the_check_names_and_runs_nothing(self, probes_that_must_not_run, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            doctor.main(["--list"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.splitlines() == ["Python"]
        assert probes_that_must_not_run == []

    def test_an_unknown_argument_is_refused_not_ignored(self, probes_that_must_not_run, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            doctor.main(["--json"])
        assert exc.value.code == 2
        assert "unrecognized arguments: --json" in capsys.readouterr().err
        assert probes_that_must_not_run == []


def test_a_bare_invocation_still_runs_every_row(monkeypatch) -> None:
    ran: list[str] = []

    def probe() -> str:
        ran.append("ran")
        return doctor._pass("Python")

    monkeypatch.setattr(doctor, "check_python_version", probe)
    monkeypatch.setattr(doctor, "CHECKS", (("Python", "check_python_version"),))
    with pytest.raises(SystemExit) as exc:
        doctor.main([])
    assert exc.value.code == 0
    assert ran == ["ran"]
