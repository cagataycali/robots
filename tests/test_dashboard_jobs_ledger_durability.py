"""The training job ledger is the only record that a run happened.

A training run outlives the page, the process and often the day. The ledger is what
turns it back into a card with a status, a loss curve and an export button, so losing
that file loses the *user's* work in the only sense that matters here: the run keeps
burning GPU hours with nothing on screen that knows about it.

Two ways it used to disappear, both silent:
* ``Path.write_text`` truncates and then writes, so a kill in that window left a half
  file — and the next submit wrote a one-entry list over it, making the loss permanent;
* an unreadable ledger returned ``[]``, which renders exactly like "no jobs yet".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.dashboard import training


@pytest.fixture
def ledger(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "nested" / "train_jobs.json"
    monkeypatch.setattr(training, "JOBS_FILE", path)
    monkeypatch.setattr(training, "_JOBS_PROBLEM", None)
    return path


def _rows(n: int) -> list[dict]:
    return [{"job_id": f"j{i}", "provider": "lerobot_local"} for i in range(n)]


class TestDurableWrite:
    def test_a_write_is_all_or_nothing(self, ledger) -> None:
        training._save_jobs(_rows(2))
        original = ledger.read_text(encoding="utf-8")

        # a serialization that blows up half way through must leave the previous
        # ledger exactly as it was, not a truncated file
        class _Boom:
            def __repr__(self) -> str:
                raise RuntimeError("cannot serialize me")

        def _explode(*a, **k):
            raise RuntimeError("disk went away mid-write")

        import json as real_json

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(real_json, "dump", _explode)
            training._save_jobs(_rows(9))

        assert ledger.read_text(encoding="utf-8") == original
        assert json.loads(original) == _rows(2)

    def test_no_temp_files_are_left_behind(self, ledger) -> None:
        training._save_jobs(_rows(1))
        assert [p.name for p in ledger.parent.iterdir()] == [ledger.name]

    def test_it_creates_its_directory(self, ledger) -> None:
        assert not ledger.parent.exists()
        training._save_jobs(_rows(1))
        assert json.loads(ledger.read_text(encoding="utf-8")) == _rows(1)

    def test_the_ledger_is_capped_but_keeps_the_newest(self, ledger) -> None:
        training._save_jobs(_rows(140))
        kept = json.loads(ledger.read_text(encoding="utf-8"))
        assert len(kept) == 100
        assert kept[-1]["job_id"] == "j139", "the cap must drop the oldest, not the newest"


class TestUnreadableLedgerIsNotSilent:
    def test_a_corrupt_ledger_is_preserved_not_overwritten(self, ledger) -> None:
        ledger.parent.mkdir(parents=True)
        ledger.write_text('[{"job_id": "half-writ', encoding="utf-8")  # truncated

        assert training._load_jobs() == []
        problem = training.jobs_problem()
        assert problem and "could not be read" in problem

        # the bytes still exist somewhere a human can look at them
        kept = list(ledger.parent.glob("train_jobs.json.corrupt-*"))
        assert len(kept) == 1
        assert kept[0].read_text(encoding="utf-8") == '[{"job_id": "half-writ'

    def test_an_empty_history_is_not_reported_as_a_problem(self, ledger) -> None:
        assert training._load_jobs() == []
        assert training.jobs_problem() is None

    def test_a_healthy_read_clears_a_previous_problem(self, ledger) -> None:
        monkey_problem = "stale complaint from an earlier read"
        training._JOBS_PROBLEM = monkey_problem
        training._save_jobs(_rows(1))
        assert training._load_jobs() == _rows(1)
        assert training.jobs_problem() is None

    def test_a_ledger_that_is_not_a_list_says_what_it_is(self, ledger) -> None:
        ledger.parent.mkdir(parents=True)
        ledger.write_text('{"job_id": "not-a-list"}', encoding="utf-8")
        assert training._load_jobs() == []
        assert "dict" in (training.jobs_problem() or "")

    def test_an_unquarantinable_ledger_is_never_overwritten(self, ledger, monkeypatch) -> None:
        # if we cannot even move the bad file aside, saving over it would replace runs
        # we cannot read with the one we just started
        ledger.parent.mkdir(parents=True)
        ledger.write_text("{{{ not json", encoding="utf-8")
        monkeypatch.setattr(training, "_quarantine", lambda p: (_ for _ in ()).throw(OSError("read-only fs")))

        assert training._load_jobs() == []
        assert "refusing to overwrite" in (training.jobs_problem() or "")
        training._save_jobs(_rows(1))
        assert ledger.read_text(encoding="utf-8") == "{{{ not json"

    def test_a_run_recorded_after_a_corrupt_read_still_lands(self, ledger) -> None:
        # the quarantine path is a recovery, not a dead end: once the bad file is out
        # of the way, the next submit must be remembered
        ledger.parent.mkdir(parents=True)
        ledger.write_text("nonsense", encoding="utf-8")
        assert training._load_jobs() == []
        training._save_jobs(_rows(1))
        assert training._load_jobs() == _rows(1)
        assert training.jobs_problem() is None
