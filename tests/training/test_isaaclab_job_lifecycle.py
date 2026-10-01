"""An Isaac Lab job ends when it should, is found where it was written, and exports the task asked for.

Three lifecycle faults, each measured on Isaac Lab 3.0 through the tool:

* ``timeout_s`` was enforced only inside ``status()``. An agent that died
  after launching ``timeout_s=30`` left the run training - 438 iterations and
  98 s later it still held the GPU - until someone polled.
* A relative jobs dir (``jobs_dir=`` or ``$STRANDS_ISAACLAB_JOBS``) was handed
  to a wrapper running in the output_dir, so the exit status landed in a
  directory that did not exist and every run ended ``error`` / "killed".
* Export took the newest run of ANY task in output_dir, by mtime: "export
  Isaac-Cartpole" returned the Cartpole-Camera policy with ``success``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from strands_robots.training import _isaaclab_runtime as runtime
from strands_robots.training.isaaclab import RUN_RECORD_FILE, IsaacLabTrainer
from tests.training.test_isaaclab import _poll, _spec, _trainer, fake_python  # noqa: F401


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
    return any(not state.startswith("Z") for state in out)


class TestTheDeadlineHoldsWithoutAPoll:
    def test_the_whole_process_group_is_gone_after_the_deadline(self, tmp_path: Path) -> None:
        # A child that forks a grandchild, like Isaac Lab starting Kit.
        proc = runtime.launch(
            ["/bin/sh", "-c", "sleep 60 & sleep 60"],
            cwd=tmp_path,
            log_path=tmp_path / "log",
            exit_file=tmp_path / "exit_code",
            timeout_s=1,
            timed_out_file=tmp_path / "timed_out",
        )
        deadline = time.monotonic() + 10
        while _group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        proc.wait(timeout=5)
        assert not _group_alive(proc.pid)
        assert (tmp_path / "timed_out").is_file()
        assert not (tmp_path / "exit_code").exists()  # killed, not finished

    def test_a_run_that_ends_first_writes_its_status_and_the_watchdog_leaves(self, tmp_path: Path) -> None:
        proc = runtime.launch(
            ["/bin/true"],
            cwd=tmp_path,
            log_path=tmp_path / "log",
            exit_file=tmp_path / "exit_code",
            timeout_s=30,
            timed_out_file=tmp_path / "timed_out",
        )
        proc.wait(timeout=5)
        assert (tmp_path / "exit_code").read_text().strip() == "0"
        deadline = time.monotonic() + 5
        while _group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _group_alive(proc.pid) and not (tmp_path / "timed_out").exists()

    def test_a_deadline_needs_a_marker_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timed_out_file"):
            runtime.launch(["/bin/true"], cwd=tmp_path, log_path=tmp_path / "l", exit_file=tmp_path / "e", timeout_s=5)

    def test_a_timed_out_job_reports_timeout_to_a_later_poll(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        trainer = _trainer()
        launched = trainer.train(_spec(tmp_path, steps=2, timeout_s=1))
        job = json.loads((trainer._jobs_dir / launched.job_id / "job.json").read_text())
        deadline = time.monotonic() + 10
        while _group_alive(int(job["pid"])) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _group_alive(int(job["pid"])), "the run outlived its deadline with nobody polling"
        result = trainer.status(launched.job_id)
        assert result.status == "error" and result.metrics.get("failure") == "timeout", result


class TestARelativeJobsDirIsResolved:
    def test_a_run_launched_with_a_relative_jobs_dir_succeeds(
        self,
        fake_python: Path,  # noqa: F811
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(runtime.JOBS_DIR_ENV, "rel_jobs")
        assert runtime.default_jobs_dir() == tmp_path / "rel_jobs"
        trainer = _trainer()
        result = _poll(trainer, trainer.train(_spec(tmp_path, steps=2)).job_id)
        assert result.status == "success", result.message


def _run(root: Path, name: str, task: str) -> Path:
    run = root / "logs" / "rsl_rl" / "cartpole" / name
    run.mkdir(parents=True)
    (run / "model_1.pt").write_bytes(b"")
    (run / RUN_RECORD_FILE).write_text(json.dumps({"task": task}))
    return run


class TestExportTakesTheTaskItIsAskedFor:
    def test_the_newest_run_of_the_task_wins_whatever_was_touched_last(self, tmp_path: Path) -> None:
        plain = _run(tmp_path, "2026-09-29_01-00-00_isaaclab-a", "Isaac-Cartpole")
        camera = _run(tmp_path, "2026-09-29_02-00-00_isaaclab-b", "Isaac-Cartpole-Camera")
        os.utime(plain, (time.time() + 60, time.time() + 60))  # e.g. a later play() of the older run
        trainer = IsaacLabTrainer(python="/bin/true", jobs_dir=str(tmp_path / "jobs"))
        assert trainer.latest_checkpoint(str(tmp_path)) == str(camera)
        assert trainer.latest_checkpoint(str(tmp_path), task="Isaac-Cartpole") == str(plain)
        assert trainer.latest_checkpoint(str(tmp_path), task="Isaac-Ant") is None

    def test_export_of_one_task_never_returns_another(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        from tests.training.test_rsl_rl_actor_export import write_rsl_rl_run

        plain = _run(tmp_path, "2026-09-29_01-00-00_isaaclab-a", "Isaac-Cartpole")
        camera = _run(tmp_path, "2026-09-29_02-00-00_isaaclab-b", "Isaac-Cartpole-Camera")
        (plain / "model_1.pt").unlink()
        write_rsl_rl_run(plain, iteration=1)
        trainer = IsaacLabTrainer(python="/bin/true", jobs_dir=str(tmp_path / "jobs"))
        spec = _spec(tmp_path)
        spec.output_dir = str(tmp_path)
        out = trainer.export(spec, str(camera))  # what train_policy hands it: the newest run of any task
        meta = json.loads((Path(out) / "policy_meta.json").read_text())
        assert meta["task"] == "Isaac-Cartpole" and Path(out).parent == plain
        spec.extra = {"task": "Isaac-Ant"}
        with pytest.raises(FileNotFoundError, match="no Isaac-Ant run"):
            trainer.export(spec, str(camera))
        del torch

    def test_the_task_is_read_from_the_job_record_when_the_run_has_none(self, tmp_path: Path) -> None:
        jobs = tmp_path / "jobs"
        (jobs / "isaaclab-20260929-010000-0123456789ab").mkdir(parents=True)
        (jobs / "isaaclab-20260929-010000-0123456789ab" / "job.json").write_text(json.dumps({"task": "Isaac-Ant"}))
        run = tmp_path / "2026-09-29_01-00-00_isaaclab-20260929-010000-0123456789ab"
        run.mkdir()
        assert IsaacLabTrainer(python="/bin/true", jobs_dir=str(jobs)).run_task(run) == "Isaac-Ant"
