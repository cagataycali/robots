"""The ``isaaclab`` trainer drives a separate Isaac Lab interpreter, and reads its log.

No GPU and no Isaac Lab: the interpreter is a fake ``python`` script that prints
an rsl_rl-shaped log, writes a ``model_<it>.pt`` into the run directory Isaac Lab
would use, and exits with a chosen status. What is under test is everything on
this side of ``argv``: the preflight, the argv built from a spec, the job record,
the verdict read back from the log and the exit status, the timeout that stops a
run, and the agent tool's envelope around both.
"""

from __future__ import annotations

import json
import re
import stat
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.train_policy import train_policy
from strands_robots.training import TrainSpec, create_trainer, list_trainers
from strands_robots.training import _isaaclab_runtime as runtime
from strands_robots.training.isaaclab import IsaacLabTrainer, latest_model, parse_rsl_rl_log

# One rsl_rl iteration block, as Isaac Lab 3.0.0rc1 prints it (ANSI bold kept).
_ITERATION = """\
################################################################################
\x1b[1m                            Learning iteration {it}/{total}                            \x1b[0m

                            Total steps: {total_steps}
                       Steps per second: {sps}
                            Mean reward: {reward}
                    Mean episode length: 10.34
"""

# The fake interpreter. It honours ``-m isaaclab train``, records its argv, and
# behaves per $FAKE_ISAACLAB_MODE: ok | fail | hang | slow.
_FAKE = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys, time
    from pathlib import Path

    argv = sys.argv[1:]
    assert argv[:3] == ["-m", "isaaclab", "train"], argv
    Path(os.environ["FAKE_ISAACLAB_ARGV"]).write_text(json.dumps(argv))
    flags = dict(zip(argv[3::2], argv[4::2]))
    mode = os.environ.get("FAKE_ISAACLAB_MODE", "ok")
    total = int(flags["--max_iterations"])
    root = Path.cwd() / "logs" / "rsl_rl" / "cartpole"
    run = root / ("2026-09-29_01-00-00_" + flags["--run_name"])
    run.mkdir(parents=True)
    print("[INFO] Logging experiment in directory: " + str(root), flush=True)
    block = {block!r}
    for it in range(total):
        print(block.format(it=it, total=total, total_steps=(it + 1) * 1000, sps=1000 + it, reward=0.1 * it), flush=True)
        (run / ("model_%d.pt" % it)).write_bytes(b"")
        if mode == "slow":
            time.sleep(0.2)
    if mode == "hang":
        time.sleep(600)
    if mode == "fail":
        print("Traceback (most recent call last):\\nRuntimeError: CUDA error", flush=True)
        sys.exit(3)
    print("Training time: 1.25 seconds", flush=True)
    """
)


@pytest.fixture
def fake_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake Isaac Lab interpreter, named by ISAACLAB_PYTHON, with the EULA accepted."""
    import sys

    script = tmp_path / "isaaclab-venv" / "bin" / "python"
    script.parent.mkdir(parents=True)
    script.write_text(_FAKE.format(python=sys.executable, block=_ITERATION))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(runtime.ISAACLAB_PYTHON_ENV, str(script))
    monkeypatch.setenv(runtime.EULA_ENV, "YES")
    monkeypatch.setenv(runtime.JOBS_DIR_ENV, str(tmp_path / "jobs"))
    monkeypatch.setenv("FAKE_ISAACLAB_ARGV", str(tmp_path / "argv.json"))
    return script


def _spec(tmp_path: Path, steps: int = 3, **extra: Any) -> TrainSpec:
    return TrainSpec(output_dir=str(tmp_path / "out"), steps=steps, seed=7, extra={"task": "Isaac-Cartpole", **extra})


def _trainer(**kwargs: Any) -> IsaacLabTrainer:
    return IsaacLabTrainer(poll_interval_s=0.05, **kwargs)


def _poll(trainer: IsaacLabTrainer, job_id: str, *, until: float = 30.0) -> Any:
    deadline = time.monotonic() + until
    result = trainer.status(job_id)
    while result.status == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        result = trainer.status(job_id)
    return result


def _json_block(envelope: dict[str, Any]) -> dict[str, Any]:
    return next(item["json"] for item in envelope["content"] if "json" in item)


class TestTheProviderIsRegistered:
    def test_list_and_create_resolve_it(self) -> None:
        assert "isaaclab" in list_trainers()
        assert isinstance(create_trainer("isaaclab"), IsaacLabTrainer)

    def test_it_reads_no_dataset(self) -> None:
        assert IsaacLabTrainer.requires_dataset is False


class TestAMissingRuntimeIsReportedBeforeAnythingLaunches:
    def test_no_interpreter_names_the_variable_and_the_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(runtime.ISAACLAB_PYTHON_ENV, raising=False)
        monkeypatch.setenv(runtime.EULA_ENV, "YES")
        problems = _trainer().validate(_spec(tmp_path))
        assert len(problems) == 1, problems
        assert "ISAACLAB_PYTHON" in problems[0] and "docs/learn/training/isaaclab.md" in problems[0]

    def test_an_interpreter_that_does_not_exist_is_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(runtime.EULA_ENV, "YES")
        missing = str(tmp_path / "nope" / "python")
        problems = _trainer(python=missing).validate(_spec(tmp_path))
        assert problems == [f"isaaclab: ISAACLAB_PYTHON={missing!r} does not exist or is not a file"]

    def test_an_unaccepted_eula_is_refused_not_accepted_for_the_operator(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(runtime.EULA_ENV)
        problems = _trainer().validate(_spec(tmp_path))
        assert len(problems) == 1 and "OMNI_KIT_ACCEPT_EULA=YES" in problems[0], problems
        result = _trainer().train(_spec(tmp_path))
        assert result.status == "error" and not (tmp_path / "jobs").exists()

    def test_the_tool_reports_it_without_asking_for_a_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(runtime.ISAACLAB_PYTHON_ENV, raising=False)
        envelope = train_policy(
            action="validate", provider="isaaclab", output_dir=str(tmp_path), extra={"task": "Isaac-Cartpole"}
        )
        text = envelope["content"][0]["text"]
        assert envelope["status"] == "error"
        assert "ISAACLAB_PYTHON" in text and "data source" not in text


class TestTheSpecIsGraded:
    @pytest.mark.parametrize(
        ("extra", "needle"),
        [
            ({}, "extra['task'] is required"),
            ({"task": "--help"}, "extra['task'] must be an Isaac Lab task id"),
            ({"task": "Isaac Cartpole"}, "extra['task'] must be an Isaac Lab task id"),
            ({"task": "Isaac-Cartpole", "num_envs": 0}, "extra['num_envs'] must be a positive integer"),
            ({"task": "Isaac-Cartpole", "num_envs": True}, "extra['num_envs'] must be a positive integer"),
            ({"task": "Isaac-Cartpole", "physics": "physx --x"}, "extra['physics'] must be a physics preset"),
            ({"task": "Isaac-Cartpole", "rl_library": "skrl"}, "extra['rl_library'] 'skrl' is not supported"),
            ({"task": "Isaac-Cartpole", "wait": "yes"}, "extra['wait'] must be a boolean"),
            ({"task": "Isaac-Cartpole", "timeout_s": 0}, "extra['timeout_s'] must be"),
            ({"task": "Isaac-Cartpole", "python": "/bin/sh"}, "extra key(s) ['python'] are not read"),
        ],
    )
    def test_an_unusable_extra_is_refused_by_name(
        self, fake_python: Path, tmp_path: Path, extra: dict[str, Any], needle: str
    ) -> None:
        spec = TrainSpec(output_dir=str(tmp_path / "out"), steps=3, extra=extra)
        problems = _trainer().validate(spec)
        assert any(needle in p for p in problems), problems

    def test_resume_is_refused_rather_than_silently_starting_over(self, fake_python: Path, tmp_path: Path) -> None:
        spec = _spec(tmp_path)
        spec.resume = True
        assert any("resuming a run is not supported" in p for p in _trainer().validate(spec))

    def test_a_valid_spec_has_no_problems(self, fake_python: Path, tmp_path: Path) -> None:
        assert _trainer().validate(_spec(tmp_path, num_envs=64, physics="newton_mjwarp")) == []


class TestTheArgvCarriesTheSpec:
    def test_every_forwarded_field_reaches_its_flag(self, fake_python: Path, tmp_path: Path) -> None:
        spec = _spec(tmp_path, steps=50, num_envs=4096, physics="isaacsim_physx")
        spec.learning_rate = 3e-4
        cmd = _trainer().build_command(spec, "isaaclab-20260929-010000-0123456789ab")
        assert cmd[:4] == [str(fake_python), "-m", "isaaclab", "train"]
        flags = dict(zip(cmd[4::2], cmd[5::2], strict=False))
        assert flags["--task"] == "Isaac-Cartpole"
        assert flags["--max_iterations"] == "50"
        assert flags["--num_envs"] == "4096"
        assert flags["--seed"] == "7"
        assert flags["--rl_library"] == "rsl_rl"
        assert flags["--visualizer"] == "none"
        assert flags["--run_name"] == "isaaclab-20260929-010000-0123456789ab"
        assert "physics=isaacsim_physx" in cmd
        assert "agent.algorithm.learning_rate=0.0003" in cmd

    def test_an_unstated_knob_is_left_to_the_task(self, fake_python: Path, tmp_path: Path) -> None:
        spec = _spec(tmp_path)
        spec.seed = None
        cmd = _trainer().build_command(spec, "isaaclab-20260929-010000-0123456789ab")
        assert "--num_envs" not in cmd and "--seed" not in cmd
        assert not [token for token in cmd if token.startswith(("physics=", "agent."))]


class TestARunIsJudgedFromItsLogAndExitStatus:
    def test_a_finished_run_is_success_with_its_checkpoint(self, fake_python: Path, tmp_path: Path) -> None:
        trainer = _trainer()
        launched = trainer.train(_spec(tmp_path, steps=4))
        assert launched.status == "running" and launched.job_id.startswith("isaaclab-")
        result = _poll(trainer, launched.job_id)
        assert result.status == "success", result.message
        assert result.metrics["latest_iteration"] == 3
        assert result.metrics["first_reward"] == 0.0 and result.metrics["latest_reward"] == pytest.approx(0.3)
        assert result.metrics["learning"] is True
        assert result.metrics["steps_per_s"] == 1003 and result.metrics["total_steps"] == 4000
        assert result.metrics["training_time_s"] == 1.25 and result.metrics["exit_code"] == 0
        assert result.checkpoint_dir and result.checkpoint_dir.endswith("_" + launched.job_id)
        assert result.metrics["latest_model"] == str(Path(result.checkpoint_dir) / "model_3.pt")
        assert trainer.latest_checkpoint(str(tmp_path / "out")) == result.checkpoint_dir

    def test_a_fresh_trainer_in_another_call_polls_the_same_job(self, fake_python: Path, tmp_path: Path) -> None:
        launched = _trainer().train(_spec(tmp_path))
        assert _poll(_trainer(), launched.job_id).status == "success"

    def test_the_child_does_not_see_this_interpreters_packages(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYTHONPATH", str(tmp_path / "strands-site-packages"))
        assert "PYTHONPATH" not in runtime.child_env()
        assert _poll(_trainer(), _trainer().train(_spec(tmp_path)).job_id).status == "success"

    def test_a_crashed_run_is_an_error_with_the_log_tail(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "fail")
        trainer = _trainer()
        result = _poll(trainer, trainer.train(_spec(tmp_path)).job_id)
        assert result.status == "error"
        assert "exited 3" in result.message and "CUDA error" in result.message
        assert result.metrics["exit_code"] == 3

    def test_wait_blocks_until_the_verdict(self, fake_python: Path, tmp_path: Path) -> None:
        result = _trainer().train(_spec(tmp_path, wait=True))
        assert result.status == "success" and result.metrics["latest_iteration"] == 2

    def test_a_run_past_its_timeout_is_stopped_and_reported(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        started = time.monotonic()
        result = _trainer().train(_spec(tmp_path, wait=True, timeout_s=1.5))
        assert result.status == "error" and "timeout_s" in result.message
        assert time.monotonic() - started < 30
        assert not runtime.process_alive(result.metrics["pid"])

    def test_a_killed_run_has_no_exit_status_and_is_an_error(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        trainer = _trainer()
        launched = trainer.train(_spec(tmp_path))
        time.sleep(0.5)
        runtime.terminate(launched.metrics["pid"], grace_s=5)
        result = _poll(trainer, launched.job_id)
        assert result.status == "error" and "without an exit status" in result.message

    def test_a_running_job_reports_progress(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        trainer = _trainer()
        launched = trainer.train(_spec(tmp_path, steps=2))
        try:
            deadline = time.monotonic() + 20
            result = trainer.status(launched.job_id)
            while result.metrics.get("latest_iteration") != 1 and time.monotonic() < deadline:
                time.sleep(0.05)
                result = trainer.status(launched.job_id)
            assert result.status == "running" and result.metrics["liveness_ok"] is True
            assert result.message == "iteration 1/2"
        finally:
            runtime.terminate(launched.metrics["pid"], grace_s=5)

    @pytest.mark.parametrize("job_id", ["../../etc", "isaaclab-x", "", "isaaclab-20260929-010000-0123456789ab/.."])
    def test_a_job_id_that_is_not_one_is_refused(self, fake_python: Path, job_id: str) -> None:
        result = _trainer().status(job_id)
        assert result.status == "error" and "is not an Isaac Lab job id" in result.message

    def test_an_unknown_job_is_an_error(self, fake_python: Path) -> None:
        result = _trainer().status("isaaclab-20260929-010000-0123456789ab")
        assert result.status == "error" and "no job" in result.message


class TestTheAgentToolDrivesTheLifecycle:
    def test_train_then_status_through_train_policy(self, fake_python: Path, tmp_path: Path) -> None:
        launched = train_policy(
            action="train",
            provider="isaaclab",
            output_dir=str(tmp_path / "out"),
            steps=3,
            seed=1,
            extra={"task": "Isaac-Cartpole", "num_envs": 16},
        )
        assert launched["status"] == "success", launched
        job = _json_block(launched)
        assert job["status"] == "running"
        assert "train_policy(action='status', provider='isaaclab'" in launched["content"][0]["text"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            polled = train_policy(action="status", provider="isaaclab", job_id=job["job_id"])
            if _json_block(polled)["status"] != "running":
                break
            time.sleep(0.05)
        block = _json_block(polled)
        assert block["status"] == "success", polled
        assert block["checkpoint_dir"] and block["checkpoint_dir"].endswith(job["job_id"])
        assert block["metrics"]["latest_iteration"] == 2
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[argv.index("--num_envs") + 1] == "16"


class TestTheLogParser:
    def test_an_empty_log_reports_nothing_learned(self) -> None:
        metrics = parse_rsl_rl_log("")
        assert metrics["latest_iteration"] is None and metrics["learning"] is False

    def test_a_falling_reward_is_not_learning(self) -> None:
        text = "".join(
            _ITERATION.format(it=i, total=3, total_steps=i, sps=1, reward=r) for i, r in enumerate((1.0, 0.5, -0.2))
        )
        metrics = parse_rsl_rl_log(runtime_strip(text))
        assert metrics["best_reward"] == 1.0 and metrics["latest_reward"] == -0.2 and metrics["learning"] is False

    def test_the_latest_model_is_the_highest_iteration_not_the_newest_name(self, tmp_path: Path) -> None:
        for it in (0, 50, 9, 100):
            (tmp_path / f"model_{it}.pt").write_bytes(b"")
        (tmp_path / "model_best.pt").write_bytes(b"")
        assert latest_model(str(tmp_path)) == str(tmp_path / "model_100.pt")


def runtime_strip(text: str) -> str:
    """Remove the ANSI escapes the way the trainer does before parsing."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
