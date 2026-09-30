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
from strands_robots.training.isaaclab import (
    RUN_RECORD_FILE,
    IsaacLabTrainer,
    classify_failure,
    latest_model,
    parse_rsl_rl_log,
)

# One rsl_rl iteration block, as Isaac Lab 3.0.0rc1 prints it (ANSI bold kept).
_ITERATION = """\
################################################################################
\x1b[1m                            Learning iteration {it}/{total}                            \x1b[0m

                            Total steps: {total_steps}
                       Steps per second: {sps}
                            Mean reward: {reward}
                    Mean episode length: 10.34
"""

# The fake interpreter. It honours ``-m isaaclab train`` and ``play``, records
# its argv, and behaves per $FAKE_ISAACLAB_MODE: ok | fail | nan | hang | slow |
# play_fail.
_FAKE = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys, time
    from pathlib import Path

    argv = sys.argv[1:]
    if argv[:3] == ["-m", "isaaclab", "play"]:
        # Playback: write the clip where Isaac Lab writes it, beside the
        # checkpoint, export the actor, and say so the way its recorder does.
        Path(os.environ["FAKE_ISAACLAB_ARGV"] + ".play").write_text(json.dumps(argv))
        if os.environ.get("FAKE_ISAACLAB_MODE") == "play_fail":
            print("Traceback (most recent call last):\\nRuntimeError: no display", flush=True)
            sys.exit(2)
        ckpt = Path(argv[argv.index("--checkpoint") + 1])
        n = int(argv[argv.index("--video_length") + 1])
        clip = ckpt.parent / "videos" / "play" / ("clip_" + ckpt.stem + "_0000.mp4")
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"mp4")
        (ckpt.parent / "exported").mkdir(exist_ok=True)
        (ckpt.parent / "exported" / "policy.pt").write_bytes(b"jit")
        print("[INFO]: [VideoRecorder] Wrote %d frames to %s" % (n, clip), flush=True)
        sys.exit(0)
    assert argv[:3] == ["-m", "isaaclab", "train"], argv
    Path(os.environ["FAKE_ISAACLAB_ARGV"]).write_text(json.dumps(argv))
    flags = dict(zip(argv[3::2], argv[4::2]))
    mode = os.environ.get("FAKE_ISAACLAB_MODE", "ok")
    total = int(flags["--max_iterations"])
    root = Path.cwd() / "logs" / "rsl_rl" / "cartpole"
    run = root / ("2026-09-29_01-00-00_" + flags["--run_name"])
    run.mkdir(parents=True)
    if "--export_io_descriptors" in argv:
        if os.environ.get("FAKE_IO_DESCRIPTORS"):
            (run / "io_descriptors").mkdir()
            (run / "io_descriptors" / "IO_descriptors.yaml").write_text(os.environ["FAKE_IO_DESCRIPTORS"])
        else:
            print("[WARNING] IO descriptors are only supported for manager based RL environments.", flush=True)
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
    if mode == "nan":
        # What a diverged run looks like with a block-buffered stdout: the
        # traceback first, then iteration metrics flushed after it.
        print("Traceback (most recent call last):", flush=True)
        print("ValueError: The observation group 'policy' returned by the environment contains NaN values.", flush=True)
        print(block.format(it=total, total=total, total_steps=0, sps=1, reward="nan"), flush=True)
        sys.exit(1)
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

    def test_a_nan_reward_is_read_and_the_run_is_diverged(self) -> None:
        text = "".join(
            _ITERATION.format(it=i, total=4, total_steps=i, sps=1, reward=r)
            for i, r in enumerate(("1.5", "2.0", "nan", "nan"))
        )
        metrics = parse_rsl_rl_log(runtime_strip(text))
        assert metrics["latest_reward"] != metrics["latest_reward"]  # NaN, not the stale 2.0
        assert metrics["diverged"] is True and metrics["diverged_at_iteration"] == 2
        assert metrics["best_reward"] == 2.0 and metrics["best_iteration"] == 1
        assert metrics["learning"] is False

    def test_counts_with_thousands_separators_are_read_whole(self) -> None:
        text = runtime_strip(_ITERATION.format(it=0, total=1, total_steps="1,234,567", sps="98,304", reward=1))
        metrics = parse_rsl_rl_log(text)
        assert metrics["total_steps"] == 1_234_567 and metrics["steps_per_s"] == 98_304

    def test_a_curriculum_dip_that_recovers_is_learning(self) -> None:
        # A penalty ramp drags the reward down for a while; ten-iteration
        # windows at each end judge the run, not two single samples.
        rewards = [-0.2] + [-3.0] * 9 + [-2.5] * 20 + [-1.0] * 10
        text = "".join(
            _ITERATION.format(it=i, total=len(rewards), total_steps=i, sps=1, reward=r) for i, r in enumerate(rewards)
        )
        metrics = parse_rsl_rl_log(runtime_strip(text))
        assert metrics["latest_reward"] < metrics["first_reward"]
        assert metrics["learning"] is True and metrics["reward_trend"] == pytest.approx(1.72)

    def test_task_metrics_and_the_success_rate_are_read(self) -> None:
        text = "".join(
            _ITERATION.format(it=i, total=3, total_steps=i, sps=1, reward=-0.25 - 0.1 * i)
            + f"                   Metrics/success_rate: {sr}\n"
            + f"    Metrics/ee_pose/position_error: {err}\n"
            + "       Episode_Termination/time_out: 1.0000\n"
            for i, (sr, err) in enumerate(((0.10, 0.30), (0.95, 0.06), (0.93, 0.07)))
        )
        metrics = parse_rsl_rl_log(runtime_strip(text))
        assert metrics["success_rate"] == {"latest": 0.93, "max": 0.95}
        assert metrics["task_metrics"]["Metrics/ee_pose/position_error"] == {
            "first": 0.30,
            "latest": 0.07,
            "min": 0.06,
            "max": 0.30,
        }
        assert metrics["task_metrics"]["Episode_Termination/time_out"]["latest"] == 1.0


class TestAFailedRunNamesItsCause:
    @pytest.mark.parametrize(
        ("line", "failure"),
        [
            ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.12 GiB.", "cuda_oom"),
            (
                "ValueError: The observation group 'policy' returned by the environment contains NaN values.",
                "nan_observation",
            ),
            ("gymnasium.error.NameNotFound: Environment `Isaac-Does-Not-Exist` doesn't exist.", "unknown_task"),
            ("ValueError: Unknown preset(s): no_such_preset", "unknown_physics_preset"),
            ("RuntimeError: something else", "exception"),
        ],
    )
    def test_the_last_exception_line_is_classified(self, line: str, failure: str) -> None:
        text = 'Traceback (most recent call last):\n  File "x.py", line 1\n    raise X(\n' + line + "\n"
        text += "".join(_ITERATION.format(it=i, total=2, total_steps=i, sps=1, reward=1) for i in range(2))
        assert classify_failure(runtime_strip(text), {}) == (failure, line)

    def test_a_clean_log_names_no_failure(self) -> None:
        assert classify_failure(
            runtime_strip(_ITERATION.format(it=0, total=1, total_steps=1, sps=1, reward=1)), {}
        ) == (
            None,
            None,
        )

    def test_the_cause_comes_first_even_above_buffered_metrics(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "nan")
        trainer = _trainer()
        result = _poll(trainer, trainer.train(_spec(tmp_path, steps=3)).job_id)
        assert result.status == "error"
        assert result.metrics["failure"] == "nan_observation"
        assert result.metrics["error"].startswith("ValueError: The observation group 'policy'")
        first_line = result.message.splitlines()[0]
        assert "exited 1" in first_line and "contains NaN values" in first_line and "physics preset" in first_line

    def test_the_child_writes_its_log_unbuffered(self) -> None:
        assert runtime.child_env()["PYTHONUNBUFFERED"] == "1"


class TestARunCanBeStopped:
    def test_stop_ends_a_running_job_and_keeps_its_checkpoints(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        trainer = _trainer()
        launched = trainer.train(_spec(tmp_path, steps=2))
        deadline = time.monotonic() + 20
        while trainer.status(launched.job_id).metrics.get("latest_model") is None and time.monotonic() < deadline:
            time.sleep(0.05)
        stopped = trainer.stop(launched.job_id)
        assert stopped.status == "stopped" and "stopped by request at iteration 1" in stopped.message
        assert not runtime.process_alive(launched.metrics["pid"])
        assert stopped.metrics["latest_model"] and stopped.metrics["latest_model"].endswith("model_1.pt")
        assert _trainer().status(launched.job_id).status == "stopped"

    def test_stop_through_the_tool(self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        job = _json_block(
            train_policy(
                action="train",
                provider="isaaclab",
                output_dir=str(tmp_path / "out"),
                steps=2,
                extra={"task": "Isaac-Cartpole"},
            )
        )
        stopped = train_policy(action="stop", provider="isaaclab", job_id=job["job_id"])
        assert stopped["status"] == "success" and _json_block(stopped)["status"] == "stopped"
        assert not runtime.process_alive(job["metrics"]["pid"])

    def test_stopping_an_ended_job_reports_it_unchanged(self, fake_python: Path, tmp_path: Path) -> None:
        trainer = _trainer()
        launched = trainer.train(_spec(tmp_path))
        assert _poll(trainer, launched.job_id).status == "success"
        result = trainer.stop(launched.job_id)
        assert result.status == "success" and "had already ended" in result.message

    def test_a_trainer_without_runs_in_flight_says_so(self) -> None:
        result = train_policy(action="stop", provider="mock", job_id="job-1")
        assert result["status"] == "error" and "stop() is not supported" in result["content"][0]["text"]

    def test_stop_needs_a_job_id(self) -> None:
        result = train_policy(action="stop", provider="isaaclab")
        assert result["status"] == "error" and "requires job_id" in result["content"][0]["text"]


class TestASecondLaunchIntoTheSameRunIsRefused:
    def test_a_live_job_on_the_same_task_and_output_dir_is_named(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        trainer = _trainer()
        first = trainer.train(_spec(tmp_path))
        try:
            second = _trainer().train(_spec(tmp_path))
            assert second.status == "error" and second.job_id == first.job_id
            assert "is already training Isaac-Cartpole" in second.message
            other = _trainer().train(
                TrainSpec(output_dir=str(tmp_path / "elsewhere"), steps=3, extra={"task": "Isaac-Cartpole"})
            )
            assert other.status == "running"
            trainer.stop(other.job_id)
        finally:
            trainer.stop(first.job_id)
        assert _trainer().train(_spec(tmp_path)).status == "running"


class TestARunRemembersHowItWasTrained:
    def test_the_run_record_sits_beside_the_checkpoints(self, fake_python: Path, tmp_path: Path) -> None:
        trainer = _trainer()
        spec = _spec(tmp_path, num_envs=64, physics="isaacsim_physx")
        spec.learning_rate = 1e-3
        result = _poll(trainer, trainer.train(spec).job_id)
        assert result.status == "success"
        record = json.loads((Path(result.checkpoint_dir) / RUN_RECORD_FILE).read_text())
        assert record["task"] == "Isaac-Cartpole" and record["physics"] == "isaacsim_physx"
        assert record["num_envs"] == 64 and record["seed"] == 7 and record["iterations"] == 3
        assert record["overrides"] == ["physics=isaacsim_physx", "agent.algorithm.learning_rate=0.001"]
        assert record["job_id"] == result.job_id


class TestAnUnknownTaskIsRefusedBeforeLaunch:
    @pytest.fixture
    def registered(self, fake_python: Path) -> Path:
        site = fake_python.parent.parent / "lib" / "python3.12" / "site-packages" / "isaaclab_tasks"
        (site / "manager_based" / "classic").mkdir(parents=True)
        (site / "__init__.py").write_text("")
        (site / "manager_based" / "classic" / "__init__.py").write_text(
            'import gymnasium as gym\n\ngym.register(\n    id="Isaac-Cartpole",\n    entry_point="x",\n)\n'
            'gym.register(id="Isaac-Cartpole-Direct", entry_point="x")\n'
        )
        return site

    def test_a_misspelt_task_is_refused_with_close_matches(self, registered: Path, tmp_path: Path) -> None:
        spec = TrainSpec(output_dir=str(tmp_path / "out"), steps=3, extra={"task": "Isaac-Cartpol"})
        problems = _trainer().validate(spec)
        assert any("'Isaac-Cartpol' is not registered" in p and "Isaac-Cartpole" in p for p in problems), problems

    def test_a_registered_task_passes(self, registered: Path, tmp_path: Path) -> None:
        assert _trainer().validate(_spec(tmp_path)) == []

    def test_an_install_without_task_packages_is_not_second_guessed(self, fake_python: Path, tmp_path: Path) -> None:
        spec = TrainSpec(output_dir=str(tmp_path / "out"), steps=3, extra={"task": "Isaac-Anything"})
        assert _trainer().validate(spec) == []
        assert runtime.registered_tasks(str(fake_python)) is None


class TestATrainedRunCanBePlayedBack:
    def test_play_records_a_clip_with_the_physics_the_run_trained_on(self, fake_python: Path, tmp_path: Path) -> None:
        trainer = _trainer()
        trained = _poll(trainer, trainer.train(_spec(tmp_path, steps=3, physics="isaacsim_physx")).job_id)
        assert trained.status == "success"
        played = trainer.play(trained.job_id, num_envs=4, video_length=30, wait=True)
        assert played.status == "success", played.message
        assert played.metrics["video"].endswith("videos/play/clip_model_2_0000.mp4")
        assert played.metrics["video_frames"] == 30 and Path(played.metrics["video"]).is_file()
        assert played.exported_model == str(Path(trained.checkpoint_dir) / "exported" / "policy.pt")
        argv = json.loads((tmp_path / "argv.json.play").read_text())
        assert argv[argv.index("--checkpoint") + 1] == trained.metrics["latest_model"]
        assert argv[argv.index("--task") + 1] == "Isaac-Cartpole" and argv[argv.index("--num_envs") + 1] == "4"
        assert "physics=isaacsim_physx" in argv and argv[argv.index("--visualizer") + 1] == "kit"
        assert "--video" in argv

    def test_play_through_the_tool_polls_like_a_run(self, fake_python: Path, tmp_path: Path) -> None:
        trainer = _trainer()
        trained = _poll(trainer, trainer.train(_spec(tmp_path)).job_id)
        launched = train_policy(action="play", provider="isaaclab", job_id=trained.job_id, extra={"video_length": 10})
        assert launched["status"] == "success" and _json_block(launched)["status"] == "running"
        result = _poll(_trainer(), _json_block(launched)["job_id"])
        assert (
            result.status == "success" and result.metrics["kind"] == "play" and result.metrics["of"] == trained.job_id
        )

    def test_a_failed_playback_names_its_cause(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _trainer()
        trained = _poll(trainer, trainer.train(_spec(tmp_path)).job_id)
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "play_fail")
        result = trainer.play(trained.job_id, wait=True)
        assert result.status == "error" and result.metrics["failure"] == "exception"
        assert "RuntimeError: no display" in result.message

    def test_a_run_still_training_or_without_a_checkpoint_is_not_played(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_ISAACLAB_MODE", "hang")
        trainer = _trainer()
        live = trainer.train(_spec(tmp_path))
        try:
            assert "still training" in trainer.play(live.job_id).message
        finally:
            trainer.stop(live.job_id)
        assert trainer.play("isaaclab-20260929-010000-0123456789ab").status == "error"

    @pytest.mark.parametrize(
        ("extra", "needle"), [({"video_length": 0}, "video_length"), ({"fps": 30}, "does not read")]
    )
    def test_bad_play_options_are_refused(
        self, fake_python: Path, tmp_path: Path, extra: dict[str, Any], needle: str
    ) -> None:
        trained = _poll(_trainer(), _trainer().train(_spec(tmp_path)).job_id)
        result = train_policy(action="play", provider="isaaclab", job_id=trained.job_id, extra=extra)
        assert result["status"] == "error" and needle in result["content"][0]["text"]

    def test_a_playback_does_not_block_the_next_training_run(self, fake_python: Path, tmp_path: Path) -> None:
        trainer = _trainer()
        trained = _poll(trainer, trainer.train(_spec(tmp_path)).job_id)
        trainer.play(trained.job_id, wait=True)
        assert trainer.train(_spec(tmp_path)).status == "running"

    def test_a_trainer_without_a_simulator_says_so(self) -> None:
        result = train_policy(action="play", provider="mock", job_id="job-1")
        assert result["status"] == "error" and "play() is not supported" in result["content"][0]["text"]


class TestATrainedRunExportsWhatCreatePolicyLoads:
    def test_export_converts_the_newest_checkpoint_and_names_the_rl_provider(
        self, fake_python: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        from tests.training.test_isaaclab_deploy_contract import fake_io_descriptors
        from tests.training.test_rsl_rl_actor_export import write_rsl_rl_run

        monkeypatch.setenv("FAKE_IO_DESCRIPTORS", json.dumps(fake_io_descriptors()))
        trainer = _trainer()
        trained = _poll(trainer, trainer.train(_spec(tmp_path, physics="isaacsim_physx")).job_id)
        write_rsl_rl_run(Path(trained.checkpoint_dir), iteration=2, normalize=True)
        exported = train_policy(
            action="export",
            provider="isaaclab",
            output_dir=str(tmp_path / "out"),
            steps=3,
            extra={"task": "Isaac-Cartpole"},
        )
        assert exported["status"] == "success", exported
        text = exported["content"][0]["text"]
        path = _json_block(exported)["exported_model"]
        assert f"create_policy('rl', checkpoint_dir='{path}')" in text
        meta = json.loads((Path(path) / "policy_meta.json").read_text())
        assert meta["provider"] == "rsl_rl" and meta["task"] == "Isaac-Cartpole" and meta["physics"] == "isaacsim_physx"
        assert meta["source_checkpoint"].endswith("model_2.pt")
        from strands_robots.policies import create_policy

        policy = create_policy("rl", checkpoint_dir=path)
        policy.set_robot_state_keys(["slider_to_cart", "cart_to_pole"])
        import asyncio

        action = asyncio.run(policy.get_actions({"policy_obs": [0.1, -0.2, 0.3, 0.0]}, ""))[0]
        assert set(action) == {"slider_to_cart", "cart_to_pole"} and all(isinstance(v, float) for v in action.values())
        del torch


def runtime_strip(text: str) -> str:
    """Remove the ANSI escapes the way the trainer does before parsing."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
