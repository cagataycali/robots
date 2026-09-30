"""Isaac Lab trainer - reinforcement learning on thousands of GPU environments.

Transport-shaped provider: like
:class:`~strands_robots.training.sagemaker.SagemakerTrainer` it imports no
training library. It launches ``python -m isaaclab train`` in Isaac Lab's own
interpreter (see :mod:`strands_robots.training._isaaclab_runtime` for why that
interpreter is separate) and answers :meth:`IsaacLabTrainer.status` by reading
the run's log and exit status. The run outlives the call that started it, so
:meth:`IsaacLabTrainer.train` returns ``running`` with a ``job_id`` unless
``extra['wait']`` asks it to block.

Spec mapping:

* ``steps`` -> ``--max_iterations`` (rsl_rl PPO iterations, each
  ``num_envs x num_steps_per_env`` environment steps).
* ``seed`` -> ``--seed``.
* ``learning_rate`` -> the ``agent.algorithm.learning_rate=<lr>`` override;
  omitted, the task's own agent config.
* ``output_dir`` -> the working directory; Isaac Lab writes
  ``logs/rsl_rl/<experiment>/<time>_<job_id>/`` under it, with
  ``model_<iteration>.pt`` checkpoints and ``params/``.
* ``extra['task']`` (required) -> ``--task``, an Isaac Lab task id such as
  ``Isaac-Cartpole`` or ``Isaac-Velocity-Flat-G1``.
* ``extra['num_envs']`` -> ``--num_envs``; omitted, the task's own default.
* ``extra['physics']`` -> the ``physics=<name>`` override
  (``newton_mjwarp``, ``isaacsim_physx``, ``ovphysx``); omitted, the task's
  own default.
* ``extra['rl_library']`` -> ``--rl_library``; only ``rsl_rl``, the library
  whose log :meth:`IsaacLabTrainer.status` reads.
* ``extra['wait']`` - block in :meth:`IsaacLabTrainer.train` until the run
  ends.
* ``extra['timeout_s']`` - wall-clock limit in seconds, after which the run's
  process group is stopped and the job reports ``error``. Enforced while
  :meth:`IsaacLabTrainer.train` waits and on every
  :meth:`IsaacLabTrainer.status` poll.

The run is always headless (``--visualizer none``).
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from strands_robots.training import _isaaclab_runtime as runtime
from strands_robots.training.base import Trainer, TrainResult, TrainSpec
from strands_robots.utils import (
    boolean_flag_error,
    positive_count_error,
    positive_finite_number_error,
    refusal_repr,
)

logger = logging.getLogger(__name__)

#: The ``extra`` keys this provider reads. Any other key is refused, because a
#: key Isaac Lab never receives would train the default while reporting success.
ACCEPTED_EXTRA_KEYS: tuple[str, ...] = ("task", "num_envs", "physics", "rl_library", "wait", "timeout_s")

#: RL libraries whose training log :meth:`IsaacLabTrainer.status` can read.
SUPPORTED_RL_LIBRARIES: tuple[str, ...] = ("rsl_rl",)

# Isaac Lab task ids are gym ids: ``Isaac-Cartpole``, ``Isaac-Velocity-Flat-G1``,
# ``IsaacContrib-Stack-Cube-SO101-v0``. The first character is a letter, so a
# task cannot read as a flag.
_TASK_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")

# A physics preset name, passed as the ``physics=<name>`` override.
_PHYSICS_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}\Z")

# ``isaaclab-<UTC stamp>-<12 hex>``; also the ``--run_name`` of the run, so the
# run directory can be found from the job alone. Anchored so a job id read back
# from an agent cannot name a path outside the jobs directory.
_JOB_ID_RE = re.compile(r"^isaaclab-\d{8}-\d{6}-[0-9a-f]{12}\Z")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ITERATION_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
# A logged number: rsl_rl prints ``nan`` / ``inf`` for a diverged run, and those
# must be read, not skipped - skipping them made a dead run look healthy.
_NUMBER = r"([-+]?(?:nan|inf|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))"
# ``_EOL`` ends a match at the end of its line, without ``$``'s match before a
# trailing newline.
_EOL = r"[ \t\r]*(?=\n|\Z)"
_REWARD_RE = re.compile(r"Mean reward: " + _NUMBER + _EOL, re.I)
# Whole-line counts, with or without thousands separators, so ``1,234,567``
# reads as that number rather than as ``1``.
_STEPS_PER_S_RE = re.compile(r"Steps per second: (\d[\d,]*)" + _EOL)
_TOTAL_STEPS_RE = re.compile(r"Total steps: (\d[\d,]*)" + _EOL)
# Isaac Lab's own per-task terms: ``Metrics/success_rate: 0.98``. These say
# whether a task is solved when the reward is shaped by a penalty curriculum.
_TASK_METRIC_RE = re.compile(
    r"^\s*((?:Metrics|Curriculum|Episode_Termination)/[\w/.-]+): " + _NUMBER + _EOL, re.M | re.I
)
# The last line of a Python traceback: an unindented ``module.SomeError: ...``.
_EXCEPTION_RE = re.compile(
    r"^((?:[A-Za-z_][\w]*\.)*[A-Za-z_]\w*(?:Error|Exception|NotFound|Interrupt))(?::[ \t]?(.*))?" + _EOL, re.M
)
_LOG_DIR_RE = re.compile(r"Logging experiment in directory: (\S+)")
_TRAINING_TIME_RE = re.compile(r"Training time: (\d+(?:\.\d+)?) seconds")
_MODEL_RE = re.compile(r"^model_(\d+)\.pt\Z")
# Isaac Lab's video recorder: ``[VideoRecorder] Wrote 120 frames to <path>.mp4``.
_VIDEO_RE = re.compile(r"\[VideoRecorder\] Wrote (\d+) frames to (\S+\.mp4)")

_JOB_FILE = "job.json"
_LOG_FILE = "train.log"
_TIMED_OUT_FILE = "timed_out"
_STOPPED_FILE = "stopped"
#: Written next to the checkpoints, so a run remembers how it was trained:
#: which task, which physics preset, how many environments and which overrides.
RUN_RECORD_FILE = "strands_run.json"
_TAIL_LINES = 12

#: How the end of a failed run is classified, with the next step for each.
FAILURE_HINTS: dict[str, str] = {
    "cuda_oom": "the GPU ran out of memory - lower extra['num_envs']",
    "nan_observation": "the simulation produced NaN observations - try another physics preset or a lower learning_rate",
    "diverged": "the mean reward became NaN or infinite - lower learning_rate or try another physics preset",
    "unknown_task": "Isaac Lab has no such task id - check extra['task']",
    "unknown_physics_preset": "the task has no such physics preset - check extra['physics']",
    "timeout": "the run passed its timeout_s",
    "stopped": "the run was stopped by request",
    "killed": "the run was killed from outside",
    "exception": "the run raised an exception",
    "exit_status": "the run exited without reporting a training time",
}

# Iterations averaged at each end of the run for the ``learning`` verdict. One
# iteration is noise; a penalty curriculum dips the reward for tens of them.
_TREND_WINDOW = 10

# Popen handles of runs this process launched, so a finished child is reaped
# rather than left a zombie. A run launched by another process is judged by pid.
_CHILDREN: dict[str, subprocess.Popen[bytes]] = {}


class IsaacLabTrainer(Trainer):
    """Train an RL policy with Isaac Lab, out of process, and poll it by job id.

    Args:
        python: Interpreter of the Isaac Lab virtual environment. Defaults to
            ``$ISAACLAB_PYTHON``. Never taken from the spec, so an agent cannot
            choose which program runs.
        jobs_dir: Directory job records are written under. Defaults to
            ``$STRANDS_ISAACLAB_JOBS``, else
            ``~/.cache/strands_robots/isaaclab/jobs``.
        poll_interval_s: Seconds between checks while ``extra['wait']`` blocks.
    """

    #: Isaac Lab tasks build their own environments; no dataset is read.
    requires_dataset = False

    def __init__(
        self,
        python: str | None = None,
        jobs_dir: str | None = None,
        poll_interval_s: float = 2.0,
    ) -> None:
        self._python = runtime.resolve_python(python)
        self._jobs_dir = runtime.default_jobs_dir(jobs_dir)
        self._poll_interval_s = poll_interval_s

    @property
    def provider_name(self) -> str:
        """``"isaaclab"``."""
        return "isaaclab"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """One RTX-class GPU; measured 3.5 GB for 4096 G1 environments on Newton."""
        return {"min_gpus": 1, "min_vram_gb": 8, "multinode": False}

    def validate(self, spec: TrainSpec) -> list[str]:
        """Report why *spec* cannot launch an Isaac Lab run; empty when it can.

        Read-only: checks the input safety of every field, that the Isaac Lab
        interpreter exists and the EULA is accepted, ``steps``, ``seed``,
        ``learning_rate``, ``resume`` and each ``extra`` key. No process is started.
        """
        ctx = self.provider_name
        problems = self._security_problems(spec)
        problems.extend(runtime.runtime_problems(self._python, context=ctx))
        if not spec.output_dir:
            problems.append(f"{ctx}: output_dir is required (Isaac Lab writes logs/ and checkpoints under it)")
        steps_error = positive_count_error(spec.steps, "steps", ctx)
        if steps_error is not None:
            problems.append(steps_error)
        problems.extend(self._seed_problems(spec))
        problems.extend(self._learning_rate_problems(spec))
        resume_problems = self._resume_problems(spec)
        problems.extend(resume_problems)
        if not resume_problems and spec.resume:
            problems.append(f"{ctx}: resuming a run is not supported yet - start a new run with resume=False")
        problems.extend(_extra_problems(spec.extra or {}, ctx))
        task = (spec.extra or {}).get("task")
        if isinstance(task, str) and _TASK_RE.match(task) and self._python:
            known = runtime.registered_tasks(self._python)
            if known and task not in known:
                close = difflib.get_close_matches(task, sorted(known), n=3, cutoff=0.6)
                hint = f"; did you mean {close}?" if close else ""
                problems.append(
                    f"{ctx}: extra['task'] {task!r} is not registered by the Isaac Lab install at {self._python} "
                    f"({len(known)} tasks){hint}. A task from your own package is trained once the operator "
                    f"names that package in ${runtime.TASK_PACKAGES_ENV} ('module:register_fn', importable in "
                    "the Isaac Lab venv)"
                )
            owner = runtime.task_package_of(self._python, task)
            if owner is not None and owner[1] is None:
                problems.append(
                    f"{ctx}: {task!r} is registered by {owner[0]!r}, which ${runtime.TASK_PACKAGES_ENV} names "
                    f"without the function that registers it; name it as '{owner[0]}:<function>' so Isaac Lab "
                    "can call it through --external_callback before looking the task up"
                )
        return problems

    def _external_callback(self, task: str) -> str | None:
        """``module.function`` Isaac Lab must call to register *task*, for an operator package's task."""
        owner = runtime.task_package_of(str(self._python), task) if self._python else None
        return f"{owner[0]}.{owner[1]}" if owner is not None and owner[1] else None

    def build_command(self, spec: TrainSpec, job_id: str) -> list[str]:
        """Return the ``python -m isaaclab train`` argv for *spec*.

        Only call on a spec :meth:`validate` accepted.

        Args:
            spec: The validated spec.
            job_id: The job id, passed as ``--run_name``.

        Returns:
            The argv, interpreter first.
        """
        extra = spec.extra or {}
        cmd = [
            str(self._python),
            "-m",
            "isaaclab",
            "train",
            "--rl_library",
            str(extra.get("rl_library", "rsl_rl")),
            "--task",
            str(extra["task"]),
            "--max_iterations",
            str(spec.steps),
            "--visualizer",
            "none",
            "--run_name",
            job_id,
        ]
        if "num_envs" in extra:
            cmd += ["--num_envs", str(extra["num_envs"])]
        if spec.seed is not None:
            cmd += ["--seed", str(spec.seed)]
        if (callback := self._external_callback(str(extra["task"]))) is not None:
            cmd += ["--external_callback", callback]
        if "physics" in extra:
            cmd.append(f"physics={extra['physics']}")
        if spec.learning_rate is not None:
            cmd.append(f"agent.algorithm.learning_rate={spec.learning_rate!r}")
        return cmd

    def train(self, spec: TrainSpec) -> TrainResult:
        """Launch the run and return ``running``, or its verdict under ``extra['wait']``.

        Validates first and fails closed. The job record (argv, pid, working
        directory, deadline) is written before this returns, so any process
        can poll the job with :meth:`status`.
        """
        problems = self.validate(spec)
        if problems:
            return TrainResult(status="error", job_id="", message="validation failed: " + "; ".join(problems))
        extra = spec.extra or {}
        work_dir = Path(spec.output_dir).expanduser().resolve()
        running = self._running_job_for(work_dir, str(extra["task"]))
        if running is not None:
            return TrainResult(
                status="error",
                job_id=running,
                message=(
                    f"{self.provider_name}: job {running} is already training {extra['task']} in {work_dir}; "
                    f"poll it with action='status', stop it with action='stop', or use another output_dir"
                ),
            )
        job_id = f"isaaclab-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(spec, job_id)
        proc = runtime.launch(
            cmd, cwd=work_dir, log_path=job_dir / _LOG_FILE, exit_file=job_dir / runtime.EXIT_CODE_FILE
        )
        _CHILDREN[job_id] = proc
        started = time.time()
        timeout_s = extra.get("timeout_s")
        record = {
            "job_id": job_id,
            "pid": proc.pid,
            "cmd": cmd,
            "cwd": str(work_dir),
            "started": started,
            "deadline": started + float(timeout_s) if timeout_s is not None else None,
            "task": extra["task"],
            "max_iterations": spec.steps,
            "run": run_record(spec),
        }
        (job_dir / _JOB_FILE).write_text(json.dumps(record, indent=1), encoding="utf-8")
        logger.info("isaaclab: launched %s (pid %d): %s", job_id, proc.pid, " ".join(cmd))
        if extra.get("wait"):
            result = self.status(job_id)
            while result.status == "running":
                time.sleep(self._poll_interval_s)
                result = self.status(job_id)
            return result
        return TrainResult(
            status="running",
            job_id=job_id,
            message=f"launched Isaac Lab {extra['task']} (pid {proc.pid}); log: {job_dir / _LOG_FILE}",
            metrics={"pid": proc.pid, "max_iterations": spec.steps},
        )

    def status(self, job_id: str) -> TrainResult:
        """Return the run's verdict from its log and exit status.

        ``running`` while the process lives, ``success`` once it exited 0
        after rsl_rl printed its training time, ``error`` otherwise - with the
        log tail as the message. Metrics carry the latest iteration, rewards
        (first, latest, best), environment steps per second and total steps,
        so a live run can be told from a learning one. Stops the run first if
        its ``timeout_s`` deadline has passed.
        """
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return TrainResult(
                status="error",
                job_id=str(job_id),
                message=f"{self.provider_name}: {refusal_repr(job_id)} is not an Isaac Lab job id",
            )
        job_dir = self._jobs_dir / job_id
        try:
            record = json.loads((job_dir / _JOB_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return TrainResult(
                status="error", job_id=job_id, message=f"{self.provider_name}: no job {job_id} under {self._jobs_dir}"
            )
        child = _CHILDREN.get(job_id)
        if child is not None and child.poll() is not None:
            _CHILDREN.pop(job_id, None)
        pid = int(record["pid"])
        alive = runtime.process_alive(pid)
        deadline = record.get("deadline")
        if alive and deadline is not None and time.time() > deadline:
            logger.warning("isaaclab: %s passed its timeout; stopping pid %d", job_id, pid)
            (job_dir / _TIMED_OUT_FILE).write_text(str(time.time()), encoding="utf-8")
            runtime.terminate(pid)
            if child is not None:
                child.poll()
            alive = runtime.process_alive(pid)

        text = _read_log(job_dir / _LOG_FILE)
        exit_code = _read_exit_code(job_dir / runtime.EXIT_CODE_FILE)
        if record.get("kind") == "play":
            return self._play_status(job_id, job_dir, record, text, exit_code, alive)
        metrics = parse_rsl_rl_log(text)
        metrics.update(
            {
                "max_iterations": record.get("max_iterations"),
                "elapsed_s": round(time.time() - float(record["started"]), 1),
                "pid": pid,
                "exit_code": exit_code,
                "liveness_ok": alive,
            }
        )
        run_dir = find_run_dir(text, job_id)
        model = latest_model(run_dir)
        metrics["latest_model"] = model
        tail = "\n".join(text.strip().splitlines()[-_TAIL_LINES:])
        _write_run_record(run_dir, record)

        if alive:
            return TrainResult(
                status="running",
                job_id=job_id,
                checkpoint_dir=run_dir,
                metrics=metrics,
                message=_running_line(metrics, record),
            )
        at = f"iteration {metrics['latest_iteration']}" if metrics["latest_iteration"] is not None else "startup"
        if (job_dir / _STOPPED_FILE).exists():
            metrics["failure"] = "stopped"
            return TrainResult(
                status="stopped",
                job_id=job_id,
                checkpoint_dir=run_dir,
                metrics=metrics,
                message=f"{self.provider_name}: stopped by request at {at}; latest checkpoint {model}",
            )
        failure, error_line = classify_failure(text, metrics)
        if (job_dir / _TIMED_OUT_FILE).exists():
            failure, summary = "timeout", f"stopped at {at} after its timeout_s"
        elif exit_code is None:
            failure, summary = failure or "killed", f"the run ended at {at} without an exit status (killed)"
        else:
            summary = f"Isaac Lab exited {exit_code} at {at}"
            if failure is None and (exit_code != 0 or metrics["training_time_s"] is None):
                failure = "exit_status"
        if failure is not None:
            metrics["failure"] = failure
            metrics["error"] = error_line or summary
            cause = f": {error_line}" if error_line else ""
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=run_dir,
                metrics=metrics,
                message=(
                    f"{self.provider_name}: {summary}{cause} ({FAILURE_HINTS[failure]}); "
                    f"latest checkpoint {model}; log tail:\n{tail}"
                ),
            )
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=run_dir,
            metrics=metrics,
            message=(
                f"finished {metrics['latest_iteration'] + 1 if metrics['latest_iteration'] is not None else 0}"
                f"/{record.get('max_iterations')} iterations in {metrics['training_time_s']} s; "
                f"{_verdict_line(metrics)}; latest checkpoint {model}"
            ),
        )

    def play(
        self,
        job_id: str,
        *,
        num_envs: int = 16,
        video_length: int = 200,
        timeout_s: float | None = 900.0,
        wait: bool = False,
    ) -> TrainResult:
        """Play a trained run's latest checkpoint back in Isaac Lab and record a video.

        Launches ``python -m isaaclab play`` on the run's newest
        ``model_<iteration>.pt``, with the same task and the same physics
        preset the run trained with (read from its record - a policy replayed
        on another preset can fall at once), rendering with the Kit visualizer
        so the clip has frames. Isaac Lab also exports the actor it loaded to
        ``<run>/exported/policy.pt`` (TorchScript) and ``policy.onnx``.

        Args:
            job_id: The training job whose run to play.
            num_envs: Environments to play (the video shows the first camera view).
            video_length: Frames in the clip.
            timeout_s: Wall-clock limit for the playback, like ``extra['timeout_s']``.
            wait: Block until the playback ends.

        Returns:
            ``running`` with the playback's own ``job_id``, whose :meth:`status`
            reports ``metrics['video']`` once written; or its verdict under ``wait``.
        """
        ctx = self.provider_name
        trained = self.status(job_id)
        if trained.status == "running":
            return TrainResult(
                status="error", job_id=job_id, message=f"{ctx}: job {job_id} is still training; stop it or wait"
            )
        if trained.metrics.get("kind") == "play":
            return TrainResult(
                status="error", job_id=job_id, message=f"{ctx}: {job_id} is a playback, not a training job"
            )
        model = trained.metrics.get("latest_model")
        if not trained.checkpoint_dir or not model:
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"{ctx}: job {job_id} has no checkpoint to play ({trained.message})",
            )
        for value, name in ((num_envs, "num_envs"), (video_length, "video_length")):
            error = positive_count_error(value, name, ctx)
            if error is not None:
                return TrainResult(status="error", job_id=job_id, message=error)
        if timeout_s is not None and (error := positive_finite_number_error(timeout_s, "timeout_s", ctx)):
            return TrainResult(status="error", job_id=job_id, message=error)
        problems = runtime.runtime_problems(self._python, context=ctx)
        if problems:
            return TrainResult(status="error", job_id=job_id, message="; ".join(problems))
        record = json.loads((self._jobs_dir / job_id / _JOB_FILE).read_text(encoding="utf-8"))
        run = record.get("run") or {}
        cmd = [
            str(self._python),
            "-m",
            "isaaclab",
            "play",
            "--rl_library",
            str(run.get("rl_library", "rsl_rl")),
            "--task",
            str(record["task"]),
            "--num_envs",
            str(num_envs),
            "--checkpoint",
            str(model),
            "--video",
            "--video_length",
            str(video_length),
            "--visualizer",
            "kit",
        ]
        if (callback := self._external_callback(str(record["task"]))) is not None:
            cmd += ["--external_callback", callback]
        if run.get("physics"):
            cmd.append(f"physics={run['physics']}")
        play_id = f"isaaclab-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
        play_dir = self._jobs_dir / play_id
        play_dir.mkdir(parents=True, exist_ok=False)
        proc = runtime.launch(
            cmd, cwd=Path(record["cwd"]), log_path=play_dir / _LOG_FILE, exit_file=play_dir / runtime.EXIT_CODE_FILE
        )
        _CHILDREN[play_id] = proc
        started = time.time()
        (play_dir / _JOB_FILE).write_text(
            json.dumps(
                {
                    "job_id": play_id,
                    "kind": "play",
                    "of": job_id,
                    "pid": proc.pid,
                    "cmd": cmd,
                    "cwd": record["cwd"],
                    "started": started,
                    "deadline": started + float(timeout_s) if timeout_s is not None else None,
                    "task": record["task"],
                    "run_dir": trained.checkpoint_dir,
                    "checkpoint": model,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info("isaaclab: playing %s (pid %d): %s", model, proc.pid, " ".join(cmd))
        if wait:
            result = self.status(play_id)
            while result.status == "running":
                time.sleep(self._poll_interval_s)
                result = self.status(play_id)
            return result
        return TrainResult(
            status="running",
            job_id=play_id,
            checkpoint_dir=trained.checkpoint_dir,
            metrics={"pid": proc.pid, "kind": "play", "of": job_id, "checkpoint": model},
            message=f"playing {Path(model).name} of {job_id} ({record['task']}, {num_envs} envs, {video_length} frames)",
        )

    def _play_status(
        self, job_id: str, job_dir: Path, record: dict[str, Any], text: str, exit_code: int | None, alive: bool
    ) -> TrainResult:
        """The verdict of a playback launched by :meth:`play`: the video it wrote, or why it did not."""
        run_dir = record.get("run_dir")
        videos = _VIDEO_RE.findall(text)
        exported = Path(run_dir) / "exported" / "policy.pt" if run_dir else None
        metrics: dict[str, Any] = {
            "kind": "play",
            "of": record.get("of"),
            "checkpoint": record.get("checkpoint"),
            "video": videos[-1][1] if videos else None,
            "video_frames": int(videos[-1][0]) if videos else None,
            "exported_jit": str(exported) if exported and exported.is_file() else None,
            "pid": int(record["pid"]),
            "exit_code": exit_code,
            "liveness_ok": alive,
            "elapsed_s": round(time.time() - float(record["started"]), 1),
        }
        if alive:
            return TrainResult(
                status="running", job_id=job_id, checkpoint_dir=run_dir, metrics=metrics, message="playing back"
            )
        if exit_code == 0 and metrics["video"] and Path(metrics["video"]).is_file():
            return TrainResult(
                status="success",
                job_id=job_id,
                checkpoint_dir=run_dir,
                exported_model=metrics["exported_jit"],
                metrics=metrics,
                message=f"wrote {metrics['video_frames']} frames to {metrics['video']}",
            )
        failure, error_line = classify_failure(text, metrics)
        if (job_dir / _TIMED_OUT_FILE).exists():
            failure = "timeout"
        elif (job_dir / _STOPPED_FILE).exists():
            failure = "stopped"
        elif exit_code is None:
            failure = failure or "killed"
        elif failure is None:
            failure, error_line = "exit_status", f"Isaac Lab play exited {exit_code} without writing a video"
        metrics["failure"] = failure
        metrics["error"] = error_line or FAILURE_HINTS[failure]
        tail = "\n".join(text.strip().splitlines()[-_TAIL_LINES:])
        return TrainResult(
            status="stopped" if failure == "stopped" else "error",
            job_id=job_id,
            checkpoint_dir=run_dir,
            metrics=metrics,
            message=f"{self.provider_name}: playback failed: {metrics['error']} ({FAILURE_HINTS[failure]}); log tail:\n{tail}",
        )

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Convert the run's newest rsl_rl checkpoint into what ``create_policy("rl")`` loads.

        Writes ``<run>/strands_policy/policy.pt`` + ``policy_meta.json``
        (``provider="rsl_rl"``: rsl_rl's MLP with the run's own activation and
        observation normalizer, rebuilt without rsl_rl or Isaac Lab), with the
        run record's task and physics preset in the metadata.

        Args:
            spec: The validated spec (unused beyond the gate ``train_policy`` runs).
            checkpoint_dir: The run directory :meth:`latest_checkpoint` returned.

        Returns:
            The ``strands_policy`` directory.

        Raises:
            FileNotFoundError: If the run directory holds no ``model_<iteration>.pt``.
        """
        from strands_robots.training.rl import rsl_rl

        del spec
        model = latest_model(checkpoint_dir)
        if model is None:
            raise FileNotFoundError(f"{self.provider_name}: no model_<iteration>.pt in {checkpoint_dir}")
        run: dict[str, Any] = {}
        record_path = Path(checkpoint_dir) / RUN_RECORD_FILE
        if record_path.is_file():
            run = json.loads(record_path.read_text(encoding="utf-8"))
        extra = {k: run[k] for k in ("task", "physics", "num_envs", "job_id", "overrides") if k in run}
        return rsl_rl.convert_checkpoint(model, str(Path(checkpoint_dir) / "strands_policy"), extra_meta=extra)

    def stop(self, job_id: str) -> TrainResult:
        """Stop a running job and return its verdict, ``stopped``.

        Marks the job as stopped by request, then stops its process group
        (SIGTERM, then SIGKILL). Checkpoints already written stay usable. A job
        that has already ended is reported as it ended, unchanged.
        """
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            return self.status(job_id)
        job_dir = self._jobs_dir / job_id
        try:
            record = json.loads((job_dir / _JOB_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.status(job_id)
        pid = int(record["pid"])
        if not runtime.process_alive(pid):
            result = self.status(job_id)
            result.message = f"{self.provider_name}: job {job_id} had already ended - {result.message}"
            return result
        (job_dir / _STOPPED_FILE).write_text(str(time.time()), encoding="utf-8")
        logger.info("isaaclab: stopping %s (pid %d) by request", job_id, pid)
        runtime.terminate(pid)
        child = _CHILDREN.pop(job_id, None)
        if child is not None:
            child.poll()
        return self.status(job_id)

    def _running_job_for(self, work_dir: Path, task: str) -> str | None:
        """Return the id of a live job already training *task* in *work_dir*."""
        if not self._jobs_dir.is_dir():
            return None
        for job_file in self._jobs_dir.glob(f"isaaclab-*/{_JOB_FILE}"):
            try:
                record = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("kind") == "play" or record.get("cwd") != str(work_dir) or record.get("task") != task:
                continue
            if runtime.process_alive(int(record.get("pid", 0))) and not (job_file.parent / _STOPPED_FILE).exists():
                return str(record.get("job_id"))
        return None

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the newest rsl_rl run directory under ``output_dir`` holding a ``model_*.pt``."""
        root = Path(output_dir).expanduser() / "logs" / "rsl_rl"
        if not root.is_dir():
            return None
        runs = [d for d in root.glob("*/*") if d.is_dir() and latest_model(str(d))]
        if not runs:
            return None
        return str(max(runs, key=lambda d: d.stat().st_mtime))


def _extra_problems(extra: dict[str, Any], ctx: str) -> list[str]:
    """Report every ``extra`` key or value this provider cannot forward."""
    problems: list[str] = []
    unknown = sorted(str(k) for k in extra if k not in ACCEPTED_EXTRA_KEYS)
    if unknown:
        problems.append(
            f"{ctx}: extra key(s) {unknown} are not read by this provider; accepted keys are "
            f"{list(ACCEPTED_EXTRA_KEYS)}"
        )
    task = extra.get("task")
    if task is None:
        problems.append(f"{ctx}: extra['task'] is required - an Isaac Lab task id such as 'Isaac-Cartpole'")
    elif not isinstance(task, str) or not _TASK_RE.match(task):
        problems.append(f"{ctx}: extra['task'] must be an Isaac Lab task id (letters first), got {refusal_repr(task)}")
    if "num_envs" in extra:
        error = positive_count_error(extra["num_envs"], "extra['num_envs']", ctx)
        if error is not None:
            problems.append(error)
    if "physics" in extra:
        physics = extra["physics"]
        if not isinstance(physics, str) or not _PHYSICS_RE.match(physics):
            problems.append(
                f"{ctx}: extra['physics'] must be a physics preset name such as 'newton_mjwarp' or "
                f"'isaacsim_physx', got {refusal_repr(physics)}"
            )
    if "rl_library" in extra and extra["rl_library"] not in SUPPORTED_RL_LIBRARIES:
        problems.append(
            f"{ctx}: extra['rl_library'] {refusal_repr(extra['rl_library'])} is not supported; status() reads "
            f"the training log of {list(SUPPORTED_RL_LIBRARIES)} only"
        )
    if "wait" in extra:
        error = boolean_flag_error(extra["wait"], "extra['wait']", ctx)
        if error is not None:
            problems.append(error)
    if "timeout_s" in extra:
        error = positive_finite_number_error(extra["timeout_s"], "extra['timeout_s']", ctx)
        if error is not None:
            problems.append(error)
    return problems


def _read_log(path: Path) -> str:
    try:
        return _ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""


def _read_exit_code(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def parse_rsl_rl_log(text: str) -> dict[str, Any]:
    """Read the progress an rsl_rl training log reports.

    Args:
        text: The log, ANSI escapes removed.

    Returns:
        ``latest_iteration`` (``None`` before the first),
        ``first_reward`` / ``latest_reward`` / ``best_reward`` (mean episode
        reward; ``best`` over finite values) and ``best_iteration``,
        ``steps_per_s`` and ``total_steps`` of the latest iteration,
        ``training_time_s`` once rsl_rl reports it, ``diverged`` with
        ``diverged_at_iteration`` once a reward is NaN or infinite,
        ``reward_trend`` (mean of the last ten finite rewards minus the mean of
        the first ten, fewer on a short run) and ``learning`` - whether that
        trend is positive after at least two iterations. ``task_metrics`` holds
        every ``Metrics/``, ``Curriculum/`` and ``Episode_Termination/`` term as
        ``{"first", "latest", "min", "max"}``, and ``success_rate`` repeats
        ``Metrics/success_rate`` when the task reports one: on a task whose
        penalties ramp up, it is the number that says the task is solved.
    """
    starts = [(m.start(), int(m.group(1))) for m in _ITERATION_RE.finditer(text)]
    blocks = [
        (it, text[pos : starts[k + 1][0] if k + 1 < len(starts) else len(text)]) for k, (pos, it) in enumerate(starts)
    ]
    rewards: list[tuple[int, float]] = []
    task_metrics: dict[str, dict[str, float]] = {}
    for it, block in blocks:
        reward = _REWARD_RE.search(block)
        if reward:
            rewards.append((it, float(reward.group(1))))
        for name, raw in _TASK_METRIC_RE.findall(block):
            value = float(raw)
            entry = task_metrics.setdefault(name, {"first": value, "latest": value, "min": value, "max": value})
            entry["latest"] = value
            if math.isfinite(value):
                entry["min"] = value if not math.isfinite(entry["min"]) else min(entry["min"], value)
                entry["max"] = value if not math.isfinite(entry["max"]) else max(entry["max"], value)
    if not blocks:
        rewards = [(0, float(m)) for m in _REWARD_RE.findall(text)]
    finite = [(it, r) for it, r in rewards if math.isfinite(r)]
    diverged_at = next((it for it, r in rewards if not math.isfinite(r)), None)
    best = max(finite, key=lambda item: item[1]) if finite else None
    window = max(1, min(_TREND_WINDOW, len(finite) // 2))
    trend = (
        sum(r for _, r in finite[-window:]) / window - sum(r for _, r in finite[:window]) / window
        if len(finite) >= 2
        else None
    )
    steps_per_s = [int(m.replace(",", "")) for m in _STEPS_PER_S_RE.findall(text)]
    total_steps = [int(m.replace(",", "")) for m in _TOTAL_STEPS_RE.findall(text)]
    training_time = _TRAINING_TIME_RE.search(text)
    success = task_metrics.get("Metrics/success_rate")
    return {
        "latest_iteration": starts[-1][1] if starts else None,
        "first_reward": rewards[0][1] if rewards else None,
        "latest_reward": rewards[-1][1] if rewards else None,
        "best_reward": best[1] if best else None,
        "best_iteration": best[0] if best and blocks else None,
        "reward_trend": round(trend, 6) if trend is not None else None,
        "diverged": diverged_at is not None,
        "diverged_at_iteration": diverged_at if blocks else None,
        "steps_per_s": steps_per_s[-1] if steps_per_s else None,
        "total_steps": total_steps[-1] if total_steps else None,
        "training_time_s": float(training_time.group(1)) if training_time else None,
        "learning": diverged_at is None and trend is not None and trend > 0,
        "task_metrics": task_metrics,
        "success_rate": {"latest": success["latest"], "max": success["max"]} if success else None,
    }


def classify_failure(text: str, metrics: dict[str, Any]) -> tuple[str | None, str | None]:
    """Name why a run failed from its log: a :data:`FAILURE_HINTS` key and the line that says so.

    The last exception line of the log wins, so a traceback printed above a
    tail of iteration metrics is still the one reported. ``(None, None)``
    when the log names no failure.
    """
    exceptions = [m for m in _EXCEPTION_RE.finditer(text) if not m.group(0).startswith(("Traceback", "raise"))]
    line = exceptions[-1].group(0).strip() if exceptions else None
    if line and len(line) > 300:
        line = line[:297] + "..."
    lowered = (line or "").lower()
    if "outofmemory" in lowered or "out of memory" in lowered:
        return "cuda_oom", line
    if "nan values" in lowered or "contains nan" in lowered:
        return "nan_observation", line
    if "namenotfound" in lowered or ("environment" in lowered and "doesn't exist" in lowered):
        return "unknown_task", line
    if "unknown preset" in lowered:
        return "unknown_physics_preset", line
    if metrics.get("diverged"):
        at = metrics.get("diverged_at_iteration")
        return "diverged", line or f"the mean reward became NaN at iteration {at}"
    if line:
        return "exception", line
    return None, None


def run_record(spec: TrainSpec) -> dict[str, Any]:
    """Return how *spec* trains: task, physics preset, environments, seed and overrides.

    Written as :data:`RUN_RECORD_FILE` next to the checkpoints, so replaying a
    checkpoint can use the simulator it was trained in - a policy trained on
    PhysX and replayed on the task's default Newton preset falls within a second.
    """
    extra = spec.extra or {}
    overrides = []
    if "physics" in extra:
        overrides.append(f"physics={extra['physics']}")
    if spec.learning_rate is not None:
        overrides.append(f"agent.algorithm.learning_rate={spec.learning_rate!r}")
    return {
        "task": extra.get("task"),
        "physics": extra.get("physics"),
        "num_envs": extra.get("num_envs"),
        "seed": spec.seed,
        "iterations": spec.steps,
        "rl_library": extra.get("rl_library", "rsl_rl"),
        "overrides": overrides,
    }


def _write_run_record(run_dir: str | None, record: dict[str, Any]) -> None:
    """Put the job's run record beside its checkpoints once the run directory exists."""
    if not run_dir or "run" not in record:
        return
    path = Path(run_dir) / RUN_RECORD_FILE
    if path.exists():
        return
    try:
        path.write_text(json.dumps({"job_id": record.get("job_id"), **record["run"]}, indent=1), encoding="utf-8")
    except OSError as exc:
        logger.warning("isaaclab: could not write %s: %s", path, exc)


def _running_line(metrics: dict[str, Any], record: dict[str, Any]) -> str:
    """``iteration N/M``, plus the task's success rate and any divergence seen so far."""
    line = f"iteration {metrics['latest_iteration']}/{record.get('max_iterations')}"
    if metrics.get("success_rate"):
        line += f"; success_rate {metrics['success_rate']['latest']:.3f}"
    if metrics.get("diverged"):
        line += f"; the mean reward became NaN at iteration {metrics['diverged_at_iteration']}"
    return line


def _verdict_line(metrics: dict[str, Any]) -> str:
    """One line on whether the run learned, preferring the task's success rate over reward."""
    success = metrics.get("success_rate")
    if success:
        return f"success_rate {success['latest']:.3f} (best {success['max']:.3f})"
    if metrics.get("best_reward") is None:
        return "no reward logged"
    return (
        f"mean reward {metrics['first_reward']} -> {metrics['latest_reward']} "
        f"(best {metrics['best_reward']} at iteration {metrics['best_iteration']})"
    )


def find_run_dir(text: str, job_id: str) -> str | None:
    """Return the rsl_rl run directory of *job_id*, from the experiment root its log names."""
    match = _LOG_DIR_RE.search(text)
    if not match:
        return None
    root = Path(match.group(1))
    if not root.is_dir():
        return None
    runs = [d for d in root.iterdir() if d.is_dir() and d.name.endswith(f"_{job_id}")]
    return str(runs[0]) if runs else None


def latest_model(run_dir: str | None) -> str | None:
    """Return the ``model_<iteration>.pt`` with the highest iteration in *run_dir*."""
    if not run_dir:
        return None
    models = []
    for path in Path(run_dir).glob("model_*.pt"):
        match = _MODEL_RE.match(path.name)
        if match:
            models.append((int(match.group(1)), path))
    return str(max(models)[1]) if models else None
