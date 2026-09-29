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

import json
import logging
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
_REWARD_RE = re.compile(r"Mean reward: (-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_STEPS_PER_S_RE = re.compile(r"Steps per second: (\d+)")
_TOTAL_STEPS_RE = re.compile(r"Total steps: (\d+)")
_LOG_DIR_RE = re.compile(r"Logging experiment in directory: (\S+)")
_TRAINING_TIME_RE = re.compile(r"Training time: (\d+(?:\.\d+)?) seconds")
_MODEL_RE = re.compile(r"^model_(\d+)\.pt\Z")

_JOB_FILE = "job.json"
_LOG_FILE = "train.log"
_TIMED_OUT_FILE = "timed_out"
_TAIL_LINES = 12

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
        return problems

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
        job_id = f"isaaclab-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        work_dir = Path(spec.output_dir).expanduser().resolve()
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

        if alive:
            return TrainResult(
                status="running",
                job_id=job_id,
                checkpoint_dir=run_dir,
                metrics=metrics,
                message=f"iteration {metrics['latest_iteration']}/{record.get('max_iterations')}",
            )
        if (job_dir / _TIMED_OUT_FILE).exists():
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=run_dir,
                metrics=metrics,
                message=f"{self.provider_name}: stopped after its timeout_s; log tail:\n{tail}",
            )
        if exit_code is None:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=run_dir,
                metrics=metrics,
                message=f"{self.provider_name}: the run ended without an exit status (killed); log tail:\n{tail}",
            )
        if exit_code != 0 or metrics["training_time_s"] is None:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=run_dir,
                metrics=metrics,
                message=f"{self.provider_name}: Isaac Lab exited {exit_code}; log tail:\n{tail}",
            )
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=run_dir,
            metrics=metrics,
            message=(
                f"finished {metrics['latest_iteration'] + 1 if metrics['latest_iteration'] is not None else 0}"
                f"/{record.get('max_iterations')} iterations in {metrics['training_time_s']} s; "
                f"latest checkpoint {model}"
            ),
        )

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
        ``latest_iteration`` (0-based, ``None`` before the first),
        ``first_reward`` / ``latest_reward`` / ``best_reward`` (mean episode
        reward), ``steps_per_s`` and ``total_steps`` of the latest iteration,
        ``training_time_s`` once rsl_rl reports it, and ``learning`` - whether
        the latest mean reward is above the first after at least two
        iterations.
    """
    iterations = [int(m.group(1)) for m in _ITERATION_RE.finditer(text)]
    rewards = [float(m) for m in _REWARD_RE.findall(text)]
    steps_per_s = [int(m) for m in _STEPS_PER_S_RE.findall(text)]
    total_steps = [int(m) for m in _TOTAL_STEPS_RE.findall(text)]
    training_time = _TRAINING_TIME_RE.search(text)
    return {
        "latest_iteration": iterations[-1] if iterations else None,
        "first_reward": rewards[0] if rewards else None,
        "latest_reward": rewards[-1] if rewards else None,
        "best_reward": max(rewards) if rewards else None,
        "steps_per_s": steps_per_s[-1] if steps_per_s else None,
        "total_steps": total_steps[-1] if total_steps else None,
        "training_time_s": float(training_time.group(1)) if training_time else None,
        "learning": len(rewards) >= 2 and rewards[-1] > rewards[0],
    }


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
