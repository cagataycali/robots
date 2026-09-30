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
  process group is stopped and the job reports ``error`` (``failure:
  "timeout"``). The launch wrapper enforces it itself, so it holds when nobody
  polls - an agent that died no longer leaves the run holding the GPU - and
  :meth:`IsaacLabTrainer.status` checks it again on every poll.

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
from collections.abc import Sequence
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
ACCEPTED_EXTRA_KEYS: tuple[str, ...] = (
    "task",
    "num_envs",
    "physics",
    "rl_library",
    "wait",
    "timeout_s",
    "overrides",
    "agent",
    "device",
    "video",
    "video_length",
    "video_interval",
    "deterministic",
)

#: A Hydra override path strands forwards: into the env cfg or the agent cfg.
_OVERRIDE_PATH_RE = re.compile(r"^(env|agent)(\.[A-Za-z_][A-Za-z0-9_]*)+\Z")
#: An override value that reaches Hydra as one token, unquoted.
_OVERRIDE_STR_RE = re.compile(r"^[A-Za-z0-9_.:/+-]{1,128}\Z")
#: ``extra['agent']``: the registry kwarg naming an agent config, e.g.
#: ``rsl_rl_recurrent_cfg_entry_point`` (recurrent / symmetry / distillation).
_AGENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}_cfg_entry_point\Z")
_DEVICE_RE = re.compile(r"^(cpu|cuda(:[0-9]{1,2})?)\Z")
#: ``TrainSpec.save_freq`` left at this default means "unset": Isaac Lab keeps
#: the task's own ``save_interval`` (50 for most rsl_rl configs).
_DEFAULT_SAVE_FREQ = 1_000

# Runs in the Isaac Lab interpreter (no Kit, about 5 s): loads the task's env
# and agent configs and reports every override path that names no field,
# because Hydra itself accepts an unknown ``env.`` path - ``env.episode_lenght_s=3``
# trains with exit 0, adds a new field and leaves ``episode_length_s`` alone.
_CFG_CHECK = """
import difflib, json, sys
import isaaclab_tasks  # noqa: F401 - registers the tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
task, agent, paths = json.loads(sys.argv[1])
out = {"problems": {}, "schedule": None}
try:
    cfgs = {"env": load_cfg_from_registry(task, "env_cfg_entry_point"), "agent": load_cfg_from_registry(task, agent)}
except Exception as exc:  # noqa: BLE001
    out["load_error"] = f"{type(exc).__name__}: {exc}"
    print("STRANDS_CFG_CHECK " + json.dumps(out))
    sys.exit(0)
out["schedule"] = getattr(getattr(cfgs["agent"], "algorithm", None), "schedule", None)
for path in paths:
    root, *parts = path.split(".")
    obj = cfgs[root]
    for i, part in enumerate(parts):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
            continue
        if not isinstance(obj, dict) and hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        names = list(obj) if isinstance(obj, dict) else [n for n in dir(obj) if not n.startswith("_")]
        out["problems"][path] = {
            "missing": ".".join([root, *parts[: i + 1]]),
            "close": difflib.get_close_matches(part, names, 3),
        }
        break
print("STRANDS_CFG_CHECK " + json.dumps(out))
"""

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
        problems.extend(self._resume_problems(spec))
        problems.extend(self._launch_topology_problems(spec))
        problems.extend(self._save_freq_problems(spec))
        problems.extend(_extra_problems(spec.extra or {}, ctx))
        problems.extend(self._forwarding_problems(spec))
        task = (spec.extra or {}).get("task")
        if isinstance(task, str) and _TASK_RE.match(task) and self._python:
            known = runtime.registered_tasks(self._python)
            if known and task not in known:
                close = difflib.get_close_matches(task, sorted(known), n=3, cutoff=0.6)
                hint = f"; did you mean {close}?" if close else ""
                problems.append(
                    f"{ctx}: extra['task'] {task!r} is not registered by the Isaac Lab install at {self._python} "
                    f"({len(known)} tasks){hint}"
                )
        return problems

    def _save_freq_problems(self, spec: TrainSpec) -> list[str]:
        """``save_freq`` becomes ``agent.save_interval`` (iterations); left at its default it is not sent."""
        problems = self._checkpoint_cadence_problems(spec)
        if not problems and spec.save_freq not in (0, _DEFAULT_SAVE_FREQ) and spec.save_freq > spec.steps:
            problems.append(
                f"{self.provider_name}: save_freq {spec.save_freq} is more iterations than steps={spec.steps}; "
                "no checkpoint but the last would be written"
            )
        return problems

    def _start_checkpoint(self, spec: TrainSpec) -> str | None:
        """The ``model_<iteration>.pt`` a run starts from: ``base_model``, or the newest one under ``resume``."""
        if spec.base_model:
            path = Path(spec.base_model).expanduser()
            return str(path.resolve()) if path.is_file() else latest_model(str(path.resolve()))
        if spec.resume:
            task = (spec.extra or {}).get("task")
            run_dir = self.latest_checkpoint(spec.output_dir, task=str(task)) if task else None
            return latest_model(run_dir) if run_dir else None
        return None

    def _forwarding_problems(self, spec: TrainSpec) -> list[str]:
        """Report TrainSpec fields and ``extra`` knobs that cannot be forwarded as asked."""
        ctx = self.provider_name
        problems: list[str] = []
        if spec.num_gpus != 1 or spec.num_nodes != 1:
            problems.append(
                f"{ctx}: num_gpus={spec.num_gpus} / num_nodes={spec.num_nodes} would run on one GPU here - this "
                "provider launches one process; Isaac Lab's multi-GPU run is `python -m torch.distributed.run "
                "--nproc_per_node=N -m isaaclab train ... --distributed`, which strands does not start"
            )
        if spec.base_model and spec.resume:
            problems.append(f"{ctx}: pass base_model or resume=True, not both - both name the checkpoint to start from")
        elif spec.base_model:
            path = Path(spec.base_model).expanduser()
            if not (path.is_file() and path.suffix == ".pt") and not (path.is_dir() and latest_model(str(path))):
                problems.append(
                    f"{ctx}: base_model {spec.base_model!r} is not an rsl_rl model_<iteration>.pt nor a run directory "
                    "holding one (Isaac Lab starts from --checkpoint <file>; Hub ids are not fetched)"
                )
        elif spec.resume and isinstance((spec.extra or {}).get("task"), str) and self._start_checkpoint(spec) is None:
            problems.append(
                f"{ctx}: resume=True but {spec.output_dir} holds no {(spec.extra or {}).get('task')} run with a "
                "model_<iteration>.pt to resume from"
            )
        overrides = (spec.extra or {}).get("overrides")
        paths = [str(k) for k in overrides] if isinstance(overrides, dict) else []
        if (
            (paths or "agent" in (spec.extra or {}))
            and not problems
            and self._python
            and not _extra_problems(spec.extra or {}, ctx)
        ):
            problems.extend(self._cfg_path_problems(spec, paths))
        return problems

    def _cfg_path_problems(self, spec: TrainSpec, paths: list[str]) -> list[str]:
        """Check override paths and the agent entry point against the task's real configs (see ``_CFG_CHECK``)."""
        extra = spec.extra or {}
        payload = json.dumps([extra["task"], extra.get("agent", "rsl_rl_cfg_entry_point"), paths])
        try:
            done = subprocess.run(  # noqa: S603 - argv, no shell; the interpreter is the operator's
                [str(self._python), "-c", _CFG_CHECK, payload],
                env=runtime.child_env(),
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [f"{self.provider_name}: could not check the overrides against {extra['task']}'s configs: {exc}"]
        line = next(
            (ln for ln in done.stdout.decode(errors="replace").splitlines() if ln.startswith("STRANDS_CFG_CHECK ")), ""
        )
        if not line:
            tail = (done.stdout + done.stderr).decode(errors="replace")[-400:]
            return [f"{self.provider_name}: the config check exited {done.returncode} with no answer: ...{tail}"]
        report = json.loads(line.removeprefix("STRANDS_CFG_CHECK "))
        if report.get("load_error"):
            return [
                f"{self.provider_name}: {extra['task']} has no loadable "
                f"{extra.get('agent', 'rsl_rl_cfg_entry_point')!r} config: {report['load_error']}"
            ]
        problems = []
        for path, info in sorted(report.get("problems", {}).items()):
            hint = f"; did you mean {info['close']}?" if info.get("close") else ""
            problems.append(
                f"{self.provider_name}: extra['overrides'] path {path!r} names no field ({info['missing']!r} does not "
                f"exist in {extra['task']}'s config){hint} - Hydra would add it silently and train the default"
            )
        return problems

    def _forwarded(self, spec: TrainSpec) -> tuple[list[str], list[str]]:
        """The flags and Hydra overrides *spec* becomes, beyond the fixed argv.

        One place, so the launched argv and the run record (which ``play``
        replays) cannot disagree.
        """
        extra = spec.extra or {}
        flags: list[str] = []
        if "num_envs" in extra:
            flags += ["--num_envs", str(extra["num_envs"])]
        if spec.seed is not None:
            flags += ["--seed", str(spec.seed)]
        if extra.get("agent"):
            flags += ["--agent", str(extra["agent"])]
        if extra.get("device"):
            flags += ["--device", str(extra["device"])]
        if extra.get("deterministic"):
            flags.append("--deterministic")
        if extra.get("video"):
            flags.append("--video")
            for key in ("video_length", "video_interval"):
                if key in extra:
                    flags += [f"--{key}", str(extra[key])]
        if (checkpoint := self._start_checkpoint(spec)) is not None:
            flags += ["--checkpoint", checkpoint]
        overrides: list[str] = []
        if "physics" in extra:
            overrides.append(f"physics={extra['physics']}")
        user = dict(extra.get("overrides") or {})
        if spec.learning_rate is not None:
            overrides.append(f"agent.algorithm.learning_rate={spec.learning_rate!r}")
            # rsl_rl's adaptive schedule (most Isaac Lab PPO configs) replaces the
            # rate from the first iteration and caps it at 1e-2, so a requested
            # rate is pinned unless the caller chose a schedule themselves.
            if "agent.algorithm.schedule" not in user:
                overrides.append("agent.algorithm.schedule=fixed")
        if spec.save_freq not in (0, _DEFAULT_SAVE_FREQ) and "agent.save_interval" not in user:
            overrides.append(f"agent.save_interval={int(spec.save_freq)}")
        overrides += [f"{path}={_hydra_value(value)}" for path, value in user.items()]
        return flags, overrides

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
        flags, overrides = self._forwarded(spec)
        return cmd + flags + overrides

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
        timeout_s = extra.get("timeout_s")
        proc = runtime.launch(
            cmd,
            cwd=work_dir,
            log_path=job_dir / _LOG_FILE,
            exit_file=job_dir / runtime.EXIT_CODE_FILE,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
            timed_out_file=job_dir / _TIMED_OUT_FILE,
        )
        _CHILDREN[job_id] = proc
        started = time.time()
        record = {
            "job_id": job_id,
            "pid": proc.pid,
            "cmd": cmd,
            "cwd": str(work_dir),
            "started": started,
            "deadline": started + float(timeout_s) if timeout_s is not None else None,
            "task": extra["task"],
            # rsl_rl continues a --checkpoint run's iteration count (model_4.pt
            # + 3 iterations logs "iteration 4/7"), so the run ends at start + steps.
            "max_iterations": spec.steps + (start := _checkpoint_iteration(self._start_checkpoint(spec))),
            "start_iteration": start,
            "run": run_record(spec, overrides=self._forwarded(spec)[1], checkpoint=self._start_checkpoint(spec)),
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
        if record.get("kind") == "record":
            return self._record_status(job_id, job_dir, record, text, exit_code, alive)
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
        if run.get("agent"):
            cmd += ["--agent", str(run["agent"])]
        # The environment the policy trained in: its physics preset and every
        # env.* override (terrain, episode length, randomization...). agent.*
        # overrides shaped the training only and are not re-applied.
        replay = [o for o in run.get("overrides") or [] if o.startswith(("physics=", "env."))]
        if run.get("physics") and not any(o.startswith("physics=") for o in replay):
            replay.insert(0, f"physics={run['physics']}")
        cmd += replay
        play_id = f"isaaclab-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
        play_dir = self._jobs_dir / play_id
        play_dir.mkdir(parents=True, exist_ok=False)
        proc = runtime.launch(
            cmd,
            cwd=Path(record["cwd"]),
            log_path=play_dir / _LOG_FILE,
            exit_file=play_dir / runtime.EXIT_CODE_FILE,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
            timed_out_file=play_dir / _TIMED_OUT_FILE,
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

    def record(
        self,
        job_id: str,
        dataset_dir: str,
        *,
        repo_id: str = "local/isaaclab_rollout",
        episodes: int = 4,
        frames: int = 300,
        camera: bool = True,
        camera_eye: Sequence[float] = (2.5, -2.5, 1.5),
        camera_target: Sequence[float] = (0.0, 0.0, 0.3),
        width: int = 320,
        height: int = 240,
        task_description: str | None = None,
        timeout_s: float | None = None,
        wait: bool = False,
    ) -> TrainResult:
        """Roll a finished run's policy out in Isaac Lab and record it as a LeRobotDataset.

        Isaac Lab's own ``play`` writes one viewport mp4 and has no route to a
        dataset, so every Isaac Lab dataset strands produced took an
        out-of-tree harness running strands inside the Isaac Lab interpreter.
        This launches ``_isaaclab_record_runner.py`` - shipped with strands,
        run by ``ISAACLAB_PYTHON``, importing no strands - which rolls the
        newest checkpoint out in ``episodes`` parallel environments (env ``i``
        is episode ``i``, from one reset to its first ``done`` or ``frames``)
        in the physics preset and ``env.*`` overrides the run trained with, and
        writes each episode's arrays. :meth:`status` of the returned job then
        converts them here, with :class:`~strands_robots.dataset_recorder.DatasetRecorder`:

        * ``observation.state`` - the robot's joint positions by name, then the
          policy's own observation vector, the root position (from the env
          origin) and the root quaternion as Isaac Lab reports it (x-y-z-w);
        * ``action`` - the policy's action, named by the action terms' joints;
        * ``observation.images.camera`` - a fixed camera per env at
          ``camera_eye`` looking at ``camera_target`` (both from the env
          origin), when ``camera``.

        Returns:
            ``running`` with the recording's own ``job_id``, or its verdict under
            ``wait``; the verdict's ``metrics['dataset']`` is the dataset root.
        """
        ctx = self.provider_name
        trained = self.status(job_id)
        if trained.status == "running":
            return TrainResult(status="error", job_id=job_id, message=f"{ctx}: job {job_id} is still training")
        if trained.metrics.get("kind") in ("play", "record"):
            return TrainResult(status="error", job_id=job_id, message=f"{ctx}: {job_id} is not a training job")
        model = trained.metrics.get("latest_model")
        if not trained.checkpoint_dir or not model:
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"{ctx}: job {job_id} has no checkpoint to record ({trained.message})",
            )
        for value, name in ((episodes, "episodes"), (frames, "frames"), (width, "width"), (height, "height")):
            if (error := positive_count_error(value, name, ctx)) is not None:
                return TrainResult(status="error", job_id=job_id, message=error)
        if error := boolean_flag_error(camera, "camera", ctx):
            return TrainResult(status="error", job_id=job_id, message=error)
        points = {}
        for point, name in ((camera_eye, "camera_eye"), (camera_target, "camera_target")):
            if not (
                isinstance(point, Sequence)
                and len(point) == 3
                and all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in point)
            ):
                return TrainResult(status="error", job_id=job_id, message=f"{ctx}: {name} must be three finite numbers")
            points[name] = ",".join(repr(float(v)) for v in point)
        if not isinstance(dataset_dir, str) or not dataset_dir.strip():
            return TrainResult(status="error", job_id=job_id, message=f"{ctx}: dataset_dir must be a directory path")
        if not isinstance(repo_id, str) or not re.match(r"^[\w.-]+/[\w.-]+\Z", repo_id):
            return TrainResult(status="error", job_id=job_id, message=f"{ctx}: repo_id must be '<owner>/<name>'")
        if timeout_s is not None and (error := positive_finite_number_error(timeout_s, "timeout_s", ctx)):
            return TrainResult(status="error", job_id=job_id, message=error)
        problems = runtime.runtime_problems(self._python, context=ctx)
        if problems:
            return TrainResult(status="error", job_id=job_id, message="; ".join(problems))
        record = json.loads((self._jobs_dir / job_id / _JOB_FILE).read_text(encoding="utf-8"))
        run = record.get("run") or {}
        if run.get("rl_library", "rsl_rl") != "rsl_rl":
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"{ctx}: recording replays rsl_rl policies; this run used {run['rl_library']}",
            )
        rec_id = f"isaaclab-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
        rec_dir = self._jobs_dir / rec_id
        rec_dir.mkdir(parents=True, exist_ok=False)
        runner_script = Path(__file__).with_name("_isaaclab_record_runner.py")
        cmd = [
            str(self._python),
            str(runner_script),
            "--task",
            str(record["task"]),
            "--checkpoint",
            str(model),
            "--agent",
            str(run.get("agent") or "rsl_rl_cfg_entry_point"),
            "--episodes",
            str(int(episodes)),
            "--frames",
            str(int(frames)),
            "--camera",
            "fixed" if camera else "none",
            "--eye",
            points["camera_eye"],
            "--target",
            points["camera_target"],
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
            "--out",
            str(rec_dir / "rollout"),
        ]
        if run.get("seed") is not None:
            cmd += ["--seed", str(run["seed"])]
        # The environment the policy trained in, as play() replays it.
        replay = [o for o in run.get("overrides") or [] if o.startswith(("physics=", "env."))]
        if run.get("physics") and not any(o.startswith("physics=") for o in replay):
            replay.insert(0, f"physics={run['physics']}")
        for override in replay:
            cmd += ["--override", override]
        proc = runtime.launch(
            cmd,
            cwd=Path(record["cwd"]),
            log_path=rec_dir / _LOG_FILE,
            exit_file=rec_dir / runtime.EXIT_CODE_FILE,
            timeout_s=float(timeout_s) if timeout_s is not None else None,
            timed_out_file=rec_dir / _TIMED_OUT_FILE,
        )
        _CHILDREN[rec_id] = proc
        started = time.time()
        (rec_dir / _JOB_FILE).write_text(
            json.dumps(
                {
                    "job_id": rec_id,
                    "kind": "record",
                    "of": job_id,
                    "pid": proc.pid,
                    "cmd": cmd,
                    "cwd": record["cwd"],
                    "started": started,
                    "deadline": started + float(timeout_s) if timeout_s is not None else None,
                    "task": record["task"],
                    "run_dir": trained.checkpoint_dir,
                    "checkpoint": model,
                    "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
                    "repo_id": repo_id,
                    "task_description": task_description or f"{record['task']}: rollout of a trained rsl_rl policy",
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info("isaaclab: recording %s (pid %d): %s", model, proc.pid, " ".join(cmd))
        if wait:
            result = self.status(rec_id)
            while result.status == "running":
                time.sleep(self._poll_interval_s)
                result = self.status(rec_id)
            return result
        return TrainResult(
            status="running",
            job_id=rec_id,
            checkpoint_dir=trained.checkpoint_dir,
            metrics={"pid": proc.pid, "kind": "record", "of": job_id, "checkpoint": model},
            message=f"recording {episodes} episodes of {Path(model).name} ({record['task']})",
        )

    def _record_status(
        self, job_id: str, job_dir: Path, record: dict[str, Any], text: str, exit_code: int | None, alive: bool
    ) -> TrainResult:
        """The verdict of a recording: the LeRobotDataset it wrote, converted here once the rollout ends."""
        metrics: dict[str, Any] = {"kind": "record", "of": record.get("of"), "checkpoint": record.get("checkpoint")}
        tail = "\n".join(text.strip().splitlines()[-_TAIL_LINES:])
        if alive:
            return TrainResult(status="running", job_id=job_id, metrics=metrics, message="recording the rollout")
        done_file = job_dir / "dataset.json"
        if done_file.is_file():
            metrics.update(json.loads(done_file.read_text(encoding="utf-8")))
            return TrainResult(
                status="success",
                job_id=job_id,
                checkpoint_dir=record.get("run_dir"),
                metrics=metrics,
                message=_recorded_line(metrics),
            )
        raw = job_dir / "rollout"
        if exit_code != 0 or not (raw / "meta.json").is_file():
            failure, error_line = classify_failure(text, {})
            if (job_dir / _TIMED_OUT_FILE).exists():
                failure = "timeout"
            metrics["failure"] = failure or "exit_status"
            metrics["error"] = error_line or f"the rollout exited {exit_code} without its episodes"
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=record.get("run_dir"),
                metrics=metrics,
                message=f"{self.provider_name}: recording failed: {metrics['error']}; log tail:\n{tail}",
            )
        try:
            summary = rollout_to_dataset(
                raw, record["dataset_dir"], repo_id=record["repo_id"], task=record["task_description"]
            )
        except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
            metrics.update(failure="dataset", error=f"{type(exc).__name__}: {exc}")
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=record.get("run_dir"),
                metrics=metrics,
                message=f"{self.provider_name}: the rollout finished but the dataset could not be written: {metrics['error']}",
            )
        done_file.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        metrics.update(summary)
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=record.get("run_dir"),
            metrics=metrics,
            message=_recorded_line(metrics),
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
            spec: The validated spec. Its ``extra['task']`` picks the run: when
                *checkpoint_dir* trained another task, the newest run of the
                asked-for task under ``spec.output_dir`` is exported instead.
            checkpoint_dir: The run directory :meth:`latest_checkpoint` returned.

        Returns:
            The ``strands_policy`` directory.

        Raises:
            FileNotFoundError: If the run directory holds no ``model_<iteration>.pt``,
                or ``spec.output_dir`` holds no run of the asked-for task.
        """
        from strands_robots.training.rl import rsl_rl

        task = (spec.extra or {}).get("task") if spec is not None else None
        if task and self.run_task(checkpoint_dir) != task:
            # The caller's pick is not a run of the task it asked for (the
            # train_policy tool asks for the newest run of ANY task): take the
            # newest run of this one, or refuse.
            chosen = self.latest_checkpoint(spec.output_dir, task=str(task)) if spec is not None else None
            if chosen is None:
                raise FileNotFoundError(
                    f"{self.provider_name}: no {task} run under {spec.output_dir if spec else checkpoint_dir} "
                    f"to export ({checkpoint_dir} trained {self.run_task(checkpoint_dir) or 'an unrecorded task'})"
                )
            checkpoint_dir = chosen
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

    def latest_checkpoint(self, output_dir: str, task: str | None = None) -> str | None:
        """Return the newest rsl_rl run directory under ``output_dir`` holding a ``model_*.pt``.

        Newest by the run's own start time - the ``YYYY-MM-DD_HH-MM-SS`` prefix
        Isaac Lab names the folder with - not by modification time, which a
        later play, export or copy of an older run bumps. With *task*, only
        runs of that task qualify (see :meth:`run_task`): one ``output_dir``
        holding an ``Isaac-Cartpole`` and an ``Isaac-Cartpole-Camera`` run
        exported the camera policy when the plain one was asked for.
        """
        root = Path(output_dir).expanduser() / "logs" / "rsl_rl"
        if not root.is_dir():
            return None
        runs = [d for d in root.glob("*/*") if d.is_dir() and latest_model(str(d))]
        if task is not None:
            runs = [d for d in runs if self.run_task(d) == task]
        if not runs:
            return None
        return str(max(runs, key=lambda d: (d.name[:19], d.stat().st_mtime)))

    def run_task(self, run_dir: str | Path) -> str | None:
        """The Isaac Lab task a run directory trained, or ``None`` when nothing records it.

        Read from the run record beside the checkpoints, else from the job
        record whose id the folder name ends with (``--run_name`` is the job id).
        """
        run_dir = Path(run_dir)
        record_path = run_dir / RUN_RECORD_FILE
        try:
            task = json.loads(record_path.read_text(encoding="utf-8")).get("task")
        except (OSError, ValueError):
            task = None
        if task:
            return str(task)
        job_id = run_dir.name[20:]
        if job_id.startswith("isaaclab-"):
            try:
                job = json.loads((self._jobs_dir / job_id / _JOB_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return str(job["task"]) if job.get("task") else None
        return None


def _checkpoint_iteration(path: str | None) -> int:
    """The iteration a ``model_<iteration>.pt`` was saved at; 0 for none."""
    match = re.search(r"model_(\d+)\.pt\Z", path or "")
    return int(match.group(1)) if match else 0


def _hydra_value(value: Any) -> str:
    """One override value as Hydra reads it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value) if isinstance(value, float) else str(value)


def _override_problems(overrides: Any, ctx: str) -> list[str]:
    """Shape of ``extra['overrides']``: ``{"env.a.b" | "agent.x.y": scalar}``."""
    if not isinstance(overrides, dict) or not overrides:
        return [f"{ctx}: extra['overrides'] must be a non-empty dict of 'env.*' / 'agent.*' path -> value"]
    problems = []
    for path, value in overrides.items():
        if not isinstance(path, str) or not _OVERRIDE_PATH_RE.match(path):
            problems.append(
                f"{ctx}: extra['overrides'] key {refusal_repr(path)} is not an 'env.<field>...' or "
                "'agent.<field>...' path"
            )
        elif path.startswith("agent.algorithm.learning_rate"):
            problems.append(f"{ctx}: set the learning rate with TrainSpec.learning_rate, not extra['overrides']")
        ok = (isinstance(value, bool) or (isinstance(value, int | float) and math.isfinite(value))) or (
            isinstance(value, str) and _OVERRIDE_STR_RE.match(value)
        )
        if not ok:
            problems.append(
                f"{ctx}: extra['overrides'][{path!r}] must be a finite number, a bool or a plain token, "
                f"got {refusal_repr(value)}"
            )
    return problems


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
    if "overrides" in extra:
        problems.extend(_override_problems(extra["overrides"], ctx))
    if "agent" in extra and not (isinstance(extra["agent"], str) and _AGENT_RE.match(extra["agent"])):
        problems.append(
            f"{ctx}: extra['agent'] must name an agent config entry point such as "
            f"'rsl_rl_recurrent_cfg_entry_point', got {refusal_repr(extra['agent'])}"
        )
    if "device" in extra and not (isinstance(extra["device"], str) and _DEVICE_RE.match(extra["device"])):
        problems.append(
            f"{ctx}: extra['device'] must be 'cpu', 'cuda' or 'cuda:<n>', got {refusal_repr(extra['device'])}"
        )
    for key in ("video", "deterministic"):
        if key in extra and (error := boolean_flag_error(extra[key], f"extra['{key}']", ctx)) is not None:
            problems.append(error)
    for key in ("video_length", "video_interval"):
        if key in extra:
            if (error := positive_count_error(extra[key], f"extra['{key}']", ctx)) is not None:
                problems.append(error)
            elif not extra.get("video"):
                problems.append(f"{ctx}: extra['{key}'] is read only with extra['video']=True")
    return problems


def rollout_to_dataset(raw_dir: str | Path, dataset_dir: str, *, repo_id: str, task: str) -> dict[str, Any]:
    """Convert what ``_isaaclab_record_runner.py`` wrote into a LeRobotDataset at *dataset_dir*.

    One dataset episode per rollout episode, with ``observation.state`` = joint
    positions by name + ``policy_obs`` + ``root_pos`` + ``root_quat`` (x-y-z-w),
    ``action`` named by the runner's action names, and
    ``observation.images.camera`` when the rollout had a camera. An episode
    with no frames is skipped. Returns what the dataset holds.
    """
    import numpy as np

    from strands_robots.dataset_recorder import DatasetRecorder

    raw = Path(raw_dir)
    meta = json.loads((raw / "meta.json").read_text(encoding="utf-8"))
    joints, actions = list(meta["joint_names"]), list(meta["action_names"])
    cam = meta.get("camera")
    recorder = DatasetRecorder.create(
        repo_id=repo_id,
        fps=int(meta["fps"]),
        robot_type=str(meta["task"]),
        joint_names=joints,
        action_names=actions,
        camera_keys=["camera"] if cam else None,
        camera_dims={"camera": (int(cam["height"]), int(cam["width"]))} if cam else None,
        extra_state_specs=[
            ("policy_obs", [str(k) for k in range(int(meta["policy_obs_dim"]))]),
            ("root_pos", ["x", "y", "z"]),
            ("root_quat", ["x", "y", "z", "w"]),
        ],
        task=task,
        root=dataset_dir,
        use_videos=bool(cam),
        overwrite=True,
    )
    saved, frames = [], 0
    for i in range(int(meta["episodes"])):
        ep = np.load(raw / f"episode_{i:03d}.npz")
        n = int(ep["joint_pos"].shape[0])
        if n == 0:
            continue
        for t in range(n):
            observation: dict[str, Any] = {name: float(ep["joint_pos"][t, k]) for k, name in enumerate(joints)}
            observation.update(policy_obs=ep["policy_obs"][t], root_pos=ep["root_pos"][t], root_quat=ep["root_quat"][t])
            if cam:
                observation["camera"] = ep["image"][t]
            recorder.add_frame(
                observation,
                {name: float(ep["action"][t, k]) for k, name in enumerate(actions)},
                task=task,
                camera_keys=["camera"] if cam else None,
            )
        recorder.save_episode()
        saved.append(i)
        frames += n
    recorder.finalize()
    return {
        "dataset": str(Path(dataset_dir)),
        "repo_id": repo_id,
        "episodes": len(saved),
        "frames": frames,
        "fps": int(meta["fps"]),
        "episode_len": meta["episode_len"],
        "episode_return": meta["episode_return"],
        "episode_end": meta["episode_end"],
        "camera": bool(cam),
    }


def _recorded_line(metrics: dict[str, Any]) -> str:
    returns = metrics.get("episode_return") or []
    mean = f"; mean return {sum(returns) / len(returns):.3f}" if returns else ""
    return (
        f"recorded {metrics.get('episodes')} episodes, {metrics.get('frames')} frames at {metrics.get('fps')} fps "
        f"into {metrics.get('dataset')} ({metrics.get('repo_id')}){mean}"
    )


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


def run_record(spec: TrainSpec, overrides: list[str] | None = None, checkpoint: str | None = None) -> dict[str, Any]:
    """Return how *spec* trains: task, physics preset, environments, seed and overrides.

    Written as :data:`RUN_RECORD_FILE` next to the checkpoints, so replaying a
    checkpoint can use the simulator it was trained in - a policy trained on
    PhysX and replayed on the task's default Newton preset falls within a second.
    """
    extra = spec.extra or {}
    if overrides is None:
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
        "agent": extra.get("agent"),
        "checkpoint": checkpoint,
        "overrides": list(overrides),
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
