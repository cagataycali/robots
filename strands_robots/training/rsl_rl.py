"""rsl_rl trainer on mjlab - GPU-vectorized PPO from a TrainSpec, ONNX out.

The *local* :class:`~strands_robots.training.base.Trainer` for the
``rsl_rl_onnx`` provider. It imports mjlab's own training entry point
(:func:`mjlab.scripts.train.run_train`) and calls it in-process, exactly as the
lerobot trainer calls ``lerobot.scripts.train.train``; nothing about PPO is
reimplemented here. What this module owns is the mapping from the
provider-agnostic :class:`TrainSpec` onto mjlab's typed config, the checkpoint
discovery, and the export step that turns the run's last ``model_*.pt`` into
the ONNX file :class:`~strands_robots.policies.rsl_rl_onnx.RslRlOnnxPolicy`
loads, so the agent's ``train_policy -> run_policy`` loop closes on one name::

    train_policy(provider="rsl_rl", extra={"task": "Strands-Reach-SO101"},
                 steps=30, batch_size=256, output_dir="runs/reach")
    run_policy(sim, policy_provider="rsl_rl_onnx",
               policy_config={"onnx_path": <exported_model>})

Field mapping (``TrainSpec`` -> mjlab):

====================  ============================================
``steps``             ``agent.max_iterations`` (PPO iterations)
``global_batch_size`` ``env.scene.num_envs`` (parallel worlds)
``save_freq``         ``agent.save_interval``
``learning_rate``     ``agent.algorithm.learning_rate``
``seed``              ``agent.seed`` (and ``env.seed``)
``output_dir``        ``log_root`` (runs land in ``<output_dir>/<experiment>/<stamp>_<run>``)
``resume``            ``agent.resume`` (latest run under ``output_dir``)
``extra["task"]``     mjlab task id; derived from ``embodiment`` when absent
``extra["run_name"]`` run suffix (default ``strands``)
``extra["device"]``   torch device (default ``cuda:0``; ``cpu`` is allowed but slow)
====================  ============================================

There is no dataset: RL trains against the simulator, so ``dataset_root`` is
neither required nor read. The task must be one mjlab's registry knows,
including the strands-robots tasks registered by
:mod:`strands_robots.training.mjlab_tasks`.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec
from strands_robots.utils import refusal_repr

#: The run name becomes a directory under the operator's ``output_dir``: a plain token only
#: (letters, digits, ``_``, ``-``; no leading ``-`` or ``.``), so an agent-supplied value can
#: neither traverse (``..``, ``/``, ``\\``) nor hide. Checked in :meth:`RslRlTrainer.validate`
#: and again at the write site.
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*\Z")


def run_name_problem(value: Any) -> str | None:
    """Error text when ``extra["run_name"]`` is not a plain token usable as one path segment."""
    if not isinstance(value, str) or not _RUN_NAME_RE.match(value):
        return (
            f"extra run_name {refusal_repr(value)} is not allowed (must match {_RUN_NAME_RE.pattern}: "
            "letters, digits, '_' and '-', no leading '-' or '.', no path separators)"
        )
    return None


logger = logging.getLogger(__name__)

#: embodiment -> default task when ``extra["task"]`` is not given.
DEFAULT_TASK_FOR_EMBODIMENT: dict[str, str] = {
    "so101": "Strands-Reach-SO101",
    "so-101": "Strands-Reach-SO101",
    "unitree_g1": "Mjlab-Velocity-Flat-Unitree-G1",
    "g1": "Mjlab-Velocity-Flat-Unitree-G1",
    "unitree_go1": "Mjlab-Velocity-Flat-Unitree-Go1",
    "go1": "Mjlab-Velocity-Flat-Unitree-Go1",
}

_MODEL_RE = re.compile(r"^model_(\d+)\.pt\Z")

#: Scalars read back from the run's TensorBoard log for the metrics verdict.
_METRIC_TAGS: tuple[str, ...] = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/value_function",
    "Loss/surrogate",
)


def _task_ids() -> list[str]:
    """Every task id mjlab's registry knows, including strands-robots' own."""
    import mjlab.tasks  # noqa: F401 - populates the registry
    from mjlab.tasks.registry import list_tasks

    from strands_robots.training import mjlab_tasks

    mjlab_tasks.register_all()
    return list(list_tasks())


def _pin_cuda_device(device: str) -> None:
    """Point mjlab at ``device`` through ``CUDA_VISIBLE_DEVICES``, which is where it reads it.

    An explicit ``cpu`` hides every GPU; ``cuda:N`` pins ordinal ``N``; a value
    already in the environment wins, since the operator set it on purpose.
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"


class RslRlTrainer(Trainer):
    """Train an rsl_rl PPO actor in mjlab and export it to ONNX."""

    #: RL against the simulator: no dataset is read.
    requires_dataset = False

    @property
    def provider_name(self) -> str:
        """Provider identity - pairs with ``RslRlOnnxPolicy.provider_name``."""
        return "rsl_rl_onnx"

    # ------------------------------------------------------------------ spec
    @staticmethod
    def task_for(spec: TrainSpec) -> str | None:
        """The mjlab task id *spec* names: ``extra["task"]`` first, then the embodiment table."""
        task = (spec.extra or {}).get("task")
        if task:
            return str(task)
        if spec.embodiment:
            return DEFAULT_TASK_FOR_EMBODIMENT.get(spec.embodiment.lower())
        return None

    def validate(self, spec: TrainSpec) -> list[str]:
        """Pure preflight: task resolvable, sizes positive, output_dir set; no GPU touched."""
        problems: list[str] = self._security_problems(spec)
        # The run name is one path segment under output_dir: refuse anything else here, before
        # any config is built (review on #4229: an agent-supplied value could traverse out).
        if (run_name := (spec.extra or {}).get("run_name")) is not None:
            if problem := run_name_problem(run_name):
                problems.append(problem)
        # Shared domains go through the Trainer gates (one owner per field).
        problems.extend(self._checkpoint_cadence_problems(spec))
        problems.extend(self._seed_problems(spec))
        problems.extend(self._resume_problems(spec))

        if not spec.output_dir:
            problems.append("output_dir is required (mjlab log_root; the run and its ONNX land under it)")

        task = self.task_for(spec)
        if not task:
            problems.append(
                "a task is required: extra={'task': <mjlab task id>} or an embodiment in "
                f"{sorted(DEFAULT_TASK_FOR_EMBODIMENT)}"
            )
        else:
            try:
                known = _task_ids()
            except ImportError as exc:  # mjlab not installed
                problems.append(f"mjlab is not importable ({exc}); install the [sim-mjlab] extra")
            else:
                if task not in known:
                    problems.append(f"unknown mjlab task {task!r}; known: {sorted(known)}")

        if spec.method != "full":
            problems.append(f"rsl_rl trains the actor from scratch; method must be 'full', got {spec.method!r}")
        if spec.dataset_root:
            problems.append("rsl_rl is RL against the simulator: dataset_root must be empty (nothing reads it)")

        problems.extend(self._run_size_problems(spec))
        problems.extend(self._learning_rate_problems(spec))
        return problems

    # ---------------------------------------------------------------- paths
    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Newest run dir under *output_dir* that holds a ``model_*.pt`` (stat only)."""
        root = Path(output_dir)
        if not root.is_dir():
            return None
        best: tuple[float, Path] | None = None
        for model in root.glob("**/model_*.pt"):
            m = _MODEL_RE.match(model.name)
            if not m:
                continue
            key = model.stat().st_mtime
            if best is None or key > best[0]:
                best = (key, model.parent)
        return str(best[1]) if best else None

    @staticmethod
    def latest_model_file(run_dir: str | Path) -> Path | None:
        """The highest-iteration ``model_<n>.pt`` in *run_dir*."""
        cands = []
        for p in Path(run_dir).glob("model_*.pt"):
            m = _MODEL_RE.match(p.name)
            if m:
                cands.append((int(m.group(1)), p))
        return max(cands)[1] if cands else None

    @staticmethod
    def _read_task(run_dir: Path) -> str | None:
        marker = run_dir / "strands_task.txt"
        return marker.read_text(encoding="utf-8").strip() if marker.is_file() else None

    # ---------------------------------------------------------------- train
    def train(self, spec: TrainSpec) -> TrainResult:
        """Run PPO in-process via mjlab's ``run_train``; export the last checkpoint to ONNX."""
        problems = self.validate(spec)
        if problems:
            return TrainResult(status="error", job_id="", message="validation failed: " + "; ".join(problems))
        self.prepare(spec)

        task = self.task_for(spec)
        assert task is not None
        extra = spec.extra or {}
        device = str(extra.get("device", "cuda:0"))
        run_name = extra.get("run_name", "strands")
        if problem := run_name_problem(run_name):
            # validate() refuses this first; the write site holds the same line on its own.
            raise ValueError(problem)
        run_name = str(run_name)

        import torch  # noqa: F401 - rsl_rl needs torch imported first
        from mjlab.scripts.train import TrainConfig, run_train

        cfg = TrainConfig.from_task(task)
        cfg.env.scene.num_envs = int(spec.global_batch_size)
        cfg.agent.max_iterations = int(spec.steps)
        cfg.agent.save_interval = max(1, min(spec.save_freq, int(spec.steps)))  # admitted by the cadence gate
        cfg.agent.run_name = run_name
        cfg.agent.resume = spec.resume
        if spec.seed is not None:
            cfg.agent.seed = spec.seed
        if spec.learning_rate is not None:
            cfg.agent.algorithm.learning_rate = float(spec.learning_rate)
        cfg.agent.logger = "tensorboard"
        cfg = TrainConfig(
            env=cfg.env,
            agent=cfg.agent,
            log_root=str(Path(spec.output_dir).resolve()),
        )

        log_root = Path(cfg.log_root) / cfg.agent.experiment_name
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = log_root / f"{stamp}_{run_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "strands_task.txt").write_text(task + "\n", encoding="utf-8")
        job_id = log_dir.name

        _pin_cuda_device(device)

        t0 = time.monotonic()
        logger.info("rsl_rl: %s num_envs=%d iterations=%d -> %s", task, cfg.env.scene.num_envs, spec.steps, log_dir)
        run_train(task, cfg, log_dir)
        wall_s = time.monotonic() - t0

        model = self.latest_model_file(log_dir)
        if model is None:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=str(log_dir),
                message=f"training returned but wrote no model_*.pt under {log_dir}",
            )
        metrics = self._metrics_from_tensorboard(log_dir)
        metrics.update(
            {
                "task": task,
                "iterations": int(spec.steps),
                "num_envs": int(cfg.env.scene.num_envs),
                "wall_s": round(wall_s, 1),
                "last_model": model.name,
            }
        )
        exported = self.export(spec, str(log_dir))
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=str(log_dir),
            exported_model=exported,
            metrics=metrics,
            message=(
                f"{task}: {spec.steps} PPO iterations x {cfg.env.scene.num_envs} envs in {wall_s:.0f} s; "
                f"mean_reward={metrics.get('Train/mean_reward', 'n/a')} -> {Path(exported).name}"
            ),
        )

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Export the highest-iteration ``model_*.pt`` in *checkpoint_dir* to ONNX (+ mjlab metadata)."""
        from strands_robots.training.mjlab_tasks.export import export_checkpoint

        run_dir = Path(checkpoint_dir)
        model = self.latest_model_file(run_dir)
        if model is None:
            raise FileNotFoundError(f"no model_*.pt under {run_dir}")
        task = self.task_for(spec) or self._read_task(run_dir)
        if not task:
            raise ValueError("cannot export: task unknown (pass extra={'task': ...} or keep strands_task.txt)")
        device = str((spec.extra or {}).get("device", "cuda:0"))
        out = run_dir / f"{model.stem}.onnx"
        if out.is_file() and out.stat().st_mtime >= model.stat().st_mtime:
            return str(out)
        return str(export_checkpoint(task, model, out, device=device, run_name=run_dir.name))

    # -------------------------------------------------------------- metrics
    @staticmethod
    def _metrics_from_tensorboard(run_dir: Path) -> dict[str, Any]:
        """Last value of each :data:`_METRIC_TAGS` scalar in the run's event file, if readable."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return {}
        try:
            acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
            acc.Reload()
            tags = set(acc.Tags().get("scalars", []))
        except Exception as exc:  # pragma: no cover - corrupt/absent event file
            logger.debug("tensorboard read failed: %s", exc)
            return {}
        out: dict[str, Any] = {}
        for tag in _METRIC_TAGS:
            if tag in tags:
                events = acc.Scalars(tag)
                if events:
                    out[tag] = round(float(events[-1].value), 4)
        for tag in sorted(tags):
            if tag.startswith("Metrics/"):
                events = acc.Scalars(tag)
                if events:
                    out[tag] = round(float(events[-1].value), 4)
        return out

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """One CUDA GPU; MuJoCo-Warp needs little VRAM at a few hundred envs."""
        return {"min_gpus": 1, "min_vram_gb": 8, "multinode": False}
