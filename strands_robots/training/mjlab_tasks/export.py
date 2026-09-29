"""Export an rsl_rl checkpoint of an mjlab task to ONNX with mjlab metadata.

mjlab's velocity / manipulation runners export on every ``save``; the default
runner (used by ``Strands-Reach-SO101``) does not. This helper does the same
export for any registered task and any ``model_*.pt`` so
``strands_robots.policies.rsl_rl_onnx`` can consume it on either engine.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any


def export_checkpoint(
    task: str,
    checkpoint: str | Path,
    out: str | Path | None = None,
    *,
    device: str = "cuda:0",
    run_name: str = "local",
) -> Path:
    """Load ``checkpoint`` for ``task`` into a 1-env runner and write ``out`` (.onnx).

    Returns the ONNX path. Imports torch / mjlab lazily (heavy, GPU).
    """
    import torch  # noqa: F401  (rsl_rl needs torch first)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from strands_robots.training import mjlab_tasks

    mjlab_tasks.register_all()
    checkpoint = Path(checkpoint)
    if out is None:
        out = checkpoint.parent / f"{checkpoint.parent.name}.onnx"
    out = Path(out)

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        _export_dynamic_batch(runner, out)
        attach_metadata_to_onnx(str(out), get_base_metadata(env, run_name))
    finally:
        env.close()
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m strands_robots.training.mjlab_tasks.export TASK CHECKPOINT [--out]``."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task")
    p.add_argument("checkpoint")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)
    print(export_checkpoint(a.task, a.checkpoint, a.out, device=a.device))


def _export_dynamic_batch(runner: Any, out: Path) -> None:
    """``runner.export_policy_to_onnx`` with a dynamic batch axis.

    mjlab exports with ``dynamic_axes={}``, which bakes ``obs: [1, obs_dim]`` into
    the graph, so one session call can only score one world. The vectorized
    evaluator (``vec_eval``) feeds ``(N, obs_dim)``; exporting the batch axis as
    dynamic costs nothing at N=1 and lets ``rsl_rl_onnx.act_batch`` run once per tick.
    """
    import os

    import torch

    onnx_model = runner.alg.get_policy().as_onnx(verbose=False)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(out.parent, exist_ok=True)
    dyn = {name: {0: "batch"} for name in [*onnx_model.input_names, *onnx_model.output_names]}
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        str(out),
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes=dyn,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
