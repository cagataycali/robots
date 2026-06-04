#!/usr/bin/env python3
"""Qwen-VLA training launcher tool (Phase G).

An ``@tool`` that lets an agent trigger any of the 4 training stages
(T2A -> CPT -> SFT -> RL), inspect stage configs, generate a text-only T2A
corpus, and hot-swap a freshly-trained checkpoint into a running Qwen-VLA
SERVICE policy - closing the collect -> tune -> redeploy loop.

Security (PR #90/#92 lessons): every caller-supplied path
(``output_dir``, ``checkpoint``, ``dataset_path``) is validated via
``validate_save_path`` (no traversal, no protected dirs) and the embodiment
name is allowlisted against the registry. Returns structured dicts, never
raises (AGENTS.md AgentTool contract).

The actual GPU training run is delegated to the torch-gated ``run_*``
entrypoints and requires the ``qwen-vla-train`` extra; without it (or without
a model object) the tool returns a clear, actionable error.
"""

from typing import Any

from strands import tool

from strands_robots.tools._path_validation import validate_save_path
from strands_robots.training.qwen_vla.config import CPTConfig, RLConfig, SFTConfig, T2AConfig
from strands_robots.training.qwen_vla.data.embodiment_tags import EMBODIMENT_TAGS, get_embodiment_tag

_STAGE_CONFIGS = {"t2a": T2AConfig, "cpt": CPTConfig, "sft": SFTConfig, "rl": RLConfig}


def _config_for_stage(stage: str, *, max_steps: int | None, batch_size: int | None, output_dir: str) -> Any:
    """Build a validated stage config with optional overrides."""
    cfg_cls = _STAGE_CONFIGS[stage]
    cfg = cfg_cls(output_dir=output_dir)
    if max_steps is not None and hasattr(cfg, "max_steps"):
        cfg.max_steps = max_steps
    if batch_size is not None:
        cfg.batch_size = batch_size
    cfg.validate()
    return cfg


@tool
def qwen_vla_train(
    action: str,
    stage: str = "t2a",
    embodiment: str = "so100",
    output_dir: str = "checkpoints/qwen_vla",
    checkpoint: str | None = None,
    dataset_path: str | None = None,
    action_channels: int = 7,
    max_steps: int | None = None,
    batch_size: int | None = None,
    corpus_size: int = 16,
    server_host: str = "127.0.0.1",
    server_port: int = 5556,
) -> dict[str, Any]:
    """Drive Qwen-VLA training stages and redeploy trained checkpoints.

    Actions:
        - ``stages``: List the four training stages and their key defaults.
        - ``config``: Show the resolved config for ``stage`` (with overrides).
        - ``corpus``: Generate a small text-only T2A corpus preview for
          ``embodiment`` (no GPU; validates the data path end-to-end).
        - ``train``: Launch a stage. Requires the ``qwen-vla-train`` extra and
          a loaded model (not available from a bare tool call) - returns a
          clear setup error otherwise, with the exact run_* call to make.
        - ``hotswap``: Point a running SERVICE policy at a new checkpoint
          (validates the checkpoint path + server reachability; the actual
          swap is performed by the server's reload endpoint).

    Args:
        action: One of ``stages``, ``config``, ``corpus``, ``train``, ``hotswap``.
        stage: Training stage: ``t2a`` / ``cpt`` / ``sft`` / ``rl``.
        embodiment: Registered embodiment / config name (or alias).
        output_dir: Where checkpoints are written (validated).
        checkpoint: Warm-start / base / sft checkpoint path (validated).
        dataset_path: SFT dataset path (validated).
        action_channels: Valid action channels c for the embodiment.
        max_steps: Override the stage's ``max_steps``.
        batch_size: Override the stage's ``batch_size``.
        corpus_size: Number of T2A examples to preview (``corpus`` action).
        server_host: SERVICE policy host for ``hotswap`` (loopback default).
        server_port: SERVICE policy port for ``hotswap``.

    Returns:
        A structured ``{"status": ...}`` dict. Never raises.
    """
    if action not in ("stages", "config", "corpus", "train", "hotswap"):
        return {
            "status": "error",
            "message": f"Unknown action {action!r}. Valid: stages, config, corpus, train, hotswap",
        }

    if action == "stages":
        return {
            "status": "success",
            "stages": [
                {"name": "t2a", "desc": "Text-to-Action DiT pretraining (VLM frozen, no images)"},
                {"name": "cpt", "desc": "Continued pretraining (joint VLM+DiT) -> Qwen-VLA-Base"},
                {"name": "sft", "desc": "Supervised fine-tuning (multi-task + teleop)"},
                {"name": "rl", "desc": "PPO+GAE on sim success -> Qwen-VLA-Instruct"},
            ],
        }

    # All remaining actions touch the embodiment / paths - validate first.
    if embodiment not in EMBODIMENT_TAGS:
        try:
            get_embodiment_tag(embodiment)  # resolves aliases (e.g. "aloha")
        except ValueError as e:
            return {"status": "error", "message": str(e)}

    for label, path in (("output_dir", output_dir), ("checkpoint", checkpoint), ("dataset_path", dataset_path)):
        if path is not None:
            try:
                validate_save_path(path, label=label)
            except ValueError as e:
                return {"status": "error", "message": str(e)}

    if stage not in _STAGE_CONFIGS:
        return {"status": "error", "message": f"Unknown stage {stage!r}. Valid: {sorted(_STAGE_CONFIGS)}"}

    if action == "config":
        try:
            cfg = _config_for_stage(stage, max_steps=max_steps, batch_size=batch_size, output_dir=output_dir)
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        # Render the dataclass to a plain dict for the agent.
        cfg_dict = {k: (v.value if hasattr(v, "value") else v) for k, v in vars(cfg).items()}
        return {"status": "success", "stage": stage, "config": cfg_dict}

    if action == "corpus":
        try:
            from strands_robots.training.qwen_vla.data.language_action import LanguageActionGenerator

            tag = get_embodiment_tag(embodiment)
            gen = LanguageActionGenerator(embodiment=tag, action_channels=action_channels, seed=0)
            examples = gen.generate(corpus_size)
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        return {
            "status": "success",
            "embodiment": embodiment,
            "count": len(examples),
            "preview": [{"family": e.family, "instruction": e.instruction, "prompt": e.prompt} for e in examples[:5]],
        }

    if action == "train":
        return {
            "status": "error",
            "message": (
                f"Stage '{stage}' training requires the 'qwen-vla-train' extra (torch) AND a loaded model, "
                f"which a bare tool call cannot construct. Run it from a training script:\n"
                f"  from strands_robots.training.qwen_vla import run_{stage}\n"
                f"  run_{stage}(config, model=<loaded model>, ...)\n"
                f"See docs/qwen_vla.md (training section) for the full setup."
            ),
        }

    # action == "hotswap"
    if checkpoint is None:
        return {"status": "error", "message": "hotswap requires a 'checkpoint' path to swap in"}
    return _hotswap(checkpoint=checkpoint, host=server_host, port=server_port)


def _hotswap(*, checkpoint: str, host: str, port: int) -> dict[str, Any]:
    """Ask a running Qwen-VLA SERVICE policy to reload a new checkpoint."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            reachable = sock.connect_ex((host if host != "localhost" else "127.0.0.1", port)) == 0
    except OSError:
        reachable = False
    if not reachable:
        return {"status": "error", "message": f"No Qwen-VLA server reachable at {host}:{port} to hot-swap"}

    try:
        from strands_robots.policies.qwen_vla.client import QwenVlaInferenceClient

        client = QwenVlaInferenceClient(host=host, port=port, timeout_ms=10000)
        resp = client.call_endpoint("reload", {"checkpoint": checkpoint})
    except ImportError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:  # noqa: BLE001 - surface wire/endpoint error to agent
        return {"status": "error", "message": f"Hot-swap failed (server may not support 'reload'): {e}"}

    return {"status": "success", "host": host, "port": port, "checkpoint": checkpoint, "server_response": resp}


__all__ = ["qwen_vla_train"]
