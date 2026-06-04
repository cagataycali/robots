"""Stage 3: Supervised Fine-Tuning (SFT) on curated demos (section 4).

Specializes Qwen-VLA-Base on target-task demos via two tracks:
(a) multi-task (VQA + grounding + manipulation + navigation) and
(b) real-robot teleop from this repo's ``DatasetRecorder`` output (adapted by
:class:`~strands_robots.training.qwen_vla.data.lerobot_adapter.LeRobotAdapter`).
Loss weights 0.1 VL / 1.0 action; H=16 manip / 8 nav waypoints.

Orchestration is torch-gated via :func:`run_sft`.
"""

import logging

from strands_robots.training.qwen_vla.config import SFTConfig

logger = logging.getLogger(__name__)


def run_sft(config: SFTConfig, *, model=None, dataset=None):
    """Run the Stage-3 SFT training loop.

    Args:
        config: Stage-3 config. ``base_checkpoint`` loads Qwen-VLA-Base.
        model: A loaded Qwen-VLA training model. Required.
        dataset: An iterable of adapted ``QwenVlaSample`` (e.g. the
            ``LeRobotAdapter`` output over a DatasetRecorder corpus). Required.

    Returns:
        Summary dict for the SFT run.

    Raises:
        ImportError: If the ``qwen-vla-train`` extra (torch) is missing.
        ValueError: If *model* or *dataset* is None or the config is invalid.
    """
    config.validate()
    if model is None:
        raise ValueError("run_sft requires a loaded model")
    if dataset is None:
        raise ValueError("run_sft requires a dataset of adapted QwenVlaSample")

    from strands_robots.utils import require_optional

    require_optional("torch", extra="qwen-vla-train", purpose="Qwen-VLA SFT training")

    if config.base_checkpoint:
        model.load_checkpoint(config.base_checkpoint)
    logger.info(
        "SFT start: steps=%d, multi_task=%s, manip_H=%d, nav_H=%d",
        config.max_steps,
        config.multi_task,
        config.chunk_size,
        config.nav_chunk_size,
    )

    final_loss = float("nan")
    data_iter = iter(dataset)
    for step in range(config.max_steps):
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            sample = next(data_iter)
        action_loss = model.flow_matching_loss(sample)
        vl_loss = model.vl_cotraining_loss(sample) if config.multi_task else 0.0
        total = config.action_loss_weight * action_loss + config.vl_loss_weight * vl_loss
        model.optimizer_step(total)
        final_loss = float(total)

    return {"stage": "sft", "steps": config.max_steps, "final_loss": final_loss, "output_dir": config.output_dir}


__all__ = ["run_sft"]
