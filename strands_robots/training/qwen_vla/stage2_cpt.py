"""Stage 2: Continued Pretraining (CPT) - joint VLM+DiT (section 5.2.1).

Unfreezes both modules and grounds the action prior in vision on the
heterogeneous mixture (Phase B). Uses the Beta timestep distribution and the
VL co-training loss (eq. 3, weight 0.1 VL / 1.0 action) to prevent
catastrophic forgetting. Zero-padding projection handles heterogeneous action
dims (section 5.2.2). Produces **Qwen-VLA-Base**.

Orchestration is torch-gated via :func:`run_cpt`.
"""

import logging
from typing import Any

from strands_robots.training.qwen_vla.config import CPTConfig
from strands_robots.training.qwen_vla.data.mixture import MixtureSampler

logger = logging.getLogger(__name__)


def run_cpt(config: CPTConfig, *, model: Any = None, data_sources: MixtureSampler | None = None) -> dict[str, Any]:
    """Run the Stage-2 CPT training loop (joint VLM + DiT).

    Args:
        config: Stage-2 config. ``warmup_checkpoint`` warm-starts the DiT from
            the Phase-C T2A decoder.
        model: A loaded Qwen-VLA training model (VLM + DiT). Required.
        data_sources: Mixture sampler over the heterogeneous corpora; defaults
            to the Table-1 proportions.

    Returns:
        Summary dict including the produced ``qwen_vla_base`` output dir.

    Raises:
        ImportError: If the ``qwen-vla-train`` extra (torch) is missing.
        ValueError: If *model* is None or the config is invalid.
    """
    config.validate()
    if model is None:
        raise ValueError("run_cpt requires a loaded model (VLM + DiT)")

    from strands_robots.utils import require_optional

    require_optional("torch", extra="qwen-vla-train", purpose="Qwen-VLA CPT training")

    sources = data_sources or MixtureSampler.from_default_mixture(seed=config.seed)
    logger.info(
        "CPT start: steps=%d, dist=%s, vl_weight=%.2f, warmup=%s, mixture=%s",
        config.max_steps,
        config.timestep_dist,
        config.vl_loss_weight,
        config.warmup_checkpoint,
        sources.probabilities,
    )
    if config.warmup_checkpoint:
        model.load_dit_warmstart(config.warmup_checkpoint)

    final_loss = float("nan")
    for step in range(config.max_steps):
        source = sources.sample()
        batch = model.sample_batch(source, config.batch_size)
        action_loss = model.flow_matching_loss(batch)
        vl_loss = model.vl_cotraining_loss(batch)
        total = config.action_loss_weight * action_loss + config.vl_loss_weight * vl_loss
        model.optimizer_step(total)
        final_loss = float(total)
        if step % 100 == 0:
            logger.info("CPT step %d/%d loss=%.4f (src=%s)", step, config.max_steps, final_loss, source)

    return {"stage": "cpt", "steps": config.max_steps, "final_loss": final_loss, "output_dir": config.output_dir}


__all__ = ["run_cpt"]
