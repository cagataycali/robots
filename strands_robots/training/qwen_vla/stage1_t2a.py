"""Stage 1: Text-to-Action (T2A) DiT pretraining (section 3.2.3 / 5.2.1).

Trains the DiT action prior from language alone - VLM frozen, no images,
full-sequence prediction (+4.9pp over chunk per the ablation), Sigmoid-Normal
timestep distribution. Produces a decoder warm-start for Phase D (CPT).

The orchestration (config wiring, mixture sampling, loss assembly) is
structured so it can be exercised without torch; the actual optimizer step is
import-guarded behind the ``qwen-vla-train`` extra via :func:`run_t2a`.
"""

import logging

import numpy as np

from strands_robots.training.qwen_vla.config import T2AConfig
from strands_robots.training.qwen_vla.data.embodiment_tags import EmbodimentTag
from strands_robots.training.qwen_vla.data.language_action import LanguageActionGenerator
from strands_robots.training.qwen_vla.data.mixture import MixtureSampler
from strands_robots.training.qwen_vla.flow_matching import (
    interpolate,
    masked_flow_matching_loss,
    sample_timesteps,
    target_velocity,
)

logger = logging.getLogger(__name__)


def build_t2a_batch(
    generator: LanguageActionGenerator,
    embodiment: EmbodimentTag,
    config: T2AConfig,
    *,
    action_channels: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Assemble one T2A flow-matching training batch (no torch).

    Draws ``config.batch_size`` language-action examples, builds the
    straight-line flow interpolation, the target velocity, and the validity
    mask. Returns the arrays a torch step would consume. Exposed (and tested)
    independently so the data-flow is validated without a GPU.

    Args:
        generator: The text-only T2A corpus generator.
        embodiment: Embodiment tag (horizon H).
        config: Stage-1 config (batch_size, action_dim K, timestep_dist).
        action_channels: Valid action channels c (<= K).
        rng: Seeded NumPy generator.

    Returns:
        ``{"x_t", "target", "mask", "timesteps"}`` with batched shapes.
    """
    from strands_robots.policies.qwen_vla.normalize import build_channel_mask, pad_to_width

    h, k = embodiment.chunk_size, config.action_dim
    examples = generator.generate(config.batch_size)

    x1 = np.stack([pad_to_width(ex.action, k) for ex in examples])  # (B, H, K) targets
    x0 = rng.standard_normal(size=x1.shape)  # (B, H, K) noise
    t = sample_timesteps(config.batch_size, config.timestep_dist, rng=rng)
    x_t = interpolate(x0, x1, t)
    target = target_velocity(x0, x1)
    mask = np.stack([build_channel_mask(c=action_channels, k=k, h_task=h, h=h) for _ in examples])

    return {"x_t": x_t, "target": target, "mask": mask, "timesteps": t}


def run_t2a(
    config: T2AConfig,
    embodiment: EmbodimentTag,
    *,
    action_channels: int,
    model=None,
):
    """Run the Stage-1 T2A training loop.

    Requires the ``qwen-vla-train`` extra (torch) and a model exposing the DiT
    action expert. Validates the config, builds the synthetic/real mixture, and
    iterates ``config.max_steps`` flow-matching updates. The VLM stays frozen
    (``config.freeze_vlm``); only the DiT is optimized.

    Args:
        config: Stage-1 config.
        embodiment: Embodiment tag.
        action_channels: Valid action channels c.
        model: A loaded Qwen-VLA training model (DiT exposed). Required.

    Returns:
        A summary dict ``{"stage", "steps", "final_loss", "output_dir"}``.

    Raises:
        ImportError: If torch (the train extra) is not installed.
        ValueError: If *model* is None or the config is invalid.
    """
    config.validate()
    if model is None:
        raise ValueError("run_t2a requires a loaded model exposing the DiT action expert")

    from strands_robots.utils import require_optional

    require_optional("torch", extra="qwen-vla-train", purpose="Qwen-VLA T2A training")

    rng = np.random.default_rng(config.seed)
    generator = LanguageActionGenerator(embodiment=embodiment, action_channels=action_channels, seed=config.seed)
    # Mixture: synthetic_fraction synthetic + remainder "real" (caller supplies
    # real sources by extending the sampler; default is all-synthetic here).
    mixture = MixtureSampler.from_default_mixture(seed=config.seed)
    logger.info(
        "T2A start: steps=%d, batch=%d, dist=%s, synthetic_fraction=%.2f, mixture=%s",
        config.max_steps,
        config.batch_size,
        config.timestep_dist,
        config.synthetic_fraction,
        mixture.probabilities,
    )

    final_loss = float("nan")
    for step in range(config.max_steps):
        batch = build_t2a_batch(generator, embodiment, config, action_channels=action_channels, rng=rng)
        # The model performs: pred = dit(x_t, t, language_cond); we score with
        # the same masked CFM loss the NumPy reference validates.
        pred = model.predict_velocity(batch["x_t"], batch["timesteps"])
        final_loss = masked_flow_matching_loss(pred, batch["target"], batch["mask"])
        model.optimizer_step(final_loss)
        if step % 100 == 0:
            logger.info("T2A step %d/%d loss=%.4f", step, config.max_steps, final_loss)

    return {"stage": "t2a", "steps": config.max_steps, "final_loss": final_loss, "output_dir": config.output_dir}


__all__ = ["build_t2a_batch", "run_t2a"]
