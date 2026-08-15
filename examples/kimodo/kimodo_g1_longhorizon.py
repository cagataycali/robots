"""Kimodo long-horizon motion generation via strands_robots.

Demonstrates that ALL motion families we generate in
``~/kimodo-longhorizon/`` (patrol / warmup / dance / combat / social /
workflow / mixed — 25 atomic slugs) are reachable through the standard
:class:`~strands_robots.policies.kimodo.KimodoPolicy` interface.

The composer templates from ``kimodo-longhorizon/composer.py`` are ported
here as pure-Python (no library dependency) so any user can:

1. Pick a family (or ``random``).
2. Get a list of prompts (one per atomic slug).
3. Feed each prompt to ``KimodoPolicy`` in turn; when the prompt changes
   the policy re-samples internally (see ``KimodoPolicy._synthesise``).
4. Concatenate the produced frames — that's a long-horizon trajectory.

Run:
    python examples/kimodo/kimodo_g1_longhorizon.py --family mixed --seed 0
    python examples/kimodo/kimodo_g1_longhorizon.py --family all       # one per family
    python examples/kimodo/kimodo_g1_longhorizon.py --slugs walk,turn,wave,bow

For the physics-tracked version (Kimodo → WBC/PD → MuJoCo), see
``kimodo_g1_walking.py`` and swap the single ``prompt`` for the list this
composer emits.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from pathlib import Path

# The 25 atomic motion slugs Kimodo has been validated on. Each maps to a
# natural-language prompt that the diffusion sampler understands well.
SLUG_PROMPTS: dict[str, str] = {
    "arms_up": "raise both arms straight up overhead and hold",
    "bow": "bow forward from the waist in a formal greeting",
    "boxing": "throw a series of boxing jabs and crosses",
    "clap": "clap hands together rhythmically in front of the chest",
    "crouch": "crouch down low to the ground and hold",
    "dance": "perform a rhythmic dance step with the whole body",
    "jump": "jump straight up in place and land softly",
    "kick": "throw a front kick with the right leg",
    "meditate": "stand still with arms relaxed at the sides in a meditative pose",
    "pickup": "reach down and pick up an object from the floor",
    "pull": "pull an object toward the body with both hands",
    "push": "push an object away with both hands",
    "reach_forward": "reach both arms forward at shoulder height",
    "run": "run forward at a steady pace",
    "salute": "raise the right hand in a formal salute",
    "sidestep": "sidestep to the right with balanced steps",
    "squats": "perform a set of bodyweight squats",
    "stretch": "stretch both arms overhead and sideways",
    "sway": "sway the torso gently from side to side",
    "tpose": "stand in a t-pose with arms straight out to the sides",
    "turn": "turn ninety degrees to the right in place",
    "twist": "twist the torso left and right rhythmically",
    "walk": "walk forward with confident strides",
    "walk_backward": "walk backward carefully",
    "wave": "wave the right hand in a friendly greeting",
}

# Sub-pools mirror ``kimodo-longhorizon/composer.py`` exactly so users can
# reproduce the streaming dataset families through the library.
POOLS: dict[str, list[str]] = {
    "locomotion": ["walk", "walk_backward", "run", "turn", "sidestep", "crouch", "jump"],
    "fitness": ["stretch", "squats", "reach_forward", "jump", "arms_up", "twist", "sway"],
    "dance": ["dance", "sway", "twist", "clap", "jump", "arms_up"],
    "combat": ["boxing", "kick", "crouch", "sidestep", "walk", "jump", "push"],
    "social": ["wave", "bow", "clap", "salute", "meditate", "arms_up"],
    "manip": ["reach_forward", "pickup", "push", "pull", "crouch", "walk"],
    "still": ["tpose", "meditate", "sway", "arms_up"],
}

FAMILY_TEMPLATES: dict[str, tuple[str, tuple[int, int]]] = {
    "patrol":   ("locomotion", (4, 7)),
    "warmup":   ("fitness",    (4, 6)),
    "dance":    ("dance",      (5, 8)),
    "combat":   ("combat",     (4, 7)),
    "social":   ("social",     (4, 6)),
    "workflow": ("manip",      (4, 6)),
    "mixed":    ("*",          (5, 8)),  # cross-family
}


def _pick(pool: list[str], k: int, rng: random.Random) -> list[str]:
    """Pick k from pool with no immediate repeats."""
    out: list[str] = []
    last = None
    for _ in range(k):
        cand = rng.choice(pool)
        while cand == last and len(pool) > 1:
            cand = rng.choice(pool)
        out.append(cand)
        last = cand
    return out


def compose(family: str, rng: random.Random) -> tuple[list[str], str]:
    """Compose a slug chain for the given family.

    Returns:
        ``(slugs, description)`` where ``slugs`` is the list of atomic
        motion slugs and ``description`` is a natural-language long-horizon
        task string.
    """
    if family not in FAMILY_TEMPLATES:
        raise ValueError(
            f"unknown family {family!r}; valid: {sorted(FAMILY_TEMPLATES)}"
        )
    pool_name, (kmin, kmax) = FAMILY_TEMPLATES[family]
    k = rng.randint(kmin, kmax)
    if pool_name == "*":
        all_slugs = list(SLUG_PROMPTS)
        slugs = _pick(all_slugs, k, rng)
    else:
        slugs = _pick(POOLS[pool_name], k, rng)
    prose = {
        "patrol": "patrol the area",
        "warmup": "warmup routine",
        "dance": "perform a dance sequence",
        "combat": "combat drill",
        "social": "social greeting",
        "workflow": "workflow of manipulation",
        "mixed": "mixed multi-task sequence",
    }[family]
    desc = f"{prose}: " + " → ".join(s.replace("_", " ") for s in slugs)
    return slugs, desc


async def run_chain(
    slugs: list[str],
    seed: int = 0,
    diffusion_steps: int = 32,
    guidance_scale: float = 2.5,
    stub: bool = False,
) -> list[list[dict[str, float]]]:
    """Run the full slug chain through KimodoPolicy.

    Args:
        slugs: Atomic motion slugs to sequence (each becomes one prompt).
        seed: Base seed; each slug offsets by its index for reproducibility.
        diffusion_steps: Kimodo sampler steps per prompt.
        guidance_scale: Classifier-free guidance strength.
        stub: If True, use a zero-motion stub agent (no torch / weights;
            for CI and API-shape verification).

    Returns:
        A list where element ``i`` is the list of per-frame action-dicts
        Kimodo produced for slug ``slugs[i]``.
    """
    # Late import: heavy deps (diffusers/torch) only loaded when actually
    # asked to run the real sampler.
    from strands_robots.policies.kimodo import (
        KIMODO_G1_JOINTS,
        KimodoConfig,
        KimodoPolicy,
    )

    if stub:
        # Local zero-motion stub — proves the API surface + composer without
        # pulling torch/diffusers/CUDA. Exercises get_actions + prompt-swap.
        import numpy as np

        class _Stub:
            """Emits ``num_frames`` frames of zeros (with a valid root)."""

            def sample(
                self,
                prompt: str,
                diffusion_steps: int,
                guidance_scale: float,
                seed: int | None,
                num_frames: int,
            ):
                qpos = np.zeros((num_frames, 7 + len(KIMODO_G1_JOINTS)), dtype=np.float64)
                qpos[:, 3] = 1.0  # unit quaternion (w=1)
                return qpos

        policy = KimodoPolicy(config=KimodoConfig(), motion_agent=_Stub())
    else:
        policy = KimodoPolicy(
            config=KimodoConfig(
                diffusion_steps=diffusion_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        )

    all_frames: list[list[dict[str, float]]] = []
    for i, slug in enumerate(slugs):
        prompt = SLUG_PROMPTS[slug]
        # Give Kimodo a fresh seed per slug so successive prompts don't
        # collapse to the same buffer.
        policy.reset(seed=seed + i)

        slug_frames: list[dict[str, float]] = []
        # KimodoPolicy synthesises on first call, streams thereafter. Ask
        # for ~1 second of motion per slug (Kimodo default 120 frames @
        # 30 Hz ≈ 4 s; grabbing the first 30 is a fair sample).
        target_frames = 30
        for _ in range(target_frames):
            actions = await policy.get_actions({}, prompt)
            slug_frames.append(actions[0])
        all_frames.append(slug_frames)

    return all_frames


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--family",
        choices=sorted(FAMILY_TEMPLATES) + ["all"],
        default="mixed",
        help="Motion family template; 'all' runs one chain per family.",
    )
    p.add_argument(
        "--slugs",
        default="",
        help="Comma-separated atomic slugs to override the family template.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diffusion-steps", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument(
        "--stub",
        action="store_true",
        help="Use zero-motion stub agent (no torch/diffusers/CUDA required).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("kimodo_longhorizon_example")
    args = _parse_args()

    rng = random.Random(args.seed)

    families = sorted(FAMILY_TEMPLATES) if args.family == "all" else [args.family]

    for fam in families:
        if args.slugs:
            slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
            for s in slugs:
                if s not in SLUG_PROMPTS:
                    raise SystemExit(f"unknown slug {s!r}; valid: {sorted(SLUG_PROMPTS)}")
            desc = "custom chain: " + " → ".join(s.replace("_", " ") for s in slugs)
        else:
            slugs, desc = compose(fam, rng)

        log.info("family=%s  task=%s", fam, desc)
        log.info("slugs=%s", slugs)

        chain = asyncio.run(
            run_chain(
                slugs=slugs,
                seed=args.seed,
                diffusion_steps=args.diffusion_steps,
                guidance_scale=args.guidance_scale,
                stub=args.stub,
            )
        )

        total_frames = sum(len(seg) for seg in chain)
        log.info(
            "family=%s produced %d segments, %d total frames, %d joints/frame",
            fam,
            len(chain),
            total_frames,
            len(chain[0][0]) if chain and chain[0] else 0,
        )

        if args.slugs:
            break  # custom slugs are not per-family


if __name__ == "__main__":
    main()
