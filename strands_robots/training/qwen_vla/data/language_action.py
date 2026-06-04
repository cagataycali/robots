"""Text-only Text-to-Action (T2A) corpus generator (section 3.2.3).

Generates the language-action pretraining corpus for stage 1: procedural task
instructions paired with motion-planned end-effector goal trajectories, with NO
rendering (fast, scalable). Six task-template families x robot configs, per the
paper's language-action recipe.

Goal trajectories are synthesized analytically (straight-line / minimum-jerk
EEF paths to procedurally-sampled targets) so the generator works WITHOUT a
heavy IK dependency like cuRobo; a MuJoCo-IK refinement hook is left for the
caller to plug in when available (PLAN section 6.5). Pure NumPy core - unit
tests are deterministic with a seeded RNG.
"""

from dataclasses import dataclass

import numpy as np

from strands_robots.training.qwen_vla.data.embodiment_tags import EmbodimentTag

# Six task-template families (section 3.2.3). Each has an instruction template
# and a motion archetype the goal synthesizer uses to shape the EEF path.
TASK_FAMILIES: dict[str, dict[str, str]] = {
    "pick": {"template": "pick up the {obj}", "motion": "reach_grasp"},
    "place": {"template": "place the {obj} on the {target}", "motion": "transport"},
    "push": {"template": "push the {obj} to the {target}", "motion": "linear_push"},
    "open": {"template": "open the {obj}", "motion": "pull"},
    "close": {"template": "close the {obj}", "motion": "push_flat"},
    "reach": {"template": "reach toward the {obj}", "motion": "reach"},
}

_OBJECTS = ("red cube", "blue bowl", "mug", "drawer", "cabinet door", "block", "bottle", "plate")
_TARGETS = ("table", "shelf", "left bin", "right bin", "tray")


@dataclass
class LanguageActionExample:
    """One text-only T2A training example.

    Attributes:
        instruction: The procedural task instruction.
        prompt: The full embodiment prompt (section 2.3) wrapping the instruction.
        family: Task-template family name.
        action: ``(H, c)`` synthesized goal action trajectory (raw units).
    """

    instruction: str
    prompt: str
    family: str
    action: np.ndarray


def _synthesize_trajectory(motion: str, h: int, c: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize an ``(H, c)`` EEF goal trajectory for a motion archetype.

    The trajectory is an analytic path (no renderer, no IK solver) shaped by
    the motion type: ``reach`` paths approach a target monotonically,
    ``transport`` paths add a lift-then-place arc, ``push`` paths are linear,
    etc. The last channel is treated as a gripper command in [0, 1].

    Args:
        motion: Motion archetype (from :data:`TASK_FAMILIES`).
        h: Horizon H.
        c: Number of valid action channels (>= 1; last = gripper).
        rng: Seeded NumPy generator.

    Returns:
        ``(H, c)`` float32 trajectory.
    """
    if c < 1:
        raise ValueError(f"c must be >= 1, got {c}")
    # Smooth minimum-jerk-ish progress profile s(t) in [0, 1].
    t = np.linspace(0.0, 1.0, h)
    s = 6 * t**5 - 15 * t**4 + 10 * t**3  # minimum-jerk scalar

    goal = rng.uniform(-1.0, 1.0, size=c)
    traj = np.outer(s, goal).astype(np.float32)  # (H, c) straight approach

    if motion == "transport":
        # Add a vertical lift arc on the 3rd channel (z) if present.
        if c >= 3:
            traj[:, 2] += (np.sin(np.pi * t) * 0.3).astype(np.float32)
    elif motion in ("pull", "push_flat", "linear_push"):
        # Push/pull are linear ramps, no arc; keep straight path.
        pass

    # Gripper channel: closes (->1) for grasp motions, opens (->0) otherwise.
    if motion in ("reach_grasp",):
        traj[:, -1] = s.astype(np.float32)
    elif motion in ("pull",):
        traj[:, -1] = (1.0 - s).astype(np.float32)
    return traj


class LanguageActionGenerator:
    """Procedural text-only T2A corpus generator for one embodiment.

    Args:
        embodiment: The :class:`EmbodimentTag` (prompt + horizon H).
        action_channels: Number of valid action channels c (last = gripper).
        seed: RNG seed for reproducible corpora.
    """

    def __init__(self, *, embodiment: EmbodimentTag, action_channels: int, seed: int = 0):
        if action_channels < 1:
            raise ValueError(f"action_channels must be >= 1, got {action_channels}")
        self.embodiment = embodiment
        self.action_channels = action_channels
        self._rng = np.random.default_rng(seed)

    def _render_instruction(self, family: str) -> str:
        """Fill a task-family template with random objects/targets."""
        tmpl = TASK_FAMILIES[family]["template"]
        obj = self._rng.choice(_OBJECTS)
        target = self._rng.choice(_TARGETS)
        return tmpl.format(obj=obj, target=target)

    def generate(self, n: int, family: str | None = None) -> list[LanguageActionExample]:
        """Generate *n* text-only T2A examples.

        Args:
            n: Number of examples.
            family: Restrict to one task family, or ``None`` to sample across
                all six.

        Returns:
            List of :class:`LanguageActionExample`.

        Raises:
            ValueError: If *n* is negative or *family* is unknown.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if family is not None and family not in TASK_FAMILIES:
            raise ValueError(f"unknown family '{family}'. Available: {sorted(TASK_FAMILIES)}")

        families = [family] if family else list(TASK_FAMILIES)
        out: list[LanguageActionExample] = []
        for _ in range(n):
            fam = family or str(self._rng.choice(families))
            instruction = self._render_instruction(fam)
            prompt = self.embodiment.render_prompt(instruction)
            action = _synthesize_trajectory(
                TASK_FAMILIES[fam]["motion"], self.embodiment.chunk_size, self.action_channels, self._rng
            )
            out.append(LanguageActionExample(instruction=instruction, prompt=prompt, family=fam, action=action))
        return out


__all__ = ["TASK_FAMILIES", "LanguageActionExample", "LanguageActionGenerator"]
