"""Qwen-VLA embodiment-aware prompt builder.

Qwen-VLA's *only* platform-specific interface is a natural-language
**embodiment prompt** prepended to the task instruction (arXiv:2605.30280v2
section 2.3). Deploying to a new robot is therefore a matter of describing it
in text - no per-embodiment model heads, no architecture change.

This module is a pure function with no heavy dependencies so it can be unit
tested in isolation and reused by both the inference provider
(:mod:`strands_robots.policies.qwen_vla.data_config`) and the training
pipeline (``training/qwen_vla/data/embodiment_tags.py``).

Template (verbatim from section 2.3 of the paper)::

    The robot is {robot_tag} with {single arm / dual arms}[, waist][, and
    mobile base]. The control frequency is {FPS} Hz. Please predict the next
    {chunk_size} control actions to execute the following task:
    {ori_instruction}.
"""


def _arm_phrase(arm_config: str) -> str:
    """Return the arm clause for the embodiment prompt.

    Args:
        arm_config: ``"single"`` or ``"dual"``.

    Returns:
        ``"single arm"`` or ``"dual arms"``.

    Raises:
        ValueError: If *arm_config* is neither ``"single"`` nor ``"dual"``.
            We raise (rather than defaulting) because a wrong arm count
            silently mis-describes the robot to the model and degrades
            action quality - exactly the "no silent defaults" rule in
            AGENTS.md.
    """
    if arm_config == "single":
        return "single arm"
    if arm_config == "dual":
        return "dual arms"
    raise ValueError(f"arm_config must be 'single' or 'dual', got {arm_config!r}")


def build_embodiment_prompt(
    *,
    robot_tag: str,
    arm_config: str,
    fps: int,
    chunk_size: int,
    instruction: str,
    has_waist: bool = False,
    has_mobile_base: bool = False,
) -> str:
    """Build the Qwen-VLA embodiment-aware prompt (section 2.3 template).

    The prompt is the sole platform-specific input to Qwen-VLA. It encodes
    the morphology (arm count, optional waist, optional mobile base), the
    control frequency, the action-chunk length, and the task instruction.

    Args:
        robot_tag: Short robot identifier (e.g. ``"so100"``, ``"aloha"``,
            ``"unitree_g1"``).
        arm_config: ``"single"`` or ``"dual"``.
        fps: Control frequency in Hz (must be a positive integer).
        chunk_size: Number of control actions to predict (the action horizon
            H; the paper uses H=16 for manipulation, 8 for navigation).
        instruction: The original task instruction (e.g. ``"pick up the red
            cube"``). Trailing periods are normalised so the template's own
            closing period is not doubled.
        has_waist: Whether the robot has a controllable waist joint.
        has_mobile_base: Whether the robot has a mobile base.

    Returns:
        The fully-rendered embodiment prompt string.

    Raises:
        ValueError: If *robot_tag* or *instruction* is empty, *fps* /
            *chunk_size* is not a positive integer, or *arm_config* is
            invalid.
    """
    if not robot_tag:
        raise ValueError("robot_tag must not be empty")
    if not instruction or not instruction.strip():
        raise ValueError("instruction must not be empty")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps!r}")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")

    arm = _arm_phrase(arm_config)

    # Morphology clauses are appended in the paper's order: arm, then waist,
    # then mobile base. Each is optional and only the present ones render.
    morphology = arm
    if has_waist:
        morphology += ", waist"
    if has_mobile_base:
        # The paper renders this as ", and mobile base" so the clause list
        # reads naturally regardless of whether a waist clause precedes it.
        morphology += ", and mobile base"

    # Normalise the instruction's trailing period: the template supplies the
    # closing "." so we strip a user-provided one to avoid "task..".
    clean_instruction = instruction.strip().rstrip(".")

    return (
        f"The robot is {robot_tag} with {morphology}. "
        f"The control frequency is {fps} Hz. "
        f"Please predict the next {chunk_size} control actions to execute "
        f"the following task: {clean_instruction}."
    )


__all__ = ["build_embodiment_prompt"]
