"""Laya (convaiinnovations/laya) as a System 1 typed-decision policy.

Laya is a text-only, non-autoregressive decision model: one forward pass of a
ModernBERT-class encoder answers a fixed set of typed questions (``choice`` /
``score`` / ``noul``) about a text state with calibrated probabilities. It has
no vision and no continuous regression head, so this provider does not and
cannot regress joint targets. What it does, per control tick:

1. serialize the observation (joint angles in degrees, gripper percent, the
   instruction and, with a privileged world reader, cube / gripper poses,
   distance and contacts) to a JSON state
   (:mod:`~strands_robots.policies.laya.state_text`);
2. ask Laya which joint to move, in which direction, by how much, and
   optionally whether that step makes progress / whether the cube is in the
   jaws;
3. decode ONE discrete primitive and turn it into a joint-target action dict
   (:mod:`~strands_robots.policies.laya.primitives`).

``get_actions`` returns exactly one action, so the runner re-queries every
tick (the same seam the queued VLA providers use). The ``confidence_gate``
turns Laya's calibrated confidence into an abstain: below the gate the policy
holds and records why, which is the property under test as a System 1 gate in
front of a System 2 policy.

Research provider: zero-shot Laya has never seen a robot, so treat it as a
baseline for the calibration study rather than a controller. See
``examples/laya/`` for the agent flow and the experiment runner.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, ClassVar

from strands_robots.policies._state_keys import observation_joint_keys
from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error, require_optional

from .primitives import (
    DEFAULT_GRIPPER_STEP_PCT,
    DEFAULT_STEP_DEG,
    GRIPPER_JOINT,
    HOLD,
    SIZE_LABELS,
    Primitive,
    apply_primitive,
    decode_answers,
    gripper_pct_from_rad,
)
from .state_text import QUESTION_PROFILES, build_questions, labels_for_keys, serialize_state

logger = logging.getLogger(__name__)

#: Laya checkpoints reachable through ``Router.predict(model=...)``.
LAYA_MODELS: tuple[str, ...] = ("english", "multilingual", "typed-decisions")

#: so101 sim gripper actuator range in radians (MJCF joint range; closed = low, open = high).
SO101_SIM_GRIPPER_RANGE_RAD: tuple[float, float] = (-0.1745, 1.7453)

#: Signature of a privileged world reader: returns the dict
#: :func:`~strands_robots.policies.laya.state_text.serialize_state` accepts as
#: ``world`` (``gripper_xyz_m``, ``cube_xyz_m``, ``contacts``), or ``None``
#: when nothing could be read this tick.
WorldReader = Callable[[], dict[str, Any] | None]


class LayaPolicy(Policy):
    """Typed-decision System 1 policy: text state in, one discrete joint primitive out.

    Args:
        model: Laya checkpoint: ``"english"`` (ModernBERT-large 421M),
            ``"multilingual"`` (mmBERT-base 322M) or ``"typed-decisions"``.
        max_len: Tokenizer max length forwarded to ``Router.predict``; ``None``
            uses the checkpoint default (512 / 1024).
        step_deg: Arm step per size label in degrees, e.g.
            ``{"small": 2, "medium": 5, "large": 10}``.
        gripper_step_pct: Gripper step per primitive in percent of its range.
        confidence_gate: When set, the policy HOLDS on a tick whose gate
            probability is below this value. The gate reads ``progress_ok``
            when the profile asks it, else the chosen joint's probability.
            ``None`` disables the gate.
        questions_profile: One of
            :data:`~strands_robots.policies.laya.state_text.QUESTION_PROFILES`.
        privileged: Whether the state text may carry simulator ground truth
            (cube / gripper poses). Requires a reader installed through
            :meth:`set_world_reader`; ``True`` without one degrades to joints
            only and logs once. ``False`` never calls the reader.
        joint_labels: ``{state_key: label}`` override for the question
            vocabulary; ``None`` derives so101 labels.
        gripper_range: ``(closed, open)`` actuator values of the gripper.
            Overwritten by the model's ctrl range when the MuJoCo engine binds
            :meth:`set_sim_context`.
        device: Torch device for the Laya router (``"cuda"``, ``"cpu"`` or
            ``None`` for Laya's own default).
        router: An object with ``predict(state, questions, model=, max_len=)``
            returning Laya's result dict. Tests inject a fake here; ``None``
            builds ``laya.Router(preload=False, device=device)`` lazily on the
            first tick.

    Raises:
        ValueError: On an unknown ``model`` or ``questions_profile``, a
            ``step_deg`` missing a size label, or a non-finite gate.
    """

    #: The instruction is part of the state text Laya reads.
    reads_instruction: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model: str = "english",
        max_len: int | None = None,
        step_deg: dict[str, float] | None = None,
        gripper_step_pct: float = DEFAULT_GRIPPER_STEP_PCT,
        confidence_gate: float | None = None,
        questions_profile: str = "joint_direction_size",
        privileged: bool = True,
        joint_labels: dict[str, str] | None = None,
        gripper_range: tuple[float, float] | list[float] = SO101_SIM_GRIPPER_RANGE_RAD,
        device: str | None = None,
        router: Any | None = None,
        **_ignored: Any,
    ) -> None:
        if model not in LAYA_MODELS:
            raise ValueError(f"LayaPolicy: unknown model {model!r}; expected one of {LAYA_MODELS}")
        if questions_profile not in QUESTION_PROFILES:
            raise ValueError(
                f"LayaPolicy: unknown questions_profile {questions_profile!r}; expected one of {sorted(QUESTION_PROFILES)}"
            )
        steps = dict(DEFAULT_STEP_DEG) if step_deg is None else {k: float(v) for k, v in step_deg.items()}
        missing = [s for s in SIZE_LABELS if s not in steps]
        if missing:
            raise ValueError(f"LayaPolicy: step_deg is missing size labels {missing}")
        if confidence_gate is not None and not (0.0 <= float(confidence_gate) <= 1.0):
            raise ValueError(f"LayaPolicy: confidence_gate must be in [0, 1] or None, got {confidence_gate!r}")
        if len(gripper_range) != 2:
            raise ValueError(f"LayaPolicy: gripper_range must be (closed, open), got {gripper_range!r}")
        self.model = model
        self.max_len = max_len
        self.step_deg = steps
        self.gripper_step_pct = float(gripper_step_pct)
        self.confidence_gate = None if confidence_gate is None else float(confidence_gate)
        self.questions_profile = questions_profile
        self.privileged = bool(privileged)
        self.joint_labels = joint_labels
        self.gripper_range: tuple[float, float] = (float(gripper_range[0]), float(gripper_range[1]))
        self.device = device
        self._router = router
        self.robot_state_keys: list[str] = []
        self._labels: dict[str, str] = {}
        self._ctrl_bounds: dict[str, tuple[float, float]] = {}
        self._world_reader: WorldReader | None = None
        self._warned_no_reader = False
        #: Full record of the most recent tick (state, questions, answers,
        #: primitive, confidences, gated, latency_ms) for experiment logging.
        self.last_tick: dict[str, Any] | None = None
        self.tick_index = 0

    @property
    def provider_name(self) -> str:
        """Registry name of this provider."""
        return "laya"

    @property
    def requires_images(self) -> bool:
        """Laya has no image encoder: never render cameras for it."""
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Bind the joint keys and derive their question labels.

        Raises:
            ValueError: If ``robot_state_keys`` is not an ordered list of
                distinct non-blank names (:func:`~strands_robots.utils.name_list_error`),
                or two keys resolve to the same label.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self.robot_state_keys = list(robot_state_keys)
        self._labels = labels_for_keys(self.robot_state_keys, self.joint_labels)

    def set_sim_context(self, model: Any, namespace: str) -> None:
        """Learn the actuator ctrl ranges from the compiled MjModel (MuJoCo binds this).

        Uses the same rule as :meth:`MockPolicy.set_sim_context`: the gripper's
        range replaces ``gripper_range`` and the arm ranges clip the targets.
        A read error leaves the policy as configured.
        """
        try:
            import mujoco  # noqa: PLC0415 - optional sim dependency

            from strands_robots.simulation.mujoco.scene_ops import (  # noqa: PLC0415
                actuator_joint_id,
                effective_ctrl_range,
            )

            bounds: dict[str, tuple[float, float]] = {}
            for key in self.robot_state_keys:
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{namespace}{key}")
                if act_id < 0:
                    continue
                jnt_id = actuator_joint_id(model, act_id, mujoco)
                held_to, _reason = effective_ctrl_range(model, mujoco, act_id, jnt_id if jnt_id >= 0 else None)
                if held_to is not None:
                    bounds[key] = (float(held_to[0]), float(held_to[1]))
        except Exception as exc:  # noqa: BLE001 - best-effort, mirrors MockPolicy
            logger.debug("LayaPolicy.set_sim_context could not read ctrlranges: %s", exc)
            return
        self._ctrl_bounds = bounds
        for key, label in self._labels.items():
            if label == GRIPPER_JOINT and key in bounds:
                self.gripper_range = bounds[key]

    def set_world_reader(self, reader: WorldReader | None) -> None:
        """Install (or clear) the privileged reader that supplies cube / gripper poses.

        Args:
            reader: Zero-argument callable returning the ``world`` dict
                :func:`~strands_robots.policies.laya.state_text.serialize_state`
                accepts, or ``None`` to clear.

        Raises:
            TypeError: If ``reader`` is neither callable nor ``None``.
        """
        if reader is not None and not callable(reader):
            raise TypeError(f"set_world_reader: reader must be callable or None, got {type(reader).__name__}")
        self._world_reader = reader

    def reset(self, seed: int | None = None) -> None:
        """Clear per-episode tick state. Laya is deterministic; ``seed`` is unused."""
        self.last_tick = None
        self.tick_index = 0

    @property
    def router(self) -> Any:
        """The Laya router, built on first use when none was injected."""
        if self._router is None:
            laya = require_optional("laya", extra="laya", purpose="the Laya typed-decision policy")
            self._router = laya.Router(preload=False, device=self.device, max_loaded=3)
        return self._router

    def _arm_labels(self) -> tuple[str, ...]:
        return tuple(label for label in self._labels.values() if label != GRIPPER_JOINT)

    def _read_world(self) -> dict[str, Any] | None:
        if not self.privileged:
            return None
        if self._world_reader is None:
            if not self._warned_no_reader:
                self._warned_no_reader = True
                logger.warning(
                    "LayaPolicy: privileged=True but no world reader is installed (set_world_reader); "
                    "the state text carries joints and gripper only."
                )
            return None
        return self._world_reader()

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """One typed-decision tick: serialize, ask, decode, apply.

        Returns:
            A single-element list: the next joint targets for every bound key.

        Raises:
            ValueError: If the observation carries no joint scalars, or Laya's
                answer is outside the offered vocabulary.
        """
        keys = observation_joint_keys(observation_dict, self.robot_state_keys)
        if not keys:
            raise ValueError("LayaPolicy: observation carries no joint state scalars")
        if not self._labels or set(keys) != set(self._labels):
            self.robot_state_keys = list(keys)
            self._labels = labels_for_keys(self.robot_state_keys, self.joint_labels)
        current = {k: float(observation_dict[k]) for k in keys}
        gripper_key = next((k for k, label in self._labels.items() if label == GRIPPER_JOINT), None)
        gripper_pct = None if gripper_key is None else gripper_pct_from_rad(current[gripper_key], self.gripper_range)
        world = self._read_world()
        state = serialize_state(
            instruction=instruction, joints_rad=current, labels=self._labels, gripper_pct=gripper_pct, world=world
        )
        questions = build_questions(self._arm_labels(), self.questions_profile)
        started = time.perf_counter()
        result = self.router.predict(state, questions, model=self.model, max_len=self.max_len)
        latency_ms = (time.perf_counter() - started) * 1000.0
        primitive, confidences = decode_answers(result["answers"], self._arm_labels())
        gate_value = confidences.get("progress_ok", confidences["joint_p"])
        gated = self.confidence_gate is not None and gate_value < self.confidence_gate
        applied: Primitive = HOLD if gated else primitive
        action = apply_primitive(
            applied,
            current,
            self._labels,
            step_deg=self.step_deg,
            gripper_step_pct=self.gripper_step_pct,
            gripper_range=self.gripper_range,
            ctrl_bounds=self._ctrl_bounds,
        )
        self.last_tick = {
            "tick": self.tick_index,
            "model": self.model,
            "state": state,
            "questions": list(questions),
            "answers": result["answers"],
            "primitive": primitive.as_dict(),
            "applied": applied.as_dict(),
            "confidences": confidences,
            "gate_value": gate_value,
            "gated": gated,
            "latency_ms": latency_ms,
        }
        self.tick_index += 1
        return [action]

    def __repr__(self) -> str:
        return (
            f"LayaPolicy(model={self.model!r}, profile={self.questions_profile!r}, "
            f"gate={self.confidence_gate!r}, privileged={self.privileged}, "
            f"reader={'yes' if self._world_reader is not None else 'no'}, keys={self.robot_state_keys})"
        )
