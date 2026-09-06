"""Configuration + YAML loader for the ProtoMotions Generalist Tracking Policy.

The upstream ONNX artifact
(``cagataydev/protomotions-gtp-unitree-g1/unified_pipeline.onnx``) ships with a
YAML sidecar (``unified_pipeline.yaml``) that pins:

* The 29 joint names in the ONNX action order.
* The 33 body names + anchor index (``torso_link`` = 16) + root index
  (``pelvis`` = 0).
* Per-joint stiffness + damping.
* Timing: ``control_dt = 0.02s`` (50Hz), ``physics_dt = 0.001s`` (1kHz),
  ``decimation = 20``.
* The future-reference lookahead schedule ``[1, 2, 4, 8]`` control steps.

:class:`ProtoMotionsConfig` is the typed dataclass representation of that
sidecar. Loading it once at construction (rather than reading the YAML on every
tick) means the hot path never touches disk, and a dimension or joint-count
error surfaces at policy build time with a clean message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands_robots.utils import (
    non_negative_whole_number_error,
    positive_finite_number_error,
    require_optional,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ProtoMotionsConfig",
    "load_config_from_yaml",
    "GTP_G1_JOINT_NAMES",
    "GTP_G1_BODY_NAMES",
    "GTP_G1_ANCHOR_BODY_INDEX",
    "GTP_G1_ROOT_BODY_INDEX",
    "GTP_G1_DEFAULT_LOOKAHEAD_STEPS",
    "GTP_G1_CONTROL_DT",
]

# ---------------------------------------------------------------------------
# Canonical constants - pinned from unified_pipeline.yaml (2026-08-14 upload)
# ---------------------------------------------------------------------------

# The 29 joint names in the order the ONNX policy emits them, matching the
# yaml `joint_names` field (id001). Order matters - ONNX output index i drives
# GTP_G1_JOINT_NAMES[i]. Kept identical to :data:`strands_robots.policies.
# kimodo.KIMODO_G1_JOINTS` so a Kimodo qpos plugs straight into a ProtoMotions
# tracker without any per-joint reordering.
GTP_G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# The 33 body names in the model's body order (yaml `body_names` id002).
# Index 0 = pelvis (root), 16 = torso_link (anchor). ``rubber_hand`` bodies are
# placeholders on the ``g1_29dof_rev_1_0`` URDF that ships fingerless - a real
# manipulation URDF would substitute finger links here without changing the
# tracker's input contract.
GTP_G1_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "head",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
)

GTP_G1_ANCHOR_BODY_INDEX = 16  # torso_link
GTP_G1_ROOT_BODY_INDEX = 0  # pelvis

# The four future-step offsets the ONNX consumes at each tick (yaml
# ``motion.future_step_indices``). Kept as a tuple so it hashes and cannot be
# mutated across policy instances.
GTP_G1_DEFAULT_LOOKAHEAD_STEPS: tuple[int, ...] = (1, 2, 4, 8)

GTP_G1_CONTROL_DT: float = 0.02  # seconds - 50Hz outer control loop

# Upstream per-joint SONIC PD gains from the yaml sidecar. Order matches
# :data:`GTP_G1_JOINT_NAMES` (id003 stiffness, id004 damping).
_G1_STIFFNESS: tuple[float, ...] = (
    40.17923847137318,
    99.09842777666113,
    40.17923847137318,
    99.09842777666113,
    28.50124619574858,
    28.50124619574858,
    40.17923847137318,
    99.09842777666113,
    40.17923847137318,
    99.09842777666113,
    28.50124619574858,
    28.50124619574858,
    40.17923847137318,
    28.50124619574858,
    28.50124619574858,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    16.77832748089279,
    16.77832748089279,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    16.77832748089279,
    16.77832748089279,
)
_G1_DAMPING: tuple[float, ...] = (
    2.5578897650279457,
    6.3088018534966395,
    2.5578897650279457,
    6.3088018534966395,
    1.814445686584846,
    1.814445686584846,
    2.5578897650279457,
    6.3088018534966395,
    2.5578897650279457,
    6.3088018534966395,
    1.814445686584846,
    1.814445686584846,
    2.5578897650279457,
    1.814445686584846,
    1.814445686584846,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    1.06814150219,
    1.06814150219,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    1.06814150219,
    1.06814150219,
)


@dataclass(frozen=True)
class ProtoMotionsConfig:
    """Frozen typed view of the ONNX ``unified_pipeline.yaml`` sidecar.

    All fields have upstream-verified defaults matching the shipped
    ``cagataydev/protomotions-gtp-unitree-g1`` weights. A caller that points at
    a different checkpoint should always pair it with a matching config.

    Attributes:
        joint_names: 29 joint names in ONNX action order.
        body_names: 33 body names in the model's rigid-body order.
        anchor_body_index: Row index of the anchor body inside ``body_names``.
        root_body_index: Row index of the root body inside ``body_names``.
        stiffness: Per-joint kp used by the deployed PD loop.
        damping: Per-joint kd used by the deployed PD loop.
        control_dt: Seconds per outer control tick, a positive finite number.
            This is the period the reference motion is RESAMPLED onto by
            :class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer`,
            so it fixes how many frames one clip becomes, and the playhead
            advances exactly one of those frames per tick.
        physics_dt: Seconds per inner physics substep. Carried for provenance;
            no reader in this package consumes it.
        decimation: Physics substeps per control tick (``control_dt /
            physics_dt``). Carried for provenance; no reader in this package
            consumes it, and the stated relation is not checked against the two
            periods.
        future_step_indices: Future-reference lookahead offsets in control
            steps.
        action_ema_alpha: Exponential-moving-average smoothing applied to the
            joint targets the PD loop receives, in ``(0, 1]``. ``1.0`` (the
            upstream default) is passthrough; a smaller value weights the
            previous target more heavily, trading tracking lag for less
            per-tick jitter. Applied by
            :meth:`~strands_robots.policies.protomotions.policy.\
ProtoMotionsPolicy.get_actions`; the raw network output continues to feed the
            historical-actions buffer, which is what the ONNX graph's
            ``historical_processed_actions`` input is defined over.
        onnx_in_names: Input tensor names in the order the exported session
            declares them, so the session's ``get_inputs()`` order can be
            validated against this tuple at load.
        onnx_out_names: Output tensor names requested from the session. Outputs
            are paired to these names rather than by position, so a config that
            declares the checkpoint's outputs in a different order still reads
            the right tensor.
        metadata: Free-form provenance carried alongside the config, populated
            from a sidecar's own ``metadata`` key. Never read by the control
            path.
    """

    joint_names: tuple[str, ...] = GTP_G1_JOINT_NAMES
    body_names: tuple[str, ...] = GTP_G1_BODY_NAMES
    anchor_body_index: int = GTP_G1_ANCHOR_BODY_INDEX
    root_body_index: int = GTP_G1_ROOT_BODY_INDEX
    stiffness: tuple[float, ...] = _G1_STIFFNESS
    damping: tuple[float, ...] = _G1_DAMPING
    control_dt: float = GTP_G1_CONTROL_DT
    physics_dt: float = 0.001
    decimation: int = 20
    future_step_indices: tuple[int, ...] = GTP_G1_DEFAULT_LOOKAHEAD_STEPS
    action_ema_alpha: float = 1.0
    # ONNX I/O names - kept as tuples so the ordering is stable and the ONNX
    # session's ``get_inputs()`` order can be validated against it at load.
    onnx_in_names: tuple[str, ...] = (
        "current_anchor_rot",
        "current_dof_pos",
        "current_dof_vel",
        "current_root_local_ang_vel",
        "historical_processed_actions",
        "mimic_future_anchor_rot",
        "mimic_future_dof_pos",
        "mimic_future_dof_vel",
    )
    onnx_out_names: tuple[str, ...] = (
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a body index, a smoothing factor or a control period no reader can honor.

        The two indices are the only fields this config resolves a body NAME
        from, and an index that misses is not a value the control path can
        report: :attr:`anchor_body_name` and :attr:`root_body_name` index a
        tuple, so a negative index silently names a real but different link,
        and the tracker then anchors on it consistently -
        :attr:`~strands_robots.policies.base.Policy.required_bodies` declares
        that link, the runtime supplies its quaternion, and the future-reference
        window reads the same row. Every stage agrees, and every stage is wrong.

        Each index goes through the shared whole-number domain BEFORE the
        ``int()`` normalisation, not after: that conversion is what laundered a
        yaml ``anchor_body_index: true`` into row 1 (``head``) and a ``2.7``
        into row 2 (``left_hip_pitch_link``).

        :attr:`action_ema_alpha` is gated for the same reason and against the
        same failure: the value is the weight the CURRENT network output carries
        in the target the PD loop receives, so ``0`` weights it not at all and
        freezes the commanded pose at the first tick's target for the whole
        clip - a tracker that reports every frame and moves through none of
        them. A negative weight drives the joint the opposite way from the
        motion it is tracking, one above ``1`` over-extrapolates past it, and
        ``nan`` propagates into the filter state and never leaves it, so every
        joint of every later tick is ``nan`` however good the network output
        is. None of the four is a smoothing factor, and all four used to be
        stored and carried into the control path.

        :attr:`control_dt` is gated for the same reason, and it is the one field
        of the timing block the control path SPENDS: it is the period the
        reference motion is resampled onto, so it decides how many frames one
        clip becomes, and the playhead advances one frame per tick. A period of
        ``1.0`` - what a sidecar's ``control_dt: true`` resolved to - resamples a
        3-second clip to 4 frames, so two of the tracker's four lookahead
        offsets read past the end from the first tick; a negative period and
        ``inf`` each collapse the clip to a SINGLE frame, which the frame clamp
        then serves for every tick of the episode - a tracker that reports a
        motion and holds one pose. ``0`` left the period undefined, and ``nan``
        reached the resampler to raise there instead, naming neither the field
        nor the sidecar it came from. All five used to be stored.

        :attr:`physics_dt` and :attr:`decimation` are deliberately NOT gated: no
        reader in this package consumes either, so refusing a value would change
        which sidecars load with no behaviour to protect - a decision this does
        not make. Their documented relation to :attr:`control_dt` is unchecked
        for the same reason.

        Raises:
            ValueError: If either body index is not a non-negative whole number,
                or is not a row of :attr:`body_names`; if
                :attr:`action_ema_alpha` is not a finite number in ``(0, 1]``;
                or if :attr:`control_dt` is not a positive finite number.
        """
        num_bodies = len(self.body_names)
        for name in ("anchor_body_index", "root_body_index"):
            raw = getattr(self, name)
            if error := non_negative_whole_number_error(raw, name, "ProtoMotionsConfig"):
                raise ValueError(error)
            index = int(raw)
            if index >= num_bodies:
                raise ValueError(
                    f"ProtoMotionsConfig.{name} must be a row of body_names "
                    f"(0..{num_bodies - 1}), got {index} for {num_bodies} bodies. "
                    "The index is an offset into body_names, so one that misses "
                    "cannot resolve the body it names."
                )
            # Normalise to a plain int (frozen -> object.__setattr__) so an
            # integral float the domain admits is stored as the row number the
            # observation lookup and the future-reference slice both index with.
            object.__setattr__(self, name, index)

        # The shared positive-finite domain covers the sign, the finiteness, the
        # non-real spellings and the ``bool`` that would act as a silent 1.0;
        # only the upper bound is this field's own, so only it is spelled here.
        if error := positive_finite_number_error(self.action_ema_alpha, "action_ema_alpha", "ProtoMotionsConfig"):
            raise ValueError(error)
        if float(self.action_ema_alpha) > 1.0:
            raise ValueError(
                "ProtoMotionsConfig: action_ema_alpha must be <= 1 (1.0 is the "
                f"unsmoothed passthrough), got {self.action_ema_alpha!r}. The value is the "
                "weight the current network output carries in the blend, so one above 1 "
                "gives the previous target a negative weight and extrapolates past the "
                "motion instead of smoothing toward it."
            )
        # Normalise to a plain float for the same reason the indices normalise
        # to int: the filter multiplies with it every tick, and a NumPy scalar
        # read from a config array would set the output dtype from the weight.
        object.__setattr__(self, "action_ema_alpha", float(self.action_ema_alpha))

        # A control period is a rate, which is precisely the shared domain's
        # subject ("the loop period is 1 / hz"): it covers the sign, the zero,
        # the finiteness, the non-real spellings and the ``bool`` that would act
        # as a silent 1.0.
        if error := positive_finite_number_error(self.control_dt, "control_dt", "ProtoMotionsConfig"):
            raise ValueError(error)
        # Normalise for the reason action_ema_alpha does: the resampler divides a
        # motion length by this value and multiplies it by a frame number.
        object.__setattr__(self, "control_dt", float(self.control_dt))

    # ------------------------------------------------------------------
    # Derived properties - computed on read, never stored, so a frozen
    # dataclass with a single source of truth for each field stays that way.
    # ------------------------------------------------------------------

    @property
    def num_dofs(self) -> int:
        """Length of :attr:`joint_names` (also the ONNX action width)."""
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        """Length of :attr:`body_names`."""
        return len(self.body_names)

    @property
    def num_future_steps(self) -> int:
        """Length of :attr:`future_step_indices`."""
        return len(self.future_step_indices)

    @property
    def anchor_body_name(self) -> str:
        """Name of the anchor body (``torso_link`` on the shipped G1 config).

        The tracker consumes this body's WORLD orientation, not the floating
        base's. Resolving the name here keeps
        :attr:`~strands_robots.policies.base.Policy.required_bodies` and the
        observation lookup reading one source of truth.

        Always resolves: :meth:`__post_init__` refuses an
        :attr:`anchor_body_index` that is not a row of :attr:`body_names`, so
        the lookup here cannot miss.
        """
        return self.body_names[self.anchor_body_index]

    @property
    def root_body_name(self) -> str:
        """Name of the root (floating-base) body - ``pelvis`` on the G1.

        Always resolves: :meth:`__post_init__` refuses a
        :attr:`root_body_index` that is not a row of :attr:`body_names`.
        """
        return self.body_names[self.root_body_index]

    @property
    def anchor_is_root(self) -> bool:
        """Whether the anchor body IS the floating base.

        Only when this holds is the observation's ``base_quat`` the anchor
        orientation. On the G1 it is ``False``: the torso differs from the
        pelvis by the three waist joints, so substituting ``base_quat`` would
        feed the tracker a silently wrong frame.
        """
        return self.anchor_body_index == self.root_body_index


def load_config_from_yaml(path: str | Path) -> ProtoMotionsConfig:
    """Parse a ``unified_pipeline.yaml`` sidecar into a typed config.

    The yaml is the artifact's source of truth. Fields absent from the yaml
    fall back to the dataclass defaults (which are themselves pinned to the
    shipped weights, so a missing block is not an error).

    An empty sidecar is the limit of "fields absent from the yaml", so it
    yields the all-defaults config exactly as ``{}`` does - the two spell the
    same information. ``~`` in ``path`` is expanded, the file is read as a
    file, and a payload that is not a mapping is reported by name rather than
    reaching the field lookups below: the reporting the sibling policy-config
    file loaders in :mod:`strands_robots.policies.kimodo.config`,
    :mod:`strands_robots.policies.motionbricks.config` and
    :mod:`strands_robots.policies.wbc.config` already give.

    The extension is deliberately not checked, unlike the two loaders that do:
    a yaml document stored under any name loads here today, and refusing one
    would stop a payload that currently works.

    Args:
        path: Path to the yaml file. ``~`` is expanded.

    Returns:
        A :class:`ProtoMotionsConfig` - validated for consistent dimensions.

    Raises:
        FileNotFoundError: If ``path`` does not name a file (a directory does
            not, so it is refused here rather than at the read).
        ImportError: If ``pyyaml`` is not installed.
        ValueError: If the file is not valid YAML, or holds a YAML value that
            is not a mapping, or contains an inconsistent dimension: a
            ``stiffness`` or ``damping`` length that is not the joint count, or
            an ``anchor_body_index`` / ``root_body_index`` that is not a row
            of ``body_names``, or an ``action_ema_alpha`` outside ``(0, 1]``
            (each refused by :meth:`ProtoMotionsConfig.__post_init__`, so a
            config built by hand reports the same value the same way).
    """
    yaml = require_optional(
        "yaml",
        pip_install="pyyaml",
        extra="protomotions",
        purpose="reading the unified_pipeline.yaml checkpoint sidecar",
    )

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ProtoMotions yaml not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]
    except yaml.YAMLError as e:  # type: ignore[attr-defined]
        raise ValueError(f"ProtoMotions yaml {path} is not valid YAML: {e}") from e
    if data is None:
        # An empty sidecar (and a comments-only one) carries the same
        # information as ``{}``: every field absent. ``{}`` already returns the
        # all-defaults config, which is what this function documents absent
        # fields to mean, so the two spellings resolve to the same config
        # rather than one of them dead-ending on ``None.get``.
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"ProtoMotions yaml {path} must contain a mapping, got {type(data).__name__}")

    joint_names = tuple(data.get("joint_names", GTP_G1_JOINT_NAMES))
    body_names = tuple(data.get("body_names", GTP_G1_BODY_NAMES))

    robot = data.get("robot", {})
    # Handed through raw, not through int(): ProtoMotionsConfig.__post_init__ is
    # the single owner of what a body index may be, and coercing here first is
    # what turned a yaml ``anchor_body_index: true`` into row 1 before the
    # domain could see it.
    anchor_idx = robot.get("anchor_body_index", GTP_G1_ANCHOR_BODY_INDEX)
    root_idx = robot.get("root_body_index", GTP_G1_ROOT_BODY_INDEX)

    control = data.get("control", {})
    stiffness = tuple(control.get("stiffness", data.get("default_joint_stiffness", _G1_STIFFNESS)))
    damping = tuple(control.get("damping", data.get("default_joint_damping", _G1_DAMPING)))
    # Handed through raw for the reason the body indices are: the config's
    # __post_init__ is the single owner of what a smoothing factor may be, and
    # ``float()`` here first is what turned a yaml ``action_ema_alpha: true``
    # into an unsmoothed 1.0 before the domain could see it.
    ema = control.get("action_ema_alpha", 1.0)

    timing = data.get("timing", {})
    # Handed through raw for the reason the body indices and the smoothing factor
    # are: the config's __post_init__ is the single owner of what a control
    # period may be, and ``float()`` here first is what turned a sidecar's
    # ``control_dt: true`` into a 1-second control period - resampling a
    # 3-second reference motion down to 4 frames - before the domain saw it.
    control_dt = timing.get("control_dt", GTP_G1_CONTROL_DT)
    # These two keep their coercion: no reader consumes either, so there is no
    # domain behind them for a value to be laundered past.
    physics_dt = float(timing.get("physics_dt", 0.001))
    decimation = int(timing.get("decimation", 20))

    motion = data.get("motion", {})
    future_steps = tuple(int(x) for x in motion.get("future_step_indices", GTP_G1_DEFAULT_LOOKAHEAD_STEPS))

    runtime = data.get("_runtime", {})
    onnx_in_names = tuple(
        runtime.get(
            "onnx_in_names",
            ProtoMotionsConfig.__dataclass_fields__["onnx_in_names"].default,
        )
    )
    onnx_out_names = tuple(
        runtime.get(
            "onnx_out_names",
            ProtoMotionsConfig.__dataclass_fields__["onnx_out_names"].default,
        )
    )

    if len(stiffness) != len(joint_names):
        raise ValueError(f"stiffness length ({len(stiffness)}) != joint count ({len(joint_names)}) in {path}.")
    if len(damping) != len(joint_names):
        raise ValueError(f"damping length ({len(damping)}) != joint count ({len(joint_names)}) in {path}.")

    cfg = ProtoMotionsConfig(
        joint_names=joint_names,
        body_names=body_names,
        anchor_body_index=anchor_idx,
        root_body_index=root_idx,
        stiffness=stiffness,
        damping=damping,
        control_dt=control_dt,
        physics_dt=physics_dt,
        decimation=decimation,
        future_step_indices=future_steps,
        action_ema_alpha=ema,
        onnx_in_names=onnx_in_names,
        onnx_out_names=onnx_out_names,
        metadata=data.get("metadata", {}),
    )
    logger.info(
        "ProtoMotionsConfig loaded from %s: %d joints, %d bodies, %d future steps @ %.0f Hz",
        path.name,
        cfg.num_dofs,
        cfg.num_bodies,
        cfg.num_future_steps,
        1.0 / cfg.control_dt,
    )
    return cfg
