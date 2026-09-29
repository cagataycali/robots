# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Roll out an rsl_rl actor that mjlab exported to ONNX (``create_policy("rsl_rl_onnx")``).

mjlab's ``MjlabOnPolicyRunner`` writes ``<run>.onnx`` next to every checkpoint
and stamps the file with the metadata this provider needs to rebuild the
actor's observation from a plain ``SimEngine.get_observation`` dict
(``mjlab/rl/exporter_utils.py``): ``observation_names`` (the term order),
``joint_names`` / ``default_joint_pos`` / ``action_scale`` (the
``JointPositionAction`` decode), ``command_names``, per-term scale / clip /
history. Nothing else is assumed: a term this module has no builder for is
refused by name at construction.

Observation terms are rebuilt from the engine's keys (``<joint>``, ``<joint>.vel``,
``base_quat`` wxyz, ``base_lin_vel`` WORLD frame, ``base_ang_vel`` BODY frame):

* ``base_lin_vel``      world linear velocity rotated into the base frame
* ``base_ang_vel``      body angular velocity as observed
* ``projected_gravity`` (0, 0, -1) rotated into the base frame
* ``joint_pos``         ``obs[j] - default_joint_pos[j]`` in the actor's joint order
* ``joint_vel``         ``obs[j.vel]``
* ``actions``           the previous raw action (zeros after ``reset``)
* ``command``           ``target_velocity`` kwarg (``[vx, vy, wz]``) or the
                        ``command`` config, else zeros
* ``ee_to_target``      target point minus the ``ee_site`` position, base frame;
                        the site pose comes from MuJoCo forward kinematics of the
                        robot's own MJCF (``robot`` config), so the term matches
                        what the training env computed from ``site_pos_w``

The action decode is mjlab's ``JointPositionActionCfg(use_default_offset=True)``:
``target = default_joint_pos + action_scale * raw``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from strands_robots.policies.base import Policy

if TYPE_CHECKING:  # pragma: no cover
    import onnxruntime as ort

logger = logging.getLogger(__name__)

_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)
_BUILDERS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "ee_to_target",
)


def _csv_floats(s: str | None) -> list[float]:
    return [float(x) for x in s.split(",")] if s else []


def _csv_strs(s: str | None) -> list[str]:
    return [x for x in s.split(",")] if s else []


def _quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector ``v`` into the frame of unit quaternion ``q`` (w, x, y, z)."""
    w, x, y, z = (float(c) for c in q_wxyz)
    qv = np.array([x, y, z], dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * w * 2.0
    c = qv * float(np.dot(qv, v)) * 2.0
    return (a - b + c).astype(np.float32)


@dataclass
class OnnxActorSpec:
    """What the ONNX file says about itself; enough to rebuild its observation."""

    path: str
    obs_dim: int
    num_actions: int
    observation_names: list[str]
    joint_names: list[str]
    default_joint_pos: list[float]
    action_scale: list[float]
    command_names: list[str] = field(default_factory=list)
    obs_scale: list[float] = field(default_factory=list)
    obs_history: list[float] = field(default_factory=list)
    obs_clip: list[tuple[float, float]] = field(default_factory=list)
    joint_stiffness: list[float] = field(default_factory=list)
    joint_damping: list[float] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)


def resolve_onnx_path(onnx_path: str) -> str:
    """A local path, or ``hf://repo_id/file.onnx`` resolved through huggingface_hub."""
    if onnx_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo_and_file = onnx_path[len("hf://") :]
        repo_id, _, filename = repo_and_file.partition("/")
        owner_repo = repo_id + "/" + filename.split("/", 1)[0]
        rest = filename.split("/", 1)[1] if "/" in filename else ""
        if not rest:
            raise ValueError(f"hf:// path needs owner/repo/file.onnx, got {onnx_path!r}")
        return hf_hub_download(owner_repo, rest)
    p = Path(os.path.expanduser(onnx_path))
    if p.is_dir():
        cands = sorted(p.glob("*.onnx"))
        if not cands:
            raise FileNotFoundError(f"no .onnx in {p}")
        p = cands[-1]
    if not p.is_file():
        raise FileNotFoundError(f"onnx_path does not exist: {p}")
    return str(p)


def load_actor_spec(onnx_path: str) -> tuple[ort.InferenceSession, OnnxActorSpec]:
    """Open the ONNX file and parse mjlab's metadata into an :class:`OnnxActorSpec`."""
    import onnxruntime as ort

    path = resolve_onnx_path(onnx_path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    md = sess.get_modelmeta().custom_metadata_map
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    clips = []
    for item in _csv_strs(md.get("observation_terms_clip")):
        lo, _, hi = item.partition(";")
        clips.append((float(lo), float(hi)))
    spec = OnnxActorSpec(
        path=path,
        obs_dim=int(inp.shape[-1]),
        num_actions=int(out.shape[-1]),
        observation_names=_csv_strs(md.get("observation_names")),
        joint_names=_csv_strs(md.get("joint_names")),
        default_joint_pos=_csv_floats(md.get("default_joint_pos")),
        action_scale=_csv_floats(md.get("action_scale")),
        command_names=_csv_strs(md.get("command_names")),
        obs_scale=_csv_floats(md.get("observation_terms_scale")),
        obs_history=_csv_floats(md.get("observation_terms_history_length")),
        obs_clip=clips,
        joint_stiffness=_csv_floats(md.get("joint_stiffness")),
        joint_damping=_csv_floats(md.get("joint_damping")),
        raw=dict(md),
    )
    if not spec.observation_names or not spec.joint_names:
        raise ValueError(
            f"{path} carries no mjlab metadata (observation_names/joint_names); only ONNX files "
            "written by mjlab's MjlabOnPolicyRunner are supported"
        )
    if len(spec.action_scale) == 1 and spec.num_actions > 1:
        spec.action_scale = spec.action_scale * spec.num_actions
    if len(spec.joint_names) != spec.num_actions or len(spec.action_scale) != spec.num_actions:
        raise ValueError(
            f"{path}: {spec.num_actions} outputs but {len(spec.joint_names)} joint_names / "
            f"{len(spec.action_scale)} action_scale entries"
        )
    return sess, spec


class RslRlOnnxPolicy(Policy):
    """Deterministic rollout of an mjlab/rsl_rl actor from its exported ONNX file.

    Args:
        onnx_path: ``<run>.onnx`` written by mjlab, a run directory holding one,
            or ``hf://owner/repo/file.onnx``.
        command: Default command vector when the caller passes no
            ``target_velocity`` / ``target_pose`` kwarg (locomotion: ``[vx, vy, wz]``).
        robot: strands-robots robot name whose MJCF gives forward kinematics for
            ``ee_to_target`` (default ``"so101"`` when that term is present).
        ee_site: MJCF site name treated as the end effector (default ``"gripper"``).
        target: Default reach target ``[x, y, z]`` in the base frame.
        **kwargs: Ignored, for factory uniformity.
    """

    reads_instruction: ClassVar[bool] = False
    instruction_free_actions: ClassVar[str | None] = "the rsl_rl actor's per-step joint targets"

    def __init__(
        self,
        onnx_path: str = "",
        command: list[float] | None = None,
        robot: str | None = None,
        ee_site: str = "gripper",
        target: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not onnx_path or not str(onnx_path).strip():
            raise ValueError(
                "onnx_path is required for the 'rsl_rl_onnx' provider: the <run>.onnx mjlab wrote "
                "next to its checkpoints, a run directory, or hf://owner/repo/file.onnx"
            )
        self._sess, self.spec = load_actor_spec(str(onnx_path).strip())
        unknown = [n for n in self.spec.observation_names if n not in _BUILDERS]
        if unknown:
            raise ValueError(
                f"{self.spec.path}: observation terms {unknown} have no builder in this provider "
                f"(known: {list(_BUILDERS)})"
            )
        if any(h > 0 for h in self.spec.obs_history):
            raise ValueError(f"{self.spec.path}: observation history stacking is not supported yet")
        self._input_name = self._sess.get_inputs()[0].name
        self._command = np.asarray(command if command is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self._target = np.asarray(target if target is not None else [0.2, 0.0, 0.15], dtype=np.float32)
        self._last_action = np.zeros(self.spec.num_actions, dtype=np.float32)
        self._default = np.asarray(self.spec.default_joint_pos, dtype=np.float32)
        self._scale = np.asarray(self.spec.action_scale, dtype=np.float32)
        self._fk = None
        self._ee_site = ee_site
        if "ee_to_target" in self.spec.observation_names:
            self._fk = _SiteFK(robot or "so101", ee_site, self.spec.joint_names)
        self.robot_state_keys: list[str] = []
        logger.info(
            "rsl_rl_onnx loaded %s: obs=%d terms=%s actions=%d joints=%s",
            self.spec.path,
            self.spec.obs_dim,
            self.spec.observation_names,
            self.spec.num_actions,
            self.spec.joint_names[:4],
        )

    @property
    def provider_name(self) -> str:
        """Provider name for identification (always ``"rsl_rl_onnx"``)."""
        return "rsl_rl_onnx"

    @property
    def requires_images(self) -> bool:
        """An rsl_rl state actor consumes scalar state only."""
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's ordered action keys (informational: the ONNX names its own joints)."""
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Clear the previous-action term at an episode boundary."""
        self._last_action[:] = 0.0

    # ------------------------------------------------------------ observation

    def _term(self, name: str, obs: dict[str, Any], kwargs: dict[str, Any]) -> np.ndarray:
        if name == "joint_pos":
            return np.asarray([float(obs[j]) for j in self.spec.joint_names], dtype=np.float32) - self._default
        if name == "joint_vel":
            return np.asarray([float(obs.get(f"{j}.vel", 0.0)) for j in self.spec.joint_names], dtype=np.float32)
        if name == "actions":
            return self._last_action
        if name == "command":
            tv = kwargs.get("target_velocity")
            if tv is not None:
                v = np.zeros_like(self._command)
                v[: len(tv)] = np.asarray(tv, dtype=np.float32)[: len(v)]
                return v
            return self._command
        quat = np.asarray(obs.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        if name == "base_lin_vel":
            return _quat_rotate_inverse(quat, np.asarray(obs.get("base_lin_vel", [0, 0, 0]), dtype=np.float64))
        if name == "base_ang_vel":
            return np.asarray(obs.get("base_ang_vel", [0, 0, 0]), dtype=np.float32)
        if name == "projected_gravity":
            return _quat_rotate_inverse(quat, _GRAVITY.astype(np.float64))
        if name == "ee_to_target":
            tp = kwargs.get("target_pose")
            target = np.asarray(tp[:3], dtype=np.float32) if tp is not None else self._target
            assert self._fk is not None
            ee = self._fk.site_pos([float(obs[j]) for j in self.spec.joint_names])
            return (target - ee).astype(np.float32)
        raise KeyError(name)

    def build_observation(self, obs: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """Rebuild the actor's flat observation vector from an engine observation dict."""
        missing = [j for j in self.spec.joint_names if j not in obs]
        if missing:
            raise ValueError(f"observation omits joints the actor was trained on: {missing}")
        parts = []
        for i, name in enumerate(self.spec.observation_names):
            t = self._term(name, obs, kwargs).astype(np.float32)
            if i < len(self.spec.obs_scale):
                t = t * np.float32(self.spec.obs_scale[i])
            if i < len(self.spec.obs_clip):
                lo, hi = self.spec.obs_clip[i]
                t = np.clip(t, lo, hi)
            parts.append(t)
        vec = np.concatenate(parts)
        if vec.shape[0] != self.spec.obs_dim:
            raise ValueError(
                f"rebuilt observation has {vec.shape[0]} values but the actor expects {self.spec.obs_dim} "
                f"(terms {self.spec.observation_names})"
            )
        return vec

    def act(self, obs: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """One forward pass; returns absolute joint targets in the actor's joint order."""
        vec = self.build_observation(obs, **kwargs)
        raw = self._sess.run(None, {self._input_name: vec[None, :]})[0][0].astype(np.float32)
        self._last_action = raw
        return self._default + self._scale * raw

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """One-tick chunk of ``{joint: target}``; ``target_velocity`` / ``target_pose`` kwargs feed the command terms."""
        target = self.act(observation_dict, **kwargs)
        return [{j: float(target[i]) for i, j in enumerate(self.spec.joint_names)}]


class _SiteFK:
    """MuJoCo forward kinematics of one site from the robot's own MJCF (CPU, ~20 us)."""

    def __init__(self, robot: str, site: str, joint_names: list[str]) -> None:
        import mujoco

        from strands_robots.assets import resolve_model_path, resolve_robot_name

        self.model = mujoco.MjModel.from_xml_path(str(resolve_model_path(resolve_robot_name(robot))))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"site {site!r} not in {robot}'s MJCF")
        self.qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in joint_names
        ]
        self._mujoco = mujoco

    def site_pos(self, q: list[float]) -> np.ndarray:
        for adr, v in zip(self.qadr, q, strict=True):
            self.data.qpos[adr] = v
        self._mujoco.mj_kinematics(self.model, self.data)
        return np.asarray(self.data.site_xpos[self.site_id], dtype=np.float32)
