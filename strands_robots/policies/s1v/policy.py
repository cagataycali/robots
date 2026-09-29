"""S1V policy provider: a System 1 vision decider driving an arm one primitive per tick.

The model (:class:`~strands_robots.policies.s1v.model.S1VDecider`) answers six typed
questions from two camera images and the joint state, in one non-autoregressive
forward pass: which joint, which direction, which step size (the choice heads) and
three yes/no questions about a candidate primitive (``cube_in_jaws``,
``progress_if_executed``, ``safe``). :class:`S1VBrain` turns the answers into one
primitive; :class:`S1VPolicy` wraps that in the ``Policy`` contract so the same
runner seam that drives FLUX 3 Action or a lerobot checkpoint drives the decider.

Instruction text is ignored in v1: the task (``"reach"`` or ``"pick"``) is a config
key, embedded as a learned token. A text encoder is the v2 route.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from strands_robots.policies._state_keys import observation_joint_keys
from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error, require_optional

from .primitives import (
    DEFAULT_GRIPPER_STEP_PCT,
    DEFAULT_STEP_DEG,
    GRIPPER_JOINT,
    HOLD,
    SIZE_LABELS,
    SO101_ARM_LABELS,
    SO101_CTRL_BOUNDS,
    SO101_GRIPPER_RANGE,
    SO101_LABELS,
    Primitive,
    apply_primitive,
    gripper_pct_from_rad,
    primitive_index,
)

logger = logging.getLogger(__name__)

TASKS = ("reach", "pick")
_CAMERA_ROLES = ("scene", "wrist")
#: Arm setpoints may lead the measured joint by at most this much (radians, ~8 deg);
#: the same anti-windup the scripted expert uses so a pinned fingertip cannot run
#: the integrator away.
WINDUP_MAX_RAD = 0.14


class S1VBrain:
    """Frozen DINOv2-small + the trained decider; ``decide`` is one tick.

    Args:
        pretrained_name_or_path: Local directory or Hugging Face repo id holding
            ``config.json`` + ``model.pt`` written by ``S1VDecider.save_pretrained``.
        device: Torch device.
        temperature_scaling: Apply the per-head temperatures fitted after training.
        confidence_gate: When set, hold instead of moving whenever the chosen
            joint's probability or the ``safe`` answer for the chosen primitive is
            below it. Opt-in and measured to cost success on this rig (reach 15/20
            ungated, 11/20 at 0.5, 2/20 at 0.7): the joint head's doubt is mostly
            between two primitives that both progress, and a hold is not free.
        cuda_graph: Capture the decider forward in a CUDA graph (fixed shapes) for
            the lowest tick latency; falls back to eager on CPU.
    """

    def __init__(
        self,
        pretrained_name_or_path: str | Path,
        *,
        device: str = "cuda",
        temperature_scaling: bool = True,
        confidence_gate: float | None = None,
        cuda_graph: bool = False,
    ) -> None:
        torch = require_optional("torch", extra="s1v", purpose="S1V decider inference")
        require_optional("transformers", extra="s1v", purpose="frozen DINOv2-small features")
        from .dataset import featurize_images, load_backbone
        from .model import CANDIDATE_NONE, CHOICE_HEADS, NOUL_HEADS, S1VDecider

        self._torch = torch
        self._featurize = featurize_images
        self._none = CANDIDATE_NONE
        self._choice_heads = tuple(CHOICE_HEADS)
        self._noul_heads = tuple(NOUL_HEADS)
        self.device = device
        self.temperature_scaling = bool(temperature_scaling)
        self.confidence_gate = None if confidence_gate is None else float(confidence_gate)
        path = _resolve_checkpoint(pretrained_name_or_path)
        t0 = time.perf_counter()
        self.model = S1VDecider.from_pretrained(path, device=device)
        self.backbone, self._mean, self._std = load_backbone(device)
        self.load_s = time.perf_counter() - t0
        self.grid = int(self.model.cfg.grid)
        self._graph = None
        if cuda_graph and str(device).startswith("cuda"):
            self._graph = self._capture()
        self.tick_ms: list[float] = []
        self.decisions: list[dict[str, Any]] = []

    def _capture(self):
        torch = self._torch
        n_tok = self.model.n_cam_tokens
        feat = self.model.cfg.feat_dim
        static = {
            "cams": torch.zeros(1, n_tok, feat, device=self.device),
            "state": torch.zeros(1, 6, device=self.device),
            "task": torch.zeros(1, dtype=torch.long, device=self.device),
            "candidate": torch.full((1,), self._none, dtype=torch.long, device=self.device),
        }
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self.model(**static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            out = self.model(**static)
        return {"graph": g, "static": static, "out": out}

    def _forward(self, cams, state, task, candidate) -> dict[str, Any]:
        torch = self._torch
        with torch.inference_mode():
            if self._graph is None:
                return self.model(cams, state, task, candidate)
            st = self._graph["static"]
            st["cams"].copy_(cams)
            st["state"].copy_(state)
            st["task"].copy_(task)
            st["candidate"].copy_(candidate)
            self._graph["graph"].replay()
            return {k: v.clone() for k, v in self._graph["out"].items()}

    def features(self, scene: np.ndarray, wrist: np.ndarray):
        """Two uint8 (224,224,3) images -> (1, 2*(1+grid^2), 384) decider input."""
        torch = self._torch
        both = self._featurize(
            self.backbone, self._mean, self._std, np.stack([scene, wrist]), grid=self.grid, batch=2
        )  # (2, tokens, 384) f16
        return torch.from_numpy(both).to(self.device).reshape(1, -1, both.shape[-1])

    def decide(
        self, scene: np.ndarray, wrist: np.ndarray, state6: Sequence[float], task: str
    ) -> tuple[Primitive, dict]:
        """One tick: images + proprio (deg x5, gripper %) -> primitive and its probabilities.

        The choice heads run with no candidate; the noul heads run once more with
        the chosen primitive as candidate so ``safe`` / ``progress_if_executed``
        speak about what is about to execute. Returns ``(primitive, info)`` where
        ``info`` carries ``p_joint``, ``p_direction``, ``p_size``, the three noul
        probabilities, ``gated`` and ``ms``.
        """
        torch = self._torch
        t0 = time.perf_counter()
        cams = self.features(scene, wrist)
        state = torch.tensor([list(map(float, state6))], device=self.device)
        task_t = torch.tensor([TASKS.index(task)], device=self.device)
        none = torch.tensor([self._none], device=self.device)
        choice = self.model.probabilities(self._forward(cams, state, task_t, none), calibrated=self.temperature_scaling)
        pj = choice["joint"][0]
        pd = choice["direction"][0]
        ps = choice["size"][0]
        j = int(pj.argmax())
        joints = (*SO101_ARM_LABELS, GRIPPER_JOINT)
        if j >= len(joints):
            prim = HOLD
        else:
            prim = Primitive(joints[j], 1.0 if int(pd.argmax()) == 0 else -1.0, SIZE_LABELS[int(ps.argmax())])
        cand = torch.tensor([primitive_index(prim)], device=self.device)
        noul = self.model.probabilities(self._forward(cams, state, task_t, cand), calibrated=self.temperature_scaling)
        info = {
            "p_joint": float(pj[j]),
            "p_direction": float(pd.max()),
            "p_size": float(ps.max()),
            "chosen": prim.as_dict(),
            "gated": False,
        }
        for q in self._noul_heads:
            info[q] = float(noul[q][0])
        if self.confidence_gate is not None and not prim.is_hold:
            if info["p_joint"] < self.confidence_gate or info["safe"] < self.confidence_gate:
                prim = HOLD
                info["gated"] = True
        info["ms"] = (time.perf_counter() - t0) * 1000.0
        self.tick_ms.append(info["ms"])
        self.decisions.append(info)
        return prim, info

    def actor(self, task: str):
        """``(observation, privileged, expert) -> primitive`` for ``expert.run_episode``."""
        from .expert import state_vector

        def act(obs: dict[str, Any], priv: Any, _expert: Primitive) -> Primitive:
            prim, _ = self.decide(obs["scene"], obs["wrist"], state_vector(priv.qpos), task)
            return prim

        return act


def _resolve_checkpoint(name_or_path: str | Path) -> Path:
    """A local directory as is; otherwise a private/public HF repo snapshot."""
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        return p
    hub = require_optional("huggingface_hub", extra="s1v", purpose="downloading the S1V checkpoint")
    return Path(hub.snapshot_download(str(name_or_path)))


class S1VPolicy(Policy):
    """System 1 vision decider as a robots policy: one joint-target dict per tick.

    Args:
        pretrained_name_or_path: Local checkpoint directory or HF repo id.
        task: ``"reach"`` or ``"pick"``; the instruction text is ignored in v1.
        device: Torch device for the decider and DINOv2-small.
        camera_map: Observation image key per role, ``{"scene": ..., "wrist": ...}``.
        step_deg: Arm step per size label in degrees (default 2/5/10).
        gripper_step_pct: Gripper step per primitive in percent of its travel.
        confidence_gate: Hold when the chosen joint's probability or the ``safe``
            answer is below this; ``None`` (default) disables the gate, which is the
            measured-better setting (see the lane report).
        temperature_scaling: Use the fitted per-head temperatures.
        cuda_graph: Capture the decider forward in a CUDA graph.
        labels: ``{state_key: joint_label}`` for the robot (default so101 sim keys
            ``"1"``..``"6"``).
        gripper_range: ``(closed, open)`` gripper joint values in robot units.
        ctrl_bounds: Optional per-key clip for arm targets.
    """

    provider_name: ClassVar[str] = "s1v"
    requires_images: ClassVar[bool] = True
    reads_instruction: ClassVar[bool] = False

    def __init__(
        self,
        pretrained_name_or_path: str,
        task: str = "pick",
        device: str = "cuda",
        camera_map: dict[str, str] | None = None,
        step_deg: dict[str, float] | None = None,
        gripper_step_pct: float = DEFAULT_GRIPPER_STEP_PCT,
        confidence_gate: float | None = None,
        temperature_scaling: bool = True,
        cuda_graph: bool = False,
        labels: dict[str, str] | None = None,
        gripper_range: tuple[float, float] | list[float] = SO101_GRIPPER_RANGE,
        ctrl_bounds: dict[str, tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        self.camera_map = {role: role for role in _CAMERA_ROLES}
        if camera_map:
            unknown = set(camera_map) - set(_CAMERA_ROLES)
            if unknown:
                raise ValueError(f"camera_map roles must be {_CAMERA_ROLES}, unknown: {sorted(unknown)}")
            self.camera_map.update({k: str(v) for k, v in camera_map.items()})
        self.task = task
        self.step_deg = dict(DEFAULT_STEP_DEG if step_deg is None else step_deg)
        self.gripper_step_pct = float(gripper_step_pct)
        self.labels = dict(SO101_LABELS if labels is None else labels)
        self.gripper_range = (float(gripper_range[0]), float(gripper_range[1]))
        self.ctrl_bounds = dict(SO101_CTRL_BOUNDS if ctrl_bounds is None else ctrl_bounds)
        self.robot_state_keys: list[str] = []
        self._setpoint: dict[str, float] | None = None
        self.brain = S1VBrain(
            pretrained_name_or_path,
            device=device,
            temperature_scaling=temperature_scaling,
            confidence_gate=confidence_gate,
            cuda_graph=cuda_graph,
        )
        self.pretrained_name_or_path = pretrained_name_or_path
        self.device = device
        logger.info("s1v: loaded %s on %s in %.1fs", pretrained_name_or_path, device, self.brain.load_s)

    @property
    def tick_ms(self) -> list[float]:
        """Per-tick wall time of features + two decider forwards."""
        return self.brain.tick_ms

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Remember the robot's joint key order.

        Raises:
            ValueError: if the list is not an ordered list of distinct non-blank names.
        """
        if error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys"):
            raise ValueError(error)
        if isinstance(robot_state_keys, str | bytes) or not isinstance(robot_state_keys, Sequence):
            raise TypeError(
                f"s1v.set_robot_state_keys expects a list of joint names, got {type(robot_state_keys).__name__}"
            )
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Forget the setpoint integrator; the next observation seeds it."""
        super().reset(seed)
        self._setpoint = None
        if seed is not None:
            self.brain._torch.manual_seed(seed)

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        """Refuse before loading weights when a required camera is missing."""
        camera_map = {role: role for role in _CAMERA_ROLES}
        camera_map.update(policy_config.get("camera_map") or {})
        missing = [f"{role}={key!r}" for role, key in camera_map.items() if key not in observation_keys]
        if missing:
            images = sorted(k for k in observation_keys if "." not in k and not k.isdigit())
            raise ValueError(
                "s1v needs two cameras named by camera_map; missing "
                f"{', '.join(missing)}. Observation keys: {images}. Add cameras with "
                "robot.add_camera(name=...) or pass camera_map={'scene': ..., 'wrist': ...}."
            )
        task = policy_config.get("task", "pick")
        if task not in TASKS:
            raise ValueError(f"s1v task must be one of {TASKS}, got {task!r}")

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Decide one primitive and return the resulting joint targets for this tick.

        Args:
            observation_dict: Flat observation: six joint floats (radians) plus the two
                camera images as ``(H, W, 3)`` uint8 arrays.
            instruction: Ignored in v1 (the task is a config key).
            **kwargs: Ignored.

        Returns:
            ``[{joint_key: target}]`` with every joint key present, python floats.
        """
        keys = observation_joint_keys(observation_dict, self.robot_state_keys)
        if len(keys) != 6:
            raise ValueError(f"s1v expects six joint values in the observation, found {len(keys)}: {keys}")
        qpos = np.array([float(observation_dict[k]) for k in keys])
        state6 = [float(np.degrees(q)) for q in qpos[:5]] + [gripper_pct_from_rad(float(qpos[5]), self.gripper_range)]
        scene = self._image(observation_dict, "scene")
        wrist = self._image(observation_dict, "wrist")
        prim, _ = self.brain.decide(scene, wrist, state6, self.task)
        if self._setpoint is None or set(self._setpoint) != set(keys):
            self._setpoint = {k: float(q) for k, q in zip(keys, qpos, strict=True)}
        labels = {k: self.labels[k] for k in keys if k in self.labels}
        action = apply_primitive(
            prim,
            self._setpoint,
            labels,
            step_deg=self.step_deg,
            gripper_step_pct=self.gripper_step_pct,
            gripper_range=self.gripper_range,
            ctrl_bounds=self.ctrl_bounds,
        )
        for k, q in zip(keys[:5], qpos[:5], strict=True):
            action[k] = float(min(max(action[k], q - WINDUP_MAX_RAD), q + WINDUP_MAX_RAD))
        self._setpoint = action
        return [action]

    def _image(self, observation: dict[str, Any], role: str) -> np.ndarray:
        key = self.camera_map[role]
        if key not in observation:
            raise ValueError(f"s1v: camera {role}={key!r} missing from the observation")
        img = np.asarray(observation[key])
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"s1v: camera {key!r} must be (H, W, 3), got {img.shape}")
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if img.shape[:2] != (224, 224):
            img = _resize_224(img)
        return img

    def __repr__(self) -> str:
        return f"S1VPolicy({self.pretrained_name_or_path!r}, task={self.task!r}, gate={self.brain.confidence_gate})"


def _resize_224(img: np.ndarray) -> np.ndarray:
    """Nearest-neighbour resize to 224x224 without an image library dependency."""
    h, w = img.shape[:2]
    ys = (np.arange(224) * h / 224).astype(int)
    xs = (np.arange(224) * w / 224).astype(int)
    return img[ys][:, xs]
