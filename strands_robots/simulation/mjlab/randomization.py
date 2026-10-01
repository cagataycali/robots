"""Per-world domain randomization for the mjlab backend.

The MuJoCo and Newton backends randomize ONE world and rebuild. mjlab runs
``num_envs`` worlds on one ``mjwarp.Model`` whose per-world fields can be
*expanded* (``Simulation.expand_model_fields``), so every world gets its own
draw in a single call and no rebuild: ``randomize(randomize_physics=True)`` on a
256-world engine produces 256 different friction/mass multipliers. The keyword
names, defaults and order mirror the MuJoCo backend so randomization code
transfers unchanged; the ``json`` block returns the applied multipliers with a
leading world axis (``mass_scales[world][body]``).

Supported axes: ``randomize_physics`` (geom friction + body mass with inertia
scaled alongside, then ``set_const`` recomputed on the GPU) and
``randomize_positions`` (each object's root position jittered per world).
``randomize_colors`` / ``randomize_lighting`` are refused explicitly: the
batched worlds have no renderer (the ``render`` fallback rasterises world 0
with ``mujoco.Renderer`` from the CPU model), so a per-world colour draw would
be a successful no-op for 255 of 256 worlds. Use the MuJoCo backend for visual
randomization; use this one for physics at scale.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.base import (
    finite_non_negative_error,
    randomization_range_error,
    randomization_seed_error,
    unknown_kwargs_error,
)
from strands_robots.utils import boolean_flag_error

logger = logging.getLogger(__name__)

_OBS_NOISE_PARAMS: tuple[str, ...] = ("joint_pos_std", "joint_vel_std", "camera_jitter_px", "seed")

_RANDOMIZE_PARAMS: tuple[str, ...] = (
    "randomize_colors",
    "randomize_lighting",
    "randomize_physics",
    "randomize_positions",
    "position_noise",
    "color_range",
    "friction_range",
    "mass_range",
    "seed",
)

# Model fields a physics draw touches, and the derived constants mjlab must
# recompute afterwards (body_mass/body_inertia -> set_const; geom_friction -> none).
_PHYSICS_FIELDS: tuple[str, ...] = (
    "geom_friction",
    "body_mass",
    "body_inertia",
    # Derived constants set_const rewrites; expanded too so each world keeps its
    # own subtree mass / inverse weights (mjlab's EventManager does the same).
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
)


class MjlabRandomizationMixin:
    """Domain randomization for :class:`MjlabEngine` (mixed into the engine class)."""

    if TYPE_CHECKING:
        _lock: threading.RLock
        _sim: Any
        _scene: Any
        _robots: dict[str, Any]
        _objects: dict[str, Any]
        num_envs: int
        device: str
        _dr_applied: dict[str, Any] | None
        _obs_noise: dict[str, float] | None
        _obs_noise_rng: np.random.Generator | None

        def _ensure_built(self) -> None: ...

    def randomize(
        self,
        randomize_colors: bool = False,
        randomize_lighting: bool = False,
        randomize_physics: bool = False,
        randomize_positions: bool = False,
        position_noise: float = 0.02,
        color_range: tuple[float, float] = (0.1, 1.0),
        friction_range: tuple[float, float] = (0.5, 1.5),
        mass_range: tuple[float, float] = (0.5, 2.0),
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Draw per-world physics / object-position randomization on the GPU model.

        Defaults differ from the MuJoCo backend in one place: ``randomize_colors``
        and ``randomize_lighting`` default to ``False`` because they are not
        supported here and a default-on refusal would make the bare
        ``randomize()`` call fail. Passing either as ``True`` returns an error
        naming the supported axes.

        Args:
            randomize_colors: Not supported (no batched renderer). ``True`` -> error.
            randomize_lighting: Not supported. ``True`` -> error.
            randomize_physics: Per world, scale every robot/object geom's sliding
                friction by a draw in ``friction_range`` and every body's mass AND
                inertia by a draw in ``mass_range`` (uniform density), then
                recompute the derived constants (``mjwarp.set_const``).
            randomize_positions: Per world, jitter each object's root position by
                ``N(0, position_noise)`` in x/y (z untouched so nothing spawns
                below the plane).
            position_noise: Std-dev in metres for ``randomize_positions``.
            color_range: Accepted for signature parity; unused.
            friction_range: ``(lo, hi)`` multiplicative range on friction.
            mass_range: ``(lo, hi)`` multiplicative range on mass; strictly positive.
            seed: Non-negative int for a reproducible draw, or None.
            **kwargs: Any other keyword is rejected by name.

        Returns:
            Status dict; ``json`` carries ``friction_scales`` (worlds x geoms),
            ``mass_scales`` (worlds x bodies), ``position_offsets``
            (object -> worlds x 3), ``num_envs`` and ``seed``.
        """
        if kwargs_error := unknown_kwargs_error("randomize", kwargs, _RANDOMIZE_PARAMS):
            return kwargs_error
        for flag_param, flag_value in (
            ("randomize_colors", randomize_colors),
            ("randomize_lighting", randomize_lighting),
            ("randomize_physics", randomize_physics),
            ("randomize_positions", randomize_positions),
        ):
            if msg := boolean_flag_error(flag_value, flag_param, "randomize"):
                return {"status": "error", "content": [{"text": msg}]}
        if randomize_colors or randomize_lighting:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "randomize_colors / randomize_lighting are not supported by the mjlab backend: "
                            "the batched worlds have no renderer, so a per-world colour draw would be a no-op. "
                            "Supported axes: randomize_physics, randomize_positions. "
                            "Use the MuJoCo backend for visual randomization."
                        )
                    }
                ],
            }
        for label, rng_range, allow_zero in (
            ("mass_range", mass_range, False),
            ("friction_range", friction_range, True),
            ("color_range", color_range, True),
        ):
            if msg := randomization_range_error(rng_range, label, allow_zero=allow_zero):
                return {"status": "error", "content": [{"text": msg}]}
        if msg := finite_non_negative_error(position_noise, "position_noise", "randomize"):
            return {"status": "error", "content": [{"text": msg}]}
        if msg := randomization_seed_error(seed, "randomize"):
            return {"status": "error", "content": [{"text": msg}]}

        with self._lock:
            if not self._robots and not self._objects:
                return {"status": "error", "content": [{"text": "Nothing to randomize: add a robot or object first."}]}
            self._ensure_built()
            rng = np.random.default_rng(seed)
            applied: dict[str, Any] = {"num_envs": int(self.num_envs), "seed": seed}
            changes: list[str] = []
            if randomize_physics:
                n_geoms, n_bodies = self._randomize_physics(rng, friction_range, mass_range, applied)
                changes.append(
                    f"Physics: {n_bodies} bodies mass-scaled, {n_geoms} geoms friction-scaled, "
                    f"independently in each of {self.num_envs} worlds"
                )
            if randomize_positions:
                n_obj = self._randomize_positions(rng, float(position_noise), applied)
                changes.append(f"Positions: {n_obj} objects jittered (sigma {position_noise} m) per world")
            if not changes:
                changes.append("No axes enabled; nothing randomized.")
            self._dr_applied = applied

        return {
            "status": "success",
            "content": [{"text": "Domain randomization applied:\n" + "\n".join(changes)}, {"json": applied}],
        }

    def _randomize_physics(
        self,
        rng: np.random.Generator,
        friction_range: tuple[float, float],
        mass_range: tuple[float, float],
        applied: dict[str, Any],
    ) -> tuple[int, int]:
        import torch
        from mjlab.managers.event_manager import RecomputeLevel

        sim = self._sim
        sim.expand_model_fields(_PHYSICS_FIELDS)  # idempotent: (N, ...) copies of the CPU model's arrays
        geom_ids: list[int] = []
        body_ids: list[int] = []
        for name in list(self._robots) + list(self._objects):
            idx = self._scene[name].indexing
            geom_ids.extend(int(g) for g in torch.as_tensor(idx.geom_ids).flatten().tolist())
            body_ids.extend(int(b) for b in torch.as_tensor(idx.body_ids).flatten().tolist())
        geom_ids = sorted(set(geom_ids))
        body_ids = sorted(set(body_ids))
        n = int(self.num_envs)
        dev = self.device

        fric = rng.uniform(friction_range[0], friction_range[1], size=(n, len(geom_ids))).astype(np.float32)
        mass = rng.uniform(mass_range[0], mass_range[1], size=(n, len(body_ids))).astype(np.float32)
        g_idx = torch.as_tensor(geom_ids, device=dev, dtype=torch.long)
        b_idx = torch.as_tensor(body_ids, device=dev, dtype=torch.long)
        w_idx = torch.arange(n, device=dev, dtype=torch.long)
        wg, gg = torch.meshgrid(w_idx, g_idx, indexing="ij")
        wb, bb = torch.meshgrid(w_idx, b_idx, indexing="ij")

        # Always scale from the compiled defaults so repeated calls do not compound.
        d_fric = sim.get_default_field("geom_friction")[g_idx, 0]  # (G,)
        d_mass = sim.get_default_field("body_mass")[b_idx]  # (B,)
        d_inert = sim.get_default_field("body_inertia")[b_idx]  # (B, 3)
        fric_t = torch.as_tensor(fric, device=dev)
        mass_t = torch.as_tensor(mass, device=dev)
        sim.model.geom_friction[wg, gg, 0] = d_fric[None, :] * fric_t
        sim.model.body_mass[wb, bb] = d_mass[None, :] * mass_t
        for axis in range(3):
            sim.model.body_inertia[wb, bb, axis] = d_inert[None, :, axis] * mass_t
        sim.recompute_constants(RecomputeLevel.set_const)
        sim.forward()

        applied["friction_scales"] = fric.round(6).tolist()
        applied["mass_scales"] = mass.round(6).tolist()
        applied["geom_ids"] = geom_ids
        applied["body_ids"] = body_ids
        return len(geom_ids), len(body_ids)

    def _randomize_positions(self, rng: np.random.Generator, sigma: float, applied: dict[str, Any]) -> int:
        import torch

        offsets: dict[str, list[list[float]]] = {}
        n = int(self.num_envs)
        for name in self._objects:
            ent = self._scene[name]
            if getattr(ent, "is_fixed_base", False):
                continue
            root = ent.data.default_root_state.clone()  # (N, 13)
            root[:, :3] += self._scene.env_origins
            d = np.zeros((n, 3), dtype=np.float32)
            d[:, :2] = rng.normal(0.0, sigma, size=(n, 2))
            root[:, :3] += torch.as_tensor(d, device=root.device)
            ent.write_root_state_to_sim(root)
            offsets[name] = d.round(6).tolist()
        self._sim.forward()
        applied["position_offsets"] = offsets
        return len(offsets)

    def set_obs_noise(
        self,
        joint_pos_std: float = 0.0,
        joint_vel_std: float = 0.0,
        camera_jitter_px: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Configure additive Gaussian sensor noise on observations (same contract as the other backends).

        Applied to every world in :meth:`get_observation_batch` and to world 0 in
        :meth:`get_observation`; ``camera_jitter_px`` is accepted for parity but has
        no effect here because the batched worlds have no camera stream. All zeros
        (the default) switches the noise off.

        Args:
            joint_pos_std: Std-dev in radians added to every joint position.
            joint_vel_std: Std-dev in rad/s added to every joint velocity.
            camera_jitter_px: Accepted for parity; no camera stream to jitter.
            seed: Non-negative int for a reproducible noise stream, or None.
            **kwargs: Any other keyword is rejected by name.

        Returns:
            Status dict describing the configured noise.
        """
        if kwargs_error := unknown_kwargs_error("set_obs_noise", kwargs, _OBS_NOISE_PARAMS):
            return kwargs_error
        for label, value in (
            ("joint_pos_std", joint_pos_std),
            ("joint_vel_std", joint_vel_std),
            ("camera_jitter_px", camera_jitter_px),
        ):
            if msg := finite_non_negative_error(value, label, "set_obs_noise"):
                return {"status": "error", "content": [{"text": msg}]}
        if msg := randomization_seed_error(seed, "set_obs_noise"):
            return {"status": "error", "content": [{"text": msg}]}
        with self._lock:
            if joint_pos_std == 0.0 and joint_vel_std == 0.0 and camera_jitter_px == 0.0:
                self._obs_noise = None
                self._obs_noise_rng = None
                return {"status": "success", "content": [{"text": "Sensor noise disabled."}]}
            self._obs_noise = {
                "joint_pos_std": float(joint_pos_std),
                "joint_vel_std": float(joint_vel_std),
                "camera_jitter_px": float(camera_jitter_px),
            }
            self._obs_noise_rng = np.random.default_rng(seed)
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Sensor noise: joint_pos_std={joint_pos_std}, joint_vel_std={joint_vel_std}, "
                        f"camera_jitter_px={camera_jitter_px} (no camera stream on this backend)"
                    )
                }
            ],
        }

    def _apply_obs_noise_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Add the configured Gaussian noise to ``(N, ...)`` joint tensors; base_* keys untouched."""
        noise = getattr(self, "_obs_noise", None)
        rng = getattr(self, "_obs_noise_rng", None)
        if not noise or rng is None:
            return batch
        import torch

        out: dict[str, Any] = {}
        for k, v in batch.items():
            if k.startswith("base_") or not hasattr(v, "shape"):
                out[k] = v
                continue
            std = noise["joint_vel_std"] if k.endswith(".vel") else noise["joint_pos_std"]
            if std <= 0.0:
                out[k] = v
                continue
            eps = rng.normal(0.0, std, size=tuple(v.shape)).astype(np.float32)
            out[k] = v + torch.as_tensor(eps, device=v.device, dtype=v.dtype)
        return out
