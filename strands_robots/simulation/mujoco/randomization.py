"""Domain randomization mixin."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, _ensure_mujoco

logger = logging.getLogger(__name__)


class RandomizationMixin:
    """Domain randomization mixed into ``Simulation``.

    Recolors geoms, perturbs lighting, and scales body mass (with a matching
    inertia scale, so randomized bodies stay physically consistent) and geom
    friction by a random factor inside a user-supplied range.

    **Coupling** (see simulation.py top-level docstring): mixin reaches
    into ``self._world``, ``self._lock``, and the host's
    ``_require_no_running_policy`` / ``_require_world`` helpers. ``TYPE_CHECKING``
    stubs below exist so mypy accepts those lookups; they are a
    documentary contract, not an enforceable protocol.
    """

    if TYPE_CHECKING:
        import threading

        from strands_robots.simulation.models import SimWorld

        _lock: "threading.RLock"
        _world: "SimWorld | None"

        def _require_no_running_policy(
            self, action_name: str, robot_name: str | None = None
        ) -> dict[str, Any] | None: ...
        def _require_world(self) -> dict[str, Any] | None: ...

    def randomize(
        self,
        randomize_colors: bool = True,
        randomize_lighting: bool = True,
        randomize_physics: bool = False,
        randomize_positions: bool = False,
        position_noise: float = 0.02,
        color_range: tuple[float, float] = (0.1, 1.0),
        friction_range: tuple[float, float] = (0.5, 1.5),
        mass_range: tuple[float, float] = (0.5, 2.0),
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply domain randomization to the scene.

        Each flag is opt-in per-axis. Defaults:
          - ``randomize_colors=True`` - geom RGB re-sampled in ``color_range``.
          - ``randomize_lighting=True`` - light pos jittered ±0.5m, diffuse resampled.
          - ``randomize_physics=False`` - friction/mass left untouched unless asked.
          - ``randomize_positions=False`` - object qpos left untouched unless asked.

        "No flags" means "nothing is randomized" - the call is a no-op. This
        matches the LLM ergonomics principle: explicit is better than implicit.
        Randomization IS destructive (writes to ``model.geom_*`` / ``body_*``
        arrays and to ``data.qpos``); recompile the scene to undo.

        Args:
            randomize_colors:     Re-sample every non-ground geom's RGB (and
                                  its material colour, which overrides geom RGB
                                  in the renderer).
            randomize_lighting:   Jitter light positions + diffuse colour.
            randomize_physics:    Scale geom friction and body mass (body
                                  inertia is scaled by the same factor as the
                                  mass so each randomized body stays physically
                                  consistent).
            randomize_positions:  Add uniform noise to dynamic-object xyz.
            position_noise:       Max ± xyz offset in meters when randomising positions.
            color_range:          (lo, hi) for uniform RGB sampling.
            friction_range:       (lo, hi) multiplicative scale on friction[0].
            mass_range:           (lo, hi) multiplicative scale on body_mass.
            seed:                 Optional np.random seed for reproducibility.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # domain randomization mutates model arrays; a running policy racing with it is UB
        if err := self._require_no_running_policy("randomize"):
            return err

        rng = np.random.default_rng(seed)
        mj = _ensure_mujoco()
        model = self._world._model
        data = self._world._data
        changes = []

        with self._lock:
            if randomize_colors:
                # Recolor every geom except the ground plane. Two correctness
                # points, both previously silent:
                #   1. Robot mesh geoms are typically UNNAMED, so a truthiness
                #      check on the name skipped them entirely - the robot kept
                #      its original colors while the call reported success.
                #   2. A geom that references a material draws its colour from
                #      that material in the renderer, NOT from geom_rgba, so the
                #      recolor is visually inert unless the material is updated
                #      too. Geoms sharing one material converge to the last
                #      colour written - acceptable for domain randomization.
                n_recolored = 0
                for i in range(model.ngeom):
                    if mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) == "ground":
                        continue
                    color = rng.uniform(color_range[0], color_range[1], size=3)
                    model.geom_rgba[i, :3] = color
                    matid = int(model.geom_matid[i])
                    if matid >= 0:
                        model.mat_rgba[matid, :3] = color
                    n_recolored += 1
                changes.append(f"Colors: {n_recolored} geoms randomized")

            if randomize_lighting:
                for i in range(model.nlight):
                    model.light_pos[i] += rng.uniform(-0.5, 0.5, size=3)
                    model.light_diffuse[i] = rng.uniform(0.3, 1.0, size=3)
                changes.append(f"Lighting: {model.nlight} lights randomized")

            if randomize_physics:
                friction_scales = {}
                for i in range(model.ngeom):
                    gn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                    f = float(rng.uniform(*friction_range))
                    model.geom_friction[i, 0] *= f
                    friction_scales[gn] = f
                mass_scales = {}
                for i in range(model.nbody):
                    if model.body_mass[i] > 0:
                        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                        s = float(rng.uniform(*mass_range))
                        model.body_mass[i] *= s
                        # Inertia tracks mass for fixed geometry: scaling a
                        # rigid body's mass by ``s`` at constant shape (a uniform
                        # density change) scales its inertia tensor by the same
                        # ``s`` (I = integral of r^2 dm). Scaling mass alone
                        # leaves a physically inconsistent body - heavy in
                        # translation but with the light body's rotational
                        # resistance - which silently corrupts the dynamics the
                        # randomization is meant to perturb. Match the Newton
                        # backend, which scales both.
                        model.body_inertia[i] *= s
                        mass_scales[bn] = s
                changes.append(
                    f"Physics: {len(friction_scales)} geoms friction-scaled, {len(mass_scales)} bodies mass-scaled"
                )
                changes.append(f"   friction_scales={friction_scales}")
                changes.append(f"   mass_scales={mass_scales}")

            if randomize_positions:
                for obj_name, obj in self._world.objects.items():
                    if not obj.is_static:
                        jnt_name = f"{obj_name}_joint"
                        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                        if jnt_id >= 0:
                            qpos_addr = model.jnt_qposadr[jnt_id]
                            noise = rng.uniform(-position_noise, position_noise, size=3)
                            data.qpos[qpos_addr : qpos_addr + 3] += noise
                changes.append(f"Positions: ±{position_noise}m noise on dynamic objects")

            # Recompute derived state so the sim is left render-ready. Several
            # randomization axes mutate model arrays whose rendered/simulated
            # effect flows through data: light_pos -> data.light_xpos (the
            # array the renderer reads, NOT model.light_pos), and object qpos ->
            # body xpos. Without a forward the next render()/get_observation()
            # keeps stale derived values, so a light-position jitter is a silent
            # visual no-op until some later mj_step. Mirror the mutate-then-
            # forward contract already used by reset(), load_scene() and
            # move_object(). Guarded on ``changes`` so a no-flag call stays a
            # true no-op.
            if changes:
                mj.mj_forward(model, data)

        return {
            "status": "success",
            "content": [{"text": "Domain Randomization applied:\n" + "\n".join(changes)}],
        }
