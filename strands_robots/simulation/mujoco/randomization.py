"""Domain randomization mixin."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.base import (
    finite_non_negative_error,
    randomization_range_error,
    randomization_seed_error,
    unknown_kwargs_error,
)
from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, _ensure_mujoco

logger = logging.getLogger(__name__)

# Parameter names ``randomize`` / ``set_obs_noise`` actually honor. Both declare
# ``**kwargs`` to match the ``**kwargs``-typed SimEngine base signature, but
# neither forwards it anywhere - so anything landing there is a caller mistake
# and is rejected instead of dropped (test_domain_randomization_rejects_unknown_params
# pins these tuples to the live signatures).
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
_OBS_NOISE_PARAMS: tuple[str, ...] = (
    "joint_pos_std",
    "joint_vel_std",
    "camera_jitter_px",
    "seed",
)


class RandomizationMixin:
    """Domain randomization mixed into ``Simulation``.

    Recolors geoms, perturbs lighting, and scales body mass (with a matching
    inertia scale, so randomized bodies stay physically consistent) and geom
    friction by a random factor inside a user-supplied range.

    **Coupling** (see the :mod:`simulation` top-level docstring): mixin reaches
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
        _obs_noise: "dict[str, float] | None"
        _obs_noise_rng: "np.random.Generator | None"
        _obs_noise_seed: "int | None"
        _mj: "Any"

        def _require_no_running_policy(
            self, action_name: str, robot_name: str | None = None
        ) -> dict[str, Any] | None: ...
        def _require_world(self) -> dict[str, Any] | None: ...
        def _sync_spec_geom(self, geom_name: str, **changes: Any) -> None:
            """Provided by PhysicsMixin."""

        def _sync_spec_body(self, body_name: str, **changes: Any) -> None:
            """Provided by PhysicsMixin."""

    def _sync_randomization_to_spec(self, model: Any, colors: bool, physics: bool) -> None:
        """Mirror a randomization sample onto the live ``MjSpec``.

        ``randomize`` writes ``model.geom_rgba`` / ``mat_rgba`` /
        ``geom_friction`` / ``body_mass`` / ``body_inertia`` for immediate effect,
        but the spec is what every scene mutation recompiles from, so the sample
        was reverted by the next ``add_object`` / ``remove_object`` /
        ``add_camera`` / ``add_robot``. Keyed by NAME, never by index: a recompile
        shifts geom and body ids (AGENTS.md "Per-name state copy").

        Unnamed geoms and bodies are skipped - the spec cannot address them, and
        that is not new: their ``model.*`` values stay correct for the current
        model and are re-randomised by the next ``randomize`` call. ``lighting``
        is deliberately NOT mirrored: it is re-derived from ``light_pos``
        baselines each call and the renderer reads ``data.light_xpos``, so there
        is no cross-recompile contract to keep.

        Args:
            model: The live compiled model holding the sampled values.
            colors: Whether the colour axis was randomised.
            physics: Whether the friction/mass axis was randomised.
        """
        if self._world is None:
            return
        spec = self._world._backend_state.get("spec")
        if spec is None:
            return
        mj = self._mj
        for geom_id in range(int(model.ngeom)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            changes: dict[str, Any] = {}
            if colors:
                changes["rgba"] = [float(v) for v in model.geom_rgba[geom_id]]
            if physics:
                changes["friction"] = [float(v) for v in model.geom_friction[geom_id]]
            if changes:
                self._sync_spec_geom(name, **changes)
        if not physics:
            return
        for body_id in range(int(model.nbody)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
            if not name or float(model.body_mass[body_id]) <= 0.0:
                continue
            self._sync_spec_body(name, mass=float(model.body_mass[body_id]))

    def _dr_baseline(self, model: Any) -> dict[str, Any]:
        """Un-randomized copies of every model array ``randomize`` perturbs.

        Domain randomization must sample around the ORIGINAL model, not around
        whatever the previous call produced. These axes write ``model.*`` arrays,
        which ``reset()`` does not restore (``mj_resetData`` only touches
        ``data``), so scaling in place compounded every call: the documented
        per-episode loop (``for ep: sim.reset(); sim.randomize(...)``) drove a
        0.5 kg body to 1.2 kg after four calls with ``mass_range=(0.5, 2.0)``
        - outside the requested [0.25, 1.0] window - and to absurd values over a
        long eval run, while ``light_pos`` random-walked away from the scene.

        Snapshotted on first use and RE-MAPPED (not re-read) whenever the scene is
        recompiled, keyed on ``_recompile_generation``, so the baseline always
        describes the CURRENT model's bodies and geoms while still holding their
        un-randomised values.
        """
        assert self._world is not None  # callers hold the world guard
        # Key the cache on the recompile GENERATION, not on ``ngeom``. A shape
        # check misses any rebuild that leaves the counts unchanged: a
        # remove_object + add_object pair returns ngeom to its old value while
        # the arrays now describe DIFFERENT bodies, so the stale baseline was
        # applied to whichever body took the freed slot. Measured: an object with
        # a 0.1 kg base scaled against the removed body's 0.5 kg baseline reached
        # 0.2907 kg, outside its legal [0.05, 0.2] window.
        cache = self._world._backend_state.get("dr_baseline")
        generation = int(getattr(self._world, "_recompile_generation", 0))
        if cache is None:
            self._world._backend_state["dr_baseline"] = self._snapshot_dr_baseline(model, generation)
            return self._world._backend_state["dr_baseline"]
        if int(cache.get("generation", -1)) != generation:
            # The scene changed shape, so the cached arrays no longer line up by
            # index - but they DO still hold the pristine values, and the live
            # arrays hold randomised ones. Re-reading the live model here made the
            # current randomisation the new "original", so an eval loop that
            # churned the scene compounded without bound:
            #
            #     for ep: reset(); randomize(); (add_object + remove_object)
            #     baseline_mass[keep]  0.5000 -> 0.5134 -> 0.4078 -> 0.6664 ...
            #     live mass at ep 7    1.1423     (legal window is [0.25, 1.0])
            #
            # Carry the old values across by NAME instead, and only fall back to
            # the live model for entities the previous baseline never saw (newly
            # added bodies/geoms, which are un-randomised anyway).
            cache = self._remap_dr_baseline(model, cache, generation)
            self._world._backend_state["dr_baseline"] = cache
        return cache

    def _snapshot_dr_baseline(self, model: Any, generation: int) -> dict[str, Any]:
        """Fresh copy of every array ``randomize`` perturbs, keyed by name.

        The name maps are what let :meth:`_remap_dr_baseline` carry pristine
        values across a recompile that renumbers bodies and geoms.
        """
        mj = _ensure_mujoco()
        body_by_name: dict[str, Any] = {}
        for i in range(int(model.nbody)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
            if name:
                body_by_name[name] = (float(model.body_mass[i]), model.body_inertia[i].copy())
        geom_by_name: dict[str, Any] = {}
        for i in range(int(model.ngeom)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            if name:
                geom_by_name[name] = model.geom_friction[i].copy()
        return {
            "generation": generation,
            "ngeom": int(model.ngeom),
            "geom_friction": model.geom_friction.copy(),
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
            "light_pos": model.light_pos.copy(),
            "body_by_name": body_by_name,
            "geom_by_name": geom_by_name,
        }

    def _remap_dr_baseline(self, model: Any, cache: dict[str, Any], generation: int) -> dict[str, Any]:
        """Re-index a baseline onto a recompiled model, keeping pristine values.

        Entities present in the old baseline keep their un-randomised value at
        their NEW index; entities the baseline never saw take the live model's
        value (they were just added, so they are un-randomised by definition).
        """
        mj = _ensure_mujoco()
        fresh = self._snapshot_dr_baseline(model, generation)
        old_bodies = cache.get("body_by_name") or {}
        old_geoms = cache.get("geom_by_name") or {}
        for i in range(int(model.nbody)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
            prior = old_bodies.get(name) if name else None
            if prior is None:
                continue
            mass, inertia = prior
            fresh["body_mass"][i] = mass
            fresh["body_inertia"][i] = inertia
            fresh["body_by_name"][name] = (mass, inertia.copy())
        for i in range(int(model.ngeom)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            prior = old_geoms.get(name) if name else None
            if prior is None:
                continue
            fresh["geom_friction"][i] = prior
            fresh["geom_by_name"][name] = prior.copy()
        # ``light_pos`` has no names to key on; lights are declared by the scene
        # builder and not renumbered by object churn, so carry the old rows for
        # every light that still exists.
        old_lights = cache.get("light_pos")
        if old_lights is not None:
            shared = min(len(old_lights), len(fresh["light_pos"]))
            fresh["light_pos"][:shared] = old_lights[:shared]
        return fresh

    def reseed_obs_noise(self) -> None:
        """Restart the observation-noise stream from its configured seed.

        Called by ``reset()`` so a seeded run is reproducible PER EPISODE. The
        generator is otherwise continuous: episode 2 inherited wherever episode 1
        left off, so its noise depended on how many observations the previous
        episode happened to consume. A no-op when no seed was given (an unseeded
        stream is explicitly non-reproducible) or when noise is off.
        """
        seed = getattr(self, "_obs_noise_seed", None)
        if seed is None or getattr(self, "_obs_noise_rng", None) is None:
            return
        self._obs_noise_rng = np.random.default_rng(seed)

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
            position_noise:       Max ± xyz offset in meters when randomising
                                  positions. A finite non-negative number: a
                                  NaN half-width writes NaN into ``qpos`` and
                                  poisons every later step, a negative one
                                  inverts the sampling bounds.
            color_range:          (lo, hi) for uniform RGB sampling.
            friction_range:       (lo, hi) multiplicative scale on friction[0].
            mass_range:           (lo, hi) multiplicative scale on body_mass.
                                  Each range must be a pair of finite numbers
                                  with ``lo <= hi``, non-negative for friction
                                  and colour and strictly positive for mass -
                                  the domain :func:`~strands_robots.simulation.base.randomization_range_error`
                                  defines and the Newton backend shares. A
                                  scale a body cannot have is refused, not
                                  installed: a negative mass falls upward and a
                                  zero mass ignores gravity.
            seed:                 Optional seed for a reproducible stream; a
                                  non-negative integer, or None for fresh
                                  entropy.
            **kwargs:             Declared only to match the ``**kwargs``-typed
                                  ``SimEngine.randomize`` signature; nothing is
                                  forwarded, so any keyword arriving here is
                                  rejected with an error naming the valid
                                  parameters. A misspelled axis (e.g.
                                  ``randomize_position``) must not report
                                  success while leaving that axis untouched.

        Returns:
            Status dict listing the axes applied, or an error dict when a
            keyword is unknown, a range/noise/seed value cannot be applied, no
            world exists, or a policy is running.
        """
        if err := unknown_kwargs_error("randomize", kwargs, _RANDOMIZE_PARAMS):
            return err
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # domain randomization mutates model arrays; a running policy racing with it is UB
        if err := self._require_no_running_policy("randomize"):
            return err
        # Every numeric knob below is written straight into the live model (or
        # into ``data.qpos``), so a value with no valid sampling interval either
        # raises deep inside the mutation loop - past the tool envelope - or
        # succeeds and leaves an unphysical world reporting success. Reject at
        # the call instead, with the same accepted domain the Newton backend
        # already enforces for the three ranges it shares.
        for label, rng_range, allow_zero in (
            # A zero MASS multiplier is not a lighter body, it is a massless one
            # that ignores gravity; zero friction and zero colour are both real
            # physical settings.
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
                base = self._dr_baseline(model)
                for i in range(model.nlight):
                    model.light_pos[i] = base["light_pos"][i] + rng.uniform(-0.5, 0.5, size=3)
                    model.light_diffuse[i] = rng.uniform(0.3, 1.0, size=3)
                changes.append(f"Lighting: {model.nlight} lights randomized")

            if randomize_physics:
                base = self._dr_baseline(model)
                friction_scales = {}
                for i in range(model.ngeom):
                    gn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                    f = float(rng.uniform(*friction_range))
                    model.geom_friction[i, 0] = base["geom_friction"][i, 0] * f
                    friction_scales[gn] = f
                mass_scales = {}
                for i in range(model.nbody):
                    if model.body_mass[i] > 0:
                        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                        s = float(rng.uniform(*mass_range))
                        model.body_mass[i] = base["body_mass"][i] * s
                        # Inertia tracks mass for fixed geometry: scaling a
                        # rigid body's mass by ``s`` at constant shape (a uniform
                        # density change) scales its inertia tensor by the same
                        # ``s`` (I = integral of r^2 dm). Scaling mass alone
                        # leaves a physically inconsistent body - heavy in
                        # translation but with the light body's rotational
                        # resistance - which silently corrupts the dynamics the
                        # randomization is meant to perturb. Match the Newton
                        # backend, which scales both.
                        model.body_inertia[i] = base["body_inertia"][i] * s
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
                # body_mass / body_inertia are compile-time inputs: MuJoCo
                # derives body_subtreemass, dof_M0, dof_invweight0 and
                # body_invweight0 from them, and the constraint solver reads
                # those derived arrays. Without this refresh a mass-scaled body
                # is solved with the pre-scale impedance and sinks into the
                # floor. mj_setConst also runs the forward pass we need below.
                if randomize_physics:
                    mj.mj_setConst(model, data)
                mj.mj_forward(model, data)
                # Every write above went to ``model.*`` only, and every scene
                # mutation recompiles from ``_backend_state["spec"]`` - so one
                # add_object after a randomize silently reverted the whole
                # sample. Measured: post-DR mass 0.1307 / friction 1.2535 /
                # rgba 0.561, then post-add_object 0.1000 / 1.0000 / 0.500, i.e.
                # back to the un-randomised model. A rollout that adds a
                # distractor mid-episode trained on the WRONG domain.
                self._sync_randomization_to_spec(model, randomize_colors, randomize_physics)

        return {
            "status": "success",
            "content": [{"text": "Domain Randomization applied:\n" + "\n".join(changes)}],
        }

    def set_obs_noise(
        self,
        joint_pos_std: float = 0.0,
        joint_vel_std: float = 0.0,
        camera_jitter_px: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Configure additive Gaussian sensor noise on observations.

        Models real-encoder / real-camera measurement noise so policies trained
        on MuJoCo data do not assume noise-free sensing. Once set, the noise is
        applied on every :meth:`get_observation` / :meth:`get_robot_state` and
        every rendered camera frame (:meth:`render` and the camera frames in
        ``get_observation``) until reconfigured. Pass all-zero std to disable -
        with every std zero the noise path is an exact no-op, so leaving this
        unconfigured (the default) leaves every observation and render
        byte-for-byte unchanged. Mirrors :meth:`NewtonSimEngine.set_obs_noise`
        so an identical call behaves the same on both backends.

        Args:
            joint_pos_std: Std (radians) of Gaussian noise added to joint
                positions in ``get_observation`` and ``get_robot_state``.
            joint_vel_std: Std (rad/s) of Gaussian noise added to per-joint
                velocities - the ``<joint>.vel`` entries in ``get_observation``
                and the ``velocity`` field in ``get_robot_state``.
            camera_jitter_px: Max integer pixel shift applied to rendered
                frames (uniform in ``[-px, px]`` per axis).
            seed: Optional seed for a reproducible noise stream; a non-negative
                integer, or None for fresh entropy. Validated here rather than
                where the stream is first drawn, so an unusable seed is reported
                by the call that supplied it.
            **kwargs: Declared only to match the ``**kwargs``-typed
                ``SimEngine.set_obs_noise`` signature; nothing is forwarded, so
                any keyword arriving here is rejected with an error naming the
                valid parameters rather than reporting an all-zero (no-op) noise
                configuration as success.

        Returns:
            Status dict echoing the configured noise, or an error dict when a
            keyword is unknown or a value is negative or non-finite.
        """
        if err := unknown_kwargs_error("set_obs_noise", kwargs, _OBS_NOISE_PARAMS):
            return err
        for label, value in (
            ("joint_pos_std", joint_pos_std),
            ("joint_vel_std", joint_vel_std),
            ("camera_jitter_px", camera_jitter_px),
        ):
            if msg := finite_non_negative_error(value, label, "set_obs_noise"):
                return {"status": "error", "content": [{"text": msg}]}
        # The seed only reaches ``default_rng`` here; an unusable one would
        # otherwise raise on the first observation drawn, long after this call
        # reported the noise configured.
        if msg := randomization_seed_error(seed, "set_obs_noise"):
            return {"status": "error", "content": [{"text": msg}]}

        with self._lock:
            self._obs_noise = {
                "joint_pos_std": float(joint_pos_std),
                "joint_vel_std": float(joint_vel_std),
                "camera_jitter_px": float(camera_jitter_px),
            }
            # Retain the seed so ``reset()`` can restart the SAME stream at each
            # episode boundary. Without it a seeded eval was not reproducible
            # per episode: the generator advanced continuously across resets, so
            # episode 2 of a run drew different noise than episode 2 of a re-run
            # whose episode 1 consumed a different number of observations.
            self._obs_noise_seed = seed
            self._obs_noise_rng = np.random.default_rng(seed)
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Sensor noise: joint_pos_std={joint_pos_std}, "
                        f"joint_vel_std={joint_vel_std}, camera_jitter_px={camera_jitter_px}"
                    )
                }
            ],
        }

    def _apply_obs_noise(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return ``obs`` with configured sensor noise applied.

        ``get_observation`` returns a heterogeneous dict: scalar joint positions
        keyed by joint name, scalar per-joint velocities keyed ``<joint>.vel``,
        camera frames as ``(H, W, 3)`` uint8 arrays, and (for floating-base
        robots) ``base_quat`` / ``base_ang_vel`` list values. Position noise
        (``joint_pos_std``) applies to the position scalars, velocity noise
        (``joint_vel_std``) to the ``.vel`` scalars, and camera jitter
        (``camera_jitter_px``) to the image arrays. The floating-base list
        signals are left untouched (a quaternion would need renormalization;
        out of scope for additive scalar noise). A no-op returning the input
        unchanged when no noise is configured.
        """
        cfg = self._obs_noise or {}
        rng = self._obs_noise_rng
        if rng is None or not cfg:
            return obs
        pos_std = cfg.get("joint_pos_std", 0.0)
        vel_std = cfg.get("joint_vel_std", 0.0)
        px = cfg.get("camera_jitter_px", 0.0)
        if pos_std <= 0 and vel_std <= 0 and px <= 0:
            return obs
        out: dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                out[key] = self._maybe_jitter_frame(value) if px > 0 else value
            elif isinstance(value, float):
                if key.endswith(".vel"):
                    out[key] = value + (float(rng.normal(0.0, vel_std)) if vel_std > 0 else 0.0)
                else:
                    out[key] = value + (float(rng.normal(0.0, pos_std)) if pos_std > 0 else 0.0)
            else:
                out[key] = value
        return out

    def _apply_state_noise(self, state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Return ``get_robot_state`` output with position + velocity noise.

        Entries are ``{joint: {"position": p, "velocity": v}}``. Position noise
        uses ``joint_pos_std`` and velocity noise uses ``joint_vel_std`` from
        :meth:`set_obs_noise`. A no-op when neither std is positive.
        """
        cfg = self._obs_noise or {}
        pos_std = cfg.get("joint_pos_std", 0.0)
        vel_std = cfg.get("joint_vel_std", 0.0)
        rng = self._obs_noise_rng
        if rng is None or (pos_std <= 0 and vel_std <= 0) or not state:
            return state
        out: dict[str, dict[str, float]] = {}
        for jname, vals in state.items():
            pos = vals["position"] + (float(rng.normal(0.0, pos_std)) if pos_std > 0 else 0.0)
            vel = vals["velocity"] + (float(rng.normal(0.0, vel_std)) if vel_std > 0 else 0.0)
            out[jname] = {"position": pos, "velocity": vel}
        return out

    def _maybe_jitter_frame(self, frame: "np.ndarray") -> "np.ndarray":
        """Return ``frame`` shifted by a random integer pixel offset.

        Applies ``camera_jitter_px`` configured via :meth:`set_obs_noise` by
        rolling the image along both axes. A no-op when jitter is disabled.
        """
        px = (self._obs_noise or {}).get("camera_jitter_px", 0.0)
        rng = self._obs_noise_rng
        if px <= 0 or rng is None or frame.ndim < 2:
            return frame
        max_shift = int(px)
        if max_shift < 1:
            return frame
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        return np.roll(frame, shift=(dy, dx), axis=(0, 1))
