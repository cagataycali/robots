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
from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, _ensure_mujoco, mj_name_to_id
from strands_robots.simulation.mujoco.scene_ops import _get_spec
from strands_robots.simulation.obs_noise import ObservationNoiseMixin
from strands_robots.utils import boolean_flag_error

if TYPE_CHECKING:
    from strands_robots.simulation.models import SimWorld

logger = logging.getLogger(__name__)

# Parameter names ``randomize`` actually honors. It declares
# ``**kwargs`` to match the ``**kwargs``-typed SimEngine base signature, but
# never forwards it anywhere - so anything landing there is a caller mistake
# and is rejected instead of dropped (test_domain_randomization_rejects_unknown_params
# pins this tuple to the live signature).
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

#: Half-width, in metres, of the uniform offset ``randomize(randomize_lighting=True)``
#: applies to each light. Named once so the bound the docstring advertises and the
#: bound the sampler draws cannot drift apart.
_LIGHT_POS_JITTER_M: float = 0.5


def _authored_light_positions(world: "SimWorld", model: Any) -> "np.ndarray | None":
    """Return every light's authored position from the scene spec.

    The lighting axis needs a fixed reference to sample around, and the scene
    spec is the one the rest of the backend already treats as authoritative:
    it is what a recompile regenerates ``model.light_pos`` from, so it holds the
    authored pose no matter how many times the axis has run. ``MjSpec.lights``
    is ordered as the compiled ``model.light_pos`` rows are, including lights a
    robot spec contributes through ``spec.attach``.

    Args:
        world: The live world, whose ``_backend_state`` carries the spec.
        model: The compiled model, read only for its light count.

    Returns:
        An ``(nlight, 3)`` float array of authored positions, or ``None`` when
        no spec is tracked or it disagrees with the compiled light count - the
        caller refuses the axis rather than falling back to the live positions,
        which is what compounds.
    """
    spec = _get_spec(world)
    if spec is None:
        return None
    lights = list(spec.lights)
    if len(lights) != int(model.nlight):
        return None
    return np.array([light.pos for light in lights], dtype=np.float64).reshape(len(lights), 3)


class RandomizationMixin(ObservationNoiseMixin):
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
          - ``randomize_lighting=True`` - light pos jittered ±0.5m about its
            authored position, diffuse resampled.
          - ``randomize_physics=False`` - friction/mass left untouched unless asked.
          - ``randomize_positions=False`` - object qpos left untouched unless asked.

        Each flag selects a posture, so each is checked on the shared
        boolean-flag domain
        (:func:`~strands_robots.utils.boolean_flag_error`) before anything is
        written: an axis is not turned off by ``"false"``, ``"no"``, ``"off"``
        or ``"0"``, every one of which is a truthy non-empty string that turned
        the axis ON instead.

        "No flags" means "nothing is randomized" - the call is a no-op. This
        matches the LLM ergonomics principle: explicit is better than implicit.
        Randomization IS destructive, and it survives a :meth:`reset` but not a
        scene mutation. Every axis writes the compiled ``model``: the colour,
        friction and mass axes write their arrays, and the position axis writes
        ``model.qpos0`` (the pose a reset restores) alongside the live
        ``data.qpos``. That uniformity is what makes the axis usable from a
        rollout entry point at all -- ``run_policy`` and ``eval_policy`` reset
        before an episode's first step, so an axis a reset undoes would never
        reach a rollout.

        The compiled model is derived state, though, and every scene mutation
        rebuilds it from the scene spec, which carries the authored values --
        deliberately, because the lighting and position axes measure their
        bounded offsets from that reference. So :meth:`add_object`,
        :meth:`remove_object`, :meth:`add_camera`, :meth:`remove_camera`,
        :meth:`add_robot`, :meth:`remove_robot` and :meth:`patch_scene_mjcf`
        each restore the authored scene, and there is no ``recompile`` action to
        reach for -- any of those is the undo. Randomizing *before* one of them
        is a no-op this call cannot report: both calls return
        ``status="success"`` and the policy's first observation sees an
        unrandomized scene. Randomize after the episode's scene is built.

        Args:
            randomize_colors:     Re-sample every non-ground geom's RGB (and
                                  its material colour, which overrides geom RGB
                                  in the renderer).
            randomize_lighting:   Jitter light positions + diffuse colour. Each
                                  light's offset is drawn inside ±0.5m of its
                                  AUTHORED position -- the pose the scene spec
                                  declares, which is what a recompile restores --
                                  so repeated calls draw independent offsets
                                  inside that bound instead of compounding into
                                  a random walk, exactly as ``position_noise``
                                  does below. Measuring from the live position
                                  instead walks the light out of the scene: 50
                                  per-episode calls reach 4.7m from a light
                                  authored 3.5m up, breaching the bound by the
                                  third call. Refused (with nothing applied) on
                                  a world that tracks no spec agreeing with its
                                  compiled light count, since the bound cannot
                                  be honoured without that reference.
            randomize_physics:    Scale geom friction and body mass (body
                                  inertia is scaled by the same factor as the
                                  mass so each randomized body stays physically
                                  consistent).
            randomize_positions:  Add uniform noise to dynamic-object xyz.
            position_noise:       Max ± xyz offset in meters when randomising
                                  positions. A finite non-negative number: a
                                  NaN half-width writes NaN into ``qpos`` and
                                  poisons every later step, a negative one
                                  inverts the sampling bounds. The offset is
                                  measured from each dynamic object's commanded
                                  pose (what ``add_object`` / ``move_object``
                                  placed it at), so repeated calls draw
                                  independent offsets inside this bound instead
                                  of compounding into a random walk, and it
                                  becomes both the live pose and the pose a
                                  reset restores. Static objects have no pose
                                  DOF and are skipped; the result text reports
                                  how many objects were actually perturbed.
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
            keyword is unknown, an axis flag is not a boolean, a range/noise/seed
            value cannot be applied, the
            lighting axis cannot resolve the authored light positions it jitters
            around, no world exists, or a policy is running.
        """
        if err := unknown_kwargs_error("randomize", kwargs, _RANDOMIZE_PARAMS):
            return err
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # domain randomization mutates model arrays; a running policy racing with it is UB
        if err := self._require_no_running_policy("randomize"):
            return err
        # The four flags select which axes run, so each is checked on the shared
        # boolean-flag domain rather than read by truthiness. Randomization is
        # destructive - the physics and position axes default to OFF precisely
        # because undoing them means recompiling the scene - and every spelling
        # an operator reaches for to turn an axis off ("false", "no", "off",
        # "0") is a non-empty string, so it turned that axis ON and the call
        # reported the axis applied. The ``randomize_lighting`` refusal further
        # down branches on its own flag, so a truthy non-boolean also made it
        # describe the axis the caller had asked to skip. A misspelled axis
        # NAME is already refused above; this is the same guarantee for the
        # value that name carries.
        for flag_param, flag_value in (
            ("randomize_colors", randomize_colors),
            ("randomize_lighting", randomize_lighting),
            ("randomize_physics", randomize_physics),
            ("randomize_positions", randomize_positions),
        ):
            if msg := boolean_flag_error(flag_value, flag_param, "randomize"):
                return {"status": "error", "content": [{"text": msg}]}
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
        # Resolved here, before the first mutation, so a scene whose authored
        # light poses cannot be read is refused with nothing applied. Resolving
        # it inside the lock would leave the colour axis (which runs first)
        # already written by the time the lighting axis gives up.
        light_base: np.ndarray | None = None
        if randomize_lighting:
            light_base = _authored_light_positions(self._world, model)
            if light_base is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "randomize_lighting needs each light's authored position to jitter around, "
                                "and this world tracks no scene spec that agrees with its "
                                f"{int(model.nlight)} compiled light(s). Without that reference the offset "
                                "would be measured from wherever the previous call left the light, which "
                                "compounds instead of staying inside the documented bound. Re-create the "
                                "scene (create_world or load_scene), or randomize the other axes with "
                                "randomize_lighting=False."
                            )
                        }
                    ],
                }
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

            # Non-None exactly when ``randomize_lighting`` was requested and the
            # authored reference resolved; the validation above already returned
            # for every other case, so this is the lighting axis's guard.
            if light_base is not None:
                # Offset each light from its AUTHORED position, not from
                # wherever the previous call left it. ``+=`` on the live array
                # makes every call start from the last one's result, so the
                # displacement is a random walk rather than the bounded jitter
                # this axis documents: 50 per-episode calls reach 4.7 m from a
                # light authored 3.5 m up, and the bound is already breached by
                # the third call. ``light_diffuse`` is assigned, so the colour
                # half was always bounded. Same reference discipline as the
                # position axis below, which measures from each object's
                # commanded pose for exactly this reason, and as the Newton
                # backend, which jitters around a constant base direction.
                for i in range(model.nlight):
                    model.light_pos[i] = light_base[i] + rng.uniform(-_LIGHT_POS_JITTER_M, _LIGHT_POS_JITTER_M, size=3)
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
                # Two writes per object, because a start pose lives in two
                # places and the axis is only useful if it reaches both:
                #   * ``data.qpos`` -- the live pose, and
                #   * ``model.qpos0`` -- the pose ``mj_resetData`` restores.
                # Every rollout entry point resets before an episode's first
                # step (``PolicyRunner.evaluate`` resets at the top of each
                # episode), so a qpos-only write is undone before the policy
                # ever observes it, while the three model-array axes above
                # persist. Writing both makes this axis behave like colour,
                # friction and mass: it survives a reset and is undone by a
                # recompile.
                #
                # The offset is measured from the object's COMMANDED pose --
                # what ``add_object`` / ``move_object`` last placed it at, which
                # is what the registry holds -- not from the live pose. The
                # registry pose is a fixed reference, so each call draws an
                # independent offset bounded by ``position_noise``; measuring
                # from the live pose instead compounds every call into a random
                # walk (50 episodes at a 0.03 m half-width reach 0.12 m and put
                # a table-top object under the floor).
                n_moved = 0
                for obj_name, obj in self._world.objects.items():
                    if obj.is_static:
                        continue
                    jnt_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, f"{obj_name}_joint")
                    if jnt_id < 0:
                        continue
                    qpos_addr = model.jnt_qposadr[jnt_id]
                    noise = rng.uniform(-position_noise, position_noise, size=3)
                    start = np.asarray(obj.position, dtype=np.float64) + noise
                    data.qpos[qpos_addr : qpos_addr + 3] = start
                    model.qpos0[qpos_addr : qpos_addr + 3] = start
                    n_moved += 1
                # Report the count actually perturbed, as the colour axis does:
                # a scene whose objects are all static perturbs nothing, and
                # naming the axis without a count reads as work done.
                changes.append(f"Positions: {n_moved} dynamic objects perturbed by +/-{position_noise}m")

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
