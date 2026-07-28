"""``LiberoAdapter`` - :class:`BenchmarkProtocol` driven by a LIBERO BDDL file.

LIBERO is a suite of ~130 tabletop manipulation tasks built around a Franka
Panda. The adapter compiles the BDDL ``:goal`` into a sparse success
predicate via :mod:`strands_robots.benchmarks.libero.bddl_parser` and drives
the scene through the :class:`BenchmarkProtocol` lifecycle: scene load +
setup in :meth:`~LiberoAdapter.on_episode_start`, sparse zero-reward steps
in :meth:`~LiberoAdapter.on_step` (LIBERO has no dense reward), and goal
evaluation in :meth:`~LiberoAdapter.is_success`.

**Panda-only by design.** LIBERO's scene MJCFs and BDDL predicates reference
Panda body names (``robot0_gripper_*``); retargeting is out of scope.
Subclass and override :attr:`supported_robots` + :attr:`default_robot` if
you know what you're doing.

The adapter does NOT require the ``libero`` Python package - only a BDDL
string / file and (optionally) an MJCF scene path.
:func:`strands_robots.benchmarks.libero.suite.load_libero_suite` is the
helper that pulls in the upstream package to discover task files.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from strands_robots.benchmarks.libero.bddl_parser import (
    BDDLParseError,
    BDDLProblem,
    Node,
    compile_goal,
    parse_bddl,
    parse_bddl_file,
)
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo
from strands_robots.simulation.models import SimCamera, SimRobot
from strands_robots.utils import get_base_dir, require_optional

if TYPE_CHECKING:
    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)


class LiberoAdapter(BenchmarkProtocol):
    """Panda-only :class:`BenchmarkProtocol` driven by a parsed LIBERO BDDL task.

    Construct with a BDDL file path (``from_file``) or raw BDDL text
    (``from_text``) - direct ``__init__`` is for advanced use when you
    already have a :class:`BDDLProblem`.

    Example::

        from strands_robots.benchmarks.libero import LiberoAdapter

        adapter = LiberoAdapter.from_file(
            "libero/tasks/libero_spatial/pick_up_the_red_cube.bddl",
            scene_path="libero/assets/scenes/libero_spatial_scene.xml",
        )
        sim.register_benchmark("pick-red-cube", adapter)
        sim.evaluate_benchmark("pick-red-cube", policy_provider="mock",
                               n_episodes=10, seed=42)

    Attributes:
        max_steps: Default 720, matching NVIDIA's upstream
            ``MultiStepConfig.max_episode_steps`` for GR00T-LIBERO eval
            (LIBERO's own 300-step convention is too short for
            libero_10's long-horizon tasks). Override via ``max_steps=``.
        problem: The parsed :class:`BDDLProblem`. Stored for introspection
            (agents may read ``problem.language`` as the instruction).
    """

    max_steps: int = 720
    supported_robots_list: list[str] = ["panda"]
    default_robot_name: str = "panda"

    #: Cameras the ``libero_panda`` ``Gr00tDataConfig`` expects, keyed by the
    #: bare video keys (``image`` / ``wrist_image``). Poses are world-fixed
    #: approximations used only as a FALLBACK: scenes that declare the
    #: canonical RoboSuite cameras get them renamed to these keys via
    #: :attr:`_scene_camera_aliases`, so these static entries never install.
    #: Override by passing ``cameras={...}`` to the constructor.
    LIBERO_CAMERAS: dict[str, dict[str, Any]] = {
        "image": {
            "position": [1.0, 0.0, 1.5],
            "target": [0.0, 0.0, 0.85],
            "fov": 60.0,
            "width": 256,
            "height": 256,
        },
        "wrist_image": {
            "position": [0.0, 0.0, 1.4],
            "target": [0.0, 0.0, 0.85],
            "fov": 60.0,
            "width": 256,
            "height": 256,
        },
    }

    def __init__(
        self,
        problem: BDDLProblem,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str | None = None,
        eef_state_site_name: str | None = None,
        gripper_joint_name: str | None = None,
        state_gripper_joint_names: list[str] | None = None,
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
        scene_robot_prefix: str = "robot0_",
        scene_gripper_prefix: str = "gripper0_",
        init_states: np.ndarray | None = None,
        strict_action_controller: bool = True,
        bddl_source: str | None = None,
        bddl_path: str | None = None,
    ):
        """Construct from a pre-parsed :class:`BDDLProblem`.

        Args:
            problem: Parsed BDDL problem with a non-``None`` ``goal``.
            scene_path: Optional MJCF to ``sim.load_scene()`` on each
                episode start. ``None`` triggers ``auto_generate_scene``
                if enabled.
            max_steps: Override the class-level default.
            init_jitter: Per-episode +/- jitter (metres) applied to xy of
                every object referenced by ``(:init ...)`` clauses.
                Default ``0.0`` matches LIBERO's deterministic-reset
                convention (the checkpoint expects the exact training
                init poses); a positive value evaluates generalization at
                the cost of lower nominal success rates.
            install_cameras: When ``True`` (default), install
                :attr:`LIBERO_CAMERAS` (or ``cameras``) on episode start.
                No-ops for cameras the scene already declares.
            cameras: Override / extend :attr:`LIBERO_CAMERAS`; each value
                is forwarded as ``**kwargs`` to ``Simulation.add_camera``.
                An empty dict disables camera installation.
            eef_body_name: MuJoCo body whose pose is the FALLBACK source
                for ``state.x/y/z/roll/pitch/yaw`` when the site doesn't
                resolve. ``None`` (default) auto-resolves the canonical
                RoboSuite EEF body (``<prefix>right_hand`` ->
                ``<prefix>hand`` -> bare ``hand``) at episode start; an
                explicit string disables auto-resolution (legacy
                bare-Panda default: ``"hand"``).
            eef_state_site_name: MuJoCo *site* whose pose is read for
                ``state.x/y/z/roll/pitch/yaw``. ``None`` (default)
                auto-resolves to ``"<scene_gripper_prefix>grip_site"``
                (``"gripper0_grip_site"`` for LIBERO scenes) - the
                gripper-tip site RoboSuite's OSC reads for
                ``robot0_eef_pos``/``robot0_eef_quat``. The site sits
                ~9.7 cm below the wrist body and rotated 180 deg around
                X; reading the body instead feeds the policy
                out-of-distribution state. Body is the fallback when the
                site doesn't exist.
            gripper_joint_name: Joint whose ``qpos`` is read for
                ``state.gripper``. ``None`` (default) auto-resolves via
                ``<scene_gripper_prefix>finger_joint1`` ->
                ``<scene_robot_prefix>finger_joint1`` -> bare
                ``finger_joint1``; an explicit string disables
                auto-resolution (legacy default: ``"finger_joint1"``).
            inject_eef_state: When ``True`` (default),
                :meth:`augment_observation` injects the
                ``x/y/z/roll/pitch/yaw/gripper`` keys the ``libero_panda``
                ``Gr00tDataConfig`` expects. Set ``False`` when the sim
                already exposes them.
            auto_generate_scene: When ``True`` (default) and ``scene_path``
                is ``None``, build the scene MJCF from the BDDL via the
                upstream ``libero`` package, cached on disk. ``False``
                runs against a bare Panda instead.
            scene_cache_dir: Generated-scene cache location. Defaults to
                ``$STRANDS_BASE_DIR/scene_cache/libero/``.
            scene_camera_aliases: MJCF-camera-name -> policy-observation-key
                rename map applied to generated scenes. Default maps
                ``agentview`` -> ``image`` and ``robot0_eye_in_hand`` (with
                and without the ``_image`` suffix, covering both upstream
                naming conventions) -> ``wrist_image``. An empty dict
                disables renaming (the static :attr:`LIBERO_CAMERAS`
                fallbacks then fire, making the wrist channel a static
                view). The map is hashed into the scene-cache key so alias
                changes invalidate stale caches.
            apply_scene_keyframe: When ``True`` (default) and a scene was
                loaded, restore qpos/qvel to the canonical home state each
                episode - via the MJCF ``<keyframe>`` when one exists,
                otherwise first-episode snapshot-and-restore (see
                :meth:`_apply_canonical_state`). ``False`` disables both.
            scene_keyframe_index: Which ``<keyframe>`` to apply on the
                keyframe branch. Default ``0`` (LIBERO convention);
                ignored on the snapshot fallback.
            scene_robot_prefix: Name prefix identifying the scene-supplied
                Panda for pre-registration in ``world.robots``. Default
                ``"robot0_"`` (RoboSuite/LIBERO convention). No-ops when
                nothing matches; super() then falls back to ``add_robot``.
            scene_gripper_prefix: Name prefix identifying the
                scene-supplied gripper. Default ``"gripper0_"`` -
                RoboSuite grippers get their own namespace, separate
                from the arm's.
            init_states: Optional ``ndarray[(N, 1+nq+nv)]`` of LIBERO's
                canonical init states (row layout ``[time, qpos, qvel]``).
                One row is applied per episode, RNG-seeded; row width
                MUST equal ``1 + model.nq + model.nv`` or the apply
                raises rather than silently slicing. Populate via
                ``load_libero_suite`` or pass explicitly; without it a
                keyframe-less scene starts at ``qpos=0`` instead of the
                canonical "ready" pose the policy expects.
            strict_action_controller: When ``True`` (default), an OSC
                controller install failure raises (so the eval returns a
                structured error) instead of silently dropping every
                action; ``False`` restores best-effort logging, with the
                failure recorded on ``_action_controller_error``.
                Dependency-clash failures (e.g. the
                ``numba``/``coverage>=7`` import clash) are ALWAYS
                surfaced with a remediation hint regardless of the flag.
            bddl_source: Original BDDL text so the scene generator can
                hand ``libero`` a file. Set by :meth:`from_text`.
            bddl_path: Original BDDL file path; lets the generator skip
                the temp-file step. Set by :meth:`from_file`.

        Raises:
            ValueError: If ``problem.goal`` is ``None`` or
                ``init_jitter`` is negative.
        """
        if problem.goal is None:
            raise ValueError(f"LiberoAdapter: BDDL problem {problem.name!r} has no (:goal ...) block")
        self.problem = problem
        self.scene_path = scene_path
        self._init_jitter = float(init_jitter)
        if self._init_jitter < 0:
            raise ValueError(f"init_jitter must be >= 0, got {init_jitter}")
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self._install_cameras = bool(install_cameras)
        # Snapshot the camera config at construction time so subsequent
        # mutations to LIBERO_CAMERAS don't leak across instances.
        self._cameras: dict[str, dict[str, Any]] = (
            {k: dict(v) for k, v in cameras.items()}
            if cameras is not None
            else {k: dict(v) for k, v in self.LIBERO_CAMERAS.items()}
        )
        self._eef_body_name: str = str(eef_body_name) if eef_body_name is not None else "hand"
        self._gripper_joint_name: str = str(gripper_joint_name) if gripper_joint_name is not None else "finger_joint1"
        # EEF state site: state.x/y/z/roll/pitch/yaw must come from the
        # gripper-tip site (RoboSuite OSC convention), not the wrist body.
        # Auto-defaults to "<scene_gripper_prefix>grip_site"; a user
        # override disables auto-derivation, and an empty-string sentinel
        # forces the body fallback.
        self._user_eef_state_site_name: str | None = (
            str(eef_state_site_name) if eef_state_site_name is not None else None
        )
        # Track whether the user explicitly supplied either name so the
        # auto-resolver in _register_default_robot only overrides the
        # constructor default (None); explicit values are never touched.
        self._user_eef_body_name: str | None = str(eef_body_name) if eef_body_name is not None else None
        self._user_gripper_joint_name: str | None = str(gripper_joint_name) if gripper_joint_name is not None else None
        # State-side gripper joints: LIBERO trains state.gripper on the
        # 2-vector [finger1.qpos, finger2.qpos] whose elements have
        # opposite signs. Default auto-derives
        # ["<gripper_prefix>finger_joint1", "<gripper_prefix>finger_joint2"];
        # a user-supplied list is used as-is.
        self._user_state_gripper_joint_names: list[str] | None = (
            [str(n) for n in state_gripper_joint_names] if state_gripper_joint_names is not None else None
        )
        self._inject_eef_state = bool(inject_eef_state)
        self._auto_generate_scene = bool(auto_generate_scene)
        self._scene_cache_dir = scene_cache_dir
        # Default alias map: canonical RoboSuite camera names -> the bare
        # keys the libero_panda Gr00tDataConfig expects. Both eye_in_hand
        # spellings are mapped to cover old and new upstream conventions.
        self._scene_camera_aliases: dict[str, str] = (
            dict(scene_camera_aliases)
            if scene_camera_aliases is not None
            else {
                "agentview": "image",
                "robot0_eye_in_hand": "wrist_image",
                "robot0_eye_in_hand_image": "wrist_image",
            }
        )
        self._apply_canonical_state_enabled = bool(apply_scene_keyframe)
        self._scene_keyframe_index = int(scene_keyframe_index)
        self._scene_robot_prefix = str(scene_robot_prefix)
        self._scene_gripper_prefix = str(scene_gripper_prefix)
        # Canonical init states; when non-None this takes precedence over
        # the keyframe / snapshot branches in _apply_canonical_state.
        # Stored 2D (N, 1+nq+nv); per-episode selection is RNG-seeded.
        if init_states is not None:
            init_states_array = np.asarray(init_states, dtype=np.float64)
            if init_states_array.ndim == 1:
                # Single state - promote to 2D for uniform indexing.
                init_states_array = init_states_array[np.newaxis, :]
            if init_states_array.ndim != 2:
                raise ValueError(f"init_states must be 1D or 2D ndarray, got ndim={init_states_array.ndim}")
            self._init_states: np.ndarray | None = init_states_array
        else:
            self._init_states = None
        # Episode counter: episode 0 is pinned to init_states[0]
        # (matching prewarm's apply, so the recorder's first frame and
        # the policy's first observation agree); episodes 1+ RNG-sample.
        self._episode_count: int = 0
        # Snapshot-and-restore fallback for procedurally-generated MJCFs that
        # don't ship a <keyframe>. Captured on the first episode after super() +
        # _install_libero_cameras have run; replayed on every subsequent
        # episode so qpos/qvel land on the same canonical state every time.
        self._canonical_qpos: np.ndarray | None = None
        self._canonical_qvel: np.ndarray | None = None
        self._bddl_source = bddl_source
        self._bddl_path = bddl_path
        self._success_fn: Callable[[SimEngine], bool] = compile_goal(problem.goal)

        # Surface OSC controller install failures: strict raises,
        # non-strict records on _action_controller_error (reset at the
        # start of each install attempt).
        self._strict_action_controller = bool(strict_action_controller)
        self._action_controller_error: str | None = None

        # Diagnostic gate: STRANDS_LIBERO_STATE_LOG=1 emits one INFO line
        # per augment_observation call for the first
        # STRANDS_LIBERO_STATE_LOG_MAX (default 50) calls per episode.
        # Pairs with STRANDS_LIBERO_ACTION_LOG on the action side.
        self._state_log_enabled = os.environ.get("STRANDS_LIBERO_STATE_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            self._state_log_max = int(os.environ.get("STRANDS_LIBERO_STATE_LOG_MAX", "50"))
        except ValueError:
            logger.warning(
                "STRANDS_LIBERO_STATE_LOG_MAX=%r is not an integer; defaulting to 50",
                os.environ.get("STRANDS_LIBERO_STATE_LOG_MAX"),
            )
            self._state_log_max = 50
        self._state_log_step: int = 0

    # Construction helpers

    @classmethod
    def from_file(
        cls,
        bddl_path: str | Path,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str | None = None,
        eef_state_site_name: str | None = None,
        gripper_joint_name: str | None = None,
        state_gripper_joint_names: list[str] | None = None,
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
        scene_robot_prefix: str = "robot0_",
        scene_gripper_prefix: str = "gripper0_",
        init_states: np.ndarray | None = None,
        strict_action_controller: bool = True,
    ) -> LiberoAdapter:
        """Parse a ``.bddl`` file from disk and build an adapter.

        Raises :class:`FileNotFoundError` / :class:`BDDLParseError` on bad
        input - callers that want structured error dicts should catch and
        convert.
        """
        problem = parse_bddl_file(bddl_path)
        return cls(
            problem,
            scene_path=scene_path,
            max_steps=max_steps,
            init_jitter=init_jitter,
            install_cameras=install_cameras,
            cameras=cameras,
            eef_body_name=eef_body_name,
            eef_state_site_name=eef_state_site_name,
            gripper_joint_name=gripper_joint_name,
            state_gripper_joint_names=state_gripper_joint_names,
            inject_eef_state=inject_eef_state,
            auto_generate_scene=auto_generate_scene,
            scene_cache_dir=scene_cache_dir,
            scene_camera_aliases=scene_camera_aliases,
            apply_scene_keyframe=apply_scene_keyframe,
            scene_keyframe_index=scene_keyframe_index,
            scene_robot_prefix=scene_robot_prefix,
            scene_gripper_prefix=scene_gripper_prefix,
            init_states=init_states,
            strict_action_controller=strict_action_controller,
            bddl_path=str(bddl_path),
        )

    @classmethod
    def from_text(
        cls,
        bddl_text: str,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str | None = None,
        eef_state_site_name: str | None = None,
        gripper_joint_name: str | None = None,
        state_gripper_joint_names: list[str] | None = None,
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
        scene_robot_prefix: str = "robot0_",
        scene_gripper_prefix: str = "gripper0_",
        init_states: np.ndarray | None = None,
        strict_action_controller: bool = True,
    ) -> LiberoAdapter:
        """Parse a BDDL string directly - useful in tests."""
        problem = parse_bddl(bddl_text)
        return cls(
            problem,
            scene_path=scene_path,
            max_steps=max_steps,
            init_jitter=init_jitter,
            install_cameras=install_cameras,
            cameras=cameras,
            eef_body_name=eef_body_name,
            eef_state_site_name=eef_state_site_name,
            gripper_joint_name=gripper_joint_name,
            state_gripper_joint_names=state_gripper_joint_names,
            inject_eef_state=inject_eef_state,
            auto_generate_scene=auto_generate_scene,
            scene_cache_dir=scene_cache_dir,
            scene_camera_aliases=scene_camera_aliases,
            apply_scene_keyframe=apply_scene_keyframe,
            scene_keyframe_index=scene_keyframe_index,
            scene_robot_prefix=scene_robot_prefix,
            scene_gripper_prefix=scene_gripper_prefix,
            init_states=init_states,
            strict_action_controller=strict_action_controller,
            bddl_source=bddl_text,
        )

    # BenchmarkProtocol interface

    @property
    def supported_robots(self) -> list[str]:
        """Registry ``data_config`` names this task accepts (LIBERO is Panda-only).

        Returns a copy of the ``supported_robots_list`` constructor argument
        (default ``["panda"]``) so callers cannot mutate the adapter's list.
        """
        return list(self.supported_robots_list)

    @property
    def default_robot(self) -> str:
        """Robot :meth:`on_episode_start` loads when the sim is empty.

        The ``default_robot_name`` constructor argument (default ``"panda"``);
        must be an element of :attr:`supported_robots`.
        """
        return self.default_robot_name

    @property
    def instruction(self) -> str:
        """Language instruction from the BDDL ``:language`` clause, or ``""``."""
        return self.problem.language or ""

    @property
    def eef_state_site_name(self) -> str:
        """Resolved MuJoCo site name to read for ``state.x/y/z/roll/pitch/yaw``.

        User-supplied ``eef_state_site_name`` when set, otherwise
        ``scene_gripper_prefix + "grip_site"`` (``"gripper0_grip_site"``
        for LIBERO scenes) - the gripper-tip site RoboSuite's
        ``OperationalSpaceController`` reads for its EEF observables,
        matching what LIBERO-trained checkpoints expect.
        """
        if self._user_eef_state_site_name is not None:
            return self._user_eef_state_site_name
        return f"{self._scene_gripper_prefix}grip_site"

    @property
    def state_gripper_joint_names(self) -> list[str]:
        """Resolved 2-element list of finger joint names for ``state.gripper``.

        User-supplied ``state_gripper_joint_names`` when set, otherwise
        ``["<gripper_prefix>finger_joint1", "<gripper_prefix>finger_joint2"]``.
        LIBERO trains ``state.gripper`` on the 2-vector
        ``[finger1.qpos, finger2.qpos]`` (opposite signs by physical
        convention); the names are returned in trained-vector order.
        """
        if self._user_state_gripper_joint_names is not None:
            return list(self._user_state_gripper_joint_names)
        return [f"{self._scene_gripper_prefix}finger_joint1", f"{self._scene_gripper_prefix}finger_joint2"]

    def on_episode_start(self, sim: SimEngine, rng: random.Random) -> None:
        """Per-episode setup: resolve/load the scene, restore canonical
        state, validate Panda, install cameras + render options + OSC
        controller, then apply jitter.

        Ordering constraints:

        1. Scene resolution (auto-generate + disk cache when
           ``scene_path`` is unset).
        2. ``load_scene`` - so the base compatibility check sees the
           scene's Panda instead of loading ``default_robot``.
        3. Canonical-state apply IMMEDIATELY after ``load_scene`` +
           robot pre-register: MuJoCo initialises qpos from ``qpos0``
           and ignores MJCF ``<keyframe>`` blocks, so
           :meth:`_apply_canonical_state` must restore (or snapshot)
           the canonical pose before anything else can drift qpos.
        4. ``super().on_episode_start`` - base compat check (a spec
           recompile here preserves qpos for existing joints).
        5. ``_install_libero_cameras`` - model-side detection avoids
           recompiling on top of the just-restored state.
        6. ``_install_render_options`` - LIBERO-canonical ``mjvOption``
           (skipped on bare-Panda fallback).
        7. ``_apply_init_jitter`` - RNG-seeded jitter on top of
           canonical state.

        Ends with one zero-action settle step (when the OSC controller
        installed) so the first observation is at "init_state + 1
        control step", matching upstream LIBERO's reset semantics.
        """
        if self.scene_path is None and self._auto_generate_scene:
            try:
                self.ensure_scene()
            except Exception as e:  # noqa: BLE001 - never abort eval on a setup-time error
                logger.warning(
                    "LiberoAdapter: scene auto-generation failed (%s); falling back to bare Panda. "
                    "Install the [benchmark-libero] extra (pip install 'strands-robots[benchmark-libero]') "
                    "or pass scene_path= explicitly to silence this warning.",
                    e,
                )

        scene_was_loaded = False
        # Detect "prewarm-fresh ep0": prewarm already loaded the scene and
        # applied init_states[0]; re-running load_scene would reset MjData
        # to qpos0 and race the recorder thread into capturing gradient /
        # qpos0 frames. On the fast-path, skip load_scene (canonical-state
        # apply still runs - see below).
        backend_state = getattr(getattr(sim, "_world", None), "_backend_state", None)
        prewarm_path = backend_state.get("libero_prewarm_path") if isinstance(backend_state, dict) else None
        is_prewarm_fresh_ep0 = self._episode_count == 0 and self.scene_path and prewarm_path == self.scene_path

        # Sanity-check: a model mutated since prewarm (e.g. a stray
        # sim.add_robot changed nq) makes the flag stale - warn and fall
        # through to the full reload + canonical-state apply.
        if is_prewarm_fresh_ep0 and self._init_states is not None and self._init_states.shape[0] > 0:
            world = getattr(sim, "_world", None)
            model = getattr(world, "_model", None) if world is not None else None
            if model is not None:
                expected_width = 1 + int(getattr(model, "nq", 0)) + int(getattr(model, "nv", 0))
                actual_width = int(self._init_states[0].shape[0])
                if actual_width != expected_width:
                    logger.warning(
                        "LiberoAdapter.on_episode_start: prewarm-fresh flag is set but "
                        "model size mismatches init_states[0] (1+nq+nv=%d, init_states[0].shape=%d). "
                        "This usually means sim.add_robot or another model-mutating call ran "
                        "between prewarm() and evaluate_benchmark, recompiling the spec and "
                        "invalidating prewarm's setup. Falling through to normal lifecycle.",
                        expected_width,
                        actual_width,
                    )
                    is_prewarm_fresh_ep0 = False
                    if isinstance(backend_state, dict):
                        backend_state.pop("libero_prewarm_path", None)

        if is_prewarm_fresh_ep0:
            # Fast-path: skip load_scene (avoids a redundant spec
            # recompile) and _register_default_robot (idempotent; prewarm
            # did it). _apply_canonical_state is intentionally NOT
            # skipped: PolicyRunner calls sim.reset() between prewarm and
            # on_episode_start, wiping prewarm's init-state apply, so it
            # must be re-applied here. _episode_count advances via the
            # branch's own increment, keeping ep0 pinned to idx 0.
            logger.debug(
                "LiberoAdapter.on_episode_start: prewarm-fresh ep0 detected (path=%r); "
                "skipping load_scene + _register_default_robot "
                "(canonical-state apply still runs to restore qpos after PolicyRunner.sim.reset)",
                self.scene_path,
            )
            scene_was_loaded = True
            # Clear the flag so a subsequent fresh prewarm() (e.g. user
            # re-evaluates with a different scene) is detected fresh.
            if isinstance(backend_state, dict):
                backend_state.pop("libero_prewarm_path", None)
        elif self.scene_path:
            load_scene = getattr(sim, "load_scene", None)
            if load_scene is None:
                logger.warning(
                    "LiberoAdapter: sim has no load_scene(); skipping scene_path=%r",
                    self.scene_path,
                )
            else:
                result = load_scene(self.scene_path)
                if isinstance(result, dict) and result.get("status") == "error":
                    msg = (result.get("content") or [{}])[0].get("text", "")
                    raise RuntimeError(f"LiberoAdapter: load_scene({self.scene_path!r}) failed: {msg}")
                scene_was_loaded = True
        # Pre-register the scene-supplied robot BEFORE super() runs.
        # load_scene resets world.robots, so super() would otherwise call
        # sim.add_robot - recompiling the spec with a second Panda,
        # changing nq, and invalidating any qpos snapshot across episodes.
        if scene_was_loaded and not is_prewarm_fresh_ep0:
            self._register_default_robot(sim)
        # Apply canonical state RIGHT AFTER load_scene + pre-register so
        # the snapshot captures the post-load state before super() /
        # install_cameras can drift it. Always runs - including on the
        # prewarm-fresh-ep0 fast-path (PolicyRunner's reset() between
        # prewarm and here wipes prewarm's qpos work).
        if scene_was_loaded and self._apply_canonical_state_enabled:
            self._apply_canonical_state(sim, rng)
        super().on_episode_start(sim, rng)
        if self._install_cameras:
            self._install_libero_cameras(sim)
        if scene_was_loaded:
            # LIBERO-canonical render options (hide collision geoms,
            # site / joint / actuator / COM markers) - see
            # _install_render_options.
            self._install_render_options(sim)
            # OSC_POSE controller: converts GR00T's task-space delta-EEF
            # actions into the scene's torque-mode joint actuators;
            # without it every action key is silently dropped.
            self._install_action_controller(sim)
        if self._init_jitter > 0:
            self._apply_init_jitter(sim, rng)

        # Settle physics by one zero-action control step so the first
        # observation is at "init_state + 1 OSC step", matching upstream
        # LIBERO's reset (set_init_state + env.step(zeros)); the training
        # data was generated against the post-step state. Best-effort:
        # only runs when the OSC controller (which owns stepping) is
        # installed.
        if scene_was_loaded:
            world = getattr(sim, "_world", None)
            if world is not None:
                backend_state = getattr(world, "_backend_state", None)
                if isinstance(backend_state, dict) and "action_controller" in backend_state:
                    send_action = getattr(sim, "send_action", None)
                    if callable(send_action):
                        try:
                            send_action({})
                        except Exception as e:  # noqa: BLE001 - never abort eval on settle failure
                            logger.warning(
                                "LiberoAdapter.on_episode_start: settle send_action({}) raised %s; "
                                "first observation will be at raw init_state instead of init_state+1step",
                                e,
                            )

        # Reset per-episode state-log counter so each episode emits its
        # own first N STATE_LOG lines.
        self._state_log_step = 0

    def ensure_scene(self) -> str | None:
        """Resolve :attr:`scene_path`, auto-generating the scene MJCF if needed.

        Idempotent public entry point for the scene-resolution step that
        :meth:`on_episode_start` otherwise runs lazily; call it (then
        ``sim.load_scene`` and :meth:`prewarm`) when a driver needs the
        scene and its cameras before the eval starts. Returns the
        resolved path, or ``None`` when no BDDL source is recoverable or
        ``auto_generate_scene`` is off. Unlike the lazy path (which warns
        and falls back to a bare Panda), generation failures propagate to
        the caller.
        """
        if self.scene_path is None and self._auto_generate_scene:
            generated = self._generate_scene_from_bddl()
            if generated is not None:
                self.scene_path = generated
        return self.scene_path

    def prewarm(self, sim: SimEngine) -> None:
        """Idempotent setup that should run BEFORE ``sim.start_cameras_recording``.

        The recorder thread captures its first frame immediately - before
        :meth:`on_episode_start` runs - so two things must already be in
        place: the LIBERO ``viz_option`` (else the first frame shows
        collision capsules and site/joint/actuator markers) and a
        completed ``mj_forward`` (``MjData`` allocates ``xpos``/``xmat``
        but leaves them unset until ``mj_forward`` runs; a render before
        that returns a skybox-only gradient - no amount of per-thread
        renderer warmup fixes it).

        Runs the idempotent subset of :meth:`on_episode_start`: robot
        pre-register, camera install, render options, OSC controller,
        ``init_states[0]`` apply (so the first recorded frame shows the
        canonical ready pose), ``mj_forward``, and one main-thread warmup
        render to prime process-shared GL state.

        Recommended call order::

            sim.load_scene(spec.scene_path)
            spec.prewarm(sim)
            sim.start_cameras_recording(...)
            result = sim.evaluate_benchmark(...)

        Do NOT call ``sim.add_robot`` between ``load_scene`` and
        ``prewarm`` for LIBERO scenes: the scene already contains the
        Panda, and the recompile bumps ``model.nq`` past what
        ``init_states[0]`` was sized for (the apply then no-ops at
        WARNING). Prewarm does not replace :meth:`on_episode_start`;
        it is an early-rendering hint. No-op when ``scene_path`` is
        ``None``. Best-effort throughout - each step's failure is
        logged at WARNING and never aborts (a failure only degrades
        rendering; ``on_episode_start`` retries).
        """
        if not self.scene_path:
            logger.debug(
                "LiberoAdapter.prewarm: scene_path is None; skipping (scene auto-generation defers to on_episode_start)"
            )
            return

        # Each step is independently idempotent.
        try:
            self._register_default_robot(sim)
        except Exception as e:  # noqa: BLE001 - never abort prewarm on a single-step failure
            logger.warning("LiberoAdapter.prewarm: _register_default_robot raised: %s", e)
        if self._install_cameras:
            try:
                self._install_libero_cameras(sim)
            except Exception as e:  # noqa: BLE001
                logger.warning("LiberoAdapter.prewarm: _install_libero_cameras raised: %s", e)
        try:
            self._install_render_options(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: _install_render_options raised: %s", e)

        # OSC_POSE controller install. Stays best-effort even when
        # strict_action_controller=True: prewarm only primes rendering;
        # on_episode_start re-installs and is the authoritative gate
        # that surfaces a strict failure as an eval error.
        try:
            self._install_action_controller(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: _install_action_controller raised: %s", e)

        # Apply init_states[0] so the recorder's first frame shows the
        # canonical ready pose (pairs with the episode-0-pinned-to-idx-0
        # logic in _apply_init_state_branch).
        try:
            self._apply_init_state_for_prewarm(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: init-state apply failed: %s", e)

        # Forward the MjData so xpos/xmat reflect the ready pose before
        # the recorder thread's first render (defense-in-depth on top of
        # the mj_forward in Simulation.load_scene, and required after the
        # init-state apply above).
        try:
            self._forward_mj_data(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: mj_forward failed: %s", e)

        # One main-thread render primes process-shared GL driver state
        # (shaders, texture caches) that the recorder thread inherits;
        # its per-thread Renderer alone starts cold. Harmless if the
        # driver shares nothing. Best-effort, DEBUG on failure.
        try:
            self._warmup_render(sim)
        except Exception as e:  # noqa: BLE001
            logger.debug("LiberoAdapter.prewarm: warmup render failed: %s", e)

    def _warmup_render(self, sim: SimEngine) -> None:
        """Force one synchronous main-thread render to prime GL state.

        Renders the first configured camera (fallback ``"default"``)
        once and discards the result - only the GL state-priming
        side-effect matters. Best-effort: any failure is logged at
        DEBUG and returns silently.
        """
        render = getattr(sim, "render", None)
        if render is None:
            logger.debug("LiberoAdapter.prewarm: sim has no render(); skipping warmup")
            return
        # Pick the first declared camera; default fallback if none.
        cam_name = next(iter(self._cameras), "default") if self._cameras else "default"
        try:
            render(camera_name=cam_name, width=64, height=64)
        except Exception as e:  # noqa: BLE001 - warmup failures non-fatal
            logger.debug("LiberoAdapter.prewarm: warmup render(%r) failed: %s", cam_name, e)
            return
        logger.debug("LiberoAdapter.prewarm: warmup render(%r) primed GL state", cam_name)

    def _apply_init_state_for_prewarm(self, sim: SimEngine) -> None:
        """Write ``init_states[0]`` to ``world._data`` (best-effort).

        Non-strict mirror of :meth:`_apply_init_state_branch`: width
        mismatches, missing init_states, or missing mujoco / world /
        model / data all log-and-skip instead of raising
        (``on_episode_start`` retries strictly). Always uses index 0 and
        does NOT increment ``_episode_count`` - prewarm runs before
        episode 0.
        """
        if self._init_states is None or self._init_states.shape[0] == 0:
            logger.debug("LiberoAdapter.prewarm: no init_states; skipping init-state apply")
            return

        # Probe mujoco importability so we skip cleanly on non-MuJoCo
        # backends. try-import (not find_spec) so test fixtures can stub
        # sys.modules["mujoco"].
        try:
            import mujoco  # noqa: F401 - probe-only, real use is in _forward_mj_data
        except ImportError:
            logger.debug("LiberoAdapter.prewarm: mujoco not importable; skipping init-state apply")
            return

        world = getattr(sim, "_world", None)
        if world is None:
            return
        model = getattr(world, "_model", None)
        data = getattr(world, "_data", None)
        if model is None or data is None:
            return

        nq = int(getattr(model, "nq", 0))
        nv = int(getattr(model, "nv", 0))
        if nq == 0 or nv == 0:
            return
        state = self._init_states[0]
        expected_width = 1 + nq + nv
        if state.shape[0] != expected_width:
            # Log + skip (prewarm is a hint; on_episode_start raises on
            # the same mismatch). WARNING because this almost always
            # means a spec-recompiling call (e.g. sim.add_robot) ran
            # between load_scene and prewarm.
            logger.warning(
                "LiberoAdapter.prewarm: init_state[0] width %d != 1+nq+nv=%d; skipping init-state apply. "
                "This usually means sim.add_robot (or another spec-recompiling call) ran between "
                "sim.load_scene and spec.prewarm. Recommended call order: load_scene -> prewarm -> "
                "start_cameras_recording -> evaluate_benchmark, with NO sim.add_robot between them.",
                state.shape[0],
                expected_width,
            )
            return

        lock = getattr(sim, "_lock", None)

        def _apply() -> None:
            data.time = float(state[0])
            np.copyto(data.qpos, state[1 : 1 + nq])
            np.copyto(data.qvel, state[1 + nq :])
            # mj_forward is called by _forward_mj_data right after
            # this returns; intentionally not duplicated here.

        if lock is not None:
            with lock:
                _apply()
        else:
            _apply()

        # One-shot flag: tells on_episode_start's ep0 fast-path that
        # prewarm already loaded this scene and applied init_state[0];
        # consumed there so ep1+ follows the normal lifecycle.
        backend_state = getattr(world, "_backend_state", None)
        if isinstance(backend_state, dict):
            backend_state["libero_prewarm_path"] = self.scene_path

        logger.debug("LiberoAdapter.prewarm: applied init_state[0] (qpos[:%d] + qvel[:%d])", nq, nv)

    def _forward_mj_data(self, sim: SimEngine) -> None:
        """Run ``mujoco.mj_forward(model, data)`` if the sim has both available.

        Best-effort: missing mujoco / world / model / data debug-log and
        skip. ``mj_forward`` itself raising indicates a genuine sim-level
        bug and propagates to prewarm's catch-all.
        """
        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter._forward_mj_data: mujoco not importable; skipping")
            return

        world = getattr(sim, "_world", None)
        if world is None:
            logger.debug("LiberoAdapter._forward_mj_data: sim has no _world; skipping")
            return
        model = getattr(world, "_model", None)
        data = getattr(world, "_data", None)
        if model is None or data is None:
            logger.debug("LiberoAdapter._forward_mj_data: world missing model/data; skipping")
            return

        # mj_forward populates xpos/xquat/xmat plus other derived state.
        # The lock matches the contract of other state-mutating helpers
        # in this adapter (e.g. _apply_init_state_branch).
        lock = getattr(sim, "_lock", None)
        if lock is not None:
            with lock:
                _mj.mj_forward(model, data)
        else:
            _mj.mj_forward(model, data)

    def on_step(
        self,
        sim: SimEngine,
        obs: dict[str, Any],
        action: dict[str, Any],
    ) -> StepInfo:
        """Sparse step: zero reward, never ``done``. Success is detected by
        :meth:`is_success` at the outer eval loop."""
        return StepInfo(reward=0.0, done=False)

    def augment_observation(
        self,
        sim: SimEngine,
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject the ``x/y/z/roll/pitch/yaw/gripper`` keys the
        ``libero_panda`` ``Gr00tDataConfig`` expects.

        The policy looks up these bare keys directly in the robot
        observation; ``Simulation.get_observation()`` only returns
        joint-space readings, so without this hook the server rejects
        every request (``State key 'state.x' must be in observation``).

        Semantics:

        * EEF pose comes from the gripper-tip *site* (position) and
          wrist *body* (orientation) - see :meth:`_read_eef_pose` - with
          a body-only fallback for non-RoboSuite scenes.
        * Orientation is converted to extrinsic XYZ Euler (roll, pitch,
          yaw), matching RoboSuite's ``mat2euler(..., axes='sxyz')``.
        * ``gripper`` is the 2-vector of finger qpos; legacy fallback
          duplicates one joint's value for unknown gripper layouts.
        * Rendered ``image`` / ``wrist_image`` values are flipped
          vertically to upstream LIBERO's OpenGL bottom-row-zero
          convention (our renderer returns top-row-zero).

        Best-effort: missing sources omit their keys with a debug log.
        Keys already present in ``obs`` are never overwritten, so a
        backend that natively returns Cartesian state wins. Disable with
        ``inject_eef_state=False``.
        """
        if not self._inject_eef_state:
            return obs

        merged = dict(obs)

        # End-effector pose: site first (RoboSuite eef_pos/eef_quat
        # semantics), body fallback.
        position, quat = self._read_eef_pose(sim)
        if position is not None:
            # Don't overwrite if a backend already supplied these
            # (e.g. via a custom mapping).
            merged.setdefault("x", float(position[0]))
            merged.setdefault("y", float(position[1]))
            merged.setdefault("z", float(position[2]))
        if quat is not None:
            roll, pitch, yaw = _quat_wxyz_to_rpy_xyz(quat)
            merged.setdefault("roll", roll)
            merged.setdefault("pitch", pitch)
            merged.setdefault("yaw", yaw)

        # Gripper: LIBERO trains state.gripper on the opposite-sign
        # 2-vector of both finger qpos, read directly from
        # data.qpos[jnt_qposadr] via the canonical RoboSuite joint names.
        gripper_qpos = self._read_gripper_qpos(sim)
        if gripper_qpos is not None:
            merged.setdefault("gripper", gripper_qpos)
        else:
            # Legacy fallback - read one joint from obs and duplicate;
            # wrong for LIBERO but there is no better default for
            # unknown gripper layouts.
            gripper_value = obs.get(self._gripper_joint_name)
            if gripper_value is None:
                # Some backends namespace joint keys; try the suffix match.
                for key, val in obs.items():
                    if isinstance(key, str) and key.endswith("/" + self._gripper_joint_name):
                        gripper_value = val
                        break
            if isinstance(gripper_value, (int, float)) and not isinstance(gripper_value, bool):
                merged.setdefault("gripper", [float(gripper_value), float(gripper_value)])
            else:
                logger.debug(
                    "LiberoAdapter: gripper joints %s not found via direct mujoco lookup, "
                    "and obs key %r missing; omitting state.gripper",
                    self.state_gripper_joint_names,
                    self._gripper_joint_name,
                )

        # Flip rendered images vertically: our renderer returns
        # top-row-zero, upstream LIBERO's OffScreenRenderEnv returns
        # bottom-row-zero (OpenGL convention) and the checkpoint was
        # trained against that (plus the policy's image_rotation_180
        # flag). ascontiguousarray keeps downstream serialization
        # working - reversed views are not contiguous. Non-ndarray or
        # <2-dim values pass through untouched.
        for cam_key in ("image", "wrist_image"):
            cam_value = merged.get(cam_key)
            if isinstance(cam_value, np.ndarray) and cam_value.ndim >= 2:
                merged[cam_key] = np.ascontiguousarray(cam_value[::-1, :])

        # STATE_LOG: one structured line per call when
        # STRANDS_LIBERO_STATE_LOG=1 - the exact state values fed to the
        # policy server, for offline comparison against upstream.
        if self._state_log_enabled and self._state_log_step < self._state_log_max:
            logger.info(
                "STATE_LOG step=%d x=%s y=%s z=%s roll=%s pitch=%s yaw=%s gripper=%s obs_keys=%s",
                self._state_log_step,
                _fmt_state_value(merged.get("x")),
                _fmt_state_value(merged.get("y")),
                _fmt_state_value(merged.get("z")),
                _fmt_state_value(merged.get("roll")),
                _fmt_state_value(merged.get("pitch")),
                _fmt_state_value(merged.get("yaw")),
                _fmt_state_value(merged.get("gripper")),
                sorted(merged.keys()),
            )
            self._state_log_step += 1

        return merged

    def _read_eef_pose(self, sim: SimEngine) -> tuple[list[float] | None, list[float] | None]:
        """Read EEF position + (wxyz) quaternion for ``augment_observation``.

        Split sources matching RoboSuite exactly: POSITION from the
        gripper-tip *site* (``data.site_xpos``), ORIENTATION from the
        wrist *body* (``data.xquat``). The two points differ by ~9.7 cm
        and a 90 deg Z offset in orientation, and RoboSuite's
        ``eef_pos`` / ``eef_quat`` observables (what the dataset was
        trained on) deliberately use this split - reading both from one
        source produces out-of-distribution state.

        Returns ``(pos, quat_wxyz)``; either may be ``None`` on failure
        (logged at DEBUG; caller injects only the keys it has).
        """
        # 1. Direct-mujoco read: position from site, orientation from
        # body (the LIBERO/RoboSuite path). Each lookup can succeed or
        # fail independently.
        world = getattr(sim, "_world", None)
        model = getattr(world, "_model", None) if world is not None else None
        data = getattr(world, "_data", None) if world is not None else None

        site_pos: list[float] | None = None
        body_quat: list[float] | None = None

        if model is not None and data is not None:
            try:
                import mujoco as _mj
            except ImportError as e:
                logger.debug("LiberoAdapter: mujoco import failed in _read_eef_pose: %s", e)
                _mj = None  # type: ignore[assignment]

            if _mj is not None:
                # 1a. SITE -> position (RoboSuite eef_pos).
                site_name = self.eef_state_site_name
                if site_name:
                    try:
                        site_id = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, site_name))
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug("LiberoAdapter: mujoco site lookup failed for %r: %s", site_name, e)
                        site_id = -1
                    if site_id >= 0:
                        try:
                            pos_arr = np.asarray(data.site_xpos[site_id], dtype=np.float64)
                            site_pos = [float(c) for c in pos_arr]
                        except (AttributeError, IndexError, ValueError) as e:
                            logger.debug(
                                "LiberoAdapter: failed to read site %r position (site_id=%d): %s",
                                site_name,
                                site_id,
                                e,
                            )

                # 1b. BODY -> orientation (RoboSuite eef_quat). MuJoCo's
                # xquat is already wxyz, which _quat_wxyz_to_rpy_xyz
                # expects.
                body_name = self._eef_body_name
                if body_name:
                    try:
                        body_id = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name))
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug("LiberoAdapter: mujoco body lookup failed for %r: %s", body_name, e)
                        body_id = -1
                    if body_id >= 0:
                        try:
                            quat_arr = np.asarray(data.xquat[body_id], dtype=np.float64)
                            body_quat = [float(c) for c in quat_arr]
                        except (AttributeError, IndexError, ValueError) as e:
                            logger.debug(
                                "LiberoAdapter: failed to read body %r xquat (body_id=%d): %s",
                                body_name,
                                body_id,
                                e,
                            )

        # 2. If both were read directly, return the split-source pair
        # (this is the happy path on LIBERO scenes).
        if site_pos is not None and body_quat is not None:
            return (site_pos, body_quat)

        # 3. Body-state fallback for whichever direct read failed (e.g.
        # non-RoboSuite scenes without the canonical site).
        # sim.get_body_state is namespace-aware and returns the same
        # (pos, quat_wxyz) shape we promise here.
        get_body_state = getattr(sim, "get_body_state", None)
        fallback_pos: list[float] | None = None
        fallback_quat: list[float] | None = None
        if get_body_state is not None:
            try:
                state_result = get_body_state(body_name=self._eef_body_name)
            except Exception as e:  # noqa: BLE001 - never abort eval on a state lookup
                logger.debug("LiberoAdapter: get_body_state(%r) raised: %s", self._eef_body_name, e)
                state_result = None
            fallback_pos, fallback_quat = _extract_pose(state_result)
        else:
            logger.debug("LiberoAdapter: sim has no get_body_state(); skipping EEF state injection")

        # 4. Mix direct and fallback reads - direct wins, fallback
        # fills gaps.
        merged_pos = site_pos if site_pos is not None else fallback_pos
        merged_quat = body_quat if body_quat is not None else fallback_quat
        return (merged_pos, merged_quat)

    def _read_gripper_qpos(self, sim: SimEngine) -> list[float] | None:
        """Read both finger qpos for the LIBERO ``state.gripper`` 2-vector.

        Returns ``[finger1.qpos, finger2.qpos]`` read directly from
        ``data.qpos[jnt_qposadr]`` for
        :attr:`state_gripper_joint_names`. The two values have OPPOSITE
        signs by physical convention (mirrored joint ranges), which is
        what LIBERO's training data records.

        Returns ``None`` - never partial data - when model/data are
        unavailable, mujoco isn't importable, or any joint name doesn't
        resolve; the caller then falls back to the legacy single-joint
        duplicate path.
        """
        world = getattr(sim, "_world", None)
        model = getattr(world, "_model", None) if world is not None else None
        data = getattr(world, "_data", None) if world is not None else None
        if model is None or data is None:
            return None
        try:
            import mujoco as _mj
        except ImportError as e:
            logger.debug("LiberoAdapter: mujoco import failed in _read_gripper_qpos: %s", e)
            return None

        joint_names = self.state_gripper_joint_names
        if not joint_names:
            return None

        finger_qposes: list[float] = []
        for jname in joint_names:
            try:
                jid = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, jname))
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("LiberoAdapter: mujoco joint lookup failed for %r: %s", jname, e)
                return None
            if jid < 0:
                logger.debug(
                    "LiberoAdapter: state gripper joint %r not in compiled model (jid=%d)",
                    jname,
                    jid,
                )
                return None
            try:
                qposadr = int(model.jnt_qposadr[jid])
                finger_qposes.append(float(data.qpos[qposadr]))
            except (AttributeError, IndexError, ValueError) as e:
                logger.debug(
                    "LiberoAdapter: failed to read qpos for joint %r (jid=%d): %s",
                    jname,
                    jid,
                    e,
                )
                return None
        return finger_qposes

    def is_success(self, sim: SimEngine) -> bool:
        """Check whether the LIBERO task goal is satisfied.

        Walks the BDDL predicate tree compiled at construction time
        (:func:`compile_goal`) against the current sim state. The
        evaluator matches upstream LIBERO's ``check_ontop`` /
        ``check_contact`` semantics byte-for-byte at the moment of
        contact, including the ``_main`` body-name suffix fallback.
        """
        return bool(self._success_fn(sim))

    # Internals

    def _generate_scene_from_bddl(self) -> str | None:
        """Build the LIBERO scene MJCF from the BDDL via the upstream ``libero`` package.

        Returns the absolute path to a cached MJCF, or ``None`` when the
        BDDL source isn't recoverable (constructed without
        ``bddl_source`` / ``bddl_path``). Raises on any other failure so
        :meth:`on_episode_start` can decide whether to fall back to a
        bare Panda.

        SHA256-keyed disk cache; a cache hit never imports ``libero``.
        On miss, a headless ``ControlEnv`` is constructed (no GL
        context), the compiled MJCF extracted, camera names renamed per
        :attr:`_scene_camera_aliases`, and the XML cached. The env is
        closed after extraction.
        """
        bddl_path = self._resolve_bddl_path_for_libero()
        if bddl_path is None:
            logger.debug(
                "LiberoAdapter: no BDDL source available for scene generation - "
                "constructed from a pre-parsed BDDLProblem without bddl_source / bddl_path"
            )
            return None

        bddl_bytes = bddl_path.read_bytes()
        cache_key = self._scene_cache_key(bddl_bytes)
        cache_dir = Path(self._scene_cache_dir).expanduser() if self._scene_cache_dir else _default_scene_cache_dir()
        cache_path = cache_dir / f"{cache_key}.xml"
        if cache_path.exists():
            logger.debug("LiberoAdapter: scene cache hit %s", cache_path)
            return str(cache_path)

        # Cache miss - lazy-import libero, build the scene.
        env_wrapper = require_optional(
            "libero.libero.envs.env_wrapper",
            pip_install="libero",
            extra="benchmark-libero",
            purpose="LIBERO scene generation from BDDL",
        )
        ControlEnv = env_wrapper.ControlEnv  # type: ignore[attr-defined]

        # Renderer flags off: we only need the compiled model, not
        # frames, and use_camera_obs=False keeps reset() away from the
        # renderer too.
        env = ControlEnv(
            bddl_file_name=str(bddl_path),
            has_offscreen_renderer=False,
            has_renderer=False,
            use_camera_obs=False,
        )
        try:
            xml = _extract_compiled_mjcf(env)
        finally:
            try:
                env.close()
            except Exception as e:  # noqa: BLE001 - close errors are non-fatal
                logger.debug("LiberoAdapter: env.close() raised after extraction: %s", e)

        if self._scene_camera_aliases:
            xml = _rename_mjcf_cameras(xml, self._scene_camera_aliases)

        # No further MJCF rewrites: the cached XML matches upstream's
        # compiled model verbatim except for the camera-name aliases.
        # Visual fidelity (hiding collision geoms / markers) is handled
        # at render time via _install_render_options, which is where
        # upstream does it too - MJCF-level rgba/lighting edits were
        # tried and washed out the scene contrast.

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(xml)
        logger.info(
            "LiberoAdapter: generated scene MJCF for %s -> %s",
            self.problem.name,
            cache_path,
        )
        return str(cache_path)

    def _resolve_bddl_path_for_libero(self) -> Path | None:
        """Return a ``Path`` to a ``.bddl`` file libero can open, or ``None``.

        - If the adapter was constructed via :meth:`from_file`,
          ``self._bddl_path`` already points at a real file - reuse it.
        - If constructed via :meth:`from_text`, write the source text to
          a stable temp file (keyed by SHA256 of the text) so libero has
          a real path. The temp file lives under
          ``<scene_cache_dir>/.bddl/`` so it's cleaned up alongside the
          scene cache.
        - If neither is set, return ``None``.
        """
        if self._bddl_path is not None:
            p = Path(self._bddl_path).expanduser()
            if p.is_file():
                return p
            logger.debug(
                "LiberoAdapter: bddl_path=%s not on disk; falling back to bddl_source",
                p,
            )
        if self._bddl_source is None:
            return None

        cache_dir = Path(self._scene_cache_dir).expanduser() if self._scene_cache_dir else _default_scene_cache_dir()
        bddl_dir = cache_dir / ".bddl"
        bddl_dir.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(self._bddl_source.encode("utf-8")).hexdigest()
        tmp = bddl_dir / f"{sha}.bddl"
        if not tmp.exists():
            tmp.write_text(self._bddl_source)
        return tmp

    def _scene_cache_key(self, bddl_bytes: bytes) -> str:
        """Compute the scene-cache filename stem for ``bddl_bytes``.

        Key is ``sha256(bddl_bytes || aliases || transform_version)``:
        both the alias map and ``_LIBERO_MJCF_TRANSFORM_VERSION`` are
        hashed in so alias changes and transform-pipeline changes
        auto-invalidate stale on-disk caches. ``json.dumps(...,
        sort_keys=True)`` keeps the hash deterministic. Returns the hex
        digest without extension.
        """
        alias_repr = json.dumps(self._scene_camera_aliases, sort_keys=True).encode("utf-8")
        return hashlib.sha256(
            bddl_bytes + b"|aliases:" + alias_repr + b"|tform:" + _LIBERO_MJCF_TRANSFORM_VERSION.encode("utf-8")
        ).hexdigest()

    def _register_default_robot(self, sim: SimEngine) -> None:
        """Wrap the scene-supplied Panda in ``world.robots`` WITHOUT recompiling.

        Makes ``sim.list_robots()`` non-empty BEFORE super() runs so the
        base :class:`BenchmarkProtocol` skips its unconditional
        ``sim.add_robot`` - which would weld a second Panda into the
        spec, changing ``nq`` and blocking the ``image`` camera.
        Discovers bodies/joints/actuators by ``scene_robot_prefix`` and
        registers a :class:`SimRobot` wrapper under the key ``"robot"``.

        Best-effort: no compiled model, no prefix match, or an already
        registered ``"robot"`` all skip quietly (super() then follows
        its normal add_robot path).
        """
        world = getattr(sim, "_world", None)
        if world is None or not hasattr(world, "robots"):
            return
        if "robot" in world.robots:
            return  # super() will see it and skip its own add

        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping pre-register")
            return

        model = getattr(world, "_model", None)
        if model is None:
            logger.debug("LiberoAdapter: no compiled model; skipping pre-register")
            return

        try:
            wrapper = _build_scene_robot_wrapper(
                _mj,
                model,
                prefix=self._scene_robot_prefix,
                gripper_prefix=self._scene_gripper_prefix,
            )
        except Exception as e:  # noqa: BLE001 - never abort eval on a discovery failure
            logger.warning(
                "LiberoAdapter: scene-Panda discovery failed: %s; super() will fall back to its add_robot path",
                e,
            )
            return
        if wrapper is None:
            logger.debug(
                "LiberoAdapter: no body with prefix %r found in scene; super() will add a Panda",
                self._scene_robot_prefix,
            )
            return

        # Register under the key super() would have used so its
        # list_robots() check finds it.
        world.robots["robot"] = wrapper
        logger.debug(
            "LiberoAdapter: registered scene-supplied Panda (arm prefix=%r, gripper prefix=%r) as 'robot' "
            "(joints=%d, actuators=%d)",
            self._scene_robot_prefix,
            self._scene_gripper_prefix,
            len(wrapper.joint_names),
            len(wrapper.actuator_ids),
        )

        # The bare-Panda defaults ("hand" / "finger_joint1") don't exist
        # in RoboSuite-emitted scenes; without auto-resolution every
        # state.* key would be dropped and the policy server would
        # reject each observation. Only the constructor default (None)
        # triggers auto-resolution.
        self._resolve_scene_eef_and_gripper(_mj, model)

    def _resolve_scene_eef_and_gripper(self, mj: Any, model: Any) -> None:
        """Auto-resolve EEF body name and gripper joint name from the scene.

        First match wins, searching the canonical RoboSuite/LIBERO names:

        * EEF body: ``<scene_robot_prefix>right_hand`` ->
          ``<scene_robot_prefix>hand`` -> bare ``right_hand`` / ``hand``.
        * Gripper joint: ``<scene_gripper_prefix>finger_joint1`` ->
          ``<scene_robot_prefix>finger_joint1`` -> bare ``finger_joint1``.

        Only fires when the constructor default (``None``) was used;
        explicit user values are preserved verbatim. Best-effort: any
        lookup failure logs at DEBUG and leaves the legacy bare-Panda
        defaults in place.
        """
        prefix = self._scene_robot_prefix
        gprefix = self._scene_gripper_prefix

        if self._user_eef_body_name is None:
            eef_candidates: list[str] = []
            # Prefix-namespaced first (the case for RoboSuite/LIBERO)
            for suffix in ("right_hand", "hand", "eef"):
                if prefix:
                    eef_candidates.append(f"{prefix}{suffix}")
            # Then bare names as fallback (covers Menagerie's bare Panda)
            eef_candidates.extend(["right_hand", "hand", "eef"])
            resolved = self._first_named(mj, model, names=eef_candidates, obj=mj.mjtObj.mjOBJ_BODY)
            if resolved is not None and resolved != self._eef_body_name:
                logger.debug(
                    "LiberoAdapter: auto-resolved eef_body_name to %r (was %r); scene has prefix %r",
                    resolved,
                    self._eef_body_name,
                    prefix,
                )
                self._eef_body_name = resolved
            elif resolved is None:
                logger.debug(
                    "LiberoAdapter: no scene EEF body found among %r; keeping default %r",
                    eef_candidates,
                    self._eef_body_name,
                )

        if self._user_gripper_joint_name is None:
            grip_candidates: list[str] = []
            # Gripper namespace first (RoboSuite ``gripper0_finger_joint1``)
            if gprefix:
                grip_candidates.append(f"{gprefix}finger_joint1")
            # Robot namespace next (some custom scenes share namespaces)
            if prefix:
                grip_candidates.append(f"{prefix}finger_joint1")
            # Bare fallback (Menagerie Panda)
            grip_candidates.append("finger_joint1")
            resolved = self._first_named(mj, model, names=grip_candidates, obj=mj.mjtObj.mjOBJ_JOINT)
            if resolved is not None and resolved != self._gripper_joint_name:
                logger.debug(
                    "LiberoAdapter: auto-resolved gripper_joint_name to %r (was %r); gripper prefix %r",
                    resolved,
                    self._gripper_joint_name,
                    gprefix,
                )
                self._gripper_joint_name = resolved
            elif resolved is None:
                logger.debug(
                    "LiberoAdapter: no scene gripper joint found among %r; keeping default %r",
                    grip_candidates,
                    self._gripper_joint_name,
                )

    @staticmethod
    def _first_named(mj: Any, model: Any, *, names: list[str], obj: int) -> str | None:
        """Return the first name in ``names`` that resolves to a valid id.

        Walks ``names`` in order and returns the first one for which
        ``mj.mj_name2id(model, obj, name)`` returns a non-negative id.
        Returns ``None`` when no candidate resolves or when ``mj`` lacks
        ``mj_name2id`` (defensive against test stubs).
        """
        mj_name2id = getattr(mj, "mj_name2id", None)
        if mj_name2id is None:
            return None
        try:
            for name in names:
                if mj_name2id(model, obj, name) >= 0:
                    return name
        except Exception as e:  # noqa: BLE001 - never fatal during name resolution
            logger.debug("LiberoAdapter: mj_name2id lookup raised: %s", e)
            return None
        return None

    def _install_render_options(self, sim: SimEngine) -> None:
        """Install LIBERO-canonical render-time visualization options on ``sim``.

        Stores a ``mujoco.MjvOption`` in
        ``sim._world._backend_state["viz_option"]`` matching upstream
        LIBERO's ``OffScreenRenderEnv`` viewer setup; the render path in
        :mod:`strands_robots.simulation.mujoco.rendering` threads it to
        ``Renderer.update_scene(..., scene_option=...)`` for every
        rendered frame. Hides:

        * collision geoms (``geomgroup[0] = 0``),
        * all site markers (``sitegroup[0..5] = 0``),
        * joint / actuator / COM widgets (``mjVIS_JOINT`` /
          ``mjVIS_ACTUATOR`` / ``mjVIS_COM`` = 0).

        Render-time hiding (not MJCF rewrites) is what upstream does.
        Best-effort: missing mujoco / world / ``_backend_state`` skips
        with a debug log (defensive against stubs / non-MuJoCo
        backends).
        """
        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping render-options install")
            return

        world = getattr(sim, "_world", None)
        if world is None:
            logger.debug("LiberoAdapter: sim has no _world; skipping render-options install")
            return
        backend_state = getattr(world, "_backend_state", None)
        if not isinstance(backend_state, dict):
            logger.debug("LiberoAdapter: world._backend_state missing or not a dict; skipping")
            return

        # Building the option must not crash the eval - partial-mock
        # mujoco modules may lack these attrs; degrade to default render
        # options (the render path tolerates viz_option=None).
        try:
            opt = _mj.MjvOption()
            _mj.mjv_defaultOption(opt)
            # Hide collision geoms (group=0). MuJoCo's default after
            # mjv_defaultOption is geomgroup=[1, 1, 1, 0, 0, 0] - groups 0, 1, 2
            # visible. We turn off group 0.
            opt.geomgroup[0] = 0
            # Hide all site markers. Default after mjv_defaultOption is
            # sitegroup=[1, 1, 1, 0, 0, 0]; turn off all 6.
            for sg in range(6):
                opt.sitegroup[sg] = 0
            opt.flags[_mj.mjtVisFlag.mjVIS_JOINT] = 0
            opt.flags[_mj.mjtVisFlag.mjVIS_ACTUATOR] = 0
            opt.flags[_mj.mjtVisFlag.mjVIS_COM] = 0
        except (AttributeError, TypeError) as e:
            # Partial-mock mujoco module (test stub) - skip silently.
            logger.debug("LiberoAdapter: building MjvOption failed (%s); skipping", e)
            return

        backend_state["viz_option"] = opt
        logger.debug("LiberoAdapter: installed render options on world._backend_state['viz_option']")

    def _install_action_controller(self, sim: SimEngine) -> None:
        """Install OSC_POSE controller for GR00T task-space -> joint torques.

        GR00T-LIBERO outputs 7-dim Cartesian delta-EEF actions
        (``{x, y, z, roll, pitch, yaw, gripper}``); LIBERO scenes use
        torque-mode joint actuators, so without this controller every
        action key is silently dropped (zero torque). Installs a
        :class:`_LiberoOSCController` in
        ``world._backend_state["action_controller"]``, which the
        engine's action-dispatch path calls as
        ``controller.apply(action_dict, model, data, robot_name)``.

        Failure handling: strict mode (default) RAISES on any install
        failure so the eval returns a structured error instead of a
        misleading ``success_rate=0``; non-strict logs at WARNING and
        records the message on ``_action_controller_error``. A genuinely
        missing optional dependency (mujoco / robosuite) always degrades
        gracefully, while dependency-clash failures (e.g.
        ``numba``/``coverage>=7``) are always surfaced with a
        remediation hint.

        Lifecycle: bound to the loaded scene's compiled model; a later
        ``load_scene`` invalidates the controller's IDs, so it is
        re-installed from ``prewarm`` and every ``on_episode_start``.
        """
        # Clear any prior-episode failure tag - a re-install that now
        # succeeds must not leave a stale error around.
        self._action_controller_error = None
        try:
            controller = _LiberoOSCController.from_sim(
                sim,
                eef_site_name=f"{self._scene_gripper_prefix}grip_site",
                arm_prefix=self._scene_robot_prefix,
                gripper_prefix=self._scene_gripper_prefix,
            )
        except (ImportError, AttributeError) as e:
            # Defense-in-depth: an import/attribute error escaped from_sim's
            # own classification (it normally wraps these). Treat the known
            # numba/coverage>=7 clash as fatal (surface it); anything
            # else degrades like a missing optional dep.
            if _is_numba_coverage_clash(e):
                remediation = self._action_controller_remediation(e)
                msg = (
                    f"OSC action controller import failed ({type(e).__name__}: {e}). "
                    f"GR00T/torque actions cannot be dispatched. {remediation}"
                )
                self._action_controller_error = msg
                logger.error("LiberoAdapter._install_action_controller: %s", msg)
                raise _ControllerInstallError(msg) from e
            msg = f"OSC controller dependencies unavailable ({type(e).__name__}: {e}); GR00T actions will no-op."
            self._action_controller_error = msg
            logger.warning("LiberoAdapter._install_action_controller: %s", msg)
            return
        except _ControllerDependencyMissing as e:
            # An optional dep (mujoco / robosuite) is genuinely absent.
            # This is environmental, not a fixable setup bug, so degrade
            # gracefully even in strict mode: requiring robosuite as a
            # hard dep would break installs without the optional extras.
            msg = (
                f"{e}. GR00T actions will no-op: the OSC controller's optional "
                "dependencies (robosuite + mujoco) are not available in this environment."
            )
            self._action_controller_error = msg
            logger.warning("LiberoAdapter._install_action_controller: %s", msg)
            return
        except _ControllerInstallError as e:
            msg = (
                f"{e}. GR00T actions would no-op without the OSC controller "
                "(missing site/actuator IDs, broken import, etc.)."
            )
            self._action_controller_error = msg
            if self._strict_action_controller:
                logger.error("LiberoAdapter._install_action_controller: %s", msg)
                raise _ControllerInstallError(msg) from e
            logger.warning(
                "LiberoAdapter._install_action_controller: %s "
                "Running in non-strict mode: GR00T actions will no-op until this is resolved "
                "(set strict_action_controller=True to surface this as an eval error).",
                msg,
            )
            return
        except Exception as e:
            # Unexpected failure class. Treat the same as a controller
            # install error: strict raises, non-strict records + warns.
            msg = f"unexpected OSC controller install failure ({type(e).__name__}: {e})"
            self._action_controller_error = msg
            if self._strict_action_controller:
                logger.error("LiberoAdapter._install_action_controller: %s", msg)
                raise _ControllerInstallError(msg) from e
            logger.warning(
                "LiberoAdapter._install_action_controller: %s; "
                "GR00T actions will no-op until this is resolved "
                "(set strict_action_controller=True to surface this as an eval error).",
                msg,
            )
            return

        world = getattr(sim, "_world", None)
        if world is None:
            return
        backend_state = getattr(world, "_backend_state", None)
        if not isinstance(backend_state, dict):
            return
        backend_state["action_controller"] = controller
        logger.debug(
            "LiberoAdapter: installed OSC_POSE action_controller (eef_site=%r, arm_actuators=%d, gripper_actuators=%d)",
            controller.eef_site_name,
            len(controller.arm_actuator_ids),
            len(controller.gripper_actuator_ids),
        )

    @staticmethod
    def _action_controller_remediation(error: BaseException) -> str:
        """Build a remediation hint for a dependency-clash install failure.

        Detects the known ``numba`` / ``coverage>=7`` incompatibility
        via :func:`_is_numba_coverage_clash` and surfaces a targeted
        fix; falls back to a generic hint otherwise.
        """
        if _is_numba_coverage_clash(error):
            return (
                "This is the known numba/coverage>=7 incompatibility: numba's "
                "coverage_support module subclasses coverage.types.Tracer, which "
                "coverage>=7 removed. Remediation: uninstall the conflicting "
                "coverage from the eval environment ('pip uninstall coverage'), or "
                "pin coverage<7, or upgrade numba to a release that no longer "
                "imports coverage.types.Tracer."
            )
        return (
            "Ensure the OSC controller dependencies (robosuite and its "
            "transitive imports) are importable in this environment."
        )

    def _install_libero_cameras(self, sim: SimEngine) -> None:
        """Inject the cameras the ``libero_panda`` data_config expects.

        Without ``image`` / ``wrist_image`` in the sim, every call to a
        LIBERO policy server fails (``Video key 'video.image' must be in
        observation``). Cameras already present - in EITHER the runtime
        registry (``world.cameras``) OR the compiled MuJoCo model - are
        skipped; the model-side check matters because ``load_scene``
        resets the registry, and re-adding a scene-declared camera would
        recompile the spec and undo :meth:`_apply_canonical_state`.
        ``add_camera`` failures are logged at WARNING, never fatal.
        """
        add_camera = getattr(sim, "add_camera", None)
        if add_camera is None:
            logger.debug("LiberoAdapter: sim has no add_camera(); skipping camera install")
            return

        existing = self._existing_camera_names(sim)

        for cam_name, cam_kwargs in self._cameras.items():
            if cam_name in existing:
                logger.debug("LiberoAdapter: camera %r already in sim; skipping install", cam_name)
                # Even when add_camera is skipped, publish the configured
                # render dimensions to world.cameras - otherwise
                # model-side cameras fall through to the renderer's
                # 480x640 defaults instead of the 256x256 training
                # resolution.
                self._publish_camera_dims_to_world(sim, cam_name, cam_kwargs)
                continue
            try:
                result = add_camera(name=cam_name, **cam_kwargs)
            except Exception as e:  # noqa: BLE001 - one bad camera shouldn't kill the eval
                logger.warning("LiberoAdapter: add_camera(%r) raised: %s", cam_name, e)
                continue
            if isinstance(result, dict) and result.get("status") == "error":
                msg = (result.get("content") or [{}])[0].get("text", "")
                # "already exists" is benign - the scene XML beat us to it.
                if "already exists" in msg.lower():
                    logger.debug("LiberoAdapter: camera %r already declared by scene", cam_name)
                    # Same dimension-publish step as the skip branch above.
                    self._publish_camera_dims_to_world(sim, cam_name, cam_kwargs)
                else:
                    logger.warning("LiberoAdapter: add_camera(%r) failed: %s", cam_name, msg)

    @staticmethod
    def _publish_camera_dims_to_world(sim: SimEngine, cam_name: str, cam_kwargs: dict[str, Any]) -> None:
        """Inject a config-only :class:`SimCamera` entry into ``world.cameras``.

        Publishes render dimensions for cameras that already exist in
        the compiled model so the observation path uses the configured
        size instead of the renderer's 480x640 defaults. Only
        height/width matter - the model-compiled pose/FOV win, and the
        camera index is looked up by name downstream. Idempotent and
        best-effort (skips silently when the entry, world, or registry
        is missing).
        """
        world = getattr(sim, "_world", None)
        if world is None:
            return
        cameras_attr = getattr(world, "cameras", None)
        if not isinstance(cameras_attr, dict):
            return
        if cam_name in cameras_attr:
            # Already published (e.g. an earlier episode + scene-recompile
            # cycle). Don't overwrite the user's possibly-tweaked entry.
            return
        height = int(cam_kwargs.get("height", 256))
        width = int(cam_kwargs.get("width", 256))
        cameras_attr[cam_name] = SimCamera(
            name=cam_name,
            width=width,
            height=height,
        )
        logger.debug(
            "LiberoAdapter: published render dims for model-side camera %r (%dx%d) to world.cameras",
            cam_name,
            width,
            height,
        )

    @staticmethod
    def _existing_camera_names(sim: SimEngine) -> set[str]:
        """Union of registry-side and model-side camera names known to ``sim``.

        The model-side enumeration matters because ``load_scene`` resets
        the registry even when the MJCF declares cameras; without it the
        install would re-add scene cameras and trigger a qpos-resetting
        spec recompile. Backends without a compiled model fall back to
        the registry-only check.
        """
        names: set[str] = set()
        world = getattr(sim, "_world", None)

        # Registry-side: cameras added via sim.add_camera() previously.
        cameras_attr = getattr(world, "cameras", None) if world is not None else None
        if isinstance(cameras_attr, dict):
            names.update(cameras_attr.keys())

        # Model-side: cameras declared in a loaded scene MJCF.
        model = getattr(world, "_model", None) if world is not None else None
        if model is None:
            return names
        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping model-side camera check")
            return names
        try:
            ncam = int(getattr(model, "ncam", 0))
            for i in range(ncam):
                name = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_CAMERA, i)
                if name:
                    names.add(name)
        except Exception as e:  # noqa: BLE001 - never fatal during camera-existence check
            logger.debug("LiberoAdapter: model-side camera enumeration failed: %s", e)
        return names

    def _apply_canonical_state(self, sim: SimEngine, rng: random.Random | None = None) -> None:
        """Restore qpos / qvel to the scene's canonical home state.

        Three branches, in order of preference:

        1. **Init states** (``_init_states`` set): pick a row (episode 0
           deterministic, then RNG-seeded), validate the width equals
           ``1 + model.nq + model.nv`` (mismatch raises - never silently
           sliced), and write ``data.time / qpos / qvel`` directly.
        2. **Keyframe** (``model.nkey > 0``):
           ``mj_resetDataKeyframe(model, data, scene_keyframe_index)``.
        3. **Snapshot-and-restore** (``model.nkey == 0``): capture
           qpos/qvel on the first episode after a scene compile, restore
           it on every subsequent one (procedurally-generated MJCFs ship
           no keyframe).

        All branches end with ``mj_forward`` so derived state reflects
        the canonical qpos before the next observation / render.
        Best-effort: no compiled model or mujoco -> debug-log + skip;
        out-of-range keyframe index -> WARNING + skip; snapshot shape
        mismatch -> re-capture instead of restoring.

        Holds ``sim._lock`` if the sim exposes one to match the locking
        contract of :meth:`Simulation.reset` and :meth:`Simulation.send_action`
        - prevents racing a worker holding a stale qpos pointer.
        """
        world = getattr(sim, "_world", None)
        model = getattr(world, "_model", None) if world is not None else None
        data = getattr(world, "_data", None) if world is not None else None
        if model is None or data is None:
            logger.debug("LiberoAdapter: sim has no compiled MuJoCo model/data; skipping canonical-state apply")
            return

        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping canonical-state apply")
            return

        nkey = int(getattr(model, "nkey", 0))
        lock = getattr(sim, "_lock", None)

        # Branch 1: init_states (highest priority)
        if self._init_states is not None:
            self._apply_init_state_branch(model, data, _mj, lock, rng=rng)
            return

        # Branch 2: keyframe
        if nkey > 0:
            self._apply_keyframe_branch(sim, model, data, _mj, lock, nkey)
        # Branch 3: snapshot-and-restore
        else:
            self._apply_snapshot_branch(sim, model, data, _mj, lock)

    def _apply_init_state_branch(
        self,
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
        *,
        rng: random.Random | None,
    ) -> None:
        """Init-state branch of :meth:`_apply_canonical_state`.

        Row layout matches robosuite's ``MjSimState.from_flattened``:
        ``[time(1), qpos(nq), qvel(nv)]`` with ``na == 0`` required (no
        actuator state) - the same decomposition upstream LIBERO's
        ``set_init_state`` uses. Width mismatch raises ``RuntimeError``:
        silent slicing would produce a deeply wrong physical state and
        mask a scene-generation bug.

        Episode 0 is pinned to ``init_states[0]`` deterministically
        (matching upstream's first-episode pattern and :meth:`prewarm`);
        episodes 1+ use ``rng.randint(0, n_states-1)`` (``rng=None``
        falls back to a fresh ``random.Random()``). ``_episode_count``
        increments after every successful apply.
        """
        n_states = int(self._init_states.shape[0])  # type: ignore[union-attr]
        if n_states == 0:
            logger.debug("LiberoAdapter: empty init_states array; skipping init-state branch")
            return
        # Episode 0 = idx 0 (deterministic, matches v0.1.1 + prewarm).
        # Episodes 1+ = RNG-sampled.
        if self._episode_count == 0:
            idx = 0
        else:
            rng_local = rng if rng is not None else random.Random()
            idx = rng_local.randint(0, n_states - 1)
        state = self._init_states[idx]  # type: ignore[index]

        nq = int(model.nq)
        nv = int(model.nv)
        na = int(getattr(model, "na", 0))
        if na != 0:
            raise RuntimeError(
                f"LiberoAdapter: model has na={na} actuator state; init_state apply requires na=0. "
                f"LIBERO scenes don't carry actuator state and the flat-state layout assumes [time, qpos, qvel]."
            )

        expected_width = 1 + nq + nv
        actual_width = int(state.shape[0])
        if actual_width != expected_width:
            raise RuntimeError(
                f"LiberoAdapter: init_state width {actual_width} does not match compiled model "
                f"(1 + nq={nq} + nv={nv} = {expected_width}). The procedurally-generated MJCF "
                f"likely diverges from the upstream LIBERO scene MJCF for this BDDL task "
                f"(e.g. missing (:objects ...) declarations dropping free-joint bodies). "
                f"#168 bug I: silent slicing forbidden - fix the scene generator instead."
            )

        def _apply() -> None:
            data.time = float(state[0])
            np.copyto(data.qpos, state[1 : 1 + nq])
            np.copyto(data.qvel, state[1 + nq :])
            mj.mj_forward(model, data)

        if lock is not None:
            with lock:
                _apply()
        else:
            _apply()

        logger.debug(
            "LiberoAdapter: applied init_state[%d] (ep=%d, 1+nq+nv=%d, n_states=%d)",
            idx,
            self._episode_count,
            expected_width,
            n_states,
        )

        # Increment after successful apply so the next call is
        # "episode 1+" and gets RNG-sampled selection.
        self._episode_count += 1

    def _apply_keyframe_branch(
        self,
        sim: SimEngine,  # noqa: ARG002 - kept for symmetry with _apply_snapshot_branch
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
        nkey: int,
    ) -> None:
        """Keyframe branch of :meth:`_apply_canonical_state`."""
        if self._scene_keyframe_index < 0 or self._scene_keyframe_index >= nkey:
            logger.warning(
                "LiberoAdapter: scene_keyframe_index=%d out of range [0, %d); skipping",
                self._scene_keyframe_index,
                nkey,
            )
            return
        try:
            if lock is not None:
                with lock:
                    mj.mj_resetDataKeyframe(model, data, self._scene_keyframe_index)
                    mj.mj_forward(model, data)
            else:
                mj.mj_resetDataKeyframe(model, data, self._scene_keyframe_index)
                mj.mj_forward(model, data)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning(
                "LiberoAdapter: mj_resetDataKeyframe(%d) failed: %s",
                self._scene_keyframe_index,
                e,
            )
            return
        logger.debug(
            "LiberoAdapter: applied <keyframe> %d to canonical qpos",
            self._scene_keyframe_index,
        )

    def _apply_snapshot_branch(
        self,
        sim: SimEngine,  # noqa: ARG002 - kept for symmetry with _apply_keyframe_branch
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
    ) -> None:
        """Snapshot-and-restore branch of :meth:`_apply_canonical_state`.

        First episode: write the LIBERO-canonical Panda home pose into
        the arm qpos (mirroring upstream
        ``Robot.reset(deterministic=True)`` - without it a keyframe-less
        procedurally-generated MJCF leaves the arm at qpos=0), then
        capture qpos/qvel. Subsequent episodes: restore the cached
        snapshot via ``np.copyto`` + ``mj_forward``.
        """
        try:
            qpos = data.qpos
            qvel = data.qvel
        except AttributeError as e:
            logger.debug("LiberoAdapter: data has no qpos/qvel attrs: %s", e)
            return

        # First episode (or model recompile changed nq) -> write home
        # pose into arm joints, then capture; don't restore.
        needs_capture = (
            self._canonical_qpos is None
            or self._canonical_qpos.shape != qpos.shape
            or self._canonical_qvel is None
            or self._canonical_qvel.shape != qvel.shape
        )
        if needs_capture:
            # Write home pose before snapshotting; best-effort - on
            # failure the snapshot captures whatever qpos happens to be.
            self._write_libero_arm_home_qpos(model, data, mj, lock)
            try:
                self._canonical_qpos = np.array(qpos, copy=True)
                self._canonical_qvel = np.array(qvel, copy=True)
            except Exception as e:  # noqa: BLE001 - capture is best-effort
                logger.debug("LiberoAdapter: snapshot capture failed: %s", e)
                self._canonical_qpos = None
                self._canonical_qvel = None
                return
            logger.debug(
                "LiberoAdapter: captured canonical qpos snapshot (nq=%d, nv=%d)",
                self._canonical_qpos.shape[0],
                self._canonical_qvel.shape[0],
            )
            return

        # Subsequent episode - restore the snapshot. The needs_capture
        # check above guarantees the snapshot fields are non-None here;
        # narrow for mypy.
        assert self._canonical_qpos is not None
        assert self._canonical_qvel is not None
        canonical_qpos = self._canonical_qpos
        canonical_qvel = self._canonical_qvel
        try:
            if lock is not None:
                with lock:
                    np.copyto(qpos, canonical_qpos)
                    np.copyto(qvel, canonical_qvel)
                    mj.mj_forward(model, data)
            else:
                np.copyto(qpos, canonical_qpos)
                np.copyto(qvel, canonical_qvel)
                mj.mj_forward(model, data)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning("LiberoAdapter: snapshot restore failed: %s", e)
            return
        logger.debug("LiberoAdapter: restored canonical qpos snapshot")

    def _write_libero_arm_home_qpos(self, model: Any, data: Any, mj: Any, lock: Any) -> None:
        """Write the LIBERO Panda + gripper ready pose into ``data.qpos``.

        Mirrors upstream ``Robot.reset(deterministic=True)`` writing
        ``MountedPanda.init_qpos`` to the arm joints and
        ``SingleArm.reset`` writing ``PandaGripper.init_qpos``
        (``[0.020833, -0.020833]``) to the finger joints. Used by
        :meth:`_apply_snapshot_branch` to match upstream's BDDL-default
        start state. Best-effort: any failure logs at DEBUG and leaves
        ``data.qpos`` unmodified.
        """
        # Resolve arm joint qpos addresses (same scan as
        # _LiberoOSCController.from_sim).
        arm_qpos_addrs: list[int] = []
        gripper_qpos_addrs: list[int] = []
        njnt = int(getattr(model, "njnt", 0))
        for i in range(njnt):
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            if not isinstance(jname, str):
                continue
            if jname.startswith(self._scene_robot_prefix) and not jname.startswith(self._scene_gripper_prefix):
                arm_qpos_addrs.append(int(model.jnt_qposadr[i]))
            elif jname.startswith(self._scene_gripper_prefix):
                # Filter to finger joints only
                # (gripper0_finger_joint1, gripper0_finger_joint2).
                if "finger_joint" in jname:
                    gripper_qpos_addrs.append(int(model.jnt_qposadr[i]))
        if len(arm_qpos_addrs) != 7:
            logger.debug(
                "LiberoAdapter._write_libero_arm_home_qpos: expected 7 arm joints with prefix %r, "
                "found %d; skipping home-pose write",
                self._scene_robot_prefix,
                len(arm_qpos_addrs),
            )
            return

        home_qpos = _resolve_libero_arm_home_qpos(len(arm_qpos_addrs))
        if home_qpos is None:
            logger.debug(
                "LiberoAdapter._write_libero_arm_home_qpos: no LIBERO/robosuite home pose available; "
                "skipping home-pose write"
            )
            return

        # Resolve gripper init_qpos (PandaGripper). Same caching pattern
        # as arm home pose for cheap repeat lookup.
        gripper_init = _resolve_panda_gripper_init_qpos(len(gripper_qpos_addrs))

        def _do_write() -> None:
            for adr, v in zip(arm_qpos_addrs, home_qpos, strict=True):
                data.qpos[adr] = float(v)
            if gripper_init is not None and len(gripper_qpos_addrs) == len(gripper_init):
                for adr, v in zip(gripper_qpos_addrs, gripper_init, strict=True):
                    data.qpos[adr] = float(v)
            mj.mj_forward(model, data)

        try:
            if lock is not None:
                with lock:
                    _do_write()
            else:
                _do_write()
        except Exception as e:  # noqa: BLE001 - best-effort, never fatal
            logger.debug(
                "LiberoAdapter._write_libero_arm_home_qpos: write failed (%s); snapshot will capture pre-write qpos",
                e,
            )
            return
        logger.debug(
            "LiberoAdapter: wrote LIBERO Panda + gripper home pose into data.qpos (snapshot pre-capture); "
            "arm=%d, gripper=%d",
            len(arm_qpos_addrs),
            len(gripper_qpos_addrs),
        )

    def _apply_init_jitter(self, sim: SimEngine, rng: random.Random) -> None:
        """Apply +/- jitter to xy of every body referenced by ``(:init (on A B))``.

        Best-effort: if the sim doesn't expose ``move_object`` / ``get_body_state``,
        or the body isn't in the scene, silently skip. This matches LIBERO's
        "small random perturbation per episode" convention without requiring
        full BDDL init semantics.
        """
        move_object = getattr(sim, "move_object", None)
        if move_object is None:
            logger.debug("LiberoAdapter: sim has no move_object(); skipping init jitter")
            return
        get_body_state = getattr(sim, "get_body_state", None)
        if get_body_state is None:
            return

        # Gather the set of bodies we want to jitter - BDDL init uses the same
        # Pred grammar, so (on cube_1 table_1) means "jitter cube_1".
        from strands_robots.benchmarks.libero.bddl_parser import Pred as _Pred

        seen: set[str] = set()
        for node in self.problem.init:
            for body in _extract_init_targets(node):
                seen.add(body)
        _ = _Pred  # referenced for clarity; actual test is inside _extract_init_targets

        for body in sorted(seen):
            try:
                state = get_body_state(body_name=body)
            except Exception as e:  # noqa: BLE001 - defensive
                logger.debug("jitter lookup for %r failed: %s", body, e)
                continue
            if not isinstance(state, dict) or state.get("status") != "success":
                continue
            pos = _extract_position(state)
            if pos is None:
                continue
            jx = rng.uniform(-self._init_jitter, self._init_jitter)
            jy = rng.uniform(-self._init_jitter, self._init_jitter)
            new_pos = [pos[0] + jx, pos[1] + jy, pos[2]]
            try:
                move_object(name=body, position=new_pos)
            except Exception as e:  # noqa: BLE001 - jitter failures are not fatal
                logger.debug("jitter apply for %r failed: %s", body, e)


def _extract_init_targets(node: Node) -> list[str]:
    """Return the first-arg body name of every leaf predicate in ``node``.

    Init clauses like ``(on cube_1 table_1)`` and ``(upright bottle_1)``
    share the convention that the first argument is the "subject" body -
    the thing whose position we may want to jitter. Nested
    ``and``/``or``/``not`` are traversed; non-predicates are ignored.
    """
    from strands_robots.benchmarks.libero.bddl_parser import And, Not, Or, Pred

    if isinstance(node, Pred):
        return [node.args[0]] if node.args else []
    if isinstance(node, (And, Or)):
        out: list[str] = []
        for c in node.clauses:
            out.extend(_extract_init_targets(c))
        return out
    if isinstance(node, Not):
        return _extract_init_targets(node.clause)
    return []


def _extract_position(state: dict[str, Any]) -> list[float] | None:
    """Pull ``{"json": {"position": [...]}}`` from a status-dict payload."""
    for block in state.get("content", []) or []:
        if isinstance(block, dict) and isinstance(block.get("json"), dict):
            pos = block["json"].get("position")
            if isinstance(pos, list) and len(pos) == 3 and all(isinstance(c, (int, float)) for c in pos):
                return [float(c) for c in pos]
    return None


def _walk_predicate_tree(node: Any, sim: SimEngine) -> list[tuple[str, bool]]:
    """Walk a BDDL goal tree and return ``(repr, verdict)`` for every leaf.

    Diagnostic helper for BDDL-evaluator debugging: compiles each
    ``Pred`` leaf in isolation via ``compile_goal`` and runs it against
    the live ``sim``, annotating each leaf with the looked-up body
    positions so "name doesn't resolve", "wrong threshold", and "true
    success" are distinguishable. Returns ``(predicate, verdict)``
    tuples in traversal order; combinators become nested-string
    prefixes.
    """
    from strands_robots.benchmarks.libero.bddl_parser import And, Not, Or, Pred, compile_goal
    from strands_robots.simulation.predicates import _body_position

    out: list[tuple[str, bool]] = []

    def _arg_diag(arg: str) -> str:
        """Render a body name + its position for diagnostic output."""
        pos = _body_position(sim, arg)
        if pos is None:
            return f"{arg}=NONE"
        return f"{arg}=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"

    def visit(n: Any, prefix: str = "") -> None:
        if isinstance(n, Pred):
            try:
                verdict = bool(compile_goal(n)(sim))
            except Exception as e:  # noqa: BLE001
                out.append((f"{prefix}({n.name} {' '.join(n.args)})", False))
                logger.debug("predicate_log: leaf %r raised %s", n, e)
                return
            arg_diag = " ".join(_arg_diag(a) for a in n.args)
            out.append((f"{prefix}({n.name} {arg_diag})", verdict))
            return
        if isinstance(n, And):
            for c in n.clauses:
                visit(c, prefix + "AND/")
            return
        if isinstance(n, Or):
            for c in n.clauses:
                visit(c, prefix + "OR/")
            return
        if isinstance(n, Not):
            visit(n.clause, prefix + "NOT/")
            return

    visit(node)
    return out


def _extract_pose(state: dict[str, Any] | None) -> tuple[list[float] | None, list[float] | None]:
    """Pull ``(position, quaternion_wxyz)`` from a ``get_body_state`` payload.

    Both fields are optional; this returns ``(None, None)`` for any
    error / shape mismatch so the caller can selectively inject just
    the keys it has. The MuJoCo backend always reports both, so in
    the happy path you get both arrays back.
    """
    if not isinstance(state, dict) or state.get("status") != "success":
        return (None, None)
    pos: list[float] | None = None
    quat: list[float] | None = None
    for block in state.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        json_block = block.get("json")
        if not isinstance(json_block, dict):
            continue
        raw_pos = json_block.get("position")
        if isinstance(raw_pos, list) and len(raw_pos) == 3 and all(isinstance(c, (int, float)) for c in raw_pos):
            pos = [float(c) for c in raw_pos]
        raw_quat = json_block.get("quaternion")
        if isinstance(raw_quat, list) and len(raw_quat) == 4 and all(isinstance(c, (int, float)) for c in raw_quat):
            quat = [float(c) for c in raw_quat]
    return (pos, quat)


def _fmt_state_value(value: Any) -> str:
    """Format a state value for ``STATE_LOG`` output.

    ``"None"`` for missing keys, scalars rounded to 6dp,
    list/tuple/ndarray rounded element-wise - the same style as
    ACTION_LOG so one grep parses both diagnostic streams.
    """
    if value is None:
        return "None"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6f}"
    if isinstance(value, (list, tuple)):
        return str([round(float(v), 6) if isinstance(v, (int, float)) else v for v in value])
    if isinstance(value, np.ndarray):
        return str(np.round(value, 6).tolist())
    return repr(value)


def _quat_wxyz_to_rpy_xyz(quat_wxyz: list[float]) -> tuple[float, float, float]:
    """MuJoCo ``(w, x, y, z)`` quaternion -> extrinsic XYZ Euler ``(roll, pitch, yaw)``.

    Matches RoboSuite/LIBERO's ``mat2euler(..., axes='sxyz')``
    byte-for-byte (extrinsic XYZ, Hamilton convention):

        roll  = atan2(2(wx + yz), 1 - 2(x^2 + y^2))
        pitch = asin(2(wy - xz))
        yaw   = atan2(2(wz + xy), 1 - 2(y^2 + z^2))

    The signs matter: an intrinsic-XYZ derivation flips them and
    inverts the reported yaw relative to training data. Pure
    numpy/stdlib - deliberately does not import scipy (not a declared
    dependency).

    Gimbal lock (``|sin(pitch)| >= 1 - 1e-6``) collapses roll into yaw
    using the ``atan2(-M[1,2], M[1,1])`` resolution that matches
    scipy / robosuite.

    Returns:
        ``(roll, pitch, yaw)`` in radians: roll and yaw in (-pi, pi],
        pitch in [-pi/2, pi/2].
    """
    import math

    w, x, y, z = quat_wxyz
    # Clamp asin argument to handle minor numerical drift on unit quats.
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - x * z)))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) >= 1.0 - 1e-6:
        # Gimbal lock: roll absorbed into yaw (robosuite's fallback).
        roll = 0.0
        # M[1,2] = 2(yz - wx); M[1,1] = 1 - 2(x^2 + z^2).
        yaw = math.atan2(-2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + z * z))
    else:
        # M[2,1] = 2(yz + wx); M[2,2] = 1 - 2(x^2 + y^2).
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        # M[1,0] = 2(xy + wz); M[0,0] = 1 - 2(y^2 + z^2).
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


# Scene-generation helpers


def _default_scene_cache_dir() -> Path:
    """Filesystem location for cached LIBERO scene MJCFs.

    Uses :func:`strands_robots.utils.get_base_dir` so the cache lives
    under ``$STRANDS_BASE_DIR`` (typically ``~/.strands_robots/``)
    alongside other strands-robots state. Created on demand by
    :meth:`LiberoAdapter._generate_scene_from_bddl` - this helper just
    returns the path.
    """
    return get_base_dir() / "scene_cache" / "libero"


def _extract_compiled_mjcf(env: Any) -> str:
    """Pull the MJCF XML out of a ``libero`` ControlEnv.

    Prefers the **pre-compile** MJCF (``env.env.model.get_xml()``) over
    the post-compile ``mj_saveLastXML`` re-serialization: the latter is
    lossy on ``<compiler>`` attributes - notably dropping
    ``inertiagrouprange="0 0"``, after which reloading recomputes body
    inertias from ALL geom groups (visual meshes included) and the
    scene's masses and dynamics diverge from upstream's runtime model.
    Tries a small set of accessor fallbacks for older robosuite
    versions; never touches non-public attributes.
    """
    accessors = (
        # Preferred: pre-compile MJCF (preserves <compiler> attrs).
        lambda: env.env.model.get_xml(),
        # Older robosuite helper.
        lambda: env.env.model.get_model_xml(),  # type: ignore[attr-defined]
        # Last resort: lossy post-compile re-serialization.
        lambda: env.env.sim.model.get_xml(),
    )
    last_err: Exception | None = None
    for accessor in accessors:
        try:
            xml = accessor()
        except Exception as e:  # noqa: BLE001 - try the next accessor
            last_err = e
            continue
        if isinstance(xml, str) and xml.strip():
            return xml
    raise RuntimeError(f"could not extract MJCF from libero env (last error: {last_err!r})")


# Match a complete ``<camera ... name="OLD" ...>`` declaration so the
# rename only touches camera definitions, not e.g. material names that
# happen to share a string. Anchored on the ``camera`` element name and
# guarded by ``\s+`` to avoid partial-word matches.
_CAMERA_NAME_RE = re.compile(r'(<camera\b[^>]*\bname=")([^"]+)(")')


def _rename_mjcf_cameras(xml: str, aliases: dict[str, str]) -> str:
    """Rename ``<camera name="OLD"...>`` to ``<camera name="NEW"...>`` per ``aliases``.

    Targeted regex only - we don't parse the whole MJCF. The rename is
    safe because MuJoCo doesn't allow duplicate ``<camera>`` names within
    a model, and camera references from external code (e.g.
    ``sim.render(camera_name=...)``) come from outside the XML so they
    aren't affected.

    Names not in ``aliases`` pass through unchanged.
    """
    if not aliases:
        return xml

    def _sub(match: re.Match[str]) -> str:
        head, name, tail = match.group(1), match.group(2), match.group(3)
        return head + aliases.get(name, name) + tail

    return _CAMERA_NAME_RE.sub(_sub, xml)


# Bumped whenever the BDDL -> MJCF transform pipeline changes its
# semantics; hashed into the scene-cache key by
# LiberoAdapter._scene_cache_key so stale on-disk caches auto-invalidate
# on upgrade. Current "v4": cached MJCF is the PRE-compile XML (see
# _extract_compiled_mjcf), matching upstream verbatim except for the
# camera-name aliases; visual fixes happen at render time, not in the
# MJCF.
_LIBERO_MJCF_TRANSFORM_VERSION = "v4"


def _build_scene_robot_wrapper(
    mj: Any,
    model: Any,
    *,
    prefix: str,
    gripper_prefix: str | None = None,
) -> SimRobot | None:
    """Construct a :class:`SimRobot` for an existing scene-supplied Panda.

    Walks the compiled model for bodies / joints / actuators named with
    ``prefix`` (arm) and, when ``gripper_prefix`` is non-empty, joints
    and actuators under EITHER prefix - RoboSuite grippers live in their
    own namespace, and dropping them would strip ``state.gripper`` from
    the observation pipeline. Body discovery uses ``prefix`` only (the
    gripper mounts as a child body of the arm's last link).

    The wrapper only populates ``world.robots`` so the base protocol's
    ``list_robots()`` check passes without a spec recompile; it is NOT a
    substitute for ``Simulation.add_robot``'s wrapper. IDs are read from
    the current compiled model and go stale on a later recompile.

    Returns ``None`` when no body matches the arm prefix or the model
    lacks ``nbody`` / ``njnt`` counts (test stubs). Never raises - the
    call site catches and warns.
    """
    nbody = int(getattr(model, "nbody", 0))
    njnt = int(getattr(model, "njnt", 0))
    nu = int(getattr(model, "nu", 0))
    if nbody == 0 or njnt == 0:
        return None

    # Root body: first prefix-matching body whose parent is the world
    # body (id 0); falls back to the first match (IDs are only needed
    # for the compatibility check, not kinematic queries).
    root_body_id = -1
    for i in range(nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        if root_body_id < 0:
            root_body_id = i
        # Prefer a body whose parent is the world; treat that as canonical.
        body_parentid = getattr(model, "body_parentid", None)
        if body_parentid is not None:
            try:
                if int(body_parentid[i]) == 0:
                    root_body_id = i
                    break
            except (IndexError, TypeError):
                pass
    if root_body_id < 0:
        return None

    # Joint/actuator filter covers both the arm and gripper namespaces.
    joint_prefixes: tuple[str, ...] = (prefix,)
    if gripper_prefix:
        joint_prefixes = (prefix, gripper_prefix)

    def _starts_with_any(name: object) -> bool:
        return isinstance(name, str) and any(name.startswith(p) for p in joint_prefixes if p)

    joint_names: list[str] = []
    joint_ids: list[int] = []
    for i in range(njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if _starts_with_any(name):
            joint_names.append(name)  # type: ignore[arg-type]
            joint_ids.append(i)

    actuator_ids: list[int] = []
    for i in range(nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if _starts_with_any(name):
            actuator_ids.append(i)

    return SimRobot(
        name="robot",  # registered key matches super()'s default
        urdf_path="",  # scene-supplied, no upstream URDF
        data_config="panda",  # LIBERO is Panda-only
        body_id=root_body_id,
        joint_names=joint_names,
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        namespace=prefix,
    )


class _ControllerInstallError(RuntimeError):
    """The OSC controller can't be built although the prerequisites ARE
    present (missing site / actuator IDs, wrong arm-joint count, a
    broken-but-installed import such as the numba/coverage>=7 clash).
    In strict mode :meth:`LiberoAdapter._install_action_controller`
    re-raises this so the eval returns a structured error instead of
    silently dropping every action."""


class _ControllerDependencyMissing(_ControllerInstallError):
    """A required optional dependency (mujoco / robosuite) is genuinely
    absent. Environmental, not a fixable setup bug, so the install
    always degrades gracefully regardless of
    ``strict_action_controller`` - requiring robosuite as a hard
    dependency would break installs without the optional extras."""


def _is_numba_coverage_clash(error: BaseException) -> bool:
    """Recognise the ``numba`` / ``coverage>=7`` import incompatibility.

    ``coverage>=7`` removed ``coverage.types.Tracer``, which numba's
    coverage_support subclasses - so importing numba (pulled in
    transitively by robosuite's OSC path) raises
    ``AttributeError: module 'coverage.types' has no attribute 'Tracer'``.
    Walks the exception chain so the signature is recognised even when
    wrapped by a later ImportError.
    """
    seen: set[int] = set()
    cur: BaseException | None = error
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = str(cur)
        if "coverage.types" in text and "Tracer" in text:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class _LiberoOSCController:
    """OSC_POSE controller wrapper for GR00T-LIBERO action dispatch.

    Converts task-space delta-EEF actions
    (``{x, y, z, roll, pitch, yaw, gripper}``) into joint torques via
    RoboSuite's ``OperationalSpaceController``; the ``gripper`` channel
    is handled separately as a direct actuator write (RoboSuite's OSC
    ignores the gripper). Holds a robosuite ``MjSim`` shim around the
    sim's compiled model + data, built once per episode.

    Lifecycle: bound to one compiled model - a spec recompile stales the
    stored joint / actuator IDs, so
    ``LiberoAdapter._install_action_controller`` rebuilds the controller
    from prewarm and every on_episode_start.
    """

    # This controller drives mj_step itself: one apply() advances
    # physics by physics_substeps_per_control steps, recomputing OSC
    # torques each step. Without the flag the engine would double-step.
    owns_stepping: bool = True

    # PandaGripper.format_action ramp constant (robosuite's
    # ``self.speed = 0.01``): the normalized 2-vector current_action is
    # incremented by [-1, +1] * speed * sign(input) per substep.
    # Replicating the ramp matters - writing the raw scalar to
    # data.ctrl drives finger1 (ctrlrange [0, 0.04]) the wrong way.
    _GRIPPER_SPEED: ClassVar[float] = 0.01

    def __init__(
        self,
        controller: Any,
        sim_shim: Any,
        eef_site_name: str,
        arm_actuator_ids: list[int],
        gripper_actuator_ids: list[int],
        model: Any,
        data: Any,
        physics_substeps_per_control: int = 25,
        eef_site_id: int = -1,
        arm_qpos_addrs: list[int] | None = None,
    ) -> None:
        self.controller = controller
        self.sim_shim = sim_shim
        self.eef_site_name = eef_site_name
        self.eef_site_id = int(eef_site_id)
        self.arm_actuator_ids = list(arm_actuator_ids)
        self.arm_qpos_addrs = list(arm_qpos_addrs) if arm_qpos_addrs is not None else []
        self.gripper_actuator_ids = list(gripper_actuator_ids)
        self.model = model
        self.data = data
        # LIBERO trains at 20 Hz control with 500 Hz physics -> 25
        # physics substeps per policy action; a mismatch makes the OSC
        # under-/over-shoot its delta target every step.
        self.physics_substeps_per_control = max(1, int(physics_substeps_per_control))

        # Stateful gripper current_action, mirroring
        # PandaGripper.current_action (initialised to zeros; we init
        # directly as a 2-vector for the 2 fingers).
        self._gripper_current_action: np.ndarray = np.zeros(2, dtype=np.float64)

        # Pre-computed [-1, +1] -> [ctrl_lo, ctrl_hi] rescale per gripper
        # actuator (robosuite Manipulator.grip_action):
        #   ctrl = bias + weight * format_action_output,
        #   bias = 0.5*(hi+lo), weight = 0.5*(hi-lo).
        # ctrlrange is per-model immutable, so cached at install time.
        self._gripper_bias = np.array(
            [0.5 * (model.actuator_ctrlrange[gi, 1] + model.actuator_ctrlrange[gi, 0]) for gi in gripper_actuator_ids],
            dtype=np.float64,
        )
        self._gripper_weight = np.array(
            [0.5 * (model.actuator_ctrlrange[gi, 1] - model.actuator_ctrlrange[gi, 0]) for gi in gripper_actuator_ids],
            dtype=np.float64,
        )

        # Diagnostic gate: STRANDS_LIBERO_ACTION_LOG=1 emits one INFO
        # line per apply() for the first STRANDS_LIBERO_ACTION_LOG_MAX
        # (default 50) calls per episode.
        self._action_log_enabled = os.environ.get("STRANDS_LIBERO_ACTION_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            self._action_log_max = int(os.environ.get("STRANDS_LIBERO_ACTION_LOG_MAX", "50"))
        except ValueError:
            logger.warning(
                "STRANDS_LIBERO_ACTION_LOG_MAX=%r is not an integer; defaulting to 50",
                os.environ.get("STRANDS_LIBERO_ACTION_LOG_MAX"),
            )
            self._action_log_max = 50
        self._action_log_step: int = 0

    def reset(self) -> None:
        """Reset stateful per-episode controller state.

        Zeroes the gripper's ramp accumulator (without it the next
        episode inherits the previous episode's finger position, biasing
        every grasp) and the per-episode action-log step counter.
        """
        self._gripper_current_action.fill(0.0)
        self._action_log_step = 0

    @classmethod
    def from_sim(
        cls,
        sim: SimEngine,
        *,
        eef_site_name: str,
        arm_prefix: str,
        gripper_prefix: str,
    ) -> _LiberoOSCController:
        """Build a controller bound to ``sim``'s loaded LIBERO scene.

        Discovers the arm joints + qpos/qvel addresses, arm and gripper
        actuator IDs, the EEF site ID, and the actuator ctrlranges.

        Raises :class:`_ControllerDependencyMissing` when an optional
        dep (mujoco / robosuite) is unavailable or the sim has no
        compiled MuJoCo model - the caller degrades gracefully. Raises
        the base :class:`_ControllerInstallError` for a fixable setup
        failure (missing site / actuator IDs, wrong arm-joint count, or
        the known numba/coverage>=7 clash) - in strict mode the caller
        re-raises so the eval returns a structured error.
        """
        # Lazy imports - robosuite is a transitive dep via libero.
        # Genuinely-missing module -> _ControllerDependencyMissing
        # (graceful degrade); importable-but-broken (e.g. the
        # numba/coverage clash) -> _ControllerInstallError (surfaced).
        try:
            import mujoco as _mj
        except ModuleNotFoundError as e:
            raise _ControllerDependencyMissing(f"mujoco not available: {e}") from e
        except ImportError as e:
            if _is_numba_coverage_clash(e):
                raise _ControllerInstallError(f"mujoco import hit the numba/coverage clash: {e}") from e
            raise _ControllerDependencyMissing(f"mujoco import failed (treated as unavailable): {e}") from e
        try:
            from robosuite.controllers import (  # type: ignore[import-not-found]
                controller_factory,
                load_controller_config,
            )
            from robosuite.utils.binding_utils import MjSim  # type: ignore[import-not-found]
        except (ImportError, AttributeError) as e:
            # Robosuite OSC import chain failed: the numba/coverage>=7
            # clash is surfaced strictly (fixable environment problem);
            # anything else degrades like a missing optional dep.
            if _is_numba_coverage_clash(e):
                raise _ControllerInstallError(f"robosuite OSC import hit the numba/coverage clash: {e}") from e
            raise _ControllerDependencyMissing(
                f"robosuite OSC controller imports unavailable: {type(e).__name__}: {e}"
            ) from e

        world = getattr(sim, "_world", None)
        if world is None:
            # Only fires for non-MuJoCo backends / stub sims - a
            # "controller not applicable" degrade, not a setup bug.
            raise _ControllerDependencyMissing("sim has no _world")
        model = getattr(world, "_model", None)
        data = getattr(world, "_data", None)
        if model is None or data is None:
            raise _ControllerDependencyMissing("sim._world has no compiled MuJoCo model/data")

        # 1. Discover arm joints (robot0_joint1..7).
        arm_joint_ids: list[int] = []
        arm_qpos_addrs: list[int] = []
        arm_qvel_addrs: list[int] = []
        njnt = int(getattr(model, "njnt", 0))
        for i in range(njnt):
            jname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_JOINT, i)
            if not isinstance(jname, str) or not jname.startswith(arm_prefix):
                continue
            # Skip the gripper joints (different prefix; covered separately).
            if jname.startswith(gripper_prefix):
                continue
            arm_joint_ids.append(i)
            arm_qpos_addrs.append(int(model.jnt_qposadr[i]))
            # Each arm joint has 1 DoF (hinge), so qvel addr == joint id's
            # entry in jnt_dofadr. (For free joints this would be more
            # complex, but arm joints are hinges.)
            arm_qvel_addrs.append(int(model.jnt_dofadr[i]))
        if len(arm_joint_ids) != 7:
            # Not a RoboSuite/LIBERO Panda (or a different prefix):
            # "controller not applicable" - degrade, don't abort.
            raise _ControllerDependencyMissing(
                f"expected 7 arm joints with prefix {arm_prefix!r}, found {len(arm_joint_ids)}"
            )

        # 2. Discover arm actuator IDs (one per arm joint).
        arm_actuator_ids: list[int] = []
        nu = int(getattr(model, "nu", 0))
        for jid in arm_joint_ids:
            for ai in range(nu):
                if int(model.actuator_trnid[ai, 0]) == jid:
                    arm_actuator_ids.append(ai)
                    break
            else:
                raise _ControllerDependencyMissing(
                    f"no actuator found driving joint id={jid} (joint name "
                    f"{_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_JOINT, jid)!r})"
                )

        # 3. Discover gripper actuator IDs (any actuator with gripper_prefix
        #    in its name).
        gripper_actuator_ids: list[int] = []
        for ai in range(nu):
            aname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_ACTUATOR, ai)
            if isinstance(aname, str) and aname.startswith(gripper_prefix):
                gripper_actuator_ids.append(ai)
        if not gripper_actuator_ids:
            raise _ControllerDependencyMissing(f"no gripper actuators with prefix {gripper_prefix!r}")

        # 4. Verify EEF site exists.
        site_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, eef_site_name)
        if site_id < 0:
            raise _ControllerDependencyMissing(f"EEF site {eef_site_name!r} not found in model")

        # 5. Build robosuite MjSim shim around our model + data.
        # MjSim(model) creates its own fresh MjData, disconnected from
        # the buffer the eval is stepping - hot-patch sim_shim.data._data
        # to point at our actual data.
        sim_shim = MjSim(model)
        sim_shim.data._data = data

        # 6. Build OSC_POSE controller config + instance.
        controller_config = load_controller_config(default_controller="OSC_POSE")
        controller_config["robot_name"] = "Panda"
        controller_config["sim"] = sim_shim
        controller_config["eef_name"] = eef_site_name
        controller_config["joint_indexes"] = {
            "joints": arm_joint_ids,
            "qpos": arm_qpos_addrs,
            "qvel": arm_qvel_addrs,
        }
        # actuator_range is (low_arr, high_arr) for the ARM actuators
        # only - gripper is handled separately.
        ctrl_low = np.array(
            [float(model.actuator_ctrlrange[ai, 0]) for ai in arm_actuator_ids],
            dtype=np.float32,
        )
        ctrl_high = np.array(
            [float(model.actuator_ctrlrange[ai, 1]) for ai in arm_actuator_ids],
            dtype=np.float32,
        )
        controller_config["actuator_range"] = (ctrl_low, ctrl_high)

        # Home-qpos swap-and-restore: the OSC controller latches its
        # initial_joint / initial_ee_pos / goal_pos / goal_ori from
        # data.qpos at construction time, and upstream LIBERO constructs
        # it at the Panda ready pose (Robot.reset writes
        # MountedPanda.init_qpos BEFORE the controller factory) - so we
        # temporarily write the home pose around controller_factory and
        # restore the canonical qpos afterwards, or the controller's
        # frozen goal/nullspace state diverges from upstream and the
        # torques are wildly off. qvel is untouched (upstream doesn't
        # touch it either).
        snapshot_qpos: np.ndarray | None = None
        home_qpos = _resolve_libero_arm_home_qpos(len(arm_qpos_addrs))
        if home_qpos is not None:
            try:
                # Snapshot current canonical arm qpos so we can restore.
                snapshot_qpos = np.array(
                    [float(data.qpos[adr]) for adr in arm_qpos_addrs],
                    dtype=np.float64,
                )
                # Write home pose into arm joint addresses.
                for adr, v in zip(arm_qpos_addrs, home_qpos, strict=True):
                    data.qpos[adr] = float(v)
                # Update derived state (site_xpos, site_xmat, jacobians)
                # so the controller's __init__ ``update(force=True)``
                # reads home-pose values.
                _mj.mj_forward(model, data)
            except (AttributeError, IndexError, TypeError, ValueError) as e:
                # Defensive: if we can't snapshot, don't proceed with
                # the swap (would leak home pose into the sim).
                logger.debug(
                    "_LiberoOSCController.from_sim: failed to swap qpos to home pose for "
                    "controller construction (%s); falling back to construction-time state",
                    e,
                )
                snapshot_qpos = None

        controller = controller_factory("OSC_POSE", controller_config)

        # Restore the per-episode canonical qpos and re-forward so data
        # is byte-identical to its pre-swap state.
        if snapshot_qpos is not None:
            try:
                for adr, v in zip(arm_qpos_addrs, snapshot_qpos, strict=True):
                    data.qpos[adr] = float(v)
                _mj.mj_forward(model, data)
            except (AttributeError, IndexError, TypeError, ValueError) as e:
                # If restore fails the sim is in an unknown state;
                # raise loudly rather than silently leaving home pose.
                raise _ControllerInstallError(
                    f"failed to restore qpos after controller construction (snapshot was "
                    f"{snapshot_qpos.tolist()}): {e}. Sim may be in inconsistent state."
                ) from e

        # Physics-substeps-per-control from the sim's actual timestep:
        # LIBERO trains at 20 Hz control, so dt=0.002 (500 Hz physics)
        # gives 25 substeps, matching RoboSuite's standard step loop.
        dt = float(getattr(model.opt, "timestep", 0.002))
        substeps = max(1, int(round((1.0 / 20.0) / dt)))

        return cls(
            controller=controller,
            sim_shim=sim_shim,
            eef_site_name=eef_site_name,
            eef_site_id=int(site_id),
            arm_actuator_ids=arm_actuator_ids,
            arm_qpos_addrs=arm_qpos_addrs,
            gripper_actuator_ids=gripper_actuator_ids,
            model=model,
            data=data,
            physics_substeps_per_control=substeps,
        )

    def apply(
        self,
        action_dict: dict[str, Any],
        model: Any,
        data: Any,
        robot_name: str,  # noqa: ARG002 - kept for hook signature parity
    ) -> None:
        """Convert task-space delta-EEF action to joint torques + write data.ctrl.

        Reads ``x, y, z, roll, pitch, yaw, gripper`` from
        ``action_dict``; writes joint torques to
        ``data.ctrl[arm_actuator_ids]`` and the gripper command to
        ``data.ctrl[gripper_actuator_ids]``.

        Control-rate semantics: LIBERO trains at 20 Hz control with
        500 Hz physics -> 25 physics substeps per policy action,
        mirroring RoboSuite's step loop - ``set_goal(delta)`` once, then
        per substep ``run_controller()`` (which re-reads
        xpos/xmat/qpos/qvel and the Jacobian), ctrl writes, and
        ``mj_step``. Running OSC at the physics rate instead never
        converges. ``owns_stepping = True`` tells the engine not to
        ``mj_step`` again after this returns.

        Requires derived state (xpos/xmat) to be populated before the
        first call; the ``mj_forward`` in ``Simulation.load_scene``
        guarantees that. Best-effort against bad inputs: missing keys
        default to a no-op delta; shape mismatches log at WARNING and
        skip.
        """
        # Refresh sim_shim's view of data (controller reads from
        # sim_shim.data.qpos / xpos / xmat). MjSim shim wraps our
        # data by reference, so this is a no-op in practice but
        # makes the assumption explicit.
        self.controller.update()

        # Pack 6-dim Cartesian delta; missing keys default to 0 (no-op).
        # Each value may be a scalar or a list/array (GR00T packs all
        # channels to the training-data shape) - _to_scalar handles both.
        delta = np.array(
            [
                _to_scalar(action_dict.get("x", 0.0)),
                _to_scalar(action_dict.get("y", 0.0)),
                _to_scalar(action_dict.get("z", 0.0)),
                _to_scalar(action_dict.get("roll", 0.0)),
                _to_scalar(action_dict.get("pitch", 0.0)),
                _to_scalar(action_dict.get("yaw", 0.0)),
            ],
            dtype=np.float64,
        )
        # Default 0.5 (RLDS midway = no command) so an action dict
        # WITHOUT a gripper key produces no gripper movement: the
        # conversion below maps 0.5 -> 0 -> no ramp, holding the current
        # opening. (A 0.0 default would silently CLOSE the gripper on
        # every empty send_action({}).)
        gripper_value = _to_scalar(action_dict.get("gripper", 0.5))

        # RLDS -> robosuite/LIBERO gripper convention. The checkpoint
        # emits RLDS (0 = close, 1 = open); robosuite expects the
        # opposite sign (+1 = close, -1 = open). NVIDIA's bridge
        # (normalize + invert) combines to:
        #   gripper_out = -sign(2 * gripper_in - 1)
        #   0.0 (RLDS close) -> +1 (LIBERO close)
        #   0.5              ->  0 (no motion)
        #   1.0 (RLDS open)  -> -1 (LIBERO open)
        # Skipping this inverts every open/close command.
        gripper_value = -float(np.sign(2.0 * gripper_value - 1.0))

        # set_goal once per policy step. Subsequent run_controller
        # calls in the substep loop interpolate / hold this goal.
        try:
            self.controller.set_goal(delta)
        except Exception as e:  # noqa: BLE001 - log + skip rather than crash eval
            logger.warning(
                "_LiberoOSCController.apply: set_goal raised %s; this step's arm action will be no-op",
                e,
            )
            # Without a valid goal we still need to advance physics by the
            # full control timestep so the eval loop's timing is preserved
            # (otherwise sim time falls behind real time and benchmark
            # success criteria evaluated against ``cur_time`` go stale).
            import mujoco as mj

            for _ in range(self.physics_substeps_per_control):
                mj.mj_step(model, data)
            return

        # Cache mujoco module reference for the substep loop. Lazy import
        # is required because the OSC controller path is only exercised
        # under the `[sim-libero]` extra; the top-level adapter import
        # must work without mujoco available.
        import mujoco as mj

        n_arm = len(self.arm_actuator_ids)
        # Constant per-substep gripper ramp (see _GRIPPER_SPEED): +1
        # (close) ramps current_action toward [-1, +1], -1 (open) toward
        # [+1, -1]. Pre-computed so the inner loop is an add + clip.
        gripper_sign = float(np.sign(gripper_value))
        ramp_step = np.array([-1.0, 1.0]) * self._GRIPPER_SPEED * gripper_sign

        # Capture pre-step state for the diagnostic log (gated on
        # STRANDS_LIBERO_ACTION_LOG=1; zero cost otherwise).
        log_now = self._action_log_enabled and self._action_log_step < self._action_log_max
        if log_now:
            pre_eef_pos, pre_eef_quat = self._capture_eef_pose(data)
            pre_arm_ctrl = np.array([float(data.ctrl[ai]) for ai in self.arm_actuator_ids])
            pre_arm_qpos = (
                np.array([float(data.qpos[adr]) for adr in self.arm_qpos_addrs])
                if self.arm_qpos_addrs
                else np.zeros(n_arm)
            )
            pre_gripper_ctrl = np.array([float(data.ctrl[gi]) for gi in self.gripper_actuator_ids])
            pre_gripper_current = np.array(self._gripper_current_action)

        for _ in range(self.physics_substeps_per_control):
            # OSC: compute torques from current state (controller.update
            # is called inside run_controller via the new_update flag).
            try:
                torques = self.controller.run_controller()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "_LiberoOSCController.apply: run_controller raised %s; "
                    "leaving previous data.ctrl in place for this substep",
                    e,
                )
                torques = None

            if torques is not None:
                torques_arr = np.asarray(torques, dtype=np.float64)
                if torques_arr.shape[0] != n_arm:
                    logger.warning(
                        "_LiberoOSCController.apply: torques shape %s != %d arm actuators; skipping arm ctrl write",
                        torques_arr.shape,
                        n_arm,
                    )
                else:
                    for ai, tq in zip(self.arm_actuator_ids, torques_arr, strict=True):
                        data.ctrl[ai] = float(tq)

            # Stateful gripper ramp + bias/weight rescale, replicating
            # RoboSuite's Manipulator.grip_action per substep:
            #   current_action = clip(current_action + ramp_step, -1, 1)
            #   ctrl = bias + weight * current_action
            # Writing the raw scalar instead drives finger1 the wrong
            # way and breaks every grasp.
            self._gripper_current_action = np.clip(
                self._gripper_current_action + ramp_step,
                -1.0,
                1.0,
            )
            applied_gripper = self._gripper_bias + self._gripper_weight * self._gripper_current_action
            for gi, val in zip(self.gripper_actuator_ids, applied_gripper, strict=True):
                data.ctrl[gi] = float(val)

            mj.mj_step(model, data)

        # One structured ACTION_LOG line per apply() inside the captured
        # window: action keys, delta scale, gripper polarity, EEF
        # tracking, qpos/ctrl deltas.
        if log_now:
            post_eef_pos, post_eef_quat = self._capture_eef_pose(data)
            post_arm_ctrl = np.array([float(data.ctrl[ai]) for ai in self.arm_actuator_ids])
            post_arm_qpos = (
                np.array([float(data.qpos[adr]) for adr in self.arm_qpos_addrs])
                if self.arm_qpos_addrs
                else np.zeros(n_arm)
            )
            post_gripper_ctrl = np.array([float(data.ctrl[gi]) for gi in self.gripper_actuator_ids])
            post_gripper_current = np.array(self._gripper_current_action)
            eef_pos_delta = post_eef_pos - pre_eef_pos
            logger.info(
                "ACTION_LOG step=%d "
                "action_keys=%s "
                "delta=%s gripper_value=%.4f "
                "eef_pos_pre=%s eef_pos_post=%s eef_pos_delta=%s "
                "eef_quat_pre=%s eef_quat_post=%s "
                "arm_ctrl_pre=%s arm_ctrl_post=%s "
                "arm_qpos_pre=%s arm_qpos_post=%s "
                "gripper_ctrl_pre=%s gripper_ctrl_post=%s "
                "gripper_current_pre=%s gripper_current_post=%s",
                self._action_log_step,
                sorted(action_dict.keys()),
                np.round(delta, 6).tolist(),
                gripper_value,
                np.round(pre_eef_pos, 6).tolist(),
                np.round(post_eef_pos, 6).tolist(),
                np.round(eef_pos_delta, 6).tolist(),
                np.round(pre_eef_quat, 4).tolist(),
                np.round(post_eef_quat, 4).tolist(),
                np.round(pre_arm_ctrl, 4).tolist(),
                np.round(post_arm_ctrl, 4).tolist(),
                np.round(pre_arm_qpos, 4).tolist(),
                np.round(post_arm_qpos, 4).tolist(),
                np.round(pre_gripper_ctrl, 6).tolist(),
                np.round(post_gripper_ctrl, 6).tolist(),
                np.round(pre_gripper_current, 4).tolist(),
                np.round(post_gripper_current, 4).tolist(),
            )
            self._action_log_step += 1

    def _capture_eef_pose(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        """Read EEF position + (wxyz) quaternion from ``data`` for the log path.

        Position from ``data.site_xpos[eef_site_id]``; quaternion via
        ``mju_mat2Quat`` on ``data.site_xmat``. Returns zero-filled
        arrays when ``eef_site_id < 0`` (test-injected instances).
        """
        if self.eef_site_id < 0:
            return np.zeros(3), np.zeros(4)
        pos = np.array(data.site_xpos[self.eef_site_id], dtype=np.float64)
        xmat = np.asarray(data.site_xmat[self.eef_site_id], dtype=np.float64).reshape(9)
        quat = np.zeros(4, dtype=np.float64)
        # Lazy import - adapter must be importable without mujoco.
        import mujoco as mj

        mj.mju_mat2Quat(quat, xmat)
        return pos, quat


# Module-level cache: resolving the home pose imports the libero robot
# module (which transitively imports robosuite and loads MJCF assets);
# the result is immutable per-process, so cache after the first lookup.
_CACHED_LIBERO_HOME_QPOS: np.ndarray | None = None
_CACHED_LIBERO_HOME_QPOS_RESOLVED: bool = False


def _resolve_libero_arm_home_qpos(n_arm: int) -> np.ndarray | None:
    """Return the LIBERO-canonical Panda 7-DoF home pose, or ``None``.

    Resolution order:

    1. ``libero.libero.envs.robots.mounted_panda.MountedPanda().init_qpos``
       - stock LIBERO layout.
    2. ``libero.envs.robots.mounted_panda.MountedPanda().init_qpos``
       - LIBERO-PRO layout (flatter package structure); same canonical
       ``init_qpos`` byte-for-byte:
       ``[0, -0.161, 0, -2.4446, 0, 2.2268, pi/4]``.
    3. ``robosuite.models.robots.Panda().init_qpos`` (stock robosuite,
       slightly different: ``[0, pi/16, 0, -pi/2-pi/3, 0, pi-0.2, pi/4]``).
    4. ``None`` - caller falls back to the controller's
       construction-time default.

    This is the pose upstream's ``Robot.reset(deterministic=True)``
    writes to ``data.qpos`` before constructing the OSC controller, and
    the value the swap-and-restore in
    :meth:`_LiberoOSCController.from_sim` writes around
    ``controller_factory``.
    """
    global _CACHED_LIBERO_HOME_QPOS, _CACHED_LIBERO_HOME_QPOS_RESOLVED

    if _CACHED_LIBERO_HOME_QPOS_RESOLVED:
        if _CACHED_LIBERO_HOME_QPOS is None:
            return None
        if _CACHED_LIBERO_HOME_QPOS.shape != (n_arm,):
            # Different robot than the cached one (e.g. a 6-DoF arm).
            # Don't reuse the cached 7-DoF Panda value.
            return None
        return _CACHED_LIBERO_HOME_QPOS

    home: np.ndarray | None = None
    # Try both LIBERO package layouts. The class is identical between
    # them (same numerical init_qpos); only the import path differs.
    libero_module_paths = (
        "libero.libero.envs.robots.mounted_panda",  # stock libero
        "libero.envs.robots.mounted_panda",  # LIBERO-PRO
    )
    for module_path in libero_module_paths:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        mounted_panda_cls = getattr(module, "MountedPanda", None)
        if mounted_panda_cls is None:
            logger.debug(
                "%s exposes no MountedPanda; trying next libero layout",
                module_path,
            )
            continue
        try:
            # Instantiating triggers an MJCF asset load; the cost is
            # paid once per process thanks to the module-level cache.
            home = np.asarray(mounted_panda_cls().init_qpos, dtype=np.float64)
        except Exception as e:  # noqa: BLE001 - any failure soft-fall-through
            logger.debug(
                "MountedPanda(%s).init_qpos raised %s; trying next libero layout",
                module_path,
                e,
            )
            continue
        if home.shape != (n_arm,):
            logger.debug(
                "MountedPanda(%s).init_qpos shape %s does not match arm DoF %d; ignoring",
                module_path,
                home.shape,
                n_arm,
            )
            home = None
            continue
        # Successfully resolved.
        break

    if home is None:
        # Stock robosuite Panda fallback: init_qpos differs slightly
        # from MountedPanda's but is still closer to upstream than the
        # perturbed canonical pose.
        try:
            from robosuite.models.robots.manipulators.panda_robot import (  # type: ignore[import-not-found]
                Panda,
            )

            home = np.asarray(Panda().init_qpos, dtype=np.float64)
            if home.shape != (n_arm,):
                logger.debug(
                    "robosuite Panda.init_qpos shape %s does not match arm DoF %d; ignoring",
                    home.shape,
                    n_arm,
                )
                home = None
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug("robosuite Panda init_qpos lookup raised %s; no fallback", e)

    _CACHED_LIBERO_HOME_QPOS = home
    _CACHED_LIBERO_HOME_QPOS_RESOLVED = True
    return home


# Module-level cache for the Panda gripper init_qpos (mirrors the arm
# home pose cache above). Used by
# ``LiberoAdapter._write_libero_arm_home_qpos`` to set finger joints to
# upstream's open canonical position.
_CACHED_PANDA_GRIPPER_INIT_QPOS: np.ndarray | None = None
_CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED: bool = False


def _resolve_panda_gripper_init_qpos(n_finger: int) -> np.ndarray | None:
    """Return the LIBERO-canonical Panda gripper finger init qpos, or ``None``.

    Resolution order:

    1. ``robosuite.models.grippers.panda_gripper.PandaGripper().init_qpos``
       (used by both stock libero and LIBERO-PRO), which is
       ``[0.020833, -0.020833]`` - the canonical "open" position.
    2. ``None`` - caller leaves gripper qpos at MuJoCo's default (zero,
       fingers touching), which is far enough from the open pose to put
       the policy out-of-distribution on ``state.gripper``.

    Used by :meth:`LiberoAdapter._write_libero_arm_home_qpos`,
    mirroring upstream ``SingleArm.reset``.
    """
    global _CACHED_PANDA_GRIPPER_INIT_QPOS, _CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED

    if _CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED:
        if _CACHED_PANDA_GRIPPER_INIT_QPOS is None:
            return None
        if _CACHED_PANDA_GRIPPER_INIT_QPOS.shape != (n_finger,):
            return None
        return _CACHED_PANDA_GRIPPER_INIT_QPOS

    init_qpos: np.ndarray | None = None
    try:
        from robosuite.models.grippers.panda_gripper import (  # type: ignore[import-not-found]
            PandaGripper,
        )

        init_qpos = np.asarray(PandaGripper().init_qpos, dtype=np.float64)
        if init_qpos.shape != (n_finger,):
            logger.debug(
                "PandaGripper.init_qpos shape %s does not match finger count %d; ignoring",
                init_qpos.shape,
                n_finger,
            )
            init_qpos = None
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.debug("PandaGripper.init_qpos lookup raised %s; no fallback", e)

    _CACHED_PANDA_GRIPPER_INIT_QPOS = init_qpos
    _CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED = True
    return init_qpos


def _to_scalar(value: Any) -> float:
    """Coerce a GR00T-LIBERO action channel to a scalar float.

    GR00T sends every action key list-shaped (training-data packing);
    this centralises the coercion for all 7 channels:

    * scalar -> ``float(value)``
    * non-empty list / tuple / ndarray -> ``float(value[0])``
    * anything else -> ``0.0`` with a WARNING log.
    """
    try:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            return float(value[0])
        return float(value)
    except (TypeError, ValueError, IndexError) as e:
        logger.warning(
            "_LiberoOSCController._to_scalar: could not coerce action value %r to float (%s); "
            "treating as 0.0 for this step",
            value,
            e,
        )
        return 0.0


__all__ = [
    "BDDLParseError",
    "LiberoAdapter",
]
