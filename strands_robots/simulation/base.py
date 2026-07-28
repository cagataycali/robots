"""Simulation ABC - backend-agnostic interface for all simulation engines.

Every simulation backend (MuJoCo, Isaac, Newton) implements this interface.
Agent tools and the Robot() factory interact through these methods only -
they never touch backend-specific APIs directly.

Usage::

    from strands_robots.simulation import Simulation  # returns MuJoCo by default
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import math
import numbers
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, SupportsFloat, cast

if TYPE_CHECKING:
    import numpy as np

    from strands_robots.policies import Policy
    from strands_robots.rendering import CameraParams

# PolicyRunner and VideoConfig are imported at module level: policy_runner
# only imports SimEngine under TYPE_CHECKING, so no runtime cycle exists.
# ``OnFrame`` is deliberately NOT imported (even under TYPE_CHECKING) because
# cyclic-import linters walk TYPE_CHECKING blocks too; evaluate_benchmark
# references it as a *string* annotation instead (a no-op at runtime under
# ``from __future__ import annotations``).
from strands_robots.simulation.policy_runner import PolicyRunner, VideoConfig
from strands_robots.utils import positive_finite_number_error

logger = logging.getLogger(__name__)


# Robot-setup keyword arguments that identify a caller who confused a backend
# *constructor* with the robot-setup entry points. Backend constructors accept
# ``**kwargs`` as a cross-backend forward-compatibility sink, but that sink
# must not silently swallow an argument naming a robot to set up - these are
# rejected loudly instead of failing far downstream with a "No world" error.
_SETUP_KWARGS: tuple[str, ...] = ("robot_name", "robot")


def reject_setup_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Reject robot-setup keyword arguments passed to a backend constructor.

    A backend ``__init__`` accepts ``**kwargs`` only as a forward-compatibility
    sink; a robot-setup argument there would be silently dropped and fail far
    downstream.

    Raises:
        TypeError: If ``kwargs`` names a robot-setup argument. The message
            points at the ``Robot(name, mode="sim")`` factory and the
            ``create_world()`` + ``add_robot(name)`` sequence.
    """
    offending = [k for k in _SETUP_KWARGS if k in kwargs]
    if not offending:
        return
    names = ", ".join(repr(k) for k in offending)
    raise TypeError(
        f"Simulation backend constructor does not accept {names}: a constructor "
        'builds an empty engine, not a robot. Use Robot("so101", mode="sim") for '
        'one-step setup, or create_world() then add_robot("so101").'
    )


def unknown_kwargs_error(method: str, kwargs: Mapping[str, Any], accepted: Sequence[str]) -> dict[str, Any] | None:
    """Return a tool-envelope error for keyword arguments a method cannot use.

    Methods with a *discarding* ``**kwargs`` sink call this so a misspelled or
    invented parameter is named rather than swallowed as a successful no-op.
    (*Forwarding* sinks, whose keys belong to the callee, do not.)

    Args:
        method: Method (and action) name to quote in the message.
        kwargs: The residual keyword arguments the method would otherwise drop.
        accepted: Every keyword the method honors, including any it reads out
            of its own ``**kwargs``. Also listed as the "Valid:" hint.

    Returns:
        ``None`` when every key in ``kwargs`` is accepted, otherwise a
        ``status="error"`` result dict naming the unusable keys. An error dict
        rather than a raised exception because these methods are dispatched as
        agent tool actions, which must not raise past dispatch.
    """
    unexpected = sorted(k for k in kwargs if k not in accepted)
    if not unexpected:
        return None
    return {
        "status": "error",
        "content": [{"text": (f"Unknown parameter(s) {unexpected} for action '{method}'. Valid: {sorted(accepted)}")}],
    }


def randomization_range_error(value: Any, param: str, *, allow_zero: bool = True) -> str | None:
    """Return why a ``(lo, hi)`` randomization range cannot be applied.

    Domain randomization multiplies live physics constants (body mass, geom
    friction) and re-samples colours inside a caller-supplied range. A range
    that is not a pair of finite numbers with ``0 <= lo <= hi`` has no sampling
    interval a backend could draw from: the sampler either raises deep inside
    the mutation loop or, worse, succeeds and installs a physically impossible
    constant - a negative body mass falls *upward* under gravity and a negative
    friction coefficient is not a Coulomb model. Either way the randomized world
    no longer models anything, so the request is refused up front.

    Args:
        value: The candidate ``(lo, hi)`` pair.
        param: Parameter name to quote in the message (``"mass_range"``).
        allow_zero: Whether ``lo == 0`` is meaningful for this quantity. True
            for the ranges where zero is a real physical setting (a frictionless
            surface, a black colour channel); pass False for a multiplicative
            mass scale, where a zero multiplier leaves a massless body that
            ignores gravity instead of a lighter one.

    Returns:
        ``None`` when the range is usable, otherwise the reason as a string.
    """
    try:
        lo, hi = value
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return f"{param} must be a (lo, hi) pair of numbers, got {value!r}"
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{param} bounds must be finite, got {value!r}"
    if lo > hi:
        return f"{param} lower bound {lo} exceeds upper bound {hi}"
    if allow_zero:
        if lo < 0:
            return f"{param} bounds must be non-negative, got {value!r}"
    elif lo <= 0:
        detail = (
            "a zero scale erases the quantity it multiplies"
            if lo == 0
            else "a negative scale flips the sign of the quantity it multiplies"
        )
        return f"{param} bounds must be positive, got {value!r} ({detail})"
    return None


def finite_non_negative_error(value: Any, param: str, context: str) -> str | None:
    """Return why a magnitude parameter cannot be used as a noise/offset scale.

    Shared by the sensor-noise standard deviations and the position-jitter
    amplitude: all of them are half-widths or standard deviations, so a
    non-numeric, non-finite or negative value describes no distribution. A
    NaN amplitude propagates into ``qpos`` and poisons the whole physics state
    on the next step, and a negative half-width inverts the sampling bounds.

    Args:
        value: The candidate magnitude.
        param: Parameter name to quote in the message.
        context: Method name to prefix the message with.

    Returns:
        ``None`` when the value is a finite non-negative number, otherwise the
        reason as a string.
    """
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return f"{context}: {param} must be a number, got {value!r}"
    if not math.isfinite(fvalue) or fvalue < 0:
        return f"{context}: {param} must be a finite non-negative number, got {value!r}"
    return None


def randomization_seed_error(value: Any, context: str) -> str | None:
    """Return why a value cannot seed a reproducible randomization stream.

    The seed reaches ``numpy.random.default_rng``, which accepts only
    non-negative integers (and a few RNG objects the ``int | None`` annotations
    on these methods do not advertise). A float or string seed raises there -
    on the sensor-noise path not until the first observation is drawn, long
    after the configuring call reported success - so it is rejected at the call
    that supplied it.

    Args:
        value: The candidate seed (``None`` selects fresh entropy).
        context: Method name to prefix the message with.

    Returns:
        ``None`` when the seed is usable, otherwise the reason as a string.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return f"{context}: seed must be a non-negative integer or None, got {value!r} (None draws fresh entropy)"
    if int(value) < 0:
        return f"{context}: seed must be a non-negative integer or None, got {value!r} (None draws fresh entropy)"
    return None


def _non_finite_action_error(where: str, value: float) -> dict[str, Any]:
    """Structured rejection for a ``nan`` / ``inf`` action value.

    ``_coerce_action`` validated that every value *coerces* to a float but not
    that it is finite, so a ``nan`` was written straight into ``data.ctrl`` and
    the call still returned ``status="success"`` / "Action applied". MuJoCo then
    printed ``Nan, Inf or huge value in CTRL at ACTUATOR n. The simulation is
    unstable.`` on stderr and zeroed the control internally - so the robot did
    not move, nothing in the tool result said so, and the operator had only a
    stderr line to go on. A policy emitting ``nan`` (an un-normalised observation
    or a diverged checkpoint) is the common way to hit this, which is exactly
    when a silent no-op is most misleading.
    """
    return {
        "status": "error",
        "content": [
            {
                "text": (
                    f"send_action: action value for {where} is not finite ({value}). "
                    "A non-finite control is rejected rather than applied: MuJoCo "
                    "would report the simulation unstable and discard it, so the "
                    "robot would not move. This usually means the policy emitted "
                    "nan/inf - check the observation normalisation and the "
                    "checkpoint."
                )
            }
        ],
    }


class SimEngine(ABC):
    """Abstract base class for simulation engines.

    Defines the contract all backends (MuJoCo, Isaac, Newton) implement. This
    is the *programmatic* API - the AgentTool layer wraps it with
    tool_spec/stream for LLM access.

    Method categories: **required** (``@abstractmethod`` - the core simulation
    loop every backend must implement), **provided** (policy orchestration
    such as ``run_policy`` / ``eval_policy``, implemented once here as a
    facade over the abstract primitives; backends may override to optimise),
    and **optional** (default raises ``NotImplementedError``; backends opt in
    by overriding).

    Lifecycle::

        sim = SomeEngine()
        sim.create_world()
        sim.add_robot("so100", data_config="so100")
        sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        obs = sim.get_observation("so100")
        sim.send_action({"joint_0": 0.5}, robot_name="so100")
        sim.step(n_steps=10)
        result = sim.render(camera_name="default")
        sim.destroy()
    """

    def _init_ros_bridge(self, *, ros2_bridge: bool = False, ros2_domain: int = 0) -> None:
        """Initialize the optional ROS 2 telemetry bridge state.

        Backends that accept a ``ros2_bridge`` flag call this once from their
        own ``__init__`` (a plain method, not an ABC ``__init__`` override, so
        subclasses need not thread ``super().__init__()`` through).

        Args:
            ros2_bridge: When True, publish per-robot ``joint_states`` and
                camera ``image_raw`` on a ROS 2 domain every :meth:`step`.
                Requires ``rclpy``; an :class:`ImportError` is raised here if
                it is missing. Defaults to False - the sim never touches ROS 2.
            ros2_domain: ROS 2 domain id (``ROS_DOMAIN_ID``) to publish on.
        """
        self._ros2_bridge_enabled = bool(ros2_bridge)
        self._ros2_domain = int(ros2_domain)
        self._ros_bridge: Any = None
        if self._ros2_bridge_enabled:
            from strands_robots.simulation.ros_bridge import SimRosBridge

            self._ros_bridge = SimRosBridge(domain_id=self._ros2_domain)

    def _publish_ros_telemetry(self, *, skip_images: bool = False) -> None:
        """Publish joint_states (and camera images) for every robot once.

        No-op when the ROS 2 bridge is disabled or was never initialized.
        Called by backends from :meth:`step` after the physics tick. Per-robot
        failures (e.g. a camera that did not render) never interrupt the loop.
        """
        bridge = getattr(self, "_ros_bridge", None)
        if bridge is None:
            return
        for robot in self.list_robots():
            # Per-robot guard: a transient render/observation failure on one
            # robot must not interrupt the loop or crash the caller's step().
            # Publish what succeeds, log-and-continue on the rest.
            try:
                obs = self.get_observation(robot, skip_images=skip_images)
                names = self.robot_joint_names(robot)
                positions = [obs[j] for j in names if j in obs and isinstance(obs[j], (int, float))]
                bridge.publish_joint_states(robot, names, positions)
                if skip_images:
                    continue
                for key, value in obs.items():
                    if key in names:
                        continue
                    if hasattr(value, "ndim") and getattr(value, "ndim", 0) == 3:
                        bridge.publish_image(robot, key, value)
            except Exception:
                logger.warning(
                    "ROS 2 telemetry publish failed for robot %r; skipping this robot for this step",
                    robot,
                    exc_info=True,
                )
                continue

    def _shutdown_ros_bridge(self) -> None:
        """Tear down the ROS 2 bridge if one is active. Safe to call repeatedly."""
        bridge = getattr(self, "_ros_bridge", None)
        if bridge is not None:
            bridge.shutdown()
            self._ros_bridge = None

    def _resolve_single_robot(self, robot_name: str | None) -> str:
        """Resolve an optional robot name to a concrete one.

        An explicit name is returned unchanged. ``None`` resolves to the sole
        robot when exactly one exists.

        Raises:
            ValueError: When ``robot_name`` is None and the scene has zero
                robots or several (the message lists the candidates).
        """
        if robot_name is not None:
            return robot_name
        names = self.list_robots()
        if len(names) == 1:
            return names[0]
        if len(names) == 0:
            raise ValueError("No robots registered in the simulation. Add a robot first (add_robot or Robot factory).")
        raise ValueError(f"Multiple robots registered; specify robot_name. Available: {names}")

    def _unknown_robot_msg(self, requested: str) -> str:
        """Actionable 'robot not found' message for the backend-agnostic facade.

        Keeps the "Robot 'X' not found." prefix (the error shape the concrete
        backends also emit), then appends a close-match suggestion, the robots
        currently in the world, and the discovery action so a typo is
        recoverable in zero extra calls.
        """
        known = self.list_robots()
        msg = f"Robot '{requested}' not found."
        if known:
            matches = difflib.get_close_matches(requested, known, n=3, cutoff=0.4)
            if matches:
                msg += " Did you mean: " + ", ".join(matches) + "?"
            msg += f" Available robots: {known}. Use action='list_robots' to see all."
        else:
            msg += " No robots in the scene; add one with action='add_robot'."
        return msg

    # World lifecycle

    @abstractmethod
    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float = 1.0,
    ) -> dict[str, Any]:
        """Create a new simulation world.

        ``terrain`` (``"rough"`` = value-noise bumps, ``"stairs"`` = step
        plateaus rising along +x, ``"pyramid"`` = concentric step plateaus
        rising toward the centre, ``"slope"`` = a constant-grade ramp; see
        :mod:`strands_robots.simulation.terrain`) lays down a deterministic
        heightfield instead of the flat ground plane. Only meaningful when
        ``ground_plane=True``; defaults to ``None`` (a flat plane). Backends
        without heightfield support reject a non-None ``terrain`` with an
        actionable error rather than silently ignoring it.

        ``difficulty`` scales the terrain's peak elevation (``1.0`` = full
        height); must be finite and ``> 0``. Setting ``difficulty != 1.0``
        with no ``terrain`` is rejected rather than silently ignored.

        A floating-base robot added to a terrain world is spawned seated on
        the local terrain surface at ``add_robot`` and on ``reset()``.

        ``timestep`` (seconds) and ``gravity`` must be values the engine can
        honor, on the same terms the ``set_timestep`` / ``set_gravity``
        setters enforce: ``timestep`` a finite number ``> 0`` (``0`` is
        rejected, never coalesced to the engine default), ``gravity`` a
        3-element vector of finite numbers or a real scalar taken as the
        z-component. An unusable value is rejected with a structured error
        rather than compiled into the world. ``None`` means "use the engine
        default".
        """
        ...

    @abstractmethod
    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources."""
        ...

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset simulation to its initial state.

        Contract: on return the world must be fully consistent and
        observation-ready - derived kinematics (body/site/geom poses, camera
        transforms) must reflect the reset pose WITHOUT requiring a subsequent
        ``step()``, since ``eval_policy`` calls ``get_observation()``
        immediately after ``reset()``. A per-robot home pose captured from an
        ``add_robot(keyframe=...)`` spawn must be re-applied, so a keyframe
        pose survives a reset instead of collapsing to the zero configuration.
        """
        ...

    @abstractmethod
    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps."""
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary."""
        ...

    # Robot management

    @abstractmethod
    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        keyframe: str | int | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        ``keyframe`` (name or index) optionally spawns the robot in a
        canonical pose declared by a ``<keyframe>`` in its source model
        instead of the default all-zero configuration; the pose is sticky
        across :meth:`reset`. An unknown keyframe is a hard error naming the
        available keyframes - it never silently falls back to zeros.
        """
        ...

    @abstractmethod
    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation."""
        ...

    @abstractmethod
    def list_robots(self) -> list[str]:
        """Return ordered list of robot names currently in the world.

        Used by the backend-agnostic ``PolicyRunner`` to resolve a
        default robot when the caller omits ``robot_name``.
        """
        ...

    @abstractmethod
    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered joint names for ``robot_name``.

        Used by ``Policy.set_robot_state_keys`` to name the
        ``observation.state`` vector. Action-vector binding (``send_action``
        with a numeric vector, ``PolicyRunner.replay``) uses
        :meth:`robot_action_keys` instead - a robot's actuators are not always
        its joints. Order must match the backend's joint ordering.
        """
        ...

    def robot_action_keys(self, robot_name: str) -> list[str]:
        """Return the action keys ``send_action`` resolves for ``robot_name``.

        These are the names a policy should emit as its action-dict keys: the
        robot's *actuators*, which are NOT always its joints (passive/mimic
        joints have no driving actuator; tendon actuators are not joints).
        The default mirrors :meth:`robot_joint_names`; backends with a
        distinct actuator namespace override this to return the actuator
        short-names instead.
        """
        return self.robot_joint_names(robot_name)

    def bind_policy_sim_context(self, policy: Any, robot_name: str) -> None:
        """Give a policy the backend sim context it needs to close the loop.

        Default no-op. The MuJoCo engine overrides this to hand opt-in
        policies (those exposing ``set_sim_context``) the compiled model plus
        the robot's namespace so IK-based policies auto-configure; other
        policies are unaffected.
        """
        return None

    def _maybe_install_wbc_torque_control(self, policy: Any, robot_name: str) -> Callable[[], None] | None:
        """Hook: auto-install an action controller a policy needs to run correctly.

        Default no-op (returns ``None``). The MuJoCo engine overrides this to
        wire up the WBC torque shim on position-servo scenes. Returns an
        optional zero-arg cleanup callable that :meth:`run_policy` invokes in
        a ``finally`` block to restore the scene after the rollout.
        """
        return None

    def _preflight_policy_config(
        self,
        robot_name: str,
        policy_provider: str,
        policy_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Run a provider's pre-construction preflight before ``create_policy``.

        Resolves the provider's policy class WITHOUT instantiating it and runs
        its :meth:`~strands_robots.policies.base.Policy.preflight` hook
        against the runtime observation keys, catching a misconfiguration
        BEFORE the expensive model-weight download.

        Returns:
            A ``status=error`` dict (for the caller to return) when the
            provider's preflight rejects the configuration; ``None`` when the
            check passes, is a no-op, or the observation is not yet available.
        """
        from strands_robots.policies import preflight_policy

        obs = self.get_observation(robot_name)
        if not isinstance(obs, dict) or not obs:
            return None
        try:
            preflight_policy(policy_provider, set(obs.keys()), **(policy_config or {}))
        except ValueError as e:
            return {"status": "error", "content": [{"text": str(e)}]}
        return None

    # Object management

    @abstractmethod
    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool = False,
        mesh_path: str | None = None,
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a primitive or mesh object to the scene.

        Returns an agent-tool status dict. The ``size`` convention is
        backend-specific -- MuJoCo treats ``size`` as the **full extent in
        meters** per axis (halved internally to half-extents), whereas Newton
        consumes half-extents / radii directly; see the concrete backend's
        docstring for per-shape semantics.

        A backend MUST NOT discard ``size`` or ``color`` components the
        caller supplied: a short vector is either rejected or padded only in
        the missing *trailing* components from a documented default (MuJoCo
        completes an RGB triple with an opaque alpha and rejects every other
        count). Replacing the vector with a backend default would build a
        different object while reporting success.

        ``mass`` must be a finite number greater than zero for a dynamic
        object. A backend MUST NOT establish a body on a mass its own
        ``set_body_properties`` would refuse: a non-finite mass makes the first
        integration step produce ``nan`` and, because the solver shares one
        state vector, poisons every other body in the world too.

        ``material`` (optional): backend-specific visual material/texture
        spec. ``None`` keeps the flat ``color`` rgba. Backends that do not
        support it must reject a non-``None`` ``material`` loudly, and a
        supporting backend must reject material keys it cannot honor instead
        of dropping them.
        """
        ...

    @abstractmethod
    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene."""
        ...

    # Observation / Action

    @abstractmethod
    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get full observation for a robot: joint state + all attached cameras.

        Unified observation consumed by :class:`Policy` and
        :class:`~strands_robots.simulation.policy_runner.PolicyRunner`.
        Backends MUST return a dict with the following schema; extra keys
        are allowed.

        Schema:
            - ``"<joint_name>"`` (float): One entry per joint, keyed by the
              *short* joint name (e.g. ``"shoulder_pan"``), regardless of
              multi-robot namespacing at the physics-engine level.
            - ``"<camera_name>"`` (np.ndarray): One RGB uint8 frame of shape
              ``(H, W, 3)`` per camera associated with the robot. Cameras
              whose render fails MAY be omitted; joint state MUST still be
              returned.
            - Floating base: a robot whose root is a 6-DoF free joint does
              NOT report that joint as a scalar entry; it surfaces the full
              base pose + twist as ``"base_pos"`` (world x,y,z),
              ``"base_quat"`` (w,x,y,z), ``"base_lin_vel"`` and
              ``"base_ang_vel"``, matching :meth:`get_robot_state`'s
              ``"base"`` entry. Absent for fixed-base arms.

        Single-camera rendering is :meth:`render`'s job. For batched
        multi-robot observation, add a separate ``get_observations`` method -
        do NOT extend this one.

        Args:
            robot_name: Which robot to observe. If ``None`` and exactly one
                robot exists, that robot is used; otherwise returns ``{}``.

        Returns:
            Observation dict per schema above. Returns ``{}`` if the world
            is not yet created or ``robot_name`` is unknown.
        """
        ...

    def _ground_height_at(self, x: float, y: float) -> float:
        """Terrain surface height (world z) beneath world ``(x, y)``; ``0.0`` on flat ground.

        Default ``0.0`` -- a flat ground plane, and any backend without a
        heightfield. Backends with terrain override this so height-based
        locomotion predicates measure clearance above the *local* ground
        rather than an absolute world z. Not a public tool action.
        """
        return 0.0

    def get_ground_height(self, x: SupportsFloat, y: SupportsFloat) -> dict[str, Any]:
        """Query the terrain surface height (world z) beneath world ``(x, y)``.

        Public counterpart of :meth:`_ground_height_at`; use it to place an
        object / camera / goal *on* a ``create_world(terrain=...)``
        heightfield surface instead of buried in it.

        Returns ``0.0`` for a flat ground plane, for any backend without a
        heightfield, and before ``create_world`` - a non-terrain or
        not-yet-built world reports a flat surface rather than raising.

        Args:
            x: World x coordinate; any finite real scalar (``SupportsFloat``,
                including NumPy scalars).
            y: World y coordinate. Same accepted types as ``x``.

        Returns:
            Agent-tool status dict. On success ``content`` carries a
            ``{"json": {"x": ..., "y": ..., "height": ...}}`` block with the
            surface height in meters. Errors when ``x`` / ``y`` is not a
            finite real number.
        """
        for label, val in (("x", x), ("y", y)):
            if isinstance(val, bool) or not isinstance(val, numbers.Real) or not math.isfinite(float(val)):
                return {
                    "status": "error",
                    "content": [{"text": f"get_ground_height: {label} must be a finite number, got {val!r}."}],
                }
        fx, fy = float(x), float(y)
        height = float(self._ground_height_at(fx, fy))
        return {
            "status": "success",
            "content": [
                {"text": f"Ground height at ({fx:.4f}, {fy:.4f}) = {height:.4f}m"},
                {"json": {"x": fx, "y": fy, "height": height}},
            ],
        }

    def _coerce_action(
        self, action: dict[str, Any] | Sequence[float], robot_name: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Normalize an action into a ``{joint/actuator name: value}`` mapping.

        An ordered vector (``list`` / ``tuple`` / 1-D array) is zipped against
        ``robot_action_keys(robot_name)`` in declaration order - the robot's
        *actuator* keys, not its joint names. The vector length must match the
        actuator count exactly; a mismatch is reported as a caller error
        rather than silently truncated. A mapping is returned unchanged once
        every value is confirmed to coerce to a scalar float; a
        *single-element* sequence/array value is unwrapped to its scalar,
        while a *multi-element* value is rejected with an actionable error
        rather than raised deep in the actuator-application loop.

        Returns:
            An ``(action_dict, error)`` tuple. When ``error`` is non-None it
            is a structured ``{"status": "error", ...}`` dict and
            ``action_dict`` must be ignored. Otherwise ``action_dict`` is the
            normalized mapping.
        """
        if isinstance(action, Mapping):
            # Values are applied downstream as ``float(value)`` with no guard,
            # so validate every value coerces to a scalar float up front and
            # reject the whole action atomically (a mid-loop TypeError would
            # crash after partially applying the earlier keys). A length-1
            # sequence/array value carries exactly one scalar and is
            # unambiguous (the documented ``list[float]`` shape of
            # ``Policy.get_actions`` for a 1-DOF key), so unwrap it instead of
            # rejecting; multi-element values are still rejected atomically.
            normalized: dict[str, Any] = {}
            for key, value in action.items():
                if (
                    not isinstance(value, (str, bytes, Mapping))
                    and hasattr(value, "__len__")
                    and hasattr(value, "__getitem__")
                    and len(value) == 1
                ):
                    value = value[0]
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None, {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"send_action: action value for key '{key}' must be a "
                                    "scalar number (one value per actuator/joint), got "
                                    f"{type(value).__name__}."
                                )
                            }
                        ],
                    }
                if not math.isfinite(numeric):
                    return None, _non_finite_action_error(f"key '{key}'", numeric)
                normalized[key] = value
            return normalized, None

        # ``str``/``bytes`` are iterable but never a valid multi-joint action;
        # a scalar has no length. Reject both with an actionable message instead
        # of producing garbage character/positional keys downstream.
        if isinstance(action, (str, bytes)) or not hasattr(action, "__len__"):
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "send_action: 'action' must be a mapping of "
                            "{joint/actuator name: value} or an ordered numeric "
                            f"vector, got {type(action).__name__}."
                        )
                    }
                ],
            }

        try:
            values = [float(v) for v in action]
        except (TypeError, ValueError) as exc:
            return None, {
                "status": "error",
                "content": [{"text": f"send_action: action vector has a non-numeric entry: {exc}."}],
            }
        for index, numeric in enumerate(values):
            if not math.isfinite(numeric):
                return None, _non_finite_action_error(f"index {index}", numeric)

        action_keys = self.robot_action_keys(robot_name)
        if len(values) != len(action_keys):
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"send_action: action vector length {len(values)} does not "
                            f"match robot '{robot_name}' action-key count {len(action_keys)}. "
                            f"Action keys (in order): {action_keys}. Pass a {{name: value}} "
                            "mapping to target a subset of actuators."
                        )
                    }
                ],
            }
        return {name: value for name, value in zip(action_keys, values)}, None

    @abstractmethod
    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Apply action and advance physics by n_substeps.

        Contract: each call writes actuator/ctrl values and then runs
        ``n_substeps`` physics steps (e.g. mj_step). PolicyRunner.run()
        relies on this - it calls send_action once per control step and
        does NOT call sim.step() separately.

        Backends are responsible for internal thread-safety (e.g.
        MuJoCo acquires self._lock here). PolicyRunner does not manage
        locks.

        Returns:
            Dict with ``status`` and ``content``. When action keys cannot
            be resolved, the ``content`` list includes a ``json`` block with
            ``unresolved_keys`` so callers can self-correct.
        """
        ...

    def physics_timestep(self) -> float | None:
        """Return the physics integration timestep in seconds, or ``None``.

        Used by :class:`PolicyRunner` to convert a policy's ``control_frequency``
        into the number of physics substeps per control step
        (``round(1 / control_frequency / physics_timestep)``) so a
        position-servo robot actually tracks each action's target before the
        next action overwrites ``ctrl``. Backends that cannot report a fixed
        timestep return ``None`` and the runner falls back to ``n_substeps=1``.
        """
        return None

    # Rendering

    @abstractmethod
    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        """Render a camera view.

        Returns an agent-tool dict with ``status`` and a ``content`` list. On
        success the content holds an ``image`` block carrying PNG bytes
        (``{"image": {"format": "png", "source": {"bytes": ...}}}``); the raw
        RGB ``numpy`` arrays are available per-camera via :meth:`get_observation`.
        Resolution comes from the named camera's configuration (set via
        ``add_camera``) unless ``width``/``height`` are given; the free camera
        and model-only cameras fall back to the engine default.
        """
        ...

    # Policy orchestration (concrete facade, not abstract)

    @staticmethod
    def _resolve_horizon(
        n_steps: int | None,
        max_steps: int | None,
        control_frequency: float,
        duration: float,
        method: str = "run_policy",
    ) -> tuple[float, int | None, dict[str, Any] | None]:
        """Resolve a step horizon into a wall-clock duration.

        ``n_steps`` (primary) or the legacy ``max_steps`` alias specify the
        rollout length as a step count;
        ``duration = n_steps / control_frequency``. ``n_steps`` wins when both
        are passed. A non-positive horizon is reported as a caller error. The
        fallback ``duration`` is returned unchanged when no horizon is given,
        so the caller must still validate it with :meth:`_validate_duration`.

        Returns:
            A ``(duration, n_steps, error)`` tuple. When ``error`` is non-None
            it is a structured ``{"status": "error", ...}`` dict and the other
            fields must be ignored. Otherwise ``duration`` is the resolved
            wall-clock duration and ``n_steps`` the normalized step count (or
            ``None``).
        """
        if n_steps is None and max_steps is not None:
            n_steps = int(max_steps)
        if n_steps is not None:
            if n_steps <= 0:
                return (
                    duration,
                    n_steps,
                    {
                        "status": "error",
                        "content": [{"text": f"{method}: n_steps must be > 0, got {n_steps}."}],
                    },
                )
            # control_frequency is validated as a positive number at the public
            # entry points (run_policy / start_policy / eval_policy) via
            # _validate_positive_frequency before this helper runs, so the
            # division below is safe.
            duration = float(n_steps) / float(control_frequency)
        return duration, n_steps, None

    @staticmethod
    def _validate_action_horizon(
        action_horizon: Any, method: str, param: str = "action_horizon"
    ) -> dict[str, Any] | None:
        """Reject a non-positive-integer ``action_horizon`` at the public API.

        ``action_horizon`` is how many actions are consumed from each policy
        chunk before re-querying. A value below 1 (or a non-int) is meaningless
        and would otherwise be silently clamped to 1 by
        :func:`~strands_robots.policies.base.resolve_chunk_length`, hiding the
        caller's mistake behind a rollout that does not run the requested
        horizon. Returns a structured ``{"status": "error", ...}`` dict to
        surface, or ``None`` when the value is valid.

        Args:
            action_horizon: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.
            param: Parameter label for the error message. Multi-robot drivers
                accept a ``{robot_name: horizon}`` mapping and pass
                ``"action_horizon['alice']"`` so the message names the entry the
                caller got wrong rather than the whole mapping.

        Returns:
            A structured ``{"status": "error", ...}`` dict naming the
            offending parameter, or ``None`` when the value is valid.
        """
        if not isinstance(action_horizon, int) or action_horizon < 1:
            return {
                "status": "error",
                "content": [{"text": f"{method}: {param} must be a positive integer, got {action_horizon!r}."}],
            }
        return None

    @staticmethod
    def _validate_per_robot_mapping(
        mapping: Mapping[Any, Any], driven: Iterable[str], param: str, method: str
    ) -> dict[str, Any] | None:
        """Reject a per-robot mapping key that names no robot in this call.

        Multi-robot drivers accept ``{robot_name: value}`` overrides alongside
        the ``policies`` mapping that names the robots being driven. Reading
        those overrides with ``mapping.get(robot, default)`` silently discards
        every key that does not match a driven robot, so a typo'd or stale robot
        name left the rollout running the defaults while still reporting
        ``status="success"`` - the caller's per-robot request was never applied
        and nothing said so. Keys that ARE absent from the mapping keep their
        documented default (the mapping is an override layer, so a partial map
        is legitimate); it is the unmatched key that is a caller error.

        Args:
            mapping: The caller-supplied ``{robot_name: value}`` mapping.
            driven: Robot names being driven in this call (the authoritative
                key set - usually the ``policies`` mapping's keys).
            param: Parameter name, used in the error message.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the unmatched keys, or ``None`` when every key
            names a driven robot.
        """
        known = list(driven)
        unknown = [key for key in mapping if key not in known]
        if not unknown:
            return None
        text = f"{method}: {param} names {'robots' if len(unknown) > 1 else 'a robot'} not driven by this call: "
        text += f"{unknown!r}."
        matches: list[str] = []
        for key in unknown:
            if isinstance(key, str):
                matches += [m for m in difflib.get_close_matches(key, known, n=2, cutoff=0.6) if m not in matches]
        if matches:
            text += " Did you mean: " + ", ".join(matches) + "?"
        text += f" Robots driven by this call: {known} (the keys of 'policies')."
        return {"status": "error", "content": [{"text": text}]}

    @staticmethod
    def _validate_positive_int(value: Any, name: str, method: str) -> dict[str, Any] | None:
        """Reject a non-positive-integer count at the public API.

        Shared guard for the rollout count knobs that must be ``>= 1``
        (``n_episodes``, ``max_steps``). A zero/negative/non-int value would
        otherwise produce a degenerate rollout that still reports
        ``status="success"``.

        Returns:
            A structured ``{"status": "error", ...}`` dict naming the
            offending parameter, or ``None`` when the value is valid.
        """
        if not isinstance(value, int) or value < 1:
            return {
                "status": "error",
                "content": [{"text": f"{method}: {name} must be a positive integer, got {value!r}."}],
            }
        return None

    @staticmethod
    def _validate_control_substeps(control_substeps: Any, method: str) -> dict[str, Any] | None:
        """Reject a ``control_substeps`` override the rollout cannot honor.

        ``control_substeps`` is how many physics steps are integrated per
        applied action. ``None`` (the default) means "auto-derive from the
        backend's physics timestep" and is accepted. Any explicit value must
        be a positive integer; ``0``/negative/float/bool/non-finite values are
        rejected with a structured error rather than silently clamped or
        truncated (which under-integrates each control period so the rollout
        reports success while the policy looks like a no-op).

        Returns:
            An error dict naming the offending parameter, or ``None`` when the
            value is valid (including ``None``).
        """
        if control_substeps is None:
            return None
        if isinstance(control_substeps, bool):
            return {
                "status": "error",
                "content": [
                    {"text": f"{method}: control_substeps must be a positive integer, got {control_substeps!r}."}
                ],
            }
        return SimEngine._validate_positive_int(control_substeps, "control_substeps", method)

    def _warn_on_recording_fps_mismatch(self, control_frequency: float, method: str) -> None:
        """Warn once when the attached dataset's fps is not the control rate.

        ``start_recording`` defaults ``fps=30`` while ``run_policy`` defaults
        ``control_frequency=50.0``, and nothing compared them. lerobot's
        ``add_frame`` synthesizes ``timestamp = frame_index / meta.fps``
        unconditionally, so a rollout captured every 20 ms was written as if the
        frames were 33.3 ms apart - measured 1.667x time distortion on the two
        defaults, silently. That contradicts this module's own stated invariant
        ("the recorded control frequency IS the dataset fps") and it propagates:
        ``replay_episode`` derives its per-frame physics budget from the dataset
        fps, so a mislabelled dataset replays at the wrong speed too (measured
        max joint error 0.097 rad at fps=30 vs 0.067 at fps=50).

        Warning rather than refusing: a mismatched dataset is still readable and
        a caller may be recording a deliberately decimated stream, so this must
        not break an in-flight rollout. One-shot per recorder, so a long rollout
        does not spam the log.

        Args:
            control_frequency: The rate the rollout is actually applying actions at.
            method: Public method name, for the log line.
        """
        world = getattr(self, "_world", None)
        if world is None:
            return
        state = getattr(world, "_backend_state", None)
        if not isinstance(state, dict):
            return
        recorder = state.get("dataset_recorder")
        if recorder is None or not state.get("recording", False):
            return
        # lerobot keeps fps on the dataset's metadata; older/other shapes expose
        # it on the dataset or the recorder directly. Read all three rather than
        # assuming, because a missed read makes this check silently vacuous -
        # which is the failure mode being fixed.
        dataset = getattr(recorder, "dataset", None)
        dataset_fps = getattr(getattr(dataset, "meta", None), "fps", None)
        if dataset_fps is None:
            dataset_fps = getattr(dataset, "fps", None)
        if dataset_fps is None:
            dataset_fps = getattr(recorder, "fps", None)
        if dataset_fps is None:
            return
        try:
            dataset_fps = float(dataset_fps)
        except (TypeError, ValueError):
            return
        if dataset_fps <= 0 or math.isclose(dataset_fps, float(control_frequency), rel_tol=1e-6):
            return
        if state.get("recording_fps_mismatch_warned"):
            return
        state["recording_fps_mismatch_warned"] = True
        distortion = float(control_frequency) / dataset_fps
        logger.warning(
            "%s: recording fps (%.1f) does not match control_frequency (%.1f). Each frame captures "
            "%.4fs of simulation but is timestamped %.4fs apart, so the dataset's timebase is wrong "
            "by %.3fx - and replay_episode derives its physics budget from that fps, so replays "
            "inherit the error. Pass start_recording(fps=%.0f) to match, or run at "
            "control_frequency=%.1f.",
            method,
            dataset_fps,
            float(control_frequency),
            1.0 / float(control_frequency),
            1.0 / dataset_fps,
            distortion,
            float(control_frequency),
            dataset_fps,
        )

    @staticmethod
    def _validate_positive_frequency(control_frequency: Any, method: str) -> dict[str, Any] | None:
        """Reject a non-positive or non-numeric ``control_frequency`` at the public API.

        ``control_frequency`` (Hz) sets the control-loop rate the rollout steps
        physics at. It is used as a divisor (the per-action period is
        ``1 / control_frequency`` and ``duration = n_steps / control_frequency``)
        and is handed to :meth:`PolicyRunner`'s per-period substep computation
        (``round(1 / control_frequency / ...)``); a value ``<= 0`` or a
        non-number otherwise reaches that arithmetic deep inside the runner and
        raises a bare ``ValueError``/``TypeError``/``ZeroDivisionError`` rather
        than the structured tool-error dict the public API contracts. Any real
        scalar is accepted (``numbers.Real``), so a NumPy-scalar frequency such
        as ``np.float32(50.0)`` or ``np.int64(50)`` passes; ``bool`` is rejected
        explicitly (an ``int`` subclass, ``True`` would slip through and act as a
        silent 1 Hz) and non-finite values (``nan``/``inf``) are rejected before
        the ``<= 0`` comparison. That domain is
        :func:`~strands_robots.utils.positive_finite_number_error`, shared with
        every other rate/duration knob (including the teleop control loop, which
        divides by its ``hz`` the same way) so they cannot diverge. Returns a
        structured error dict to surface, or ``None`` when valid.

        Args:
            control_frequency: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        error = positive_finite_number_error(control_frequency, "control_frequency", method)
        if error:
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _validate_timestep(timestep: Any, method: str, param: str = "timestep") -> dict[str, Any] | None:
        """Reject a physics timestep the integrator cannot honor.

        The timestep is the ``dt`` (seconds) every physics substep advances
        by, so a non-positive or non-finite value poisons the whole world.
        ``0`` is rejected rather than coalesced to the engine default - the
        same contract the backend ``set_timestep`` setters enforce. Anything
        ``float()`` accepts is coerced (so a NumPy scalar passes); ``bool`` is
        rejected explicitly since ``True`` would act as a silent 1-second
        step. ``param`` names the parameter to quote in the message.

        Returns:
            A structured ``{"status": "error", ...}`` dict to surface, or
            ``None`` when the value is usable.
        """
        message = f"{method}: {param} must be a finite positive number, got {timestep!r}."
        if isinstance(timestep, bool):
            return {"status": "error", "content": [{"text": message}]}
        try:
            value = float(timestep)
        except (TypeError, ValueError):
            return {"status": "error", "content": [{"text": message}]}
        if not math.isfinite(value) or value <= 0:
            return {"status": "error", "content": [{"text": message}]}
        return None

    @staticmethod
    def _validate_mass(mass: Any, method: str, param: str = "mass") -> dict[str, Any] | None:
        """Reject a body mass the physics engine cannot honor.

        A dynamic body's mass is the divisor of every force applied to it, so a
        value outside ``(0, inf)`` does not merely mis-size one object - it
        poisons the whole world on the next step. ``inf`` makes the very first
        integration produce ``nan`` acceleration, and because the solver shares
        one state vector, every *other* body's ``qpos``/``qvel`` goes ``nan``
        with it. ``0`` and negatives violate MuJoCo's
        "mass and inertia of moving bodies must be larger than mjMINVAL"
        invariant, which surfaces as a compile refusal that names neither the
        parameter nor the reason. This is the same domain
        :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.set_body_properties`
        already enforces when it writes the same ``body_mass`` field, so a mass
        cannot be established at creation on terms the setter would refuse.

        Args:
            mass: The caller-supplied value. Anything ``float()`` accepts is
                coerced (so a NumPy scalar passes); ``bool`` is rejected
                explicitly since ``True`` would act as a silent 1 kg body.
            method: Public method name, used to prefix the error message.
            param: Parameter name to quote in the message.

        Returns:
            A structured ``{"status": "error", ...}`` dict to surface, or
            ``None`` when the value is usable.
        """
        if isinstance(mass, bool):
            return {
                "status": "error",
                "content": [{"text": f"{method}: '{param}' must be a positive number, got {mass!r}"}],
            }
        try:
            value = float(mass)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "content": [{"text": f"{method}: '{param}' must be a positive number, got {mass!r}"}],
            }
        if not math.isfinite(value) or value <= 0:
            return {
                "status": "error",
                "content": [{"text": f"{method}: '{param}' must be a finite number > 0, got {value}"}],
            }
        return None

    @staticmethod
    def _normalize_gravity(
        gravity: Any, method: str, param: str = "gravity"
    ) -> tuple[list[float] | None, dict[str, Any] | None]:
        """Coerce a gravity argument to three finite floats, or explain why not.

        Callers normalize through this helper and store the returned
        components, so what the result reports is what the engine received.
        ``gravity`` is a 3-element ``[x, y, z]`` sequence, or a real scalar
        taken as the z-component (``[0, 0, z]``).

        Returns:
            ``(components, None)`` with three finite floats, or
            ``(None, error_dict)`` describing what is wrong with the value.
            Exactly one element is non-``None``.
        """
        # Accept any real scalar (numbers.Real) as a z-only gravity so a value
        # computed as a NumPy scalar (np.float32 / np.int64) is treated like a
        # plain float. A NumPy array is not numbers.Real, so it still takes the
        # vector path below.
        if isinstance(gravity, numbers.Real):
            components = [0.0, 0.0, float(gravity)]
        else:
            try:
                vector = cast("Sequence[Any]", gravity)
                if len(vector) != 3:
                    return None, {
                        "status": "error",
                        "content": [
                            {"text": f"{method}: '{param}' must be a 3-element list [x,y,z], got {len(vector)}"}
                        ],
                    }
                components = [float(g) for g in vector]
            except (TypeError, ValueError) as e:
                return None, {
                    "status": "error",
                    "content": [{"text": f"{method}: '{param}' must be a 3-element list of numbers ({e})"}],
                }
        if not all(math.isfinite(g) for g in components):
            return None, {
                "status": "error",
                "content": [{"text": f"{method}: all components must be finite, got {components}"}],
            }
        return components, None

    @staticmethod
    def _validate_duration(duration: Any, method: str) -> dict[str, Any] | None:
        """Reject a rollout ``duration`` that cannot produce a single control step.

        ``duration`` is the default horizon knob: when no ``n_steps`` /
        ``max_steps`` is given, the rollout length is ``int(duration *
        control_frequency)`` control steps. A value ``<= 0`` yields zero steps,
        which used to be reported as ``status="success"`` for a rollout that
        never queried the policy and never stepped physics - and, when a
        ``video`` was requested, wrote no MP4 while still claiming success. A
        non-finite value never reached that arithmetic intact either: ``nan``
        surfaced as a bare ``ValueError`` ("cannot convert float NaN to
        integer") naming a library internal, and ``inf`` as an
        ``OverflowError``. Validating at the public entry point - before any
        policy is created or a background thread is submitted - turns all of
        these into an actionable caller error.

        The accepted domain is :func:`~strands_robots.utils.positive_finite_number_error`,
        shared with :meth:`_validate_positive_frequency` (the other knob in the
        same ``duration * control_frequency`` product) so the two cannot
        diverge: any finite positive real scalar, including a NumPy scalar such
        as ``np.float32(2.5)``; ``bool`` is rejected explicitly (an ``int``
        subclass, ``True`` would act as a silent 1 second) and ``nan``/``inf``
        are rejected before the ``<= 0`` comparison so a ``nan`` - which is
        never ``<= 0`` - cannot slip through.

        Args:
            duration: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None`` when the
            value is valid.
        """
        error = positive_finite_number_error(duration, "duration", method)
        if error:
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _validate_video_config(video: Any, method: str) -> dict[str, Any] | None:
        """Reject a ``video`` recording config the rollout cannot honor.

        ``video`` is a free-form dict, so a mistyped key has no signature to
        bounce off; checking at the public entry point turns a silently
        dropped key into an actionable error.

        Returns:
            A structured ``{"status": "error", ...}`` dict naming the
            offending key, or ``None`` when the config is valid.
        """
        video_error = VideoConfig.validation_error(video)
        if video_error is None:
            return None
        return {"status": "error", "content": [{"text": f"{method}: {video_error}"}]}

    @staticmethod
    def _validate_policy_mapping(value: Any, param: str, method: str) -> dict[str, Any] | None:
        """Reject a ``policy_config`` / ``policy_kwargs`` that cannot be splatted.

        Both parameters are opaque keyword bags reaching their consumer
        through ``**`` (``create_policy`` / ``policy.get_actions``); checking
        at the public entry point, before any policy is created or a thread is
        submitted, turns a wrong-shaped value into an actionable error instead
        of a bare ``TypeError`` deep in the call machinery. ``param`` is
        ``"policy_config"`` or ``"policy_kwargs"``.

        Returns:
            A structured ``{"status": "error", ...}`` dict to surface, or
            ``None`` when the value is valid.
        """
        from strands_robots.policies import policy_mapping_error

        message = policy_mapping_error(value, param)
        if message is None:
            return None
        return {"status": "error", "content": [{"text": f"{method}: {message}"}]}

    def run_policy(
        self,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: dict[str, Any] | None = None,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
        max_onframe_failures: int | None = None,
        control_substeps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        n_episodes: int = 1,
        reset_between: bool = True,
        async_rtc: bool | None = None,
        rtc_inference_timeout_s: float | None = None,
        wbc_install_torque_control: bool = True,
        stop_when: dict[str, Any] | Callable[[SimEngine], bool] | None = None,
    ) -> dict[str, Any]:
        """Run a policy loop in the simulation (blocking).

        Default implementation delegates to the backend-agnostic
        :class:`~strands_robots.simulation.policy_runner.PolicyRunner`.
        Backends MAY override for backend-specific optimisations.

        Every knob is validated at this entry point: an unusable value
        (non-positive ``duration`` / ``control_frequency`` /
        ``control_substeps`` / ``action_horizon`` / ``n_episodes``, a
        malformed ``video`` / ``policy_config`` / ``policy_kwargs``) is
        reported as a structured caller error rather than running a degenerate
        rollout that reports success.

        Args:
            robot_name: Robot to control.
            policy_provider: Name passed to
                :func:`strands_robots.policies.create_policy`.
            policy_config: Opaque dict of provider-specific kwargs, forwarded
                verbatim to ``create_policy``.
            instruction: Natural-language instruction for the policy.
            duration: Wall-clock seconds to run; used only when no
                ``n_steps`` / ``max_steps`` is given (the step count wins and
                ``duration`` is recomputed from it).
            control_frequency: Target Hz for policy queries.
            control_substeps: Explicit physics steps to integrate per applied
                action; ``None`` (default) derives it from
                ``control_frequency`` and the backend's physics timestep.
            action_horizon: Lower bound on actions consumed from each policy
                chunk before re-querying. The effective interval is
                ``max(action_horizon, policy.execution_horizon)``, so a
                chunk-emitting policy always keeps its full trained chunk; RTC
                policies own their own interval and ignore this.
            fast_mode: Skip real-time sleep between steps.
            video: Optional video-recording config dict. Accepted keys:
                ``path`` (str, output MP4 - required to enable recording),
                ``fps`` (int, default 30), ``camera`` (str), ``width`` (int,
                default 640), ``height`` (int, default 480). See
                :class:`~strands_robots.simulation.policy_runner.VideoConfig`.
                Any other key, or a size/fps that is not a positive whole
                number, is a caller error naming the offending key. For
                extension points beyond video, backends plug into
                ``PolicyRunner.run``'s ``on_frame`` hook via
                :meth:`_make_run_policy_hook`.
            seed: Optional master RNG seed for a reproducible rollout: reseeds
                Python / NumPy / torch / cuDNN and forwards
                ``policy.reset(seed=...)``. ``None`` (default) leaves RNG
                state untouched.
            policy_kwargs: Optional per-call goal payload forwarded verbatim
                to every ``policy.get_actions(obs, instruction,
                **policy_kwargs)`` call. Carries the well-known goal keys
                (``target_pose`` / ``target_joints`` / ``target_velocity`` /
                ``world_update``) to providers that read their goal from
                kwargs; VLA providers ignore unknown kwargs, so forwarding is
                always safe.
            n_episodes: Number of sequential episode rollouts in this single
                call (default ``1``). IMPORTANT: to record N DISTINCT dataset
                episodes pass ``n_episodes=N`` in one call - do NOT loop this
                call N times (that buffers all frames into one merged
                ``episode_index=0`` mega-episode). When ``> 1``, each rollout
                is flushed as its own dataset episode via :meth:`save_episode`
                (only while a recording is active); ``seed`` is offset per
                episode (``seed + i``) and the ``video`` path gets ``_ep{i}``
                inserted before the extension. Confirm the count after
                ``stop_recording`` with :meth:`verify_dataset_episodes`.
            reset_between: Reset the sim between episodes (default ``True``;
                never fires after the final episode). ``False`` chains
                episodes from the previous end state.
            async_rtc: ``True`` overlaps policy inference with action
                execution (latency masking); ``False`` keeps the synchronous
                chunk-then-drain loop; ``None`` (default) auto-resolves from
                ``policy.is_chunk_emitting()``. Forwarded verbatim to
                :meth:`PolicyRunner.run` - see its docstring for the full
                contract (thread safety, RTC seam blending).
            rtc_inference_timeout_s: Optional hard per-chunk timeout (seconds)
                for the async-RTC prefetch. When set, a stuck inference surfaces
                as a structured ``status=error`` result (carrying the RTC
                telemetry) instead of hanging the sim. ``None`` (default) waits
                without a deadline. Forwarded verbatim to
                :meth:`PolicyRunner.run`; ignored on the synchronous path.
            wbc_install_torque_control: When ``True`` (default), a
                :class:`~strands_robots.policies.wbc.WBCPolicy` run on a
                position-servo scene (the stock ``Robot("unitree_g1")``) gets the
                torque shim auto-installed for the duration of this call, then
                uninstalled. WBC emits joint-position targets; the stock G1's
                uniform ``kp=500`` servo would override SONIC's tuned per-joint
                PD and the gait diverges, so the documented quickstart silently
                falls over without it. Set ``False`` to manage the controller
                yourself or to drive a torque-actuated scene directly. No-op for
                non-WBC policies and on backends without the hook.
            stop_when: Optional semantic early-return condition: end the
                rollout as soon as the WORLD reaches a state, not only when
                the step budget runs out - which turns a monolithic rollout
                into a retryable primitive an agent can invoke -> inspect ->
                re-invoke. A predicate-DSL clause in the same schema as a
                benchmark spec's ``success`` clause: a single call
                ``{"predicate": "grasped", "body": "cube", "gripper_prefix":
                "so100"}`` or an ``{"all": [...]}`` / ``{"any": [...]}`` group
                of bool predicate calls. Compiled via
                :func:`~strands_robots.simulation.benchmark_spec.compile_stop_when`
                against the closed predicate registry (never ``eval`` /
                ``exec``; an unknown predicate name is rejected up front with
                the valid list), and the clause's referenced body/joint names
                are probed against the LIVE scene before the rollout starts -
                a typo'd name (or a backend without body lookups) is an
                up-front structured error instead of a clause that silently
                never fires and burns the whole budget. The compiled clause is
                evaluated against the SIM after every
                applied action - matching the benchmark semantics, not the
                observation dict - on both the synchronous and async-RTC
                paths, so the stop lands within one control step of the
                condition holding. Composes with an active recording session:
                frames are captured up to the stop, so a recorded episode's
                frame count equals the result's ``steps_used``. Programmatic
                callers may pass a callable ``(sim) -> bool`` instead of a
                dict (the tool surface accepts dicts only). ``None`` (default)
                keeps the pure step-budget horizon. The result json reports
                why the rollout ended via ``stopped_reason``.

        Returns:
            Standard status dict with an agent-consumable ``{"json": {...}}``
            content block alongside the human-readable ``text``. The json block
            carries the rollout facts as typed fields (``n_steps``,
            ``steps_used``, ``elapsed_s``, ``stopped_early``,
            ``stopped_reason`` (``"predicate"`` | ``"budget"`` |
            ``"cancelled"``; ``"error"`` on error results - so an agent
            deciding whether to retry knows WHY the rollout ended),
            ``action_errors``, ``video_path``,
            ``video_frames``, ``positional_fallback_used``,
            ``generic_state_keys_used``, ``missing_state_keys_used``, ...) so callers can self-correct
            programmatically without parsing the text. The two routing-
            degradation flags are True when the driving policy could not bind
            the observation to the model's inputs by name and silently fell
            back (a camera routed to a model image slot positionally, or
            ``observation.state`` composed from the observation's own scalar
            keys because none of ``robot_state_keys`` matched). A True flag on
            an otherwise ``success`` run is the signature of the robot moving
            on meaningless inputs. Mirrors :meth:`eval_policy`.
        """
        from strands_robots.policies import create_policy

        robot_name = self._resolve_single_robot(robot_name)

        if err := self._validate_positive_frequency(control_frequency, "run_policy"):
            return err
        # Coerce to a plain Python float now the value is validated: a NumPy
        # scalar (accepted above via numbers.Real) flows into 1 / control_frequency
        # and time.sleep(...) downstream, and time.sleep rejects a numpy.float32
        # with a bare "cannot be interpreted as an integer" TypeError.
        control_frequency = float(control_frequency)

        # accept n_steps (or legacy max_steps) as an alternate horizon
        # specification. duration = n_steps / control_frequency. If both
        # are passed, n_steps wins (primary per DoD).
        duration, n_steps, horizon_error = self._resolve_horizon(n_steps, max_steps, control_frequency, duration)
        if horizon_error is not None:
            return horizon_error

        # ``duration`` only sets the horizon when no step count was given - with
        # an ``n_steps`` the resolution above recomputes it - so validate the
        # value the rollout will actually run on, and only then.
        if n_steps is None:
            if err := self._validate_duration(duration, "run_policy"):
                return err

        if err := self._validate_positive_int(n_episodes, "n_episodes", "run_policy"):
            return err

        if err := self._validate_video_config(video, "run_policy"):
            return err
        if err := self._validate_policy_mapping(policy_config, "policy_config", "run_policy"):
            return err
        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "run_policy"):
            return err
        if err := self._validate_action_horizon(action_horizon, "run_policy"):
            return err
        if err := self._validate_control_substeps(control_substeps, "run_policy"):
            return err

        # Compile the stop_when early-return clause BEFORE any policy is
        # created (an unknown predicate name or bad kwargs is a caller error,
        # not a mid-rollout crash after an expensive weight download). The
        # tool surface only ever passes predicate-DSL dicts, resolved through
        # the closed registry - never eval/exec; programmatic callers may pass
        # a callable directly, mirroring PolicyRunner.evaluate's success_fn.
        stop_when_fn: Callable[[SimEngine], bool] | None = None
        if stop_when is not None:
            if callable(stop_when):
                stop_when_fn = stop_when
            else:
                from strands_robots.simulation.benchmark_spec import compile_stop_when

                try:
                    stop_when_fn = compile_stop_when(stop_when)
                except ValueError as e:
                    return {
                        "status": "error",
                        "content": [
                            {"text": f"run_policy: {e}"},
                            {"json": {"stopped_reason": "error", "steps_used": 0, "n_steps": 0}},
                        ],
                    }

        if robot_name not in self.list_robots():
            return {
                "status": "error",
                "content": [{"text": self._unknown_robot_msg(robot_name)}],
            }

        # Probe the clause's referenced bodies/joints against the LIVE scene.
        # compile_stop_when validates the predicate NAMES but cannot see the
        # scene: a typo'd body would compile clean, degrade to a constant
        # False at evaluation time (predicates never raise), and burn the
        # whole step budget reporting stopped_reason="budget" -
        # indistinguishable from an honest miss. Probing here (dict clauses
        # only - a programmatic callable is opaque) turns that silent
        # never-fires into an up-front structured error, including on
        # backends whose predicates cannot resolve bodies at all.
        if stop_when_fn is not None and isinstance(stop_when, dict):
            probe_err = self._stop_when_unresolved_error(stop_when)
            if probe_err is not None:
                return probe_err

        if policy_object is None:
            # Fail fast on a misconfiguration (e.g. camera names that cannot be
            # routed to the policy's declared image inputs) BEFORE the expensive
            # create_policy weight download.
            preflight_error = self._preflight_policy_config(robot_name, policy_provider, policy_config)
            if preflight_error is not None:
                return preflight_error
            policy = create_policy(policy_provider, **(policy_config or {}))
        else:
            # Pre-built policy path - skip the expensive create_policy call.
            # Caller is responsible for policy.set_robot_state_keys(...) if needed,
            # but we set it here defensively so the semantics match the provider path.
            policy = policy_object
        # set_robot_state_keys + sim-context binding are best-effort policy
        # configuration: a raising robot_action_keys must not crash the whole
        # rollout - a genuine wrong-embodiment mismatch is surfaced far more
        # actionably downstream by PolicyRunner's fail-fast probe.
        try:
            policy.set_robot_state_keys(self.robot_action_keys(robot_name))
            self.bind_policy_sim_context(policy, robot_name)
        except Exception as exc:  # noqa: BLE001 - non-fatal policy configuration
            logger.debug("policy binding for %r failed: %s", robot_name, exc)

        # Auto-install any action controller this policy needs on this scene
        # (e.g. the WBC torque shim); the cleanup callable restores the scene
        # in the finally below. Opt out with wbc_install_torque_control=False.
        controller_cleanup = (
            self._maybe_install_wbc_torque_control(policy, robot_name) if wbc_install_torque_control else None
        )

        try:
            runner = PolicyRunner(self)
            # The one point where the recording fps and the rate frames are
            # actually captured at are both known. See the helper: nothing used
            # to compare them, so the default fps=30 against the default
            # control_frequency=50.0 wrote a 1.667x-wrong timebase in silence.
            self._warn_on_recording_fps_mismatch(control_frequency, "run_policy")

            # Single-episode fast path: byte-for-byte the historical behaviour
            # (no reset, no episode-boundary flush). n_episodes defaults to 1 so
            # existing callers are completely unaffected.
            if n_episodes == 1:
                recording = self._is_recording()
                if recording:
                    logger.info(
                        "run_policy: n_episodes=1, will produce 1 dataset episode of ~%d frames "
                        "(frames buffer into the current episode and flush at save_episode/"
                        "stop_recording). To record N DISTINCT dataset episodes pass n_episodes=N "
                        "- do NOT loop the tool call.",
                        int(duration * control_frequency),
                    )
                on_frame = self._make_run_policy_hook(robot_name, instruction)
                result = runner.run(
                    robot_name,
                    policy,
                    instruction=instruction,
                    duration=duration,
                    n_steps=n_steps,
                    control_frequency=control_frequency,
                    action_horizon=action_horizon,
                    fast_mode=fast_mode,
                    video=VideoConfig.from_dict(video),
                    on_frame=on_frame,
                    max_onframe_failures=max_onframe_failures,
                    control_substeps=control_substeps,
                    policy_kwargs=policy_kwargs,
                    seed=seed,
                    async_rtc=async_rtc,
                    rtc_inference_timeout_s=rtc_inference_timeout_s,
                    stop_when=stop_when_fn,
                )
                completed = 1 if result.get("status") == "success" else 0
                contract = self._episode_contract_fields(
                    requested=1, completed=completed, saved=0, flush_deferred=recording
                )
                self._merge_json_fields(result, contract)
                return result

            # Multi-episode path: one rollout per episode, flushing a dataset
            # episode boundary (save_episode) when recording and resetting between
            # episodes. Replaces the brittle manual
            # ``for _ in range(n): run_policy(); save_episode(); reset()`` loop.
            return self._run_episodes(
                runner,
                robot_name,
                policy,
                instruction=instruction,
                duration=duration,
                n_steps=n_steps,
                control_frequency=control_frequency,
                action_horizon=action_horizon,
                fast_mode=fast_mode,
                video=video,
                max_onframe_failures=max_onframe_failures,
                control_substeps=control_substeps,
                policy_kwargs=policy_kwargs,
                seed=seed,
                n_episodes=n_episodes,
                reset_between=reset_between,
                async_rtc=async_rtc,
                rtc_inference_timeout_s=rtc_inference_timeout_s,
                stop_when=stop_when_fn,
            )
        finally:
            if controller_cleanup is not None:
                controller_cleanup()

    def _stop_when_unresolved_error(self, stop_when: dict[str, Any]) -> dict[str, Any] | None:
        """Structured error if a ``stop_when`` clause references unresolvable entities.

        Probes every body/joint name in the clause through the SAME lookup
        path the predicates use at evaluation time
        (:func:`~strands_robots.simulation.predicates.can_resolve_body` /
        :func:`~strands_robots.simulation.predicates.can_resolve_joint`,
        including the LIBERO ``<name>_main`` fallback), against the live
        scene, once, before the rollout starts. Returns ``None`` when every
        referenced entity resolves. Bodies added to the scene AFTER this
        check are out of contract - a rollout does not create bodies.
        """
        from strands_robots.simulation.benchmark_spec import stop_when_referenced_entities
        from strands_robots.simulation.predicates import can_resolve_body, can_resolve_joint, supports_body_lookup

        bodies, joints = stop_when_referenced_entities(stop_when)

        def _err(text: str) -> dict[str, Any]:
            return {
                "status": "error",
                "content": [
                    {"text": f"run_policy: {text}"},
                    {"json": {"stopped_reason": "error", "steps_used": 0, "n_steps": 0}},
                ],
            }

        if bodies and not supports_body_lookup(self):
            return _err(
                f"stop_when references bodies {bodies} but this backend has no body lookup "
                "(get_body_state), so the clause could never fire and the rollout would "
                "silently run to its step budget. Use a clause without body-referencing "
                "predicates, or a backend that supports body lookups."
            )
        missing_bodies = [b for b in bodies if not can_resolve_body(self, b)]
        if missing_bodies:
            return _err(
                f"stop_when references bodies not present in the scene: {missing_bodies}. "
                "The clause would never fire and the rollout would silently run to its "
                "step budget. Check the names against the loaded scene (get_state lists "
                "objects; describe() lists actions)."
            )
        missing_joints = [j for j in joints if not can_resolve_joint(self, j)]
        if missing_joints:
            return _err(
                f"stop_when references joints not present in the observation: {missing_joints}. "
                "The clause would never fire and the rollout would silently run to its "
                "step budget. Check the names against get_observation()'s keys "
                "(joint names are namespaced '<robot>/<joint>')."
            )
        return None

    def _run_episodes(
        self,
        runner: PolicyRunner,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str,
        duration: float,
        n_steps: int | None,
        control_frequency: float,
        action_horizon: int,
        fast_mode: bool,
        video: dict[str, Any] | None,
        max_onframe_failures: int | None,
        control_substeps: int | None,
        policy_kwargs: dict[str, Any] | None,
        seed: int | None,
        n_episodes: int,
        reset_between: bool,
        async_rtc: bool | None = None,
        rtc_inference_timeout_s: float | None = None,
        stop_when: Callable[[SimEngine], bool] | None = None,
    ) -> dict[str, Any]:
        """Run ``n_episodes`` sequential rollouts; shared multi-episode driver.

        Behind :meth:`run_policy` when ``n_episodes > 1``. Per episode it:
        (1) runs one rollout for the configured horizon, (2) flushes a dataset
        episode boundary via :meth:`save_episode` when a recording is active,
        and (3) resets the sim between episodes unless ``reset_between`` is
        ``False`` - so a single call yields N correctly delimited dataset
        episodes instead of one merged episode. Aborts early (returning a
        structured error with the episodes completed so far) if a rollout, an
        episode flush, or a reset fails.

        ``stop_when`` (already compiled to a callable by :meth:`run_policy`)
        is forwarded to every per-episode rollout, giving multi-episode
        collection a per-episode success gate: each episode ends at its own
        predicate hit (or budget), and its dataset episode is flushed with
        exactly the frames captured up to that stop.
        """
        episodes: list[dict[str, Any]] = []
        episodes_saved = 0
        total_steps = 0
        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            ep_video = self._episode_video_config(video, ep)
            on_frame = self._make_run_policy_hook(robot_name, instruction)
            result = runner.run(
                robot_name,
                policy,
                instruction=instruction,
                duration=duration,
                n_steps=n_steps,
                control_frequency=control_frequency,
                action_horizon=action_horizon,
                fast_mode=fast_mode,
                video=ep_video,
                on_frame=on_frame,
                max_onframe_failures=max_onframe_failures,
                control_substeps=control_substeps,
                policy_kwargs=policy_kwargs,
                seed=ep_seed,
                async_rtc=async_rtc,
                rtc_inference_timeout_s=rtc_inference_timeout_s,
                stop_when=stop_when,
            )
            ep_json = self._extract_json_payload(result)
            ep_record: dict[str, Any] = {"episode": ep, **ep_json}
            total_steps += int(ep_json.get("n_steps", 0) or 0)

            if result.get("status") == "error":
                ep_record["status"] = "error"
                episodes.append(ep_record)
                return self._episodes_result(
                    episodes,
                    episodes_saved,
                    total_steps,
                    n_episodes,
                    status="error",
                    extra=(
                        f"Episode {ep} rollout failed; aborting remaining "
                        f"{n_episodes - ep - 1} episode(s). {self._first_text(result)}"
                    ),
                )

            # Flush this rollout as its own dataset episode when recording.
            if self._is_recording():
                save = self.save_episode()
                if save.get("status") == "error":
                    ep_record["save_episode_error"] = self._first_text(save)
                    episodes.append(ep_record)
                    return self._episodes_result(
                        episodes,
                        episodes_saved,
                        total_steps,
                        n_episodes,
                        status="error",
                        extra=f"save_episode failed after episode {ep}: {self._first_text(save)}",
                    )
                episodes_saved += 1
                ep_record["saved"] = True

            episodes.append(ep_record)

            # Reset between episodes - never after the last one.
            if reset_between and ep < n_episodes - 1:
                reset_result = self.reset()
                if reset_result.get("status") == "error":
                    return self._episodes_result(
                        episodes,
                        episodes_saved,
                        total_steps,
                        n_episodes,
                        status="error",
                        extra=f"reset() failed after episode {ep}: {self._first_text(reset_result)}",
                    )

        return self._episodes_result(episodes, episodes_saved, total_steps, n_episodes, status="success")

    @staticmethod
    def _first_text(result: dict[str, Any]) -> str:
        """First human-readable ``text`` block from a status dict ("" if none)."""
        for blk in result.get("content", []) or []:
            if isinstance(blk, dict):
                text = blk.get("text")
                if isinstance(text, str):
                    return text
        return ""

    @staticmethod
    def _extract_json_payload(result: dict[str, Any]) -> dict[str, Any]:
        """First agent-consumable ``{"json": {...}}`` block ({} if none)."""
        for blk in result.get("content", []) or []:
            if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
                return dict(blk["json"])
        return {}

    @staticmethod
    def _merge_json_fields(result: dict[str, Any], fields: dict[str, Any]) -> None:
        """Merge ``fields`` into the result's ``{"json": {...}}`` block in place.

        Augments the first existing json content block, or appends a new one if
        the result has none. Lets :meth:`run_policy` attach the episode-contract
        fields onto a ``PolicyRunner.run`` result without rebuilding it.
        """
        for blk in result.get("content", []) or []:
            if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
                blk["json"].update(fields)
                return
        result.setdefault("content", []).append({"json": dict(fields)})

    @staticmethod
    def _episode_video_config(video: dict[str, Any] | None, episode: int) -> VideoConfig | None:
        """Per-episode :class:`VideoConfig` with ``_ep{i}`` in the filename.

        Multi-episode runs reuse one ``video`` config; without templating every
        episode would overwrite the same MP4. Inserts ``_ep{episode}`` before
        the extension so each episode gets a distinct file. Passes through
        unchanged when no video path is set.
        """
        if not video or not video.get("path"):
            return VideoConfig.from_dict(video)
        templated = dict(video)
        root, ext = os.path.splitext(str(video["path"]))
        templated["path"] = f"{root}_ep{episode}{ext or '.mp4'}"
        return VideoConfig.from_dict(templated)

    def _episodes_result(
        self,
        episodes: list[dict[str, Any]],
        episodes_saved: int,
        total_steps: int,
        n_episodes: int,
        *,
        status: str,
        extra: str = "",
    ) -> dict[str, Any]:
        """Aggregate per-episode records into one ``run_policy`` status dict.

        Mirrors the single-rollout result shape: a human-readable ``text``
        block plus an agent-consumable ``{"json": {...}}`` block carrying typed
        aggregate fields (``n_episodes_completed``, ``episodes_saved``,
        ``total_steps``, per-episode list, ``video_paths``). The payload keeps
        ONE shape across episode counts: ``stopped_reason`` / ``steps_used``
        are present here just as on the single-episode payload -
        ``stopped_reason`` is ``"error"`` on error results and otherwise the
        LAST episode's reason (why the call as a whole stopped running), with
        the per-episode attribution in ``stopped_reasons`` (aligned with
        ``episodes``); ``steps_used`` equals ``total_steps``.
        """
        completed = len(episodes)
        video_paths = [e["video_path"] for e in episodes if e.get("video_path")]
        stopped_reasons = [e.get("stopped_reason") for e in episodes]
        if status == "error":
            stopped_reason = "error"
        elif stopped_reasons and isinstance(stopped_reasons[-1], str):
            stopped_reason = stopped_reasons[-1]
        else:
            stopped_reason = "budget"
        text = (
            f"Multi-episode run_policy: {completed}/{n_episodes} episode(s) completed, "
            f"{episodes_saved} flushed to dataset, {total_steps} total steps."
        )
        if extra:
            text += f"\n{extra}"
        dataset_episode_indices: list[int] = []
        if self._is_recording():
            recorder = self._active_recorder()
            meta = getattr(getattr(recorder, "dataset", None), "meta", None)
            total_episodes = int(getattr(meta, "total_episodes", 0) or 0) if meta is not None else 0
            dataset_episode_indices = list(range(total_episodes))
        payload: dict[str, Any] = {
            "n_episodes_requested": n_episodes,
            "n_episodes_completed": completed,
            "episodes_saved": episodes_saved,
            "dataset_episode_indices": dataset_episode_indices,
            "total_steps": total_steps,
            "steps_used": total_steps,
            "stopped_reason": stopped_reason,
            "stopped_reasons": stopped_reasons,
            "episodes": episodes,
            "video_paths": video_paths,
        }
        return {"status": status, "content": [{"text": text}, {"json": payload}]}

    def _is_recording(self) -> bool:
        """Whether a dataset-recording session is active.

        Backends that support LeRobot dataset recording override this; the base
        returns ``False`` so the multi-episode :meth:`run_policy` loop only
        flushes episode boundaries on backends that actually record.
        """
        return False

    def _active_recorder(self) -> Any:
        """Return the active dataset recorder object, or ``None``.

        Backends that support LeRobot dataset recording override this to expose
        the live recorder (see the MuJoCo ``RecordingMixin``). The base has no
        recorder, so it returns ``None``. Used by :meth:`run_policy` to read the
        in-memory episode count for the episode-contract fields.
        """
        return None

    def _active_dataset_root(self) -> str | None:
        """On-disk root of the active (or most recent) recording, or ``None``.

        Backends that record override this so :meth:`verify_dataset_episodes`
        can locate the dataset parquet AFTER ``stop_recording`` has finalized it
        (the recorder object is gone by then). The base has no recorder, so it
        returns ``None``.
        """
        return None

    def verify_dataset_episodes(self, expected: int) -> dict[str, Any]:
        """Verify the recorded dataset holds exactly ``expected`` episodes.

        Reads the LeRobot dataset parquet (the on-disk ground truth, not the
        recorder's in-memory bookkeeping) for the active or
        most-recently-recorded session AND cross-checks it against the
        ``meta/info.json`` ``total_episodes`` header; both must agree with
        ``expected``. Call this AFTER :meth:`stop_recording` for a definitive
        check that a collection run produced N distinct episodes rather than
        one merged ``episode_index=0`` mega-episode.

        Returns:
            Standard status dict. ``status`` is ``"success"`` only when the
            parquet holds exactly ``expected`` episodes, the parquet agrees
            with ``meta/info.json`` (``sources_agree``), and every episode
            parquet was readable (``unreadable_files`` empty - otherwise the
            count is only a lower bound). The ``{"json": {...}}`` block
            carries ``expected``, ``actual``, ``info_total_episodes``,
            ``sources_agree``, ``episode_indices``, ``total_frames``,
            ``total_frames_per_ep``, ``unreadable_files`` and ``root``. An
            unreadable or corrupt parquet is reported as this same error dict,
            never raised.
        """
        if not isinstance(expected, int) or expected < 0:
            return {
                "status": "error",
                "content": [
                    {"text": f"verify_dataset_episodes: expected must be a non-negative int, got {expected!r}."}
                ],
            }

        root = self._active_dataset_root()
        if not root:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "verify_dataset_episodes: no active or recently-recorded dataset to verify. "
                            "Record one first (start_recording -> run_policy -> stop_recording)."
                        )
                    }
                ],
            }

        from strands_robots.dataset_recorder import read_dataset_episode_indices

        try:
            info = read_dataset_episode_indices(root)
        except (ValueError, OSError) as e:
            # OSError covers the empty/unfinalized dataset (FileNotFoundError -
            # no episode parquet yet) and an unreadable file; ValueError covers a
            # corrupt / truncated / foreign parquet (pyarrow raises ArrowInvalid,
            # a ValueError subclass). This facade is agent-callable, so both must
            # surface as a structured error dict, never as an escaping traceback.
            return {
                "status": "error",
                "content": [
                    {"text": f"verify_dataset_episodes: {e}"},
                    {
                        "json": {
                            "expected": expected,
                            "actual": 0,
                            "info_total_episodes": None,
                            "sources_agree": False,
                            "episode_indices": [],
                            "total_frames": 0,
                            "total_frames_per_ep": [],
                            "unreadable_files": [],
                            "root": str(root),
                        }
                    },
                ],
            }
        except ImportError as e:
            return {"status": "error", "content": [{"text": f"verify_dataset_episodes: {e}"}]}

        actual = info["total_episodes"]
        info_total = info.get("info_total_episodes")
        unreadable = info.get("unreadable_files") or []

        # Two independent truths must agree: the parquet episode count AND the
        # meta/info.json total_episodes header. A dataset can report the right
        # parquet count yet carry a stale/inconsistent info.json (interrupted
        # finalize), so a parquet-only check is not sufficient. sources_agree is
        # True when info.json is absent (parquet is then the sole truth) or when
        # the header matches the parquet.
        sources_agree = info_total is None or info_total == actual
        # A dataset with unreadable episode parquet files can never be certified:
        # the readable files are a LOWER BOUND on the episode count, so a count
        # that happens to equal ``expected`` proves nothing about the whole
        # dataset. Fail loud and name the broken files.
        ok = actual == expected and sources_agree and not unreadable
        status = "success" if ok else "error"

        if unreadable:
            text = (
                f"verify_dataset_episodes: UNREADABLE - {len(unreadable)} episode "
                f"parquet file(s) could not be read, so the {actual} episode(s) found "
                f"are a lower bound (expected {expected}): {'; '.join(unreadable)}. "
                f"Root: {root}"
            )
        elif not sources_agree:
            verdict = "MISMATCH"
            text = (
                f"verify_dataset_episodes: {verdict} - meta/info.json reports "
                f"{info_total} episode(s) but the parquet holds {actual}; the "
                f"dataset metadata is inconsistent (expected {expected}). "
                f"Root: {root}"
            )
        else:
            verdict = "matches" if ok else "MISMATCH"
            text = (
                f"verify_dataset_episodes: {verdict} - expected {expected}, "
                f"found {actual} episode(s) in parquet "
                f"({info['total_frames']} total frames). Root: {root}"
            )
        return {
            "status": status,
            "content": [
                {"text": text},
                {
                    "json": {
                        "expected": expected,
                        "actual": actual,
                        "info_total_episodes": info_total,
                        "sources_agree": sources_agree,
                        "episode_indices": info["episode_indices"],
                        "total_frames": info["total_frames"],
                        "total_frames_per_ep": info["frames_per_episode"],
                        "unreadable_files": list(unreadable),
                        "root": str(root),
                    }
                },
            ],
        }

    def _episode_contract_fields(
        self, *, requested: int, completed: int, saved: int, flush_deferred: bool = False
    ) -> dict[str, Any]:
        """Build the episode-count truth fields for a ``run_policy`` json block.

        Returns ``n_episodes_requested`` / ``n_episodes_completed`` /
        ``episodes_saved`` plus ``dataset_episode_indices`` (from the
        recorder's in-memory bookkeeping; ``[]`` when not recording - see
        :meth:`verify_dataset_episodes` for the on-disk truth).
        ``flush_deferred`` marks the single-episode fast path while recording:
        the frames are buffered into the CURRENT episode and flush at the next
        ``save_episode`` / ``stop_recording``, so ``episodes_saved`` is ``0``.
        """
        fields: dict[str, Any] = {
            "n_episodes_requested": requested,
            "n_episodes_completed": completed,
            "episodes_saved": saved,
            "dataset_episode_indices": [],
        }
        if flush_deferred:
            fields["episode_flush_deferred"] = True
        if self._is_recording():
            recorder = self._active_recorder()
            total = getattr(getattr(recorder, "dataset", None), "meta", None)
            total_episodes = int(getattr(total, "total_episodes", 0) or 0) if total is not None else 0
            fields["dataset_episode_indices"] = list(range(total_episodes))
        return fields

    def save_episode(self) -> dict[str, Any]:
        """Flush the current recording episode and begin a fresh one.

        Backends that support dataset recording override this (see the MuJoCo
        ``RecordingMixin``). The base has no recorder, so it returns a
        structured error rather than pretending to flush.
        """
        return {
            "status": "error",
            "content": [{"text": "save_episode: this backend does not support dataset recording."}],
        }

    def start_policy(
        self,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: dict[str, Any] | None = None,
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Start policy execution in a background thread (non-blocking).

        Default implementation: synchronous passthrough to ``run_policy``.
        Backends that support true background execution should override.
        Accepts ``n_steps`` (primary) or legacy ``max_steps`` as an alternate
        to ``duration``; see :meth:`run_policy` for conversion rules and the
        ``policy_kwargs`` goal-payload contract.
        """
        robot_name = self._resolve_single_robot(robot_name)
        return self.run_policy(
            robot_name,
            policy_provider=policy_provider,
            policy_config=policy_config,
            instruction=instruction,
            duration=duration,
            control_frequency=control_frequency,
            action_horizon=action_horizon,
            fast_mode=fast_mode,
            video=video,
            policy_object=policy_object,
            n_steps=n_steps,
            max_steps=max_steps,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )

    def replay_episode(
        self,
        repo_id: str,
        robot_name: str | None = None,
        episode: int = 0,
        root: str | None = None,
        speed: float = 1.0,
        action_key_map: list[str] | None = None,
    ) -> dict[str, Any]:
        """Replay a LeRobotDataset episode via ``PolicyRunner.replay``.

        ``speed`` is a playback-rate multiplier (1.0 = real time); it must be
        a positive number (rejected with a structured error otherwise) and
        scales only the wall-clock rate - each recorded frame always advances
        physics for a full control period derived from the dataset fps.

        ``action_key_map`` binds recorded action-vector indices to action keys
        (default: :meth:`robot_action_keys`); it must be a non-empty
        list/tuple of unique strings matching the recorded action width, and
        is rejected rather than truncated to fit. A ``"success"`` status means
        every recorded frame reached the actuators - a frame ``send_action``
        could not apply aborts the replay with the frame index, the frames
        applied so far and the unresolved keys.
        """

        return PolicyRunner(self).replay(
            repo_id,
            robot_name=robot_name,
            episode=episode,
            root=root,
            speed=speed,
            action_key_map=action_key_map,
        )

    def eval_policy(
        self,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        n_episodes: int = 1,
        max_steps: int = 300,
        success_fn: str | None = None,
        policy_object: Policy | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        action_horizon: int = 8,
        seed: int | None = None,
        async_rtc: bool = False,
        rtc_inference_timeout_s: float | None = None,
        on_frame: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Multi-episode policy evaluation via ``PolicyRunner.evaluate``.

        ``robot_name`` resolves like :meth:`run_policy`: ``None`` (default)
        auto-selects the sole robot and errors with the candidate list when
        ambiguous or impossible. ``policy_object`` also mirrors
        :meth:`run_policy`: pass an already-built ``Policy`` to skip the
        ``create_policy`` round-trip; when omitted, the policy is built from
        ``policy_provider`` / ``policy_config``.

        ``n_episodes`` and ``max_steps`` must be positive integers,
        ``control_frequency`` must be ``> 0``, and an explicit
        ``control_substeps`` must be a positive integer; unusable values are
        rejected with a structured error at the entry point (before
        ``create_policy``) rather than running a degenerate eval.
        ``control_frequency`` / ``control_substeps`` give the eval loop the
        same full-control-period servo-tracking semantics as
        :meth:`run_policy`.

        ``async_rtc`` (default ``False``) opts into overlapping policy
        inference with action-chunk execution; the default keeps the
        success-rate synchronous and bit-stable. ``rtc_inference_timeout_s``
        bounds each async inference (structured error instead of a hung
        rollout).

        ``on_frame`` is an optional ``(step, observation, action) -> None``
        hook fired per applied control step on the eval thread, immediately
        after ``sim.send_action``; ``step`` is a monotonic index continuing
        across episode boundaries. Use it for synchronous recording/telemetry
        so a daemon-thread recorder does not race sim-state mutations. A
        non-``CooperativeStop`` hook exception is logged at WARN and never
        aborts the eval; raising
        :class:`~strands_robots.simulation.policy_runner.CooperativeStop`
        stops the eval gracefully after the episodes completed so far (the
        result carries ``stopped_early=True`` and ``episodes_completed``),
        matching :meth:`run_policy`.

        ``policy_kwargs`` is the per-call goal payload forwarded verbatim to
        every ``policy.get_actions(obs, instruction, **policy_kwargs)`` call,
        exactly as on :meth:`run_policy` (goal-conditioned providers read
        ``target_velocity`` / ``target_pose`` / ``target_joints`` /
        ``world_update`` from it).

        ``success_fn`` defaults to ``None``: with no criterion,
        ``success_rate`` is a hard ``0.0`` regardless of what the policy does,
        so this case logs a warning and sets ``success_measured=false`` in the
        returned json. Pass ``success_fn="contact"`` (or a callable) to
        measure real task success.

        ``video`` optionally records one rollout MP4 PER EPISODE (same dict
        schema as :meth:`run_policy`; path validated and camera probed
        up-front). ``_ep{i}`` is inserted into the filename per episode so
        episodes never overwrite each other, and the written files are
        returned in the result json ``video_paths``. Recording is unsupported
        on the ``evaluate_benchmark`` path.
        """
        robots = self.list_robots()
        if not robots:
            return {"status": "error", "content": [{"text": "No robots in sim. Add one first."}]}
        try:
            resolved_robot = self._resolve_single_robot(robot_name)
        except ValueError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
        if resolved_robot not in robots:
            return {
                "status": "error",
                "content": [{"text": self._unknown_robot_msg(resolved_robot)}],
            }

        if err := self._validate_video_config(video, "eval_policy"):
            return err
        if err := self._validate_policy_mapping(policy_config, "policy_config", "eval_policy"):
            return err
        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "eval_policy"):
            return err
        if err := self._validate_action_horizon(action_horizon, "eval_policy"):
            return err
        if err := self._validate_positive_int(n_episodes, "n_episodes", "eval_policy"):
            return err
        if err := self._validate_positive_int(max_steps, "max_steps", "eval_policy"):
            return err
        if err := self._validate_positive_frequency(control_frequency, "eval_policy"):
            return err
        if err := self._validate_control_substeps(control_substeps, "eval_policy"):
            return err
        # Plain-float coercion of the validated NumPy-scalar case; see run_policy.
        control_frequency = float(control_frequency)

        if policy_object is None:
            from strands_robots.policies import create_policy

            # Fail fast on a misconfiguration BEFORE the create_policy download.
            preflight_error = self._preflight_policy_config(resolved_robot, policy_provider, policy_config)
            if preflight_error is not None:
                return preflight_error
            policy = create_policy(policy_provider, **(policy_config or {}))
        else:
            # Pre-built policy path - mirror run_policy. Caller may have already
            # set robot_state_keys; we set defensively so semantics match the
            # provider path.
            policy = policy_object
        policy.set_robot_state_keys(self.robot_action_keys(resolved_robot))
        self.bind_policy_sim_context(policy, resolved_robot)

        return PolicyRunner(self).evaluate(
            resolved_robot,
            policy,
            instruction=instruction,
            n_episodes=n_episodes,
            max_steps=max_steps,
            success_fn=success_fn,
            control_frequency=control_frequency,
            control_substeps=control_substeps,
            action_horizon=action_horizon,
            seed=seed,
            async_rtc=async_rtc,
            rtc_inference_timeout_s=rtc_inference_timeout_s,
            on_frame=on_frame,
            policy_kwargs=policy_kwargs,
            video=video,
        )

    # Benchmark protocol facades

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        n_episodes: int = 1,
        seed: int | None = None,
        action_horizon: int = 8,
        on_frame: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        policy_object: Policy | None = None,
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a registered :class:`BenchmarkProtocol` against the current sim.

        Benchmark-agnostic evaluation entry point. Looks up ``benchmark_name``
        in the global benchmark registry, validates robot compatibility, and
        forwards to :meth:`PolicyRunner.evaluate` with the spec.
        ``max_steps`` comes from the benchmark (not a parameter here).

        Args:
            benchmark_name: Key from :func:`register_benchmark` /
                :func:`register_benchmark_from_file`.
            robot_name: Robot to evaluate. If ``None`` and the sim has exactly
                one loaded robot, that robot is picked; otherwise returns an
                error.
            policy_provider: Policy provider name (forwarded to
                :func:`create_policy`).
            policy_config: Provider-specific kwargs.
            instruction: Natural-language instruction for the policy.
            n_episodes: Number of episodes; must be a positive integer
                (rejected with a structured error otherwise).
            seed: Master RNG seed for per-episode reproducibility.
            action_horizon: How many actions to consume from each
                ``policy.get_actions(...)`` chunk before re-querying. Default
                ``8`` matches the upstream GR00T LIBERO eval the checkpoints
                were trained against; set to ``1`` for closed-loop
                receding-horizon control ONLY for single-action policies - the
                interval is clamped up to the policy's ``execution_horizon``,
                so a chunk-emitting policy still consumes its full chunk
                open-loop. Values < 1 are rejected with a structured error.
                ``on_step`` and success/failure checks run after EACH applied
                action, so per-step rewards and early termination work
                correctly regardless of horizon.
            on_frame: Optional ``(step, observation, action) -> None`` hook
                fired per applied control step on the eval thread, immediately
                after ``sim.send_action``. Use it for synchronous recording or
                telemetry when the eval is dispatched off the script main
                thread, where a daemon-thread recorder would race sim-state
                mutations. Raising
                :class:`~strands_robots.simulation.policy_runner.CooperativeStop`
                from the hook ends the benchmark gracefully after the episodes
                completed so far - the result json carries
                ``stopped_early=True`` and ``episodes_completed`` (matching
                :meth:`run_policy` / :meth:`eval_policy`); any in-progress
                episode's partial video is closed cleanly and is NOT listed in
                ``video_paths``. A non-``CooperativeStop`` hook exception is
                logged at WARN and never aborts the eval.
            policy_kwargs: Per-call goal payload forwarded verbatim to every
                ``policy.get_actions(obs, instruction, **policy_kwargs)`` call
                (same contract as :meth:`run_policy` / :meth:`eval_policy`);
                a benchmark driving a goal-conditioned policy must pass its
                goal keys here or the policy runs with an empty goal.
            control_frequency: Target Hz for ``policy.get_actions`` calls,
                used to derive the physics substeps per action so the loop
                steps a full control period. Must be ``> 0``; defaults to
                ``50.0``. Set it to the rate the policy was trained at - the
                benchmark's ``max_steps`` maps to a wall-clock episode length
                that depends on this rate.
            control_substeps: Explicit physics substeps per action, overriding
                the ``control_frequency``-derived value (mirrors
                :meth:`eval_policy`); must be a positive integer or ``None``
                (default, auto-derive).
            policy_object: An already-built :class:`Policy` to evaluate,
                skipping the ``create_policy`` round-trip (mirrors
                :meth:`run_policy` / :meth:`eval_policy`). When ``None`` the
                policy is built from ``policy_provider`` / ``policy_config``.
            video: Optional per-episode rollout MP4 config (same dict schema
                as :meth:`run_policy` / :meth:`eval_policy`). One file per
                episode with ``_ep{i}`` inserted into the filename. Frames are
                captured synchronously on the eval thread (render is
                read-only), so recording does not perturb the bit-stable
                benchmark rollout. Written paths are returned in the result
                json ``video_paths``. ``None`` (default) records nothing.

        Returns:
            Standard status dict. On success, carries per-episode cumulative
            reward + aggregate success_rate / avg_reward / avg_steps in the
            JSON payload, plus ``video_paths`` (the per-episode MP4s written
            when ``video`` is set).
        """
        from strands_robots.policies import create_policy
        from strands_robots.simulation.benchmark import get_benchmark

        if err := self._validate_video_config(video, "evaluate_benchmark"):
            return err
        if err := self._validate_policy_mapping(policy_config, "policy_config", "evaluate_benchmark"):
            return err
        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "evaluate_benchmark"):
            return err
        if err := self._validate_action_horizon(action_horizon, "evaluate_benchmark"):
            return err
        if err := self._validate_positive_int(n_episodes, "n_episodes", "evaluate_benchmark"):
            return err
        if err := self._validate_positive_frequency(control_frequency, "evaluate_benchmark"):
            return err
        if err := self._validate_control_substeps(control_substeps, "evaluate_benchmark"):
            return err
        # Plain-float coercion of the validated NumPy-scalar case; see run_policy.
        control_frequency = float(control_frequency)

        spec = get_benchmark(benchmark_name)
        if spec is None:
            from strands_robots.simulation.benchmark import list_benchmarks as _list

            available = sorted(_list().keys())
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"evaluate_benchmark: no benchmark registered under "
                            f"{benchmark_name!r}. Registered: {available}. "
                            "Call register_benchmark_from_file or register_benchmark first."
                        )
                    }
                ],
            }

        robots = self.list_robots()
        if not robots:
            return {"status": "error", "content": [{"text": "No robots in sim. Add one first."}]}

        resolved_robot = robot_name
        if not resolved_robot:
            # Try to pick a robot. Prefer single-robot scenes; multi-robot
            # scenes require explicit selection.
            if len(robots) == 1:
                resolved_robot = robots[0]
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"evaluate_benchmark: 'robot_name' is required when the sim has "
                                f"multiple robots. Loaded: {robots}"
                            )
                        }
                    ],
                }
        if resolved_robot not in robots:
            return {
                "status": "error",
                "content": [{"text": self._unknown_robot_msg(resolved_robot)}],
            }

        if policy_object is None:
            policy = create_policy(policy_provider, **(policy_config or {}))
        else:
            # Pre-built policy path - mirror run_policy / eval_policy. Lets a
            # caller benchmark an already-loaded checkpoint (e.g. a multi-GB
            # VLA) without a create_policy round-trip / redundant reload.
            policy = policy_object
        policy.set_robot_state_keys(self.robot_action_keys(resolved_robot))
        self.bind_policy_sim_context(policy, resolved_robot)

        return PolicyRunner(self).evaluate(
            resolved_robot,
            policy,
            instruction=instruction,
            n_episodes=n_episodes,
            spec=spec,
            seed=seed,
            action_horizon=action_horizon,
            control_frequency=control_frequency,
            control_substeps=control_substeps,
            on_frame=on_frame,
            policy_kwargs=policy_kwargs,
            video=video,
        )

    def list_benchmarks(self) -> dict[str, Any]:
        """Enumerate registered benchmarks.

        Returns a standard status dict whose JSON payload contains the
        :func:`~strands_robots.simulation.benchmark.list_benchmarks`
        metadata snapshot. Safe to call from any backend; the registry is
        engine-agnostic.
        """
        from strands_robots.simulation.benchmark import list_benchmarks as _list

        snapshot = _list()
        if not snapshot:
            text = "No benchmarks registered. Use register_benchmark_from_file to add one."
        else:
            lines = [f"Registered benchmarks ({len(snapshot)}):"]
            for name, meta in snapshot.items():
                lines.append(
                    f"  • {name}: {meta['class']} "
                    f"(robots={meta['supported_robots'] or 'any'}, "
                    f"default={meta['default_robot']}, "
                    f"max_steps={meta['max_steps']})"
                )
            text = "\n".join(lines)
        return {
            "status": "success",
            "content": [{"text": text}, {"json": {"benchmarks": snapshot}}],
        }

    def register_benchmark_from_file(
        self,
        benchmark_name: str,
        spec_path: str,
    ) -> dict[str, Any]:
        """Load a declarative benchmark spec from disk and register it.

        Wraps :func:`strands_robots.simulation.benchmark_spec.register_benchmark_from_file`
        so agents can author benchmarks as YAML / JSON at runtime. Parsing
        errors surface as structured error dicts rather than exceptions.
        """
        from strands_robots.simulation.benchmark_spec import (
            register_benchmark_from_file as _register,
        )

        if not benchmark_name:
            return {
                "status": "error",
                "content": [{"text": "register_benchmark_from_file: 'benchmark_name' must be non-empty."}],
            }
        if not spec_path:
            return {
                "status": "error",
                "content": [{"text": "register_benchmark_from_file: 'spec_path' must be non-empty."}],
            }
        try:
            benchmark = _register(benchmark_name, spec_path)
        except FileNotFoundError as e:
            return {"status": "error", "content": [{"text": f"register_benchmark_from_file: {e}"}]}
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"register_benchmark_from_file: {e}"}]}
        except ImportError as e:
            # YAML support requires pyyaml; surface the install hint verbatim.
            return {"status": "error", "content": [{"text": f"{e}"}]}
        except Exception as e:  # noqa: BLE001 - defensive catch-all with clear message
            return {
                "status": "error",
                "content": [{"text": f"register_benchmark_from_file: unexpected error: {e}"}],
            }

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Registered benchmark '{benchmark_name}' from {spec_path}\n"
                        f"  class: {type(benchmark).__name__}\n"
                        f"  supported_robots: {benchmark.supported_robots or 'any'}\n"
                        f"  default_robot: {benchmark.default_robot}\n"
                        f"  max_steps: {benchmark.max_steps}"
                    )
                }
            ],
        }

    def register_builtin_benchmarks(self) -> dict[str, Any]:
        """Register the built-in benchmark specs shipped with strands_robots.

        Wraps :func:`strands_robots.simulation.builtin_benchmarks.register_builtin_benchmarks`
        so the shipped specs (e.g. ``go2_walk_forward``) become discoverable
        via :meth:`list_benchmarks` and runnable via
        :meth:`evaluate_benchmark`. Opt-in and idempotent; importing
        strands_robots performs no registry mutation.

        Returns:
            A status dict whose JSON payload carries the ``registered`` list
            of benchmark names.
        """
        from strands_robots.simulation.builtin_benchmarks import (
            register_builtin_benchmarks as _register,
        )

        names = _register()
        return {
            "status": "success",
            "content": [
                {"text": f"Registered {len(names)} built-in benchmark(s): {', '.join(names)}"},
                {"json": {"registered": names}},
            ],
        }

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Override to return an ``on_frame(step, obs, action)`` callable.

        Used by backends that want to layer in recording / telemetry without
        subclassing :class:`PolicyRunner`. Default: no hook (``None``).
        """
        return None

    # Optional overrides (have default no-op implementations)

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Load a complete scene from file. Override per backend."""
        raise NotImplementedError("load_scene not implemented by this backend")

    def randomize(self, **kwargs: Any) -> dict[str, Any]:
        """Apply domain randomization.

        Concrete backends define their own parameter signatures. Because this
        base signature is ``**kwargs``-typed, an override inherits a sink that
        would swallow any keyword it does not declare; backends must reject the
        residual keys (see :func:`unknown_kwargs_error`) so a misspelled axis
        cannot report success while leaving that axis untouched.
        Override per backend.
        """
        raise NotImplementedError("randomize not implemented by this backend")

    def set_obs_noise(self, **kwargs: Any) -> dict[str, Any]:
        """Configure additive sensor noise on observations.

        Models real-sensor measurement noise (joint encoders, camera frames)
        so policies are not trained on noise-free observations. Concrete
        backends define their own parameter signatures and, as for
        :meth:`randomize`, must reject keywords they do not declare rather than
        let this ``**kwargs``-typed signature swallow them. Override per backend.
        """
        raise NotImplementedError("set_obs_noise not implemented by this backend")

    def get_contacts(self) -> dict[str, Any]:
        """Get contact information. Override per backend."""
        raise NotImplementedError("get_contacts not implemented by this backend")

    # Raw-frame render APIs (programmatic, not tool-envelope). Optional per
    # backend, but every in-tree backend implements them; they are the
    # substrate for strands_robots.rendering.HybridCompositor.

    def get_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Render a camera to raw ``(rgb, depth)`` ndarrays.

        The numeric-array counterpart of :meth:`render` (which wraps pixels in
        the agent-tool PNG envelope); in-process consumers use this to get
        pixels without a PNG round-trip.

        Args:
            camera_name: name of a camera previously added via ``add_camera``
                (backends supporting a free camera also accept their free-cam
                tokens for the RGB path).
            width: image width in pixels; ``None`` uses the camera's
                configured resolution.
            height: image height in pixels; ``None`` uses the camera's
                configured resolution.

        Returns:
            ``(rgb, depth)`` where ``rgb`` is ``(H, W, 3) uint8`` and
            ``depth`` is ``(H, W) float32`` metric meters, or ``None`` on
            backends with no depth path (Newton). Backends must never
            substitute silently wrong pixels -- failures raise.

        Raises:
            KeyError: unknown camera name.
            ValueError: invalid render dimensions.
            RuntimeError: no world / renderer unavailable / backend render
                failure.
            NotImplementedError: backend has no raw-frame path.
        """
        raise NotImplementedError("get_frame not implemented by this backend")

    def get_camera_params(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> CameraParams:
        """Return pinhole intrinsics/extrinsics for a named camera.

        The returned :class:`strands_robots.rendering.CameraParams` carries
        the intrinsic matrix ``K`` (pixels), the world-from-camera SE(3) pose
        ``T_world_cam`` in the OpenGL optical convention (+X right, +Y up,
        **-Z forward**), the image size, and the clip planes. Backends whose
        native camera basis differs apply the fixed basis correction here so
        consumers never see a backend-specific frame.

        Args:
            camera_name: name of a camera previously added via ``add_camera``.
                Backends with a free camera also accept their free-cam tokens,
                reporting the same view :meth:`get_frame` renders.
            width: image width to compute ``K`` for; ``None`` uses the
                camera's configured resolution.
            height: image height to compute ``K`` for; ``None`` uses the
                camera's configured resolution.

        Raises:
            KeyError: unknown camera name.
            ValueError: a camera whose projection no pinhole ``K`` can
                represent (e.g. an orthographic camera), or a resolution the
                backend cannot honor.
            RuntimeError: no world created.
            NotImplementedError: backend has no camera-params path.
        """
        raise NotImplementedError("get_camera_params not implemented by this backend")

    # Hard cap on pixels per get_world_point call: bounds the per-call work an
    # LLM can request (agents ground a handful of samples per object; a whole
    # image belongs to get_frame, not this lookup).
    _WORLD_POINT_MAX_PIXELS = 1024

    def get_world_point(
        self,
        camera_name: str = "default",
        pixels: Sequence[Sequence[SupportsFloat]] | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Ground image pixels to metric world coordinates via the depth buffer.

        The perception half of deployment-shaped grounding (Harness VLA,
        arXiv:2607.08448, Appendix E.2): instead of reading privileged object
        poses (:meth:`get_body_state` -- sim-only oracle truth), the agent
        picks pixels on the visible surface of the target in the RGB frame and
        this call unprojects each one through the pixel-aligned metric depth
        buffer -- ``p_cam = depth * K^-1 @ [u, v, 1]`` in the OpenGL optical
        frame, then ``p_world = T_world_cam @ p_cam``. The same call shape
        works on hardware with an RGB-D camera, so grounding built on it
        transfers.

        Guidance for agents (the paper's localization rule):

        1. Render the camera first (``render`` / ``get_frame``) and pick
           pixels ON the visible surface of the target object.
        2. Avoid rims, edges, reflections, transparent surfaces, and
           background pixels -- depth there is unstable or belongs to
           something else.
        3. Sample SEVERAL pixels on the same surface (typically 3-9): the
           returned ``point`` is the median over the valid samples, which
           rejects stray outliers. The median is PER-COMPONENT, so on a
           strongly tilted surface the combined ``[x, y, z]`` may lie on no
           single sampled point - treat it as a robust surface estimate, not
           as one of ``points``.
        4. Pixels with no depth (background / far plane) are dropped, not
           zero-filled; check ``n_valid`` against the count you sent.
        5. Re-localize after any robot, camera, or object motion -- world
           points are snapshots, not tracks.

        Depth samples are treated as z-depth (distance along the optical
        axis), the convention every in-tree backend emits. Pixels are indexed
        ``[u, v]`` with ``u`` the column from the left and ``v`` the row from
        the top; the unprojection uses the pixel center ``(u + 0.5, v + 0.5)``.

        Atomicity: when the backend exposes an engine lock (``self._lock``,
        all in-tree backends), the frame render and the camera-params read
        happen under it, so a concurrent scene mutation cannot slip between
        the two. All failures return a structured error dict -- this is a
        tool-envelope method and never raises.

        Args:
            camera_name: a camera previously added via ``add_camera``
                (backends with a free camera also accept their free-cam
                tokens, as for :meth:`get_frame`).
            pixels: non-empty list of ``[u, v]`` pixel coordinates
                (integer-valued; at most ``_WORLD_POINT_MAX_PIXELS``).
            width: image width; ``None`` uses the camera's configured
                resolution.
            height: image height; ``None`` uses the camera's configured
                resolution.

        Returns:
            On success ``{"status": "success", "content": [{"text": ...},
            {"json": {"point": [x, y, z], "points": [...], "n_valid": int,
            "n_requested": int, "camera": str, "width": int, "height": int}}]}``
            where ``point`` is the per-component median over the valid
            samples and ``points`` is aligned with the input ``pixels``
            (``None`` where the pixel had no valid depth). Backends without a
            metric-depth path (Newton), all-invalid pixel sets, out-of-bounds
            pixels, and malformed input all return
            ``{"status": "error", "content": [{"text": ...}]}``.
        """
        # numpy stays TYPE_CHECKING-only at this module's top level; import at
        # use time like the backends' render paths do.
        import numpy as np

        def _err(msg: str) -> dict[str, Any]:
            return {"status": "error", "content": [{"text": msg}]}

        # -- Structural validation (before any render work) -- #
        # camera_name reaches backend name-lookup APIs (e.g. MuJoCo's
        # mj_name2id) that raise TypeError on non-string input, and the
        # dispatcher enforces no scalar types - so an LLM passing a camera
        # INDEX must be caught here to keep the never-raises envelope.
        if camera_name is not None and not isinstance(camera_name, str):
            return _err(
                f"get_world_point 'camera_name' must be a camera name string, got "
                f"{type(camera_name).__name__} ({camera_name!r}). Cameras are addressed by "
                "name, not index; see add_camera / get_frame."
            )
        if pixels is None or isinstance(pixels, (str, bytes)) or not hasattr(pixels, "__len__"):
            return _err(
                "get_world_point requires 'pixels': a non-empty list of [u, v] pixel coordinates, e.g. [[320, 240], [322, 238]]."
            )
        if len(pixels) == 0:
            return _err("get_world_point requires at least one [u, v] pixel; got an empty list.")
        if len(pixels) > self._WORLD_POINT_MAX_PIXELS:
            return _err(
                f"get_world_point accepts at most {self._WORLD_POINT_MAX_PIXELS} pixels per call, "
                f"got {len(pixels)}. Sample a handful of pixels on the target surface instead."
            )
        parsed: list[tuple[int, int]] = []
        for i, px in enumerate(pixels):
            if isinstance(px, (str, bytes)) or not hasattr(px, "__len__") or len(px) != 2:
                return _err(f"pixels[{i}] must be a [u, v] pair, got {px!r}.")
            coords: list[int] = []
            for axis, component in zip("uv", px, strict=True):
                if not isinstance(component, numbers.Real) or isinstance(component, bool):
                    return _err(f"pixels[{i}] {axis} must be numeric, got {type(component).__name__}.")
                value = float(component)
                if not math.isfinite(value):
                    return _err(f"pixels[{i}] {axis} must be finite, got {value}.")
                if not value.is_integer():
                    return _err(
                        f"pixels[{i}] {axis} must be an integer pixel index, got {value} "
                        "(fractional pixels are rejected, never silently truncated)."
                    )
                coords.append(int(value))
            parsed.append((coords[0], coords[1]))

        # -- Render + camera params (atomic under the engine lock) -- #
        lock = getattr(self, "_lock", None)
        ctx = lock if lock is not None else contextlib.nullcontext()
        with ctx:
            try:
                _rgb, depth = self.get_frame(camera_name, width=width, height=height)
            except NotImplementedError:
                return _err(
                    "get_world_point is unavailable: this backend has no raw-frame path (get_frame is not implemented)."
                )
            except (KeyError, ValueError, RuntimeError, TypeError) as e:
                # TypeError included as defense-in-depth for backend lookup
                # APIs that reject non-string names (the type is validated
                # above, but the envelope must hold regardless).
                return _err(f"get_world_point failed to render camera frame: {e}")
            if depth is None:
                return _err(
                    f"get_world_point is unavailable on this backend: camera '{camera_name}' produced no "
                    "metric depth (get_frame returned depth=None; e.g. Newton's ray-traced camera has no "
                    "depth output). Use a depth-capable backend (MuJoCo, Isaac) or an RGB-D camera."
                )
            h, w = int(depth.shape[0]), int(depth.shape[1])
            for i, (u, v) in enumerate(parsed):
                if not (0 <= u < w and 0 <= v < h):
                    return _err(
                        f"pixels[{i}] = [{u}, {v}] is outside the rendered {w}x{h} frame "
                        f"(valid u: 0..{w - 1}, v: 0..{h - 1})."
                    )
            try:
                cam = self.get_camera_params(camera_name, width=w, height=h)
            except NotImplementedError:
                return _err(
                    "get_world_point is unavailable: this backend has no camera-params path (get_camera_params is not implemented)."
                )
            except (KeyError, ValueError, RuntimeError, TypeError) as e:
                return _err(f"get_world_point failed to read camera parameters: {e}")

        # -- Unproject (pure math; no engine state touched past this point) -- #
        fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
        cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
        # Background convention across backends: MuJoCo pins no-geometry
        # pixels to exactly zfar; Isaac reports 0 or non-finite. A small
        # relative margin below zfar absorbs float rounding at the far plane.
        zfar_cut = float(cam.zfar) * (1.0 - 1e-6)
        points: list[list[float] | None] = []
        valid_points: list[list[float]] = []
        for u, v in parsed:
            d = float(depth[v, u])
            if not (math.isfinite(d) and 0.0 < d < zfar_cut):
                points.append(None)
                continue
            # Pixel center -> OpenGL optical frame (+X right, +Y up, -Z
            # forward): image v grows down so y flips, and z-depth lies
            # along -Z.
            x_cam = (u + 0.5 - cx) / fx * d
            y_cam = -((v + 0.5 - cy) / fy) * d
            p_world = cam.T_world_cam @ np.array([x_cam, y_cam, -d, 1.0], dtype=np.float64)
            world_xyz = [float(p_world[0]), float(p_world[1]), float(p_world[2])]
            points.append(world_xyz)
            valid_points.append(world_xyz)

        # One label per camera: every free-camera token (None / "" / "free" /
        # "default") reports as "default", so the same camera never appears
        # under two names across calls.
        camera_label = "default" if camera_name in (None, "", "free", "default") else str(camera_name)
        if not valid_points:
            return _err(
                f"get_world_point found no valid depth at any of the {len(parsed)} requested pixels via "
                f"camera '{camera_label}': every sample hit the background / far plane (zfar={cam.zfar:g} m) "
                "or had no depth. Pick pixels on the visible surface of the target object -- avoid sky, "
                "rims/edges, reflections, and background."
            )

        median = np.median(np.asarray(valid_points, dtype=np.float64), axis=0)
        point = [float(median[0]), float(median[1]), float(median[2])]
        n_valid = len(valid_points)
        text = (
            f"World point [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}] m "
            f"(median over {n_valid}/{len(parsed)} valid pixels) via camera '{camera_label}'."
        )
        return {
            "status": "success",
            "content": [
                {"text": text},
                {
                    "json": {
                        "point": point,
                        "points": points,
                        "n_valid": n_valid,
                        "n_requested": len(parsed),
                        "camera": camera_label,
                        "width": w,
                        "height": h,
                    }
                },
            ],
        }

    # Discovery / introspection

    def describe(self) -> dict[str, Any]:
        """Return a machine-readable summary of this engine's live contract.

        Agents should call this first to learn what robots exist, what
        cameras are attached, and the most commonly needed method signatures.

        Returns:
            Plain dict with keys: robots, cameras, methods, note.
        """
        return {
            "robots": self.list_robots(),
            "cameras": [],  # backends override to list camera names
            "methods": {
                "get_robot_state": "(robot_name: str) -> dict",
                "get_observation": "(robot_name: str | None = None, *, skip_images: bool = False) -> dict",
                "send_action": "(action: dict, robot_name: str | None = None, n_substeps: int = 1) -> dict",
                "add_robot": (
                    "(name: str, urdf_path=None, data_config=None, position=None, "
                    "orientation=None) -> dict  # add a robot to the scene by "
                    "registry name (or urdf_path); the first scene-construction step"
                ),
                "add_object": (
                    "(name: str, shape='box', position=None, orientation=None, "
                    "size=None, color=None, mass=0.1, is_static=None, mesh_path=None, "
                    "material=None) -> dict  # add a manipulable object "
                    "(cube/sphere/.../mesh) to the scene. material is an optional "
                    "dict for matte/textured surfaces: keys reflectance|specular|"
                    "shininess (0..1), texture (abs image path) OR builtin "
                    "(checker|gradient|flat) + rgb1/rgb2/texdim, texrepeat [u,v]; "
                    "any other key (or an empty dict) is rejected, never ignored"
                ),
                "remove_object": "(name: str) -> dict  # remove a previously added object",
                "remove_robot": (
                    "(name: str) -> dict  # remove a robot (and every scene "
                    "element it introduced) from the world; the inverse of "
                    "add_robot, completing the add/remove pair alongside "
                    "remove_object"
                ),
                "run_policy": (
                    "(robot_name: str, policy_provider='mock', n_episodes=1, "
                    "reset_between=True, stop_when=None, ...) -> dict  # "
                    "stop_when: optional semantic early-return clause in the "
                    "benchmark success: predicate DSL - a single "
                    "{'predicate': <name>, ...} call or an {'all'/'any': "
                    "[...]} group - checked against the sim after every "
                    "applied action so the rollout ends as soon as the world "
                    "reaches the state; the result json reports "
                    "stopped_reason ('predicate'|'budget'|'cancelled'; "
                    "'error' on failures) + steps_used so a caller can decide "
                    "whether to retry"
                ),
                "start_policy": "(robot_name: str, policy_provider='mock', ...) -> dict",
                "eval_policy": (
                    "(robot_name: str, policy_provider='mock', n_episodes=1, "
                    "max_steps=300, success_fn=None, ...) -> dict  # multi-episode "
                    "success-rate evaluation (the rollout sibling of run_policy)"
                ),
                "evaluate_benchmark": (
                    "(benchmark_name: str, robot_name=None, policy_provider='mock', "
                    "n_episodes=1, seed=None, video=None, ...) -> dict  # score a "
                    "registered benchmark's success/failure/dense-reward DSL over a "
                    "rollout (max_steps comes from the benchmark, not a parameter); "
                    "the DSL-scored sibling of eval_policy's success_fn"
                ),
                "list_benchmarks": (
                    "() -> dict  # enumerate registered benchmarks (names, "
                    "supported robots, default robot, max_steps) - the source of the "
                    "benchmark_name evaluate_benchmark expects"
                ),
                "register_benchmark_from_file": (
                    "(benchmark_name: str, spec_path: str) -> dict  # author a "
                    "declarative benchmark (success/failure/dense_reward predicate "
                    "DSL) as YAML/JSON at runtime and register it under benchmark_name"
                ),
                "register_builtin_benchmarks": (
                    "() -> dict  # register the shipped built-in velocity-tracking "
                    "locomotion benchmarks - the go2_walk_forward quadruped task and "
                    "the g1_walk_forward / t1_walk_forward humanoid tasks - so they "
                    "appear in list_benchmarks and can be run via evaluate_benchmark"
                ),
                "replay_episode": (
                    "(repo_id: str, robot_name=None, episode=0, root=None, "
                    "speed=1.0, action_key_map=None) -> dict  # replay a recorded "
                    "LeRobotDataset episode through the sim; action_key_map needs "
                    "one unique key per recorded action index (default: "
                    "robot_action_keys) and status='success' means every frame "
                    "reached the actuators"
                ),
                "list_robots": "() -> list[str]",
                "get_features": (
                    "(robot_name: str | None = None) -> dict  # joint / "
                    "actuator / camera / robot names of the scene (scoped to "
                    "one robot when robot_name is given) - the source of truth "
                    "for the action keys a policy must emit; consult it when "
                    "run_policy reports unresolved keys"
                ),
                "render": "(camera_name='default', width=None, height=None) -> dict",
                "create_world": (
                    "(timestep=None, gravity=None, ground_plane=True, terrain=None, "
                    "difficulty=1.0) -> dict  # create a fresh simulation world - the "
                    "world-lifecycle entry point that precedes add_robot / add_object. "
                    "gravity is [gx, gy, gz]; ground_plane lays a floor; terrain lays a "
                    "deterministic locomotion heightfield instead of the flat plane "
                    "('rough' value-noise bumps, 'stairs' step plateaus rising +x, "
                    "'pyramid' concentric steps rising to the centre, 'slope' a "
                    "constant-grade ramp); difficulty (finite, > 0; 1.0 = full height) "
                    "scales the terrain peak elevation for a curriculum without changing "
                    "the terrain kind. Backends without heightfield support reject a "
                    "non-None terrain rather than ignoring it"
                ),
                "destroy": (
                    "() -> dict  # tear down the world and release all resources "
                    "(joins any running background policy first); the inverse of "
                    "create_world, called at session end"
                ),
                "reset": "() -> dict  # during recording, flushes the buffered rollout as one episode before resetting",
                "step": "(n_steps: int = 1) -> dict",
                "get_state": (
                    "() -> dict  # snapshot of the live world: sim time, step "
                    "count, timestep, gravity, and robot / object / camera / "
                    "body / joint / actuator counts (the whole-world sibling of "
                    "get_robot_state / get_observation)"
                ),
                "load_scene": (
                    "(scene_path: str) -> dict  # load a complete scene from "
                    "an MJCF/URDF file; the alternative scene-construction "
                    "entry point to building it up with add_robot / add_object"
                ),
                "randomize": (
                    "(**kwargs) -> dict  # domain randomization (colors, "
                    "lighting, physics, positions); each backend defines its "
                    "own opt-in axes - see the backend describe() for the "
                    "concrete signature"
                ),
                "set_obs_noise": (
                    "(**kwargs) -> dict  # configure additive Gaussian sensor "
                    "noise on joint observations and rendered frames so a "
                    "policy is not evaluated on noise-free observations"
                ),
                "get_contacts": (
                    "() -> dict  # active contacts at the current step - the "
                    "physics-grounding read used to verify a grasp or detect "
                    "a collision instead of trusting a rendered caption"
                ),
            },
            "note": (
                "robot_name defaults to the sole robot when only one exists "
                "for get_observation, send_action, get_robot_state, run_policy, "
                "and start_policy. With multiple robots, pass robot_name "
                "explicitly (from the 'robots' list above)."
            ),
        }

    def cleanup(self) -> None:
        """Release all resources. Called on __del__ / context exit."""
        pass

    def __enter__(self) -> SimEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception as e:
            # Best-effort cleanup during GC - exceptions can't propagate
            # from __del__ (CPython ignores them), so log for visibility.
            logger.warning("Cleanup error during __del__: %s", e)
