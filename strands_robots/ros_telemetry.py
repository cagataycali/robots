"""Shared rclpy publisher for robot telemetry on a ROS 2 domain.

Both the simulation bridge (:class:`strands_robots.simulation.ros_bridge.SimRosBridge`)
and the hardware bridge (:class:`strands_robots.hardware_ros_bridge.HardwareRosBridge`)
advertise the same per-robot topics so a real arm and its digital twin look
identical on the ROS 2 graph:

* ``/<robot>/joint_states`` (``sensor_msgs/msg/JointState``) - joint names and
  positions.
* ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``) - one
  message per camera frame.

The ROS 2 *wire contract* - topic names, name sanitization, and inbound
``joint_command`` parsing - lives in the transport-agnostic
:class:`RosTelemetryBase`, so the rclpy and pure-RTPS
(:class:`strands_robots.hardware_rtps_bridge.HardwareRtpsBridge`) transports
are byte-identical on the graph by construction, not by two independent
codepaths happening to agree. :class:`RosTelemetryBridge` adds the rclpy
machinery on top - node ownership, per-robot publisher caching, and message
construction - so the sim and hardware bridges are thin, symmetric subclasses
that differ only in their default node name. ``rclpy`` and the ROS 2 message
packages are optional, system-provided dependencies (they are not on PyPI);
they are imported lazily through
:func:`strands_robots.utils.require_optional`, so importing this module - and
running with the bridge disabled - never requires ROS 2.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

from strands_robots.utils import (
    dds_domain_id_error,
    finite_number_error,
    positive_count_error,
    refusal_repr,
    require_optional,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

logger = logging.getLogger(__name__)

# A ROS 2 name token is ASCII ``[A-Za-z0-9_]`` with no repeated underscore, so
# the sanitiser needs the ASCII class explicitly - ``str.isalnum`` is true of
# every Unicode letter and digit and would pass a character no ROS 2 name may
# hold. The rule these two enforce on a token is the one
# :data:`~strands_robots.rtps.mangling.ROS_TOPIC_RE` enforces on the whole name.
_NON_TOKEN_CHAR_RE = re.compile(r"[^A-Za-z0-9_]")
_UNDERSCORE_RUN_RE = re.compile(r"_+")


#: Remedy for a missing ``rclpy``, shared by every surface that refuses for want
#: of it. ``rclpy`` is not published on PyPI: it arrives with a system ROS 2
#: install, which is why the ``[ros2]`` extra declares only the pip-installable
#: cyclonedds RMW binding (see ``pyproject.toml`` and ``docs/ros2-integration.md``).
#: An install hint naming a pip command for it is therefore a remedy the caller
#: can follow to no effect - ``pip install 'strands-robots[ros2]'`` exits 0 with
#: ``rclpy`` exactly as missing, and ``pip install rclpy`` fails outright - so
#: this names the step that does supply it. Callers that also offer the pure-RTPS
#: transport append that alternative, which needs no sourced distro.
ROS2_SYSTEM_INSTALL_HINT = (
    "rclpy is not published on PyPI - it ships with a system ROS 2 install "
    "(apt / RoboStack / conda).\n"
    "Source a distro in the shell that launches this process:\n"
    "  source /opt/ros/jazzy/setup.bash   # or your distro / RoboStack / conda env\n"
    "The [ros2] extra installs only the cyclonedds RMW binding; it does not "
    "provision ROS 2 by itself."
)


#: Largest KEEP_LAST history depth the rclpy transport can be built with. The
#: depth reaches the middleware as the ``depth`` field of ``rmw_qos_profile_t``,
#: whose pybind11 binding takes a signed 32-bit integer, so this is the last
#: value that converts: one more raises ``TypeError: __init__(): incompatible
#: constructor arguments`` out of ``create_publisher`` - a message naming
#: neither the parameter nor the bridge. Measured against rclpy on ROS 2 Jazzy:
#: ``2**31 - 1`` builds a publisher, ``2**31`` does not.
MAX_QOS_HISTORY_DEPTH = 2**31 - 1


def _qos_history_depth_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable KEEP_LAST history depth.

    The depth is handed to ``create_publisher`` / ``create_subscription`` as the
    ``qos_or_depth`` argument, which rclpy turns into
    ``QoSProfile(depth=value, history=KEEP_LAST)``. It is the one bridge
    parameter whose consumer is not reached from the constructor: the publishers
    are built lazily, on the first ``publish_joint_states`` / ``publish_image``
    for a robot, so an unusable depth is a construction that reports success and
    a telemetry stream that fails - or silently is not the one asked for - later,
    on the first frame.

    Two failure modes, both measured against rclpy on ROS 2 Jazzy:

    * ``0`` is **accepted**. rclpy warns "A zero depth with KEEP_LAST doesn't
      make sense; no data could be stored. This will be interpreted as
      SYSTEM_DEFAULT" and builds the publisher with the middleware default, so
      the caller's declared depth is silently not the depth in force - and
      ``warnings.warn`` shows once per location, so a second bridge in the same
      process says nothing at all. ``False`` is that same value arriving by
      another route, and ``True`` is a silent depth of 1 - one frame of history
      on a stream the caller asked to buffer.
    * every other spelling raises from *inside* rclpy, naming no parameter:
      ``-1`` gives ``ValueError: history depth must be greater than or equal to
      zero``, a float / ``str`` / ``None`` / ``nan`` / ``inf`` gives
      ``TypeError: Expected QoSProfile or int``, and so does ``np.int64(10)``,
      which names a perfectly good depth. Above
      :data:`MAX_QOS_HISTORY_DEPTH` the pybind11 conversion fails instead.

    So the floor, the ``bool`` refusal and the strict-``int`` requirement all
    come from :func:`~strands_robots.utils.positive_count_error`, the shared
    domain for a discrete count consumed directly by a C API rather than
    coerced, and only the transport's ceiling is added here - beside the
    transport that has it, in the manner of ``_transport_port_error`` in
    :mod:`strands_robots.tools.use_rosbridge`.

    Args:
        value: The caller-supplied depth.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            class name, for a constructor parameter.

    Returns:
        An error message, or ``None`` when the transport can carry the depth.
    """
    if error := positive_count_error(value, param, context):
        return error
    if value > MAX_QOS_HISTORY_DEPTH:
        return (
            f"{context}: {param} {refusal_repr(value)} is a positive integer that the rclpy transport "
            f"cannot build a publisher with (it carries 1-{MAX_QOS_HISTORY_DEPTH}; the "
            "middleware QoS profile stores the depth as a signed 32-bit integer)"
        )
    return None


#: Env var an operator sets to explicitly run an INBOUND ``joint_command``
#: surface on an unsecured DDS graph (no ``dds_security_config``). Truthy values
#: mirror the mesh insecure opt-out (``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE``):
#: ``1``, ``true``, ``yes`` (case-insensitive). This is a deliberate second
#: factor so a forgotten config cannot silently expose a drivable arm.
ROS2_INSECURE_ENV = "STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE"

#: Keys a ``dds_security_config`` dict must supply, each as a non-empty string.
#: Each names a credential the RTPS bridge wires into its DDS Security
#: participant: the identity CA, the participant's own certificate and private
#: key (identity is unprovable without the key), and the signed governance +
#: permissions documents. ``permissions_ca`` is optional; supplied, it is held
#: to the same domain, because a value the participant QoS would drop must not
#: read as "a permissions CA is configured".
_DDS_SECURITY_REQUIRED_KEYS = (
    "identity_ca",
    "certificate",
    "private_key",
    "governance",
    "permissions",
)

#: Optional ``dds_security_config`` key, graded like a required one when present.
_DDS_SECURITY_OPTIONAL_KEY = "permissions_ca"


def _credential_problem(config: Mapping[str, Any], key: str) -> str | None:
    """Why ``config[key]`` is not a usable DDS Security credential, else ``None``.

    A credential is a non-empty string: a path, or a ``file:`` / ``data:`` URI
    per the OMG DDS-Security spec. The value itself is graded rather than its
    ``str()``, because ``str(None)`` is ``"None"`` - non-empty, and therefore
    indistinguishable from a real credential to a printed-emptiness check.

    Args:
        config: The operator-supplied ``dds_security_config``.
        key: The credential key to grade.

    Returns:
        ``"absent"``, ``"empty"``, or the received type's name - what the
        refusal quotes so the operator knows which key to fix and how - or
        ``None`` when the value is a credential.
    """
    if key not in config:
        return "absent"
    value = config[key]
    if not isinstance(value, str):
        return type(value).__name__
    return None if value.strip() else "empty"


class RosTelemetryBase:
    """Transport-agnostic ROS 2 wire contract shared by every telemetry bridge.

    The rclpy bridges (:class:`RosTelemetryBridge` and its
    :class:`~strands_robots.hardware_ros_bridge.HardwareRosBridge` /
    :class:`~strands_robots.simulation.ros_bridge.SimRosBridge` subclasses) and
    the pure-RTPS bridge
    (:class:`~strands_robots.hardware_rtps_bridge.HardwareRtpsBridge`) all derive
    from this base, so the ROS 2 topic names and the inbound ``joint_command``
    parsing live in exactly one place. That makes the two transports
    byte-identical on the ROS 2 graph by construction rather than relying on two
    independent codepaths staying in sync as the contract evolves.

    This base pulls in no transport library; each subclass owns its own
    rclpy / cyclonedds machinery and message construction.
    """

    @staticmethod
    def _safe(name: str, *, fallback: str = "robot") -> str:
        """Map a robot/camera name to a valid ROS 2 topic segment.

        The segment is a name token, so it is held to the token rules the ROS 2
        mapping states and :data:`~strands_robots.rtps.mangling.ROS_TOPIC_RE`
        enforces: ASCII ``[A-Za-z0-9_]`` only, no repeated underscore, and no
        leading digit. ``str.isalnum`` is true of every Unicode letter and
        digit, so it is not that character set - a camera named ``cam2`` (with
        a superscript) survived it and produced a topic the mangling then
        refused, and one named ``0`` or ``3rd_person`` produced a token ROS 2
        refuses while the mangling let it through. Both are silent on the DDS
        side, where matching is by topic name and no subscriber ever appears.

        Args:
            name: Robot or camera name, from the caller's observation keys.
            fallback: Token used when *name* holds nothing usable, and prefixed
                to a name that starts with a digit (which ROS 2 forbids at the
                start of a token, so the digits have to be carried rather than
                dropped - two cameras named ``0`` and ``1`` must not collapse
                onto one topic).

        Returns:
            A token :data:`~strands_robots.rtps.mangling.ROS_TOPIC_RE` accepts
            between two ``/`` separators.

        Note:
            Not injective, and cannot be: two names differing only in a run of
            separators (``front  cam`` and ``front - cam``) both render
            ``front_cam``, because no valid token may carry the difference. That
            is a collision on one reachable topic rather than, as before, two
            distinct topics under names ROS 2 refuses, which no subscriber could
            read either of.
        """
        token = _UNDERSCORE_RUN_RE.sub("_", _NON_TOKEN_CHAR_RE.sub("_", name)).strip("_")
        if not token:
            return fallback
        if token[0] in "0123456789":
            return f"{fallback}_{token}"
        return token

    @staticmethod
    def _resolve_robot_name(robot: Any) -> str:
        """Namespace topics under the bound robot's name.

        Prefers the lerobot device ``.name`` (e.g. ``so101``), falling back to
        the strands tool name, so ``joint_command`` and ``joint_states`` share
        one namespace - a controller can echo our published joint names straight
        back to drive the arm.
        """
        inner = getattr(robot, "robot", None)
        return getattr(inner, "name", None) or getattr(robot, "tool_name_str", None) or "robot"

    @classmethod
    def joint_states_topic(cls, robot: str) -> str:
        """ROS 2 topic a robot's ``JointState`` telemetry is published on."""
        return f"/{cls._safe(robot)}/joint_states"

    @classmethod
    def image_topic(cls, robot: str, camera: str) -> str:
        """ROS 2 topic a robot camera's ``Image`` frames are published on."""
        return f"/{cls._safe(robot)}/{cls._safe(camera, fallback='camera')}/image_raw"

    @classmethod
    def joint_command_topic(cls, robot: str) -> str:
        """ROS 2 topic inbound ``joint_command`` messages are read from."""
        return f"/{cls._safe(robot)}/joint_command"

    @staticmethod
    def _command_namespace_error(name: Any, context: str) -> str | None:
        """Return an error message if ``command_robot_name`` cannot name a topic.

        The hardware bridges let a caller override the namespace their inbound
        ``joint_command`` topic is read from. That override is the only
        caller-supplied value this module renders into a topic through
        :meth:`_safe` - every other name reaching it is derived internally, from
        the bound robot (:meth:`_resolve_robot_name`) or from an observation key.
        So it is the one place where a value from outside is handed to a
        ``re.sub`` that assumes a string, and both non-string outcomes are worse
        than a refusal:

        * **A truthy non-``str``** raised out of :meth:`_safe` - ``TypeError:
          expected string or bytes-like object, got 'int'``, or, for ``bytes``,
          ``cannot use a string pattern on a bytes-like object``. Neither names
          the parameter. Worse, it raised from *after* the transport was built:
          on the rclpy bridge the process-wide ``ROS_DOMAIN_ID`` had been
          rewritten, the context started and the node created; on the RTPS
          bridge the ``DomainParticipant`` existed. ``__init__`` raising returns
          no object, so the ``shutdown`` that releases them is unreachable.
        * **A falsy non-``str``** (``0``, ``[]``, ``False``) was worse still: it
          is filtered by the ``command_robot_name or <derived>`` fallback, so the
          bridge reported success and subscribed under the bound robot's own
          name - a namespace the caller never asked for, silently.

        ``None`` is the documented "derive it from the bound robot" input and is
        accepted, as is ``""``, which selects the same fallback. A ``str``
        subclass is accepted: it is a string to every operation that follows.

        Args:
            name: The caller's ``command_robot_name``.
            context: Calling context, used in error text (the bridge class name).

        Returns:
            ``None`` when *name* can name a topic, otherwise the message to
            raise as :class:`ValueError`.
        """
        if name is None or isinstance(name, str):
            return None
        return (
            f"{context}: 'command_robot_name' must be a string or None, got "
            f"{refusal_repr(name)} ({type(name).__name__}); it is rendered into "
            "the '/<name>/joint_command' topic this bridge reads commands from, "
            "which only a string can name. Pass None to read them under the "
            "bound robot's own name."
        )

    # -- outbound payload gate (shared by both transports) -----------------

    @staticmethod
    def _joint_state_arrays_error(names: list[str], positions: list[float], context: str) -> str | None:
        """Return an error message if a ``JointState``'s arrays name different joints.

        ``sensor_msgs/JointState`` pairs ``name`` and ``position`` BY INDEX, so
        the two arrays are one table with two columns and only agree on which
        joint a value belongs to while they are the same length. A consumer
        reads them with ``zip(msg.name, msg.position)`` - that is what
        ``robot_state_publisher`` and every plotter do - so an array shorter by
        one does not drop one joint: it shifts every joint after the gap onto
        its neighbour's value, and the tail falls off the end unreported. The
        arm is then published in a pose it is not in, under a well-formed
        message DDS delivers and no subscriber can tell is wrong.

        Refused whole, for the reason :meth:`_command_action` refuses a
        malformed inbound command whole rather than applying part of it: a
        partial state is not a state.

        Args:
            names: ``JointState.name`` the caller supplied.
            positions: ``JointState.position`` the caller supplied.
            context: Class name of the calling bridge, for the message.

        Returns:
            ``None`` when the two arrays describe the same joints, else the
            reason they cannot be published together.
        """
        if len(names) == len(positions):
            return None
        return (
            f"{context}: publish_joint_states got {len(names)} joint name(s) and "
            f"{len(positions)} position(s). A JointState pairs the two arrays by index, so "
            "publishing them would report every joint after the first gap under its "
            "neighbour's name. Pass one position per name - build the pair together, "
            "so a joint the observation does not carry drops its name too."
        )

    # -- security / safety gate (shared by both hardware bridges) ---------

    @staticmethod
    def _validate_joint_limits(
        joint_limits: dict[str, Any] | None,
    ) -> dict[str, tuple[float, float]] | None:
        """Validate and normalize a ``joint_limits`` mapping at construction.

        Returns ``{"<motor>.pos": (min, max)}`` with floats and ``min <= max``,
        or
        ``None`` when no limits are configured. Failing fast here (rather than
        per-command) means a malformed bound surfaces at bridge construction,
        not as a silent mid-run rejection of every command.

        Raises:
            ValueError: If ``joint_limits`` is not a mapping of name to a
                ``(min, max)`` pair of finite numbers with ``min <= max``.
        """
        if joint_limits is None:
            return None
        if not isinstance(joint_limits, dict):
            raise ValueError(f"joint_limits must be a dict[str, (min, max)], got {type(joint_limits).__name__}")
        normalized: dict[str, tuple[float, float]] = {}
        for name, bounds in joint_limits.items():
            try:
                low, high = bounds
            except (TypeError, ValueError):
                raise ValueError(f"joint_limits[{name!r}] must be a (min, max) numeric pair, got {bounds!r}") from None
            # Each bound goes through the shared signed-scalar domain before the
            # ordering comparison below, which cannot see a non-finite bound:
            # every comparison against ``nan`` is False, so ``(low, nan)`` passes
            # ``low > high`` and then makes the ``low <= pos <= high`` test in
            # :meth:`_command_action` False for every position - the bridge drops
            # EVERY inbound command for that joint, which is the silent mid-run
            # rejection this validator exists to surface at construction.
            # ``(inf, inf)`` and ``(-inf, -inf)`` pass the ordering check the same
            # way and admit nothing either. The domain also answers for an int
            # past the float64 range, which raised ``OverflowError`` out of the
            # ``float()`` below rather than the documented ``ValueError``. The
            # sibling declaration of this parameter,
            # :class:`~strands_robots.simulation.isaac.delta_eef.IsaacDeltaEEFController`,
            # refuses a non-finite bound for the same reason.
            for side, bound in (("min", low), ("max", high)):
                if error := finite_number_error(bound, side, f"joint_limits[{name!r}]"):
                    raise ValueError(error)
            low, high = float(low), float(high)
            if low > high:
                raise ValueError(f"joint_limits[{name!r}] has min {low} > max {high}")
            normalized[str(name)] = (low, high)
        return normalized

    @staticmethod
    def _validate_dds_security_config(config: Any) -> dict[str, str]:
        """Validate a ``dds_security_config`` dict supplies every required key.

        Returns the config unchanged on success. The required keys
        (:data:`_DDS_SECURITY_REQUIRED_KEYS`) are the credentials a DDS Security
        participant cannot authenticate or be governed without; each must be a
        non-empty string (a path, ``file:`` / ``data:`` URI per the DDS Security
        spec). Validated at construction so a half-filled config refuses the
        bridge rather than silently degrading wire security.

        The value is graded, not its ``str()``: a printed-emptiness check read
        ``None`` as the non-empty ``"None"``, so a credential this validator had
        accepted could still be one
        :meth:`~strands_robots.hardware_rtps_bridge.HardwareRtpsBridge._build_security_qos`
        drops (it sets a property per *truthy* credential), putting a
        participant on the wire with the auth plugin loaded and no private key
        or governance; a truthy non-string was wired as its ``repr`` instead, so
        a ``bytes`` path reached cyclonedds as the literal ``"b'file:/key.pem'"``.
        Every value accepted here is therefore one the QoS keeps verbatim.

        Raises:
            ValueError: If ``config`` is not a dict, or any required key - or a
                supplied ``permissions_ca`` - is absent, empty, or not a string.
        """
        if not isinstance(config, dict):
            raise ValueError(f"dds_security_config must be a dict, got {type(config).__name__}")
        unusable = {k: problem for k in _DDS_SECURITY_REQUIRED_KEYS if (problem := _credential_problem(config, k))}
        if unusable:
            raise ValueError(
                f"dds_security_config is missing required keys: {unusable}. "
                f"All of {list(_DDS_SECURITY_REQUIRED_KEYS)} must be supplied as non-empty "
                "strings (identity CA, participant certificate + private key, governance, "
                "permissions)."
            )
        if _DDS_SECURITY_OPTIONAL_KEY in config and (
            problem := _credential_problem(config, _DDS_SECURITY_OPTIONAL_KEY)
        ):
            raise ValueError(
                f"dds_security_config[{_DDS_SECURITY_OPTIONAL_KEY!r}] is {problem}; it must be a "
                "non-empty string, or omit the key. A value the participant QoS would drop must "
                "not read as a configured permissions CA."
            )
        return config

    @staticmethod
    def _insecure_opt_out() -> bool:
        """True when the operator explicitly accepted an unsecured command surface.

        Mirrors the mesh insecure opt-out contract: ``1`` / ``true`` / ``yes``
        (case-insensitive) on :data:`ROS2_INSECURE_ENV`.
        """
        return os.getenv(ROS2_INSECURE_ENV, "").strip().lower() in ("1", "true", "yes")

    @classmethod
    def _require_secure_command_surface(
        cls,
        *,
        enable_commands: bool,
        dds_security_config: Any,
    ) -> None:
        """Refuse to expose an inbound command surface on an unsecured DDS graph.

        An inbound ``joint_command`` subscription lets any participant on the
        DDS domain drive the physical arm. When ``enable_commands`` is in
        effect, require either a ``dds_security_config`` (DDS Security: signed
        governance + permissions, authenticated identities) OR an explicit
        operator opt-out via :data:`ROS2_INSECURE_ENV` (``=1``). A telemetry-only
        bridge (``enable_commands`` False) is publish-only and is not gated.

        Raises:
            ValueError: When commands are enabled with neither a security config
                nor the explicit insecure opt-out.
        """
        if not enable_commands:
            return
        if dds_security_config:
            return
        if cls._insecure_opt_out():
            logger.warning(
                "%s: exposing an inbound joint_command surface on an UNSECURED DDS graph "
                "(%s set). Any participant on the domain can drive the arm. Provide a "
                "dds_security_config for production.",
                cls.__name__,
                ROS2_INSECURE_ENV,
            )
            return
        raise ValueError(
            "Refusing to start an inbound joint_command surface on an unsecured DDS graph. "
            "An enabled command bridge lets any DDS participant drive the physical arm. "
            "Pass a dds_security_config (identity CA, participant certificate + private key, "
            f"governance, permissions) or set {ROS2_INSECURE_ENV}=1 to explicitly accept the risk."
        )

    def _command_action(
        self,
        msg: Any,
        *,
        skip_empty: bool = False,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float] | None:
        """Parse an inbound ``joint_command`` ``JointState`` into an action dict.

        Returns ``{motor: pos}`` ready for ``Robot.send_action``, or ``None``
        when the message must be ignored. With ``skip_empty=True`` a wholly
        empty sample (a DDS dispose / keep-alive, which is not a real actuation
        request) is dropped silently; an empty or length-mismatched message is
        otherwise rejected with a warning rather than partially applied, so a
        malformed command never drives the arm to a surprising pose. A position
        that is not a readable number rejects the message the same way, and so
        does one that is readable but not finite: ``nan``/``inf`` survive
        ``float()`` and reach the actuator write, where lerobot's bounding
        clamp resolves them to an end stop rather than refusing them. Both are
        reported and return ``None`` rather than raising, so the failure costs
        exactly the one message on either transport.

        When ``joint_limits`` is supplied (``{"<motor>.pos": (min, max)}``), the
        command is range-checked against the declared bounds: if ANY commanded
        joint
        falls outside its range the ENTIRE command is rejected (returns
        ``None``) - no partial application - so one out-of-range joint can never
        drive part of the arm to a surprising pose while the rest holds. Joints
        without a declared bound are not constrained.

        Args:
            msg: The inbound ``JointState``-like message (``name``/``position``).
            skip_empty: Drop a wholly empty sample silently (DDS keep-alive).
            joint_limits: Optional ``{"<motor>.pos": (min, max)}`` clamp ranges;
                a command with any joint outside its range is rejected whole.
                Keys are matched against this message's own ``name`` entries,
                which are the ``<motor>.pos`` names the bridges publish in
                ``joint_states`` - a key that names no commanded joint
                constrains nothing.
        """
        names = list(getattr(msg, "name", []) or [])
        positions = list(getattr(msg, "position", []) or [])
        if skip_empty and not names and not positions:
            return None
        if not names or len(names) != len(positions):
            logger.warning(
                "%s: ignoring joint_command with name/position length mismatch (%d vs %d)",
                type(self).__name__,
                len(names),
                len(positions),
            )
            return None
        action: dict[str, float] = {}
        for name, pos in zip(names, positions):
            try:
                value = float(pos)
            except (TypeError, ValueError):
                # A position this cannot read is a malformed command, not a
                # reason to raise: this method's contract is an action dict or
                # None. Refusing the message WHOLE matches the length-mismatch
                # and out-of-range branches - no partial application - and it
                # keeps the failure inside the parser, where it costs exactly
                # the one message. Escaping instead reached each transport's
                # loop tolerance, and the cyclonedds loop has already taken a
                # batch by then, so the samples behind this one were dropped
                # with it.
                logger.warning(
                    "%s: ignoring joint_command with a non-numeric position for %r: %r",
                    type(self).__name__,
                    name,
                    pos,
                )
                return None
            # A readable number is not yet a usable position, and the refusal
            # ``nan``/``inf`` were left to is not one both bridges have. The
            # simulation host's ``send_action`` does refuse a non-finite action
            # value naming the joint; the hardware path does not. lerobot bounds
            # a normalized position with ``min(hi, max(lo, val))``, which
            # ``nan`` defeats in both directions - measured through
            # ``FeetechMotorsBus._unnormalize`` on an SO-101's own calibration,
            # a ``nan`` shoulder position resolved to the joint's ``range_min``
            # and an ``inf`` to its ``range_max``, so the arm drove to an end
            # stop under a ``success`` envelope with nothing logged. The
            # ``joint_limits`` branch below does catch it (``not (lo <= nan <=
            # hi)`` is true), but it is optional and defaults to ``None``.
            # Refused here, whole, on the domain those bounds already use.
            if error := finite_number_error(value, f"position for {name!r}", type(self).__name__):
                logger.warning("%s Whole command dropped, no partial application.", error)
                return None
            action[name] = value
        if joint_limits:
            for name, pos in action.items():
                bounds = joint_limits.get(name)
                if bounds is None:
                    continue
                low, high = bounds
                if not (low <= pos <= high):
                    logger.warning(
                        "%s: rejecting joint_command - %s=%.4f outside declared range "
                        "[%.4f, %.4f] (whole command dropped, no partial application)",
                        type(self).__name__,
                        name,
                        pos,
                        low,
                        high,
                    )
                    return None
        return action

    def _drive_from_command(self, robot: Any, msg: Any, *, skip_empty: bool = False) -> None:
        """Forward an inbound ``joint_command`` to ``robot.send_action``.

        Shared by both hardware bridges: parse via :meth:`_command_action`, then
        dispatch the flat ``{motor.pos: float}`` action, surfacing (never
        raising) a ``send_action`` failure so a bad command cannot kill the
        command loop.
        """
        action = self._command_action(msg, skip_empty=skip_empty, joint_limits=getattr(self, "_joint_limits", None))
        if action is None:
            return
        try:
            result = robot.send_action(action)
        except Exception:
            logger.warning("%s: send_action raised on joint_command; arm not moved", type(self).__name__, exc_info=True)
            return
        if isinstance(result, dict) and result.get("status") == "error":
            logger.warning("%s: send_action rejected joint_command: %s", type(self).__name__, result)


class RosTelemetryBridge(RosTelemetryBase):
    """A thin rclpy publisher for per-robot joint state and camera frames.

    Subclassed by :class:`SimRosBridge` and :class:`HardwareRosBridge`, which
    set a distinguishing default ``node_name`` but share every publish path so
    a simulated robot and a real one are byte-for-byte identical on the wire.
    Topic naming and the inbound command contract come from
    :class:`RosTelemetryBase`, which the pure-RTPS bridge also derives from, so
    the rclpy and cyclonedds transports advertise the same topics.

    Args:
        domain_id: ROS 2 domain (``ROS_DOMAIN_ID``) the bridge publishes on.
            Only an ``int`` in ``[0, 232]`` names a domain: RTPS derives its
            discovery ports from it, and 233 lands past the end of the port space.
        node_name: Name of the internal rclpy node.
        qos_depth: Depth of the publishers' KEEP_LAST history. Only a
            positive ``int`` up to ``MAX_QOS_HISTORY_DEPTH`` names a depth
            the transport can build a publisher with; the publishers are
            built on the first publish, so the value is checked here rather
            than surfacing mid-run from inside rclpy.

    Raises:
        ValueError: If ``domain_id`` is outside ``[0, 232]``, or if
            ``qos_depth`` is not a positive ``int`` no greater than
            :data:`MAX_QOS_HISTORY_DEPTH`. Both are checked before the
            process-wide ``ROS_DOMAIN_ID`` write, so a refused bridge leaves
            the environment as it found it.
        ImportError: When ``rclpy`` / the ROS 2 message packages are not
            importable, with an install hint (system ROS 2 or the docker image).
    """

    #: Default rclpy node name; subclasses override to identify their source.
    default_node_name = "strands_robots"

    def __init__(self, domain_id: int = 0, node_name: str | None = None, qos_depth: int = 10) -> None:
        # Refuse a domain id outside the RTPS port map BEFORE writing it. The
        # write below is process-wide and lands ahead of the rclpy import, so an
        # unusable value would otherwise outlive this call and steer every later
        # participant - a value the transport is never given the chance to reject.
        if error := dds_domain_id_error(domain_id, "domain_id", type(self).__name__):
            raise ValueError(error)

        # Refuse a history depth no publisher can be built with, in the same
        # place and for the same reasons as the domain above: it lands ahead of
        # both the process-wide write and the rclpy probe, so the refusal leaves
        # the environment as it found it and reports identically on an install
        # with the [ros2] extra and one without it. Unlike every other parameter
        # here, this one's consumer is not reached from the constructor - the
        # publishers are built on the first publish - so leaving it to the
        # transport means a bridge that reports success and then fails, or
        # quietly buffers a depth the caller did not ask for, on the first frame.
        if error := _qos_history_depth_error(qos_depth, "qos_depth", type(self).__name__):
            raise ValueError(error)

        # Pin the domain before rclpy reads it. Set it unconditionally so the
        # bridge publishes where the caller asked, not where the shell happened
        # to point.
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)

        rclpy_mod: Any = require_optional(
            "rclpy",
            system_install=ROS2_SYSTEM_INSTALL_HINT,
            purpose="the ROS 2 telemetry bridge (ros2_bridge=True)",
        )
        sensor_msgs: Any = require_optional(
            "sensor_msgs.msg", pip_install="ros-<distro>-sensor-msgs", purpose="the ROS 2 telemetry bridge"
        )
        self._rclpy = rclpy_mod
        self._JointState = sensor_msgs.JointState
        self._Image = sensor_msgs.Image

        self._owns_context = not self._rclpy.ok()
        if self._owns_context:
            self._rclpy.init()
        self._node = self._rclpy.create_node(node_name or self.default_node_name)
        self._qos_depth = qos_depth
        self._joint_pubs: dict[str, Any] = {}
        self._image_pubs: dict[str, Any] = {}

    def _now(self) -> Any:
        return self._node.get_clock().now().to_msg()

    def _joint_publisher(self, robot: str) -> Any:
        """The one publisher advertising ``robot``'s ``joint_states`` topic.

        Cached under the topic, not under *robot*: a publisher is identified by
        the topic it publishes on, and :meth:`_safe` is documented as not
        injective, so two robot names can select one topic. Keyed on the name
        instead, each spelling advertised its own publisher on that shared
        topic - one bridge appearing twice in ``ros2 topic info`` for one robot.
        """
        topic = self.joint_states_topic(robot)
        pub = self._joint_pubs.get(topic)
        if pub is None:
            pub = self._node.create_publisher(self._JointState, topic, self._qos_depth)
            self._joint_pubs[topic] = pub
        return pub

    def _image_publisher(self, robot: str, camera: str) -> Any:
        """The one publisher advertising ``robot``/``camera``'s image topic.

        Cached under the topic for the reason in :meth:`_joint_publisher`, and
        for a second one that is a wrong answer rather than a duplicate: the
        former key joined the two names with ``/``, which is a character both
        may contain, so ``("arm", "wrist/rgb")`` and ``("arm/wrist", "rgb")``
        produced one key for two different topics. The second caller was handed
        the first's publisher and its frames went out on ``/arm/wrist_rgb`` -
        a topic it never named, silently, because DDS matching is by topic name
        and the reader it expected simply never appeared.
        """
        topic = self.image_topic(robot, camera)
        pub = self._image_pubs.get(topic)
        if pub is None:
            pub = self._node.create_publisher(self._Image, topic, self._qos_depth)
            self._image_pubs[topic] = pub
        return pub

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        """Publish one ``JointState`` for ``robot`` on ``/<robot>/joint_states``.

        A ``names``/``positions`` pair of differing length is dropped whole with
        a warning rather than published misaligned - see
        :meth:`RosTelemetryBase._joint_state_arrays_error`.
        """
        if error := self._joint_state_arrays_error(list(names), list(positions), type(self).__name__):
            logger.warning("%s Whole JointState dropped, no partial publication.", error)
            return
        msg = self._JointState()
        msg.header.stamp = self._now()
        msg.header.frame_id = self._safe(robot)
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        self._joint_publisher(robot).publish(msg)

    def publish_image(self, robot: str, camera: str, image: np.ndarray) -> None:
        """Publish one RGB ``Image`` on ``/<robot>/<camera>/image_raw``.

        Args:
            robot: Robot name (topic namespace).
            camera: Camera name (topic sub-namespace).
            image: ``(H, W, 3)`` uint8 RGB frame.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return
        height, width = int(image.shape[0]), int(image.shape[1])
        msg = self._Image()
        msg.header.stamp = self._now()
        msg.header.frame_id = f"{self._safe(robot)}/{self._safe(camera, fallback='camera')}"
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = image.astype("uint8", copy=False).tobytes()
        self._image_publisher(robot, camera).publish(msg)

    def shutdown(self) -> None:
        """Destroy the node and, if this bridge initialized rclpy, shut it down.

        Best-effort, and safe to call more than once. The two steps release two
        independent resources - a node handle and the process-wide rclpy context
        - so neither failure is allowed to skip the other:

        * A node that will not be destroyed (rclpy raises ``InvalidHandle`` or
          ``RCLError`` from the C layer) is logged at debug, because the context
          shutdown below takes the node down with it. Letting it propagate left
          the context this bridge initialized running for the life of the
          process: ``_owns_context`` stays set, so the next bridge finds
          ``rclpy.ok()`` already true, disclaims ownership, and never shuts it
          down either - and both call sites tear the bridge down inside a
          suppressing block, so the leaked participant and its discovery
          threads were never reported to anyone.
        * A context that will not shut down is logged at *warning*, because
          nothing after it retries: that failure is the last word on a
          participant still on the wire.
        """
        node = getattr(self, "_node", None)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001 - the context release below is what matters
                logger.debug("%s: destroying the ROS 2 node failed", type(self).__name__, exc_info=True)
            finally:
                self._node = None
        if getattr(self, "_owns_context", False) and self._rclpy.ok():
            try:
                self._rclpy.shutdown()
            except Exception:  # noqa: BLE001 - reported, because nothing else releases the context
                logger.warning(
                    "%s: the ROS 2 context this bridge initialized could not be shut down; "
                    "its participant stays on domain %s until the process exits",
                    type(self).__name__,
                    os.environ.get("ROS_DOMAIN_ID", "?"),
                    exc_info=True,
                )
            else:
                self._owns_context = False
