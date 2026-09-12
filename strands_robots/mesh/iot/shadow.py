"""Device Shadow mirror - reflects presence/state to AWS IoT named shadows.

When the robot is online, every presence heartbeat ALSO updates the named
shadow ``presence`` for the Thing. When the robot is offline, the shadow
stays as the last reported state - late-joining operators (Bedrock agents,
fleet ops dashboards) read fleet state via ``GetThingShadow`` REST without
subscribing to MQTT.

Why mirror to a named shadow rather than the unnamed default
------------------------------------------------------------
Named shadows allow per-topic-family separation: a robot publishes its
presence to ``shadow/name/presence``, its task state to
``shadow/name/task``, its health to ``shadow/name/health``, etc. Each can
be queried independently. The unnamed default shadow becomes a single
massive blob otherwise.

Hooking in
----------
:class:`ShadowMirror` exposes :meth:`update` (call from anywhere) and a
ready-to-wire convenience :func:`enable_for_mesh` that binds it to the
heartbeat path automatically. Today the heartbeat path is
:meth:`Mesh._heartbeat_loop` which calls ``put(strands/{peer}/presence,...)``
- we hook in by registering a publish-side observer that mirrors any
presence write to the shadow update topic.

Failure mode
------------
Shadow updates are **best effort**: if the IoT transport isn't connected,
the update is dropped silently. A 0.5 Hz health update missing from the
shadow does not affect the live mesh.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.mesh.security import validate_mesh_identifier

logger = logging.getLogger(__name__)


def _validate_topic_segments(context: str, thing_name: Any, shadow_name: Any) -> None:
    """Refuse a shadow-topic segment that is not a single MQTT topic level.

    Both values are interpolated into a reserved ``$aws/things/...`` topic, so
    each must be ONE topic level. They are graded on the shared mesh-identifier
    domain (:func:`~strands_robots.mesh.security.validate_mesh_identifier`) -
    the same charset :func:`~strands_robots.mesh.core.init_mesh` already
    requires of a ``peer_id``, and for the same stated reason: ``/``, ``+`` and
    ``#`` break MQTT topic structure. A ``/`` smuggles an extra level, so the
    topic names a different Thing than the caller asked for and no longer
    matches the 7-level reserved pattern the broker accepts; ``+`` and ``#``
    are wildcards, which a publish topic may not carry at all; and an empty
    segment leaves an empty level.

    None of those reach the shadow as an error the caller can see, because
    :meth:`ShadowMirror.update` is best-effort and swallows what the transport
    says (see its ``logger.debug``). The named shadow an operator reads for
    fleet state is then simply never written, which is why the refusal belongs
    at the surface that builds the topic rather than at the publish.

    The domain is the SUPERSET of what ``init_mesh`` accepts, so every
    ``peer_id`` that reached a live mesh - including a dotted one - is accepted
    here; this cannot refuse an identifier the mesh already admitted. It is
    deliberately NOT the stricter Thing-name rule the provisioning path applies
    (which additionally rejects ``.`` for filesystem-path safety, because it
    also names a certificate file): a topic segment is not a path component,
    and refusing ``rover.01`` here would refuse a peer the mesh admitted.

    Args:
        context: The surface that received the values, used verbatim as the
            message prefix - the function name, or the class name for a
            constructor parameter.
        thing_name: Candidate AWS IoT Thing name (topic level 2).
        shadow_name: Candidate named-shadow name (topic level 5).

    Raises:
        ValidationError: Either value is not a single usable topic level.
    """
    validate_mesh_identifier(thing_name, f"{context}.thing_name")
    validate_mesh_identifier(shadow_name, f"{context}.shadow_name")


def shadow_update_topic(thing_name: str, shadow_name: str = "presence") -> str:
    """The MQTT topic that triggers an IoT named-shadow update.

    AWS IoT uses ``$aws/things/{thing}/shadow/name/{shadow}/update``.

    Args:
        thing_name: The AWS IoT Thing name. Must be a single topic level
            (:func:`~strands_robots.mesh.security.validate_mesh_identifier`).
        shadow_name: The named shadow to update. Must be a single topic level.

    Returns:
        The reserved update topic for that Thing and shadow.

    Raises:
        ValidationError: Either name is not a single usable topic level.
    """
    _validate_topic_segments("shadow_update_topic", thing_name, shadow_name)
    return f"$aws/things/{thing_name}/shadow/name/{shadow_name}/update"


def shadow_get_topic(thing_name: str, shadow_name: str = "presence") -> str:
    """The topic that triggers an IoT named-shadow GET reply.

    Args:
        thing_name: The AWS IoT Thing name. Must be a single topic level
            (:func:`~strands_robots.mesh.security.validate_mesh_identifier`).
        shadow_name: The named shadow to read. Must be a single topic level.

    Returns:
        The reserved get topic for that Thing and shadow.

    Raises:
        ValidationError: Either name is not a single usable topic level.
    """
    _validate_topic_segments("shadow_get_topic", thing_name, shadow_name)
    return f"$aws/things/{thing_name}/shadow/name/{shadow_name}/get"


class ShadowMirror:
    """Mirrors a robot's presence (or any state dict) to an AWS IoT named shadow.

    Usage::

        from strands_robots.mesh.iot import ShadowMirror
        from strands_robots.mesh.transport.factory import current_transport

        mirror = ShadowMirror(thing_name="so100-arm-01", shadow_name="presence")
        mirror.update(current_transport(), {"connected": True, "robot_type": "so100"})

    The ``update`` call wraps your dict in the canonical
    ``{"state": {"reported":...}}`` envelope and publishes via the active
    transport's ``put()``. No retain - shadows are stored server-side.
    """

    def __init__(self, thing_name: str, shadow_name: str = "presence") -> None:
        """Bind a mirror to one Thing's named shadow.

        Args:
            thing_name: The AWS IoT Thing name - normally the mesh
                ``peer_id``. Must be a single topic level
                (:func:`~strands_robots.mesh.security.validate_mesh_identifier`).
            shadow_name: The named shadow to mirror into. Must be a single
                topic level.

        Raises:
            ValidationError: Either name is not a single usable topic level.
        """
        # Named here rather than left to `shadow_update_topic` below so the
        # refusal names the call the caller actually made. The refusal precedes
        # the assignments: a mirror that cannot address a shadow must not exist
        # holding a topic `update` would publish and then swallow the rejection
        # of.
        _validate_topic_segments(type(self).__name__, thing_name, shadow_name)
        self.thing_name = thing_name
        self.shadow_name = shadow_name
        self._update_topic = shadow_update_topic(thing_name, shadow_name)

    def update(self, transport: Any, reported_state: dict[str, Any]) -> None:
        """Push *reported_state* into the named shadow's ``reported`` slot.

        Args:
            transport: An object with ``.put(topic, dict)`` - typically the
                singleton from :func:`current_transport`. Pass ``None`` to
                make this a no-op (useful when the mesh is off).
            reported_state: The dict that becomes ``state.reported`` in the
                shadow document. Must be JSON-serialisable.
        """
        if transport is None or not getattr(transport, "is_alive", lambda: False)():
            return
        envelope = {"state": {"reported": reported_state}}
        try:
            transport.put(self._update_topic, envelope)
        except Exception as exc:
            logger.debug("[shadow] update %s failed: %s", self._update_topic, exc)


def enable_for_mesh(mesh: Any) -> ShadowMirror | None:
    """Convenience wiring: add a presence-shadow mirror to a running Mesh.

    Replaces ``mesh._build_presence`` with a wrapper that, after the original
    builds the payload, also pushes it to the named shadow. The original
    behaviour (publishing to ``strands/{peer}/presence``) is preserved -
    this is purely additive.

    Returns the :class:`ShadowMirror` so callers can drive ad-hoc updates
    too. Returns ``None`` if the mesh is not running an IoT-capable
    transport (i.e. plain Zenoh - shadows aren't relevant there).
    """
    from strands_robots.mesh.transport.factory import current_backend, current_transport

    if current_backend() not in ("iot", "bridge"):
        logger.debug("[shadow] backend is %r, skipping shadow mirror", current_backend())
        return None

    mirror = ShadowMirror(thing_name=mesh.peer_id, shadow_name="presence")
    original_build = mesh._build_presence

    def _build_presence_with_shadow() -> dict[str, Any]:
        payload = original_build()
        # Best-effort shadow update; never let a shadow failure break heartbeat.
        try:
            mirror.update(current_transport(), payload)
        except Exception as exc:
            logger.debug("[shadow] mirror update raised: %s", exc)
        return payload

    mesh._build_presence = _build_presence_with_shadow  # type: ignore[method-assign]
    logger.info("[shadow] presence shadow mirror enabled for %s", mesh.peer_id)
    return mirror
