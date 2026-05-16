"""Mesh transport layer — pluggable backends behind a single Protocol.

This package abstracts the wire-level pub/sub mechanism that the :class:`Mesh`
class uses. The Zenoh implementation (:class:`ZenohTransport`) preserves the
existing behaviour of :mod:`strands_robots.mesh.session` with **zero
behavioural change**. The AWS IoT Core implementation
(:class:`IotMqttTransport`) speaks MQTT5/mTLS to AWS IoT Core, validated end
to end against a real account.

The extraction matters because :class:`Mesh` and every sensor / RPC loop in
``mesh.core`` and ``mesh.sensors`` already talk through exactly two functions:
:func:`session.put` and :func:`session.declare_subscriber`. By moving those
two functions behind a Protocol, we can swap transports without touching any
caller.

Selection
---------
The active transport is selected by the ``STRANDS_MESH_BACKEND`` environment
variable, parsed by :func:`get_transport`:

- ``zenoh`` (default) — Eclipse Zenoh, LAN multicast, the current behaviour.
- ``iot``   — AWS IoT Core MQTT5 over mTLS.
- ``bridge`` — Both Zenoh and IoT, with a topic filter for what crosses the
  WAN (Layer 3 — landing later).

Backwards compatibility
-----------------------
:mod:`strands_robots.mesh.session` is unchanged: existing callers that use
:func:`get_session` / :func:`release_session` / :func:`put` continue to work
exactly as they did before this layer was added.
"""

from strands_robots.mesh.transport.base import (
    MeshTransport,
    Sample,
    SubHandle,
)
from strands_robots.mesh.transport.iot_transport import IotMqttTransport
from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

__all__ = [
    "MeshTransport",
    "Sample",
    "SubHandle",
    "ZenohTransport",
    "IotMqttTransport",
]
