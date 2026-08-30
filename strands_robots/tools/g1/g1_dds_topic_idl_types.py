"""Agent-facing lookup for the CycloneDDS IDL types the G1 driver decodes.

``G1Driver.connect_eagerly`` walks a fixed :py:meth:`~strands_robots.drivers.g1.G1Driver._subscription_plan`
that names, per read-side topic, the ``(idl_module, idl_class)`` pair the
driver hands to ``ChannelSubscriber`` at connect time so CycloneDDS knows
how to decode the wire bytes into a typed Python object (``rt/lowstate`` →
``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowState_``, ``rt/lf/bmsstate`` →
``BmsState_`` in the same module, ``rt/utlidar/lidar_state`` →
``unitree_sdk2py.idl.unitree_go.msg.dds_.LidarState_``, and so on). The
write side follows the mirror rule: ``rt/lowcmd`` is a ``ChannelPublisher``
of ``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowCmd_``. Seven topics, seven
IDL type identifiers, all captured as ``_TOPIC_*`` and IDL-class-name
constants inside :mod:`strands_robots.drivers.g1`.

This module surfaces that mapping as an agent-facing snapshot so a caller
planning a subscribe against one of the driver's own topics can name the
same IDL type the driver would hand ``ChannelSubscriber`` - and a caller
planning a mesh publish on the write topic can name the same IDL type the
driver's own ``ChannelPublisher`` was constructed with. The verb pair
mirrors :mod:`~strands_robots.tools.g1.g1_dds_topics` (which answers
direction + role) and :mod:`~strands_robots.tools.g1.g1_motion_gates`
(which answers FSM admissibility): one snapshot lookup + one membership
decision, both read-only, neither one touching the wire.

Two things this module is deliberately *not*:

* An import path for the IDL classes themselves. The IDL modules
  (``unitree_sdk2py.idl.unitree_hg.msg.dds_`` and its
  ``unitree_go`` / ``sensor_msgs`` siblings) are the CycloneDDS-typed
  Python objects the SDK ships; this snapshot names them as strings
  rather than importing them, so ``import
  strands_robots.tools.g1.g1_dds_topic_idl_types`` pulls no
  ``unitree_sdk2py`` submodule (the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358).
  A caller that wants the actual class object calls
  ``importlib.import_module(idl_module)`` on the string this snapshot
  returns; the driver's own ``_subscription_plan`` walks the identical
  string pair and hands the resolved class to ``ChannelSubscriber``.
* A liveness or wire-touch answer. The neon bundle's
  ``g1_dds_snapshot`` verb (``cagataycali/neon-the-g1/tools/g1_dds.py``)
  is what opens a second reader against a topic and pulls a message;
  the neon bundle's ``g1_dds_discover`` verb wraps ``cyclonedds ls``
  for the "who is actually on the bus right now" question. This
  lookup answers a static question: which IDL type the driver's own
  subscription plan reaches for on each of its seven topics,
  independent of whether the robot is powered on. The wire-touching
  verbs sit behind the singleton :data:`~strands_robots.tools.g1._g1_common._DDS_INIT_LOCK`
  and are a separate port; this pair is the read-only lookup that
  precedes them.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the ``(idl_module, idl_class)`` pair the driver hands
#: ``ChannelSubscriber`` (or ``ChannelPublisher`` on the write side) for
#: each of the seven topics it opens at ``connect_eagerly`` time. The
#: read entries are the six-tuple ``G1Driver._subscription_plan``
#: returns, byte-for-byte; the write entry names the type
#: ``G1Driver._pubs[rt/lowcmd]`` is a ``ChannelPublisher`` of. The role
#: labels match :mod:`~strands_robots.tools.g1.g1_dds_topics` so a
#: caller who chained the two lookups sees the same names on both
#: sides.
#:
#: The invariant this snapshot must preserve is byte-identity of the
#: ``idl_module`` and ``idl_class`` strings with the driver's own
#: ``_subscription_plan`` return values on the read side, and with the
#: ``ChannelPublisher`` type argument on the write side. A driver-side
#: widen or narrow (a new sensor topic, or a shape change on
#: ``LowCmd_``) that does not update this snapshot leaves them out of
#: sync, and the subscribe a mesh caller opens against a driver topic
#: and the subscribe the driver itself opens on the same topic
#: silently disagree on how to decode the bytes.
_DRIVER_TOPIC_IDL_TYPES: tuple[dict[str, str], ...] = (
    {
        "topic": "rt/lowstate",
        "direction": "read",
        "role": "lowstate",
        "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "idl_class": "LowState_",
    },
    {
        "topic": "rt/lf/bmsstate",
        "direction": "read",
        "role": "battery",
        "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "idl_class": "BmsState_",
    },
    {
        "topic": "rt/utlidar/lidar_state",
        "direction": "read",
        "role": "lidar_state",
        "idl_module": "unitree_sdk2py.idl.unitree_go.msg.dds_",
        "idl_class": "LidarState_",
    },
    {
        "topic": "rt/utlidar/cloud_livox_mid360",
        "direction": "read",
        "role": "lidar_cloud",
        "idl_module": "unitree_sdk2py.idl.sensor_msgs.msg.dds_",
        "idl_class": "PointCloud2_",
    },
    {
        "topic": "rt/mainboardstate",
        "direction": "read",
        "role": "mainboard",
        "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "idl_class": "MainBoardState_",
    },
    {
        "topic": "rt/pressuresensorstate",
        "direction": "read",
        "role": "pressure",
        "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "idl_class": "PressSensorState_",
    },
    {
        "topic": "rt/lowcmd",
        "direction": "write",
        "role": "lowcmd",
        "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "idl_class": "LowCmd_",
    },
)


@tool
def g1_list_dds_topic_idl_types() -> dict[str, Any]:
    """Return the IDL type identifier the driver decodes each DDS topic with.

    Read-only. Every entry is a driver-side constant snapshot; no bus is
    touched, no driver instance is required, and neither
    ``unitree_sdk2py`` nor the CycloneDDS runtime has to be present for
    this verb to answer. The IDL-type strings are named as
    ``(idl_module, idl_class)`` string pairs rather than as imported
    class objects so the SDK-load-hygiene contract every file under
    :mod:`strands_robots.tools.g1` carries survives this port: a caller
    who wants the resolved class calls ``importlib.import_module`` on
    the ``idl_module`` string and takes ``idl_class`` off the result.

    The seven entries mirror the driver's own
    :py:meth:`~strands_robots.drivers.g1.G1Driver._subscription_plan`
    on the read side (six topics, one IDL type each) and the
    ``ChannelPublisher`` the driver constructs for ``rt/lowcmd`` on
    the write side (one topic, one IDL type). The ``direction`` and
    ``role`` fields agree with the snapshot
    :mod:`~strands_robots.tools.g1.g1_dds_topics` carries, so a caller
    who chained the two lookups reads a single answer across the two
    verbs.

    Returns:
        A dict with ``status="success"``, a ``count`` naming how many
        topics the snapshot carries (seven, matching the driver's plan),
        and a ``topics`` list of per-topic descriptors. Each descriptor
        names the DDS ``topic`` string, the wire ``direction``
        (``"read"`` or ``"write"``), the driver-side ``role`` label,
        and the ``idl_module`` / ``idl_class`` pair the driver hands
        ``ChannelSubscriber`` (or ``ChannelPublisher`` on the write
        side) at connect time.
    """
    return {
        "status": "success",
        "count": len(_DRIVER_TOPIC_IDL_TYPES),
        "topics": [dict(entry) for entry in _DRIVER_TOPIC_IDL_TYPES],
    }


@tool
def g1_topic_idl_type(topic: str) -> dict[str, Any]:
    """Decide the IDL type the driver decodes a given DDS topic with.

    Read-only. Reads the driver's constant snapshot and returns the
    same ``(idl_module, idl_class)`` pair the driver's own
    ``ChannelSubscriber`` (or ``ChannelPublisher`` on the write side)
    would be constructed with. A caller planning a
    ``g1_dds_snapshot``-shaped subscribe against a driver topic uses
    this to name the same IDL type the driver holds, so the two
    subscribes decode the wire bytes identically; a caller planning
    a mesh publish on the driver's write topic uses this to see
    which IDL type the driver's own publisher wants, so the write
    path agrees with the gate.

    The topic string must match one of the driver's ``_TOPIC_*``
    constants byte-for-byte; a mis-cased or trailing-slash variant
    is refused rather than silently resolved to a nearby topic,
    because DDS topic names are exact strings on the wire and a
    fuzzy match would open a subscriber against a different topic
    than the caller named.

    Args:
        topic: The DDS topic string to test. Must be a ``str``; a
            ``bool``, ``None``, or non-string argument is refused
            decidably with a shape-error message rather than
            resolved through Python's coercions (``True`` would
            otherwise compare equal to ``1`` and never match a
            topic string, which is a confusing refusal path).

    Returns:
        On admission, a dict with ``status="success"``, the requested
        ``topic``, the wire ``direction`` (``"read"`` or ``"write"``),
        the driver-side ``role`` label, and the ``idl_module`` /
        ``idl_class`` pair the driver decodes the topic with. On
        refusal, ``status="error"`` and a ``message`` naming why the
        argument was refused (non-string, empty-string, or off the
        driver's known-topic set) with a citation to
        ``strands-labs/robots#358``. The known-topic list is included
        in the off-set refusal so a caller can see which topics the
        driver actually opens without a second lookup.
    """
    if isinstance(topic, bool) or not isinstance(topic, str):
        return {
            "status": "error",
            "message": (f"topic must be a str; got {type(topic).__name__} {topic!r}. Refs strands-labs/robots#358."),
        }
    if topic == "":
        return {
            "status": "error",
            "message": ("topic must be a non-empty str; got the empty string. Refs strands-labs/robots#358."),
        }
    for entry in _DRIVER_TOPIC_IDL_TYPES:
        if entry["topic"] == topic:
            return {
                "status": "success",
                "topic": entry["topic"],
                "direction": entry["direction"],
                "role": entry["role"],
                "idl_module": entry["idl_module"],
                "idl_class": entry["idl_class"],
            }
    known = sorted(entry["topic"] for entry in _DRIVER_TOPIC_IDL_TYPES)
    return {
        "status": "error",
        "message": (
            f"topic {topic!r} is not in the driver's subscription set. "
            f"Known topics: {known}. Refs strands-labs/robots#358."
        ),
    }
