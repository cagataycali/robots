"""Agent-facing lookup for the DDS topic descriptions the neon catalog names.

The neon bundle's ``_dds_engine.py``
(``cagataycali/neon-the-g1/tools/_dds_engine.py``) carries a
``TOPIC_CATALOG`` dict whose per-topic value tuple is
``(idl_module, idl_class, description, category)``. The
:mod:`~strands_robots.tools.g1.g1_dds_topic_idl_types` port names the
first two positions; :mod:`~strands_robots.tools.g1.g1_dds_topic_categories`
names the fourth. The remaining third position - the plain-text
description a caller reads to name what the topic decodes at the
wire (e.g. ``"IMU, joints, motors (~1kHz)"`` for ``rt/lowstate`` or
``"Low-level motor cmd (\\U0001f6a8)"`` for ``rt/lowcmd``) - is what this
module surfaces.

This module snapshots the twenty-two topic descriptions as a
module-level constant and surfaces them as two agent-facing verbs
(:func:`g1_list_dds_topic_descriptions` returns the whole envelope;
:func:`g1_dds_topic_description_admits` decides one membership query
against the topic-name key) so a caller planning a bus-side read or
write names the intent of a topic decidably before a future
driver-side wrapper for the neon ``g1_dds_snapshot`` /
``g1_dds_subscribe`` verb dispatches. Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's generic DDS verbs open a
  second DDS reader against arbitrary topics; that second reader
  path is out of scope for this lookup, which only names the
  read-only description string. A future verb that routes a
  subscribe request through the driver's singleton reader is a
  separate port; refs strands-labs/robots#358 for the DDS-facing
  seam that would land on. This module ports the read-only lookup
  half without also introducing a second reader path the driver
  does not yet front.
* An SDK or CycloneDDS re-import. The topic-description strings are
  captured here as module-level constants snapshotted from the neon
  ``TOPIC_CATALOG`` dict; the constants live here rather than being
  re-imported from the neon module so ``import
  strands_robots.tools.g1.g1_dds_topic_descriptions`` pulls no
  ``unitree_sdk2py`` submodule (the import-hygiene contract every
  other file in this package carries, refs strands-labs/robots#358).
  A neon-side widen that adds or drops a topic description lands in
  the same PR as the neon table, or the parity test in
  ``tests/drivers/`` surfaces the drift.

What this module does not decide.

* Whether a topic is currently *live* on the bus. Topic liveness is a
  DDS-layer discovery answer; the neon bundle's ``g1_dds_discover``
  verb wraps ``cyclonedds ls`` for that. This lookup answers a static
  question: which description string the neon catalog carries per
  topic name, independent of whether the robot is powered on.
* Which IDL class decodes the topic's payload. That belongs on
  :mod:`~strands_robots.tools.g1.g1_dds_topic_idl_types`, which
  already answers it. This module carries only the plain-text
  description column.
* Which category label partitions the topic. That belongs on
  :mod:`~strands_robots.tools.g1.g1_dds_topic_categories`, which
  already answers it. A caller who wants both fields dispatches to
  each verb separately.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the neon-catalog topic-description column, keyed by
#: DDS topic name. Twenty-two entries: nine ``state``, two ``lidar``,
#: one ``joystick``, four ``control`` (each entry carries the
#: dangerous-publish marker naming the refusal list), two ``hand``
#: (the command topic also carries the dangerous-publish marker), three ``slam``, and one
#: ``config``. The strings are captured byte-for-byte from the neon
#: catalog; a neon-side widen or narrow of a description lands in
#: the same PR as the neon table (the parity test surfaces the
#: drift if the strings diverge).
_DDS_TOPIC_DESCRIPTIONS: dict[str, str] = {
    # --- READ-ONLY: robot state ---
    "rt/lowstate": "IMU, joints, motors (~1kHz)",
    "rt/lf/lowstate": "Low-freq LowState variant",
    "rt/bmsstate": "Battery management",
    "rt/lf/bmsstate": "Battery (low-freq)",
    "rt/mainboardstate": "Fan / board temps",
    "rt/pressuresensorstate": "Foot pressure sensors",
    "rt/lf/sportmodestate": "Motion state (low-freq)",
    "rt/lf/secondary_imu": "Secondary IMU",
    "rt/multiplestate": "Combined state",
    # --- READ-ONLY: LiDAR / SLAM ---
    "rt/utlidar/cloud_livox_mid360": "Livox Mid-360 point cloud",
    "rt/utlidar/lidar_state": "LiDAR sensor state",
    # --- READ-ONLY: joystick ---
    "rt/wirelesscontroller": "Remote joystick (silent unpaired)",
    # --- CONTROL (WRITE) — dangerous ---
    "rt/lowcmd": "Low-level motor cmd (\U0001f6a8)",
    "rt/armsdk": "Arm SDK override (\U0001f6a8)",
    "rt/user_lowcmd": "User low-level cmd (\U0001f6a8)",
    "rt/bmscmd": "BMS cmd (\U0001f6a8 power/reboot)",
    # --- G1 HANDS (Inspire/Unitree 5/7-DoF hand) ---
    "rt/inspire/state": "Inspire hand state (joint pos/tau)",
    "rt/inspire/cmd": "Inspire hand command (\U0001f6a8 write)",
    # --- SLAM / Odometry (experimental — topic naming varies) ---
    "rt/odom": "Robot odometry (nav_msgs)",
    "rt/unitree_slam/odom": "Unitree SLAM odometry",
    "rt/unitree_slam/global_map": "Unitree SLAM global map",
    # --- WRITE: non-motion config ---
    "rt/utlidar/switch": "LiDAR ON/OFF switch",
}


def _describe(topic: str) -> dict[str, Any]:
    """Build the per-topic descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_dds_topic_descriptions` so
    :func:`g1_dds_topic_description_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands
    in one place. Every field is a snapshot read; no bus is
    touched.
    """
    return {
        "topic": topic,
        "description": _DDS_TOPIC_DESCRIPTIONS[topic],
    }


@tool
def g1_list_dds_topic_descriptions() -> dict[str, Any]:
    """Return the DDS topic descriptions the neon catalog names.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for the neon ``g1_dds_snapshot`` / ``g1_dds_subscribe`` verb is
    called, so a caller planning a bus-side read or write names what
    the topic decodes at the wire decidably before the topic itself
    is opened.

    The envelope names twenty-two topics: nine on the read-side
    ``state`` partition, two on ``lidar``, one on ``joystick``, four
    on the ``control`` write partition (each entry carries the
    dangerous-publish marker the neon catalog uses to name the
    refusal list), two on ``hand`` (the command topic also
    dangerous-publish-marked),
    three on ``slam``, and one on ``config``.

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        topics; a ``descriptions`` list of descriptors (one per
        topic, sorted lexicographically by ``topic``) carrying
        ``topic`` and ``description``; and a ``topics`` field
        listing the topic-name keys sorted lexicographically. Every
        field is a snapshot of a neon-observed constant; no dynamic
        decode runs here.
    """
    topics = sorted(_DDS_TOPIC_DESCRIPTIONS)
    return {
        "status": "success",
        "count": len(_DDS_TOPIC_DESCRIPTIONS),
        "descriptions": [_describe(topic) for topic in topics],
        "topics": topics,
    }


@tool
def g1_dds_topic_description_admits(topic: str = "") -> dict[str, Any]:
    """Decide whether a topic name is inside the neon-catalog description table.

    Read-only. Reads the module's snapshot of the neon bundle's
    ``TOPIC_CATALOG`` topic-name keys and returns the same
    membership answer a caller planning a bus-side read or write
    against the catalog would compute. A caller with a topic
    string resolves it against the catalog before a future
    catalog-lookup verb dispatches, rather than triggering the
    neon catalog's ``KeyError`` at wire time (an unknown topic
    name is not on the neon table and the neon lookup returns
    nothing for it).

    A topic inside :data:`_DDS_TOPIC_DESCRIPTIONS` is a valid neon
    catalog entry; a topic outside is either not carried by the
    neon catalog (the neon generic ``g1_dds_snapshot`` verb still
    accepts an arbitrary topic with ``type_module`` +
    ``type_class`` overrides, but a caller relying on the neon
    catalog's built-in decode resolves the miss decidably here).

    Args:
        topic: The DDS topic name to test. Must be a non-empty
            ``str``; ``bool`` is refused (``True``/``False`` are
            not valid topic names under any DDS convention) and
            the empty string is refused as a shape error (no name
            means no membership query to answer). Non-str inputs
            are refused decidably rather than resolved through
            Python's ``str()`` coercion.

    Returns:
        A dict with ``status``; a ``query`` sub-dict carrying the
        supplied ``topic``; an ``admitted`` boolean naming whether
        the topic is a member of the twenty-two-entry catalog; and
        (when ``admitted`` is ``True``) a ``target`` sub-dict
        carrying the same descriptor
        :func:`g1_list_dds_topic_descriptions` returns for the
        topic (``topic``, ``description``). On a not-admitted
        query the dict carries a ``refusal_advice`` field naming
        the neon-side note that the topic is not on the built-in
        catalog and that a caller may still reach the bus via the
        generic ``g1_dds_snapshot`` verb by supplying
        ``type_module`` and ``type_class`` overrides. On a shape
        error (``bool``, non-str, empty string) the dict carries
        ``status="error"`` with a message naming the type refused.
    """
    if isinstance(topic, bool):
        return {
            "status": "error",
            "message": (f"topic must be str, got bool ({topic!r}). Refs strands-labs/robots#358."),
        }
    if not isinstance(topic, str):
        return {
            "status": "error",
            "message": (f"topic must be str, got {type(topic).__name__} ({topic!r}). Refs strands-labs/robots#358."),
        }
    if topic == "":
        return {
            "status": "error",
            "message": (
                "topic must be a non-empty str; an empty topic name has "
                "no membership answer to compute. Refs "
                "strands-labs/robots#358."
            ),
        }

    admitted = topic in _DDS_TOPIC_DESCRIPTIONS
    if not admitted:
        return {
            "status": "success",
            "admitted": False,
            "query": {"topic": topic},
            "refusal_advice": (
                f"topic {topic!r} is not on the neon catalog; the neon "
                "g1_dds_snapshot verb still accepts an arbitrary topic "
                "with type_module + type_class overrides, but a caller "
                "relying on the catalog's built-in decode resolves the "
                "miss here. Refs strands-labs/robots#358."
            ),
        }

    return {
        "status": "success",
        "admitted": True,
        "query": {"topic": topic},
        "target": _describe(topic),
    }
