"""Agent-facing lookup for the DDS topics ``G1Driver`` opens on the bus.

``G1Driver.connect_eagerly`` subscribes six CycloneDDS topics as the read
side (``rt/lowstate``, ``rt/lf/bmsstate``, ``rt/utlidar/lidar_state``,
``rt/utlidar/cloud_livox_mid360``, ``rt/mainboardstate``,
``rt/pressuresensorstate``) and publishes one topic as the write side
(``rt/lowcmd``). The topic names are module-level constants inside
:mod:`strands_robots.drivers.g1` (``_TOPIC_LOWSTATE`` .. ``_TOPIC_LOWCMD``),
private because the driver is the only wire path that opens them and the
mesh does not want a second subscriber path racing the singleton
``_DDS_INIT_LOCK`` these subscribes are threaded through.

This module surfaces that subscription set as an agent-facing snapshot so a
caller planning a rollout can enumerate the bus the driver would open
before it opens the bus. A reader that only wants the write topic already
has the driver's :attr:`~strands_robots.drivers.g1.G1Driver._pubs` handle
for the ``rt/lowcmd`` gate; the read set is what a caller planning a
mesh publish or a ``g1_dds_snapshot``-shaped verb (which the neon bundle's
``g1_dds.py`` exposes, refs ``cagataycali/neon-the-g1/tools/g1_dds.py``)
would compare its topic name against, to decide whether the driver is
already carrying that topic's decode or whether the caller has to bring
its own subscriber. The verb pair mirrors
:mod:`~strands_robots.tools.g1.g1_motion_gates` and
:mod:`~strands_robots.tools.g1.g1_fsm_targets`: one snapshot lookup +
one membership decision.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_dds_subscribe`` /
  ``g1_dds_snapshot`` verbs open a second DDS reader against arbitrary
  topics; that second reader path is out of scope for this lookup, which
  only names the driver's own subscription set. A future verb that
  routes a subscribe request through the driver's singleton reader is
  a separate port; refs strands-labs/robots#358 for the DDS-facing
  seam that would land on. This module ports the read-only lookup half
  without also introducing a second reader path the driver does not
  yet front.
* An SDK or CycloneDDS re-import. The topic names are captured here as
  string constants snapshotted from the driver's own
  ``_TOPIC_*`` module-level constants; the snapshot lives here rather
  than being re-imported from the driver so ``import
  strands_robots.tools.g1.g1_dds_topics`` pulls no ``unitree_sdk2py``
  submodule (the import-hygiene contract every other file in this
  package carries, refs strands-labs/robots#358). The invariant a
  driver-side widen or narrow must preserve is byte-for-byte identity
  between the ``topic`` strings surfaced here and the driver's own
  ``_TOPIC_*`` constants: a subscribe-set drift that does not update
  this snapshot leaves the two out of sync, so the read the driver
  actually opens and the read this verb reports diverge silently.

What this module does not decide.

* Whether a topic is currently *live* on the bus. Topic liveness is a
  DDS-layer discovery answer; the neon bundle's ``g1_dds_discover``
  verb wraps ``cyclonedds ls`` for that. This lookup answers a static
  question: which topics the driver's ``connect_eagerly`` code path
  subscribes on, independent of whether the robot is powered on.
* Which decoded field ends up where in the driver's cache. Each
  subscription decodes into a named attribute on the driver
  (``_imu``, ``_battery``, ``_lidar_state``, ``_lidar_summary``,
  ``_mainboard``, ``_pressure``); that mapping is per-topic and the
  driver's callbacks are the source of truth for it. A future verb
  that surfaces the cache-attribute names alongside the topic names
  would need the driver's own callback wiring as its evidence, which
  is a driver-side read separate from this lookup.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the DDS topics ``G1Driver`` opens at connect time. Each
#: entry names the topic string (matching the driver's private
#: ``_TOPIC_*`` constants byte-for-byte), the ``direction`` of the wire
#: (``"read"`` for a subscribe, ``"write"`` for a publish), and a
#: ``role`` naming what the driver decodes or writes on that topic.
#:
#: The role labels are the driver's own decode targets:
#:
#: * ``rt/lowstate`` -> ``lowstate`` (into ``_imu`` and ``_mode_machine``
#:   caches)
#: * ``rt/lf/bmsstate`` -> ``battery`` (into ``_battery`` cache; the
#:   ``lf`` prefix is the Unitree-side layout the G1 ships)
#: * ``rt/utlidar/lidar_state`` -> ``lidar_state`` (into ``_lidar_state``
#:   cache, the sensor-diagnostics envelope)
#: * ``rt/utlidar/cloud_livox_mid360`` -> ``lidar_cloud`` (into
#:   ``_lidar_summary`` cache; the point cloud is not held raw, the
#:   driver keeps the derived summary)
#: * ``rt/mainboardstate`` -> ``mainboard`` (into ``_mainboard`` cache,
#:   temperature and voltage envelope)
#: * ``rt/pressuresensorstate`` -> ``pressure`` (into ``_pressure``
#:   cache, foot force envelope)
#: * ``rt/lowcmd`` -> ``lowcmd`` (the write topic; every ``send_action``
#:   and ``run_policy`` frame publishes here through the driver's
#:   ``_pubs`` handle)
#:
#: The snapshot lives here rather than in
#: :mod:`~strands_robots.tools.g1._g1_common` because the mapping is
#: only useful for the ``g1_dds_topics``-side of the conversation. The
#: invariant this snapshot must preserve is byte-identity of the
#: ``topic`` strings with the driver's own ``_TOPIC_*`` constants; a
#: driver-side widen or narrow that updates the driver without also
#: updating this snapshot leaves them out of sync, and the read the
#: driver actually opens and the read this verb reports diverge.
_DRIVER_TOPICS: tuple[dict[str, str], ...] = (
    {"topic": "rt/lowstate", "direction": "read", "role": "lowstate"},
    {"topic": "rt/lf/bmsstate", "direction": "read", "role": "battery"},
    {"topic": "rt/utlidar/lidar_state", "direction": "read", "role": "lidar_state"},
    {"topic": "rt/utlidar/cloud_livox_mid360", "direction": "read", "role": "lidar_cloud"},
    {"topic": "rt/mainboardstate", "direction": "read", "role": "mainboard"},
    {"topic": "rt/pressuresensorstate", "direction": "read", "role": "pressure"},
    {"topic": "rt/lowcmd", "direction": "write", "role": "lowcmd"},
)

#: The two directions the ``direction`` field takes. Named here so the
#: refusal string an unknown filter would surface can quote the same
#: domain the snapshot carries, rather than restating it inline.
_VALID_DIRECTIONS: frozenset[str] = frozenset({"read", "write"})


@tool
def g1_list_dds_topics(direction: str = "") -> dict[str, Any]:
    """Return the DDS topics ``G1Driver`` opens on the bus.

    Read-only. Every entry is a driver constant snapshot; no bus is
    touched, no driver instance is required, and the CycloneDDS runtime
    does not have to be initialised for this verb to answer.

    Args:
        direction: Optional filter on the wire direction. Empty (default)
            returns every topic the driver opens. ``"read"`` returns
            only the six topics the driver subscribes at
            ``connect_eagerly`` time; ``"write"`` returns the one
            topic (``rt/lowcmd``) the driver publishes on. Any other
            value is refused with a message naming the valid set.

    Returns:
        A dict with ``status``, a ``count`` naming how many topics the
        filter matched, the requested ``direction`` (or ``None`` for the
        unfiltered case), a ``directions`` list naming both valid
        filters, and a ``topics`` list of per-topic descriptors
        (``topic``, ``direction``, ``role``). An unknown ``direction``
        carries ``status="error"`` and a refusal string that quotes the
        valid domain and cites ``strands-labs/robots#358``.
    """
    if direction and direction not in _VALID_DIRECTIONS:
        valid = sorted(_VALID_DIRECTIONS)
        return {
            "status": "error",
            "message": (
                f"unknown direction {direction!r}. Valid directions are {valid}. Refs strands-labs/robots#358."
            ),
        }
    if direction:
        topics = [dict(entry) for entry in _DRIVER_TOPICS if entry["direction"] == direction]
    else:
        topics = [dict(entry) for entry in _DRIVER_TOPICS]
    return {
        "status": "success",
        "count": len(topics),
        "direction": direction or None,
        "directions": sorted(_VALID_DIRECTIONS),
        "topics": topics,
    }


@tool
def g1_topic_role(topic: str) -> dict[str, Any]:
    """Decide the wire role ``G1Driver`` opens for a given DDS topic string.

    Read-only. Reads the driver's constant snapshot and returns the same
    ``direction`` / ``role`` answer the driver's own subscribe or publish
    call would carry. A caller planning a ``g1_dds_snapshot``-shaped
    subscribe uses this to see whether the driver already holds a
    subscriber for the topic (in which case the caller reads from the
    driver's cache instead of opening a second reader path against the
    same wire), and a caller planning a mesh publish uses it to see
    whether the target topic is the driver's write path (in which case
    the write goes through ``send_action`` / ``run_policy`` rather than
    a direct publish that would race the driver's gate).

    Args:
        topic: The DDS topic string to test. Must match one of the
            driver's ``_TOPIC_*`` constants byte-for-byte; a mis-cased
            or trailing-slash variant is refused rather than silently
            resolved to a nearby topic, because DDS topic names are
            exact strings on the wire.

    Returns:
        A dict with ``status``. On admit: the requested ``topic``, the
        ``direction`` the driver opens it in (``"read"`` or ``"write"``),
        and the ``role`` the driver decodes or writes on it. On refuse:
        ``status="error"``, a ``message`` naming the known topic set,
        and a citation to ``strands-labs/robots#358``.
    """
    if not isinstance(topic, str):
        return {
            "status": "error",
            "message": (f"topic must be a str; got {type(topic).__name__} {topic!r}. Refs strands-labs/robots#358."),
        }
    for entry in _DRIVER_TOPICS:
        if entry["topic"] == topic:
            return {
                "status": "success",
                "topic": entry["topic"],
                "direction": entry["direction"],
                "role": entry["role"],
            }
    known = sorted(entry["topic"] for entry in _DRIVER_TOPICS)
    return {
        "status": "error",
        "message": (
            f"topic {topic!r} is not in the driver's subscription set. "
            f"Known topics: {known}. Refs strands-labs/robots#358."
        ),
    }
