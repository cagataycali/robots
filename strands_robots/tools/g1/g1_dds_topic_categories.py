"""Agent-facing lookup for the DDS topic categories the neon catalog partitions.

The neon bundle's ``_dds_engine.py``
(``cagataycali/neon-the-g1/tools/_dds_engine.py``) carries a
``TOPIC_CATALOG`` dict whose per-topic value tuple ends in a category
label - one of ``state`` (read-side IMU/motor/battery snapshots),
``lidar`` (Livox Mid-360 point cloud plus the sensor's state channel),
``joystick`` (the wireless controller's read topic), ``control`` (the
low-level write topics that drive the robot at the wire),
``hand`` (the Inspire/Unitree 5/7-DoF hand's state and command
channels), ``slam`` (odometry and SLAM global-map read channels), or
``config`` (the LiDAR ON/OFF switch write). The neon generic
``g1_dds_list_topics`` verb takes a ``category`` argument and filters
its returned catalog against that partition; a caller planning a
bus-side subscribe or publish uses the partition to name the intent
of a topic before the topic name itself is chosen.

This module snapshots the seven-category partition as a module-level
constant and surfaces it as two agent-facing verbs
(:func:`g1_list_dds_topic_categories` returns the whole envelope;
:func:`g1_dds_topic_category_admits` decides one membership query)
so a caller planning a category-scoped catalog read can name the
partition decidably before a future driver-side wrapper for the neon
``g1_dds_list_topics`` verb lands. Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_dds_list_topics`` verb
  reads its own catalog dict and returns the per-topic descriptors
  filtered by category; that read path is out of scope for this
  lookup. Today's :class:`~strands_robots.drivers.g1.G1Driver` opens
  seven fixed topics
  (:mod:`~strands_robots.tools.g1.g1_dds_topics` names them) and
  writes only ``rt/lowcmd``; a future driver method that fronts the
  neon catalog's category-scoped read will cross-reference the
  membership answer this lookup returns. This module ports the
  read-only partition half without also introducing a second
  catalog-read path the driver does not yet own.
* An SDK or CycloneDDS re-import. The category names are captured
  here as string constants snapshotted from the neon bundle's
  ``TOPIC_CATALOG`` value tuples; the snapshot lives here rather
  than being re-read from the neon module so
  ``import strands_robots.tools.g1.g1_dds_topic_categories`` pulls
  no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A neon-side widen or narrow to the
  partition (a new category label, or a rename) is a caller-side
  update; the invariant a widen must preserve is byte-for-byte
  identity between the category strings surfaced here and the neon
  ``TOPIC_CATALOG`` fourth tuple element, so a rename that does not
  update this snapshot leaves the two out of sync and the caller-side
  filter and this lookup's membership answer diverge silently.

What this module does not decide.

* Which topics belong to which category at any particular moment.
  Per-topic membership is answered by
  :mod:`~strands_robots.tools.g1.g1_dds_topics` (the driver's own
  seven-topic subscription set, whose descriptors carry the same
  category label the neon catalog assigns) and by a future
  driver-side wrapper for the neon catalog's full 22-topic table.
  This lookup answers a narrower question: which category *names*
  the neon partition ships today, so a caller filtering by category
  can decide the refusal before the filter itself runs.
* Whether a category is *write-safe*. A topic inside the ``control``
  category is a topic
  ``g1_dangerous_publish_topics``
  refuses without an explicit ``unsafe=True`` override, and a topic
  inside the ``config`` category (the LiDAR switch) is a write path
  today's driver does not carry a wrapper for. This lookup does not
  carry the write-safety refusal; a caller planning a publish
  cross-checks the topic against
  ``g1_dangerous_publish_topics``
  separately.
* Whether a category is *observed live* on the bus. Category
  presence at runtime is a DDS discovery answer; the neon bundle's
  ``g1_dds_discover`` verb wraps ``cyclonedds ls`` for that. This
  lookup answers a static question: which category *labels* the neon
  catalog partitions the known topic set into, independent of
  whether any of those topics are currently published.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the seven category labels the neon bundle's
#: ``_dds_engine.TOPIC_CATALOG`` partitions its 22 catalog topics
#: into today. Each label is the fourth element of a per-topic value
#: tuple; the neon generic ``g1_dds_list_topics`` verb accepts the
#: category argument and filters against the same string set. Named
#: here as module-level constants rather than re-read from the neon
#: module so ``import strands_robots.tools.g1.g1_dds_topic_categories``
#: pulls zero ``unitree_sdk2py`` submodules - the import-hygiene
#: contract every other file in this package carries. A neon-side
#: widen or narrow updates this snapshot and the neon module in the
#: same PR, or the two silently drift.
_DDS_TOPIC_CATEGORIES: frozenset[str] = frozenset(
    {
        "state",
        "lidar",
        "joystick",
        "control",
        "hand",
        "slam",
        "config",
    }
)

#: Per-category description surfaced on the returned descriptor. Each
#: string names the wire intent the category label partitions the
#: neon catalog by - the read side (``state`` / ``lidar`` /
#: ``joystick`` / ``hand`` state / ``slam``) versus the write side
#: (``control`` / ``config`` / ``hand`` cmd). Named here rather than
#: inlined so a caller adding a category surfaces the description in
#: one place; kept plain (no emoji markers) so the source-string
#: emoji guard the repo runs at CI does not fire on the description
#: table.
_DDS_TOPIC_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "state": (
        "Read-side robot state channels: IMU, joints, motors, battery, mainboard temps, foot pressure, sportmode."
    ),
    "lidar": ("Read-side Livox Mid-360 channels: the point-cloud stream plus the LiDAR sensor's own state topic."),
    "joystick": ("Read-side wireless controller channel; silent when no controller is paired."),
    "control": (
        "Write-side low-level command topics that drive the robot "
        "at the wire; every entry is on the dangerous-publish "
        "refusal list."
    ),
    "hand": ("Inspire/Unitree 5/7-DoF hand channels; a state read plus a command write."),
    "slam": ("Read-side odometry and SLAM global-map channels; topic naming varies across firmware releases."),
    "config": ("Write-side non-motion configuration channels; the LiDAR ON/OFF switch is the only entry today."),
}

#: Per-category topic-count snapshot: how many neon-catalog topics
#: carry each category label today. Named here rather than computed
#: from a re-imported neon catalog because the whole point of the
#: port is that the snapshot lets a headless host answer without
#: pulling the neon module; a neon-side widen updates this dict in
#: the same PR as the neon table, or a category with more or fewer
#: entries surfaces the drift at CI (the parity test in
#: ``tests/drivers/`` reads this dict against the neon module when
#: the neon module is importable and skips otherwise).
_DDS_TOPIC_CATEGORY_COUNTS: dict[str, int] = {
    "state": 9,
    "lidar": 2,
    "joystick": 1,
    "control": 4,
    "hand": 2,
    "slam": 3,
    "config": 1,
}


def _describe(name: str) -> dict[str, Any]:
    """Build the per-category descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_dds_topic_categories` so
    :func:`g1_dds_topic_category_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched.
    The ``topic_count`` field carries the neon-catalog partition
    size for the category (the invariant a neon-side widen must
    preserve is that :data:`_DDS_TOPIC_CATEGORY_COUNTS` and the
    neon table agree; the parity test surfaces a drift there).
    """
    return {
        "name": name,
        "description": _DDS_TOPIC_CATEGORY_DESCRIPTIONS[name],
        "topic_count": _DDS_TOPIC_CATEGORY_COUNTS[name],
    }


@tool
def g1_list_dds_topic_categories() -> dict[str, Any]:
    """Return the DDS topic-category labels the neon catalog partitions.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for the neon ``g1_dds_list_topics`` verb is called, so a caller
    can name the category-scoped filter argument decidably before the
    filter itself runs.

    The envelope names seven partition labels: ``state`` (9 topics),
    ``lidar`` (2), ``joystick`` (1), ``control`` (4), ``hand`` (2),
    ``slam`` (3), and ``config`` (1). Each descriptor carries the
    label's plain-text description plus its neon-catalog topic
    count.

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        category labels; a ``categories`` list of descriptors (one
        per label, sorted lexicographically by ``name``) carrying
        ``name``, ``description``, and ``topic_count``; and a
        ``names`` field listing the label strings sorted
        lexicographically. Every field is a snapshot of a
        neon-observed constant; no dynamic decode runs here.
    """
    names = sorted(_DDS_TOPIC_CATEGORIES)
    return {
        "status": "success",
        "count": len(_DDS_TOPIC_CATEGORIES),
        "categories": [_describe(name) for name in names],
        "names": names,
    }


@tool
def g1_dds_topic_category_admits(name: str = "") -> dict[str, Any]:
    """Decide whether a category name is inside the neon-catalog partition.

    Read-only. Reads the module's snapshot of the neon bundle's
    ``TOPIC_CATALOG`` category labels and returns the same membership
    answer the neon ``g1_dds_list_topics`` verb's filter would
    compute. A caller with a category string resolves it against the
    partition before a future catalog-read verb dispatches, rather
    than triggering the neon filter's silent empty-result at wire
    time (an unknown category label returns zero topics under the
    neon filter, which is indistinguishable from a category that
    happens to be empty at runtime).

    A name inside :data:`_DDS_TOPIC_CATEGORIES` is a valid neon
    partition label; a name outside is a caller mistake the neon
    filter would not name at the surface. This verb answers the
    narrower membership question the neon partition answers.

    Args:
        name: The category label to test. Must be a non-empty
            ``str``; ``bool`` is refused (``True``/``False`` are not
            valid category labels under any DDS convention) and the
            empty string is refused as a shape error (no label means
            no membership query to answer). Non-str inputs are
            refused decidably rather than resolved through Python's
            ``str()`` coercion.

    Returns:
        A dict with ``status``; a ``query`` sub-dict carrying the
        supplied ``name``; an ``admitted`` boolean naming whether
        the name is a member of the seven-label partition; and
        (when ``admitted`` is ``True``) a ``target`` sub-dict
        carrying the same descriptor :func:`g1_list_dds_topic_categories`
        returns for the label (``name``, ``description``,
        ``topic_count``). On a not-admitted query the dict carries
        a ``refusal_advice`` field naming the neon-side filter note
        that an unknown category label returns zero topics under the
        neon filter, and listing the seven valid labels so the
        caller can resolve the drift without a follow-up call. On a
        shape error (``bool``, non-str, empty string) the dict
        carries ``status="error"`` with a message naming the type
        refused.
    """
    if isinstance(name, bool):
        return {
            "status": "error",
            "message": (f"name must be str, got bool ({name!r}). Refs strands-labs/robots#358."),
        }
    if not isinstance(name, str):
        return {
            "status": "error",
            "message": (f"name must be str, got {type(name).__name__} ({name!r}). Refs strands-labs/robots#358."),
        }
    if name == "":
        return {
            "status": "error",
            "message": (
                "name must be a non-empty str; an empty category label has "
                "no membership answer to compute. Refs strands-labs/robots#358."
            ),
        }

    admitted = name in _DDS_TOPIC_CATEGORIES
    if not admitted:
        return {
            "status": "success",
            "admitted": False,
            "query": {"name": name},
            "refusal_advice": (
                f"category {name!r} is not on the neon catalog partition; "
                "the neon g1_dds_list_topics filter would return zero "
                "topics for an unknown label, indistinguishable from an "
                "empty category. Known labels: "
                f"{sorted(_DDS_TOPIC_CATEGORIES)!r}. Refs "
                "strands-labs/robots#358."
            ),
        }

    return {
        "status": "success",
        "admitted": True,
        "query": {"name": name},
        "target": _describe(name),
    }
