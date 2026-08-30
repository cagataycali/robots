"""Agent-facing lookup for the DDS topics a naive publish path must refuse.

The Unitree G1 firmware exposes a handful of DDS write topics whose
payloads command the robot at the wire level: ``rt/lowcmd`` (the
:class:`~strands_robots.drivers.g1.G1Driver` write path itself,
publishing joint torque/position commands on the low-level control
loop), ``rt/armsdk`` (the arm-SDK override the neon bundle's
``g1_arm_action`` verb uses), ``rt/user_lowcmd`` (a user-shaped
alternative to ``rt/lowcmd``), ``rt/inspire/cmd`` (the Inspire hand
5/7-DoF command), and ``rt/bmscmd`` (the battery-management command
topic, which admits a reboot or power-cycle payload). Every one of
those topics is a write path that bypasses the driver's FSM gate and
the arm-SDK's own admission set at the wire.

The neon bundle's ``_dds_engine.py``
(``cagataycali/neon-the-g1/tools/_dds_engine.py``) captures the set as
``DANGEROUS_PUB_TOPICS`` and its generic ``g1_dds_publish`` verb
refuses membership on it unless the caller passes ``unsafe=True``.
This module snapshots the same five-topic set into a module-level
constant and exposes two agent-facing verbs -
:func:`g1_list_dangerous_publish_topics` (list the whole envelope) and
:func:`g1_dangerous_publish_topic_admits` (decide one query) - so a
caller planning a generic DDS publish can decide the refusal decidably
before a future driver-side wrapper for the wire actually fires.

Refs strands-labs/robots#358.

Two things this module is deliberately *not*:

* An execution path. The neon bundle's ``g1_dds_publish`` verb opens
  a second DDS writer against arbitrary topics under the same
  :func:`~strands_robots.tools.g1._g1_common.ensure_dds` singleton the
  driver holds; that second writer path is out of scope for this
  lookup. The driver at :class:`~strands_robots.drivers.g1.G1Driver`
  writes only ``rt/lowcmd`` and gates every write through
  :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`;
  a future driver method that fronts an arbitrary DDS publish will
  cross-reference this membership set at the wire refusal. This
  module ports the read-only membership half without also
  introducing a second writer path the driver does not yet own.
* An SDK or CycloneDDS re-import. The topic names are captured here
  as string constants snapshotted from the neon bundle's
  ``DANGEROUS_PUB_TOPICS`` set; the snapshot lives here rather than
  being re-read from either the SDK or the neon module so
  ``import strands_robots.tools.g1.g1_dangerous_publish_topics``
  pulls no ``unitree_sdk2py`` submodule - the import-hygiene contract
  every other file in this package carries, refs
  strands-labs/robots#358. A firmware release that widens or narrows
  the danger set is a caller-side update; when a driver-side
  publish wrapper lands, its refusal will name the same membership
  test this lookup answers.

What this module does not decide.

* Whether a topic is *published-safe* at any particular moment. A
  topic outside :data:`_DANGEROUS_PUBLISH_TOPICS` is not automatically
  safe: an unknown topic may still command the robot if the caller
  supplies a matching IDL type. This lookup answers the narrower
  membership question the neon bundle's ``DANGEROUS_PUB_TOPICS`` set
  answers: is this topic on the observed *known-dangerous* list, and
  therefore worth refusing without an explicit ``unsafe=True``
  override? Callers planning a publish still cross-check the
  topic against
  :mod:`~strands_robots.tools.g1.g1_dds_topics` (the driver's own
  subscription set) and their own knowledge of the topic's IDL.
* The driver's live FSM. A dangerous publish is dangerous regardless
  of the current FSM; this membership set does not carry the FSM
  gate's ``rt/lowcmd`` refusal (``rc=7404``) because a driver-side
  publish wrapper's own gate would layer on top of this lookup. A
  caller planning a write reads this membership set *and* consults
  :mod:`~strands_robots.tools.g1.g1_motion_gates` for the FSM-side
  refusal separately; the two questions do not collapse.
* Topic strings the neon catalog does not carry. The ``rt/lowcmd``
  entry is the same string
  :mod:`~strands_robots.tools.g1.g1_dds_topics` names on the
  driver's own write side; the invariant a future widen must
  preserve is byte-for-byte identity between the two files' topic
  strings, so a driver-side rename lands on both surfaces or the
  ``rt/lowcmd`` refusal on one side and the same-named topic on the
  other silently diverge. The four other entries
  (``rt/armsdk``, ``rt/user_lowcmd``, ``rt/inspire/cmd``,
  ``rt/bmscmd``) are wire paths the driver does not yet write; when
  a driver-side wrapper for any of them lands, its refusal will
  quote the same string this snapshot carries.
"""

from __future__ import annotations

from typing import Any

from strands import tool

#: Snapshot of the five DDS write topics the neon bundle's
#: ``_dds_engine.DANGEROUS_PUB_TOPICS`` set carries as known-dangerous
#: publish paths. Each entry is a wire-level command topic that
#: bypasses the driver's FSM gate and the arm-SDK's own admission
#: set at the wire: publishing to any of them without an explicit
#: ``unsafe=True`` override in a caller's own publish path is a
#: caller mistake the neon bundle refused, and the same membership
#: test lands here so a caller planning a generic DDS publish can
#: decide the refusal decidably before a future driver-side wrapper
#: fires.
#:
#: The names live here as a module-level ``frozenset`` rather than
#: being re-imported from the neon module so
#: ``import strands_robots.tools.g1.g1_dangerous_publish_topics``
#: pulls zero ``unitree_sdk2py`` submodules - the import-hygiene
#: contract every other file in this package carries.
_DANGEROUS_PUBLISH_TOPICS: frozenset[str] = frozenset(
    {
        "rt/lowcmd",
        "rt/armsdk",
        "rt/user_lowcmd",
        "rt/inspire/cmd",
        "rt/bmscmd",
    }
)

#: Per-topic label the returned envelopes surface as a description of
#: the wire path the topic commands. The labels are the descriptions
#: the neon bundle's ``_dds_engine.TOPIC_CATALOG`` carries for the same
#: five topics (with the neon bundle's own 🚨 markers stripped so the
#: string domain is plain text). Named here rather than re-imported
#: from the neon module so the constant snapshot is self-contained and
#: the SDK-load-hygiene contract holds; a caller widening the neon
#: table updates the two files together or the description drifts.
_TOPIC_DESCRIPTIONS: dict[str, str] = {
    "rt/lowcmd": "Low-level motor cmd (drives every joint on the low-level control loop)",
    "rt/armsdk": "Arm SDK override (the arm-SDK admission path a naive publish bypasses)",
    "rt/user_lowcmd": "User low-level cmd (an alternative rt/lowcmd path with the same wire risk)",
    "rt/inspire/cmd": "Inspire hand command (5/7-DoF hand joint cmd, publishes to the hand controller)",
    "rt/bmscmd": "BMS cmd (battery-management command, admits a reboot or power-cycle payload)",
}

#: The default advice string the returned envelopes surface as the
#: refusal a caller planning a publish would face. Kept here rather
#: than inlined so a future re-wording lands in one place instead of
#: drifting between the caller-side refusal and this lookup's
#: surfaced text. Names ``unsafe=True`` verbatim so a caller who
#: means to override sees the same argument the neon bundle's
#: ``g1_dds_publish`` verb takes; refs
#: ``cagataycali/neon-the-g1/tools/g1_dds.py`` for the caller-side
#: verb the argument gates on.
_REFUSAL_ADVICE: str = (
    "This topic commands the robot at the wire level. A publish path "
    "must refuse the write unless the caller passes an explicit "
    "unsafe=True override; refs strands-labs/robots#358 for the "
    "driver-side gate work."
)


def _describe(topic: str) -> dict[str, Any]:
    """Build the per-topic descriptor the verbs return.

    Kept here rather than inlined in
    :func:`g1_list_dangerous_publish_topics` so
    :func:`g1_dangerous_publish_topic_admits`'s admitted-path payload
    names the same fields, and so a widen to the descriptor lands in
    one place. Every field is a snapshot read; no bus is touched. The
    ``dangerous_to_publish`` flag is always ``True`` on every entry
    (membership *is* the danger contract); it is surfaced anyway so
    the returned shape mirrors the neon bundle's own
    ``list_topics`` payload, which carries the same field on every
    row it returns.
    """
    return {
        "topic": topic,
        "description": _TOPIC_DESCRIPTIONS[topic],
        "dangerous_to_publish": True,
    }


@tool
def g1_list_dangerous_publish_topics() -> dict[str, Any]:
    """Return the DDS write topics a naive publish path must refuse.

    Read-only. No driver instance, no DDS, no SDK: every field is a
    module-level constant. Useful before a future driver-side wrapper
    for a generic DDS publish is called, so a caller can compare an
    intended write topic against the set the neon bundle's
    ``g1_dds_publish`` verb refuses on without an explicit
    ``unsafe=True`` override.

    The envelope names five known-dangerous write topics:
    ``rt/lowcmd`` (the driver's own low-level motor command wire),
    ``rt/armsdk`` (the arm-SDK override), ``rt/user_lowcmd`` (a
    user-shaped alternative to ``rt/lowcmd``), ``rt/inspire/cmd``
    (the Inspire hand 5/7-DoF command), and ``rt/bmscmd`` (the
    battery-management command). Each descriptor carries the neon
    bundle's own description string plus a ``dangerous_to_publish``
    flag (always ``True`` on membership, surfaced for shape parity
    with the neon bundle's ``list_topics`` payload).

    Returns:
        A dict with ``status``; a ``count`` naming the number of
        known-dangerous topics; a ``dangerous_topics`` list of
        descriptors (one per topic, sorted lexicographically)
        carrying ``topic``, ``description``, and
        ``dangerous_to_publish``; a ``topics`` field listing the
        topic strings sorted lexicographically; and a
        ``refusal_advice`` field naming the caller-side refusal a
        publish path would surface on membership. Every field is a
        snapshot of a neon-observed constant; no dynamic decode runs
        here.
    """
    topics = sorted(_DANGEROUS_PUBLISH_TOPICS)
    return {
        "status": "success",
        "count": len(_DANGEROUS_PUBLISH_TOPICS),
        "dangerous_topics": [_describe(topic) for topic in topics],
        "topics": topics,
        "refusal_advice": _REFUSAL_ADVICE,
    }


@tool
def g1_dangerous_publish_topic_admits(topic: str = "") -> dict[str, Any]:
    """Decide whether a topic is inside the known-dangerous publish set.

    Read-only. Reads the module's snapshot of the neon bundle's
    ``DANGEROUS_PUB_TOPICS`` set and returns the same membership
    answer a caller-side publish path would compute. A caller with a
    topic string resolves it against the set before a future publish
    verb dispatches, rather than triggering the caller-side refusal
    at wire time.

    A topic inside the set is a topic the caller-side publish path
    must refuse without an explicit ``unsafe=True`` override; a topic
    outside the set is *not* automatically safe (an unknown topic may
    still command the robot if the caller supplies a matching IDL
    type). This verb answers the narrower membership question the
    neon bundle's ``DANGEROUS_PUB_TOPICS`` set answers.

    Args:
        topic: The topic string to test. Must be a non-empty ``str``;
            ``bool`` is refused (``True``/``False`` are not valid
            topic strings under any DDS convention) and the empty
            string is refused as a shape error (no topic name means
            no membership query to answer). Non-str inputs are
            refused decidably rather than resolved through Python's
            ``str()`` coercion.

    Returns:
        A dict with ``status``; a ``query`` sub-dict carrying the
        supplied ``topic``; an ``admitted`` boolean naming whether
        the topic is a member of the known-dangerous set; and (when
        ``admitted`` is ``True``) a ``target`` sub-dict carrying the
        same descriptor :func:`g1_list_dangerous_publish_topics`
        returns for the topic (``topic``, ``description``,
        ``dangerous_to_publish``). On a not-admitted query the dict
        carries a ``refusal_advice`` field naming the caller-side
        note that the topic is not on the known-dangerous list but a
        caller planning a publish still cross-checks the topic
        against its own IDL knowledge. On a shape error (``bool``,
        non-str, empty string) the dict carries ``status="error"``
        with a message naming the type refused.
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
                "topic must be a non-empty str; an empty topic name has no "
                "membership answer to compute. Refs strands-labs/robots#358."
            ),
        }

    admitted = topic in _DANGEROUS_PUBLISH_TOPICS
    if not admitted:
        return {
            "status": "success",
            "admitted": False,
            "query": {"topic": topic},
            "refusal_advice": (
                f"topic {topic!r} is not on the known-dangerous publish list; "
                "a caller planning a publish still cross-checks the topic "
                "against its own IDL knowledge before dispatch. Refs "
                "strands-labs/robots#358."
            ),
        }

    return {
        "status": "success",
        "admitted": True,
        "query": {"topic": topic},
        "target": _describe(topic),
    }
