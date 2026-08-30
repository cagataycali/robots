"""The IDL-type verbs name exactly the types ``G1Driver`` decodes.

``G1Driver._subscription_plan`` returns a six-tuple of
``(topic, (idl_module, idl_class), decoder)`` triples naming, per read
topic, which CycloneDDS-typed Python object the driver's
``ChannelSubscriber`` decodes the wire bytes into (``rt/lowstate`` →
``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowState_``, and so on). The
write side is the single ``rt/lowcmd`` topic the driver's own
``ChannelPublisher`` is constructed against, decoded as
``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowCmd_``. Two agent-facing
tools - :func:`g1_list_dds_topic_idl_types` and :func:`g1_topic_idl_type`
- exist to surface those seven IDL-type identifiers as a static snapshot
before a caller decides whether to open a matching subscribe or publish.

The IDL identifiers this tool carries are checked here against the
driver's own ``_subscription_plan`` return values rather than being
restated in the tests, so a driver-side widen or narrow of the IDL type
on any topic (a shape change on ``LowCmd_``, a new sensor topic) does
not require also editing this file. What the tests do restate is the
*shape* of each returned record and the SDK-load-hygiene contract
every file under :mod:`strands_robots.tools.g1` carries: importing the
tool module must not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_dds_topic_idl_types import (
    _DRIVER_TOPIC_IDL_TYPES,
    g1_list_dds_topic_idl_types,
    g1_topic_idl_type,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it returns the wrapped function's
    return value verbatim. This helper is where a shape drift would
    surface once, rather than at every call site.
    """
    return tool(**kwargs)


# ---------------------------------------------------------------------- #
# Import hygiene                                                         #
# ---------------------------------------------------------------------- #


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an
    office bring-up. The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the IDL-type verbs to it
    too. The snapshot names IDL modules as strings rather than
    importing them for the same reason.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_dds_topic_idl_types")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_dds_topic_idl_types imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the "
        f"SDK loads only inside function bodies "
        f"(refs strands-labs/robots#358)."
    )


# ---------------------------------------------------------------------- #
# Snapshot fidelity vs. the driver plan                                  #
# ---------------------------------------------------------------------- #


def test_the_snapshot_matches_the_driver_subscription_plan() -> None:
    """The tool's IDL-type snapshot is the driver's own read-side plan.

    The driver holds six read-side topics in ``_subscription_plan``,
    each paired with its ``(idl_module, idl_class)`` string pair. The
    tool module's :data:`_DRIVER_TOPIC_IDL_TYPES` names the same six
    read entries; a widen or narrow on the driver side surfaces here as
    a diff between the two sets, not as a diverging table the author
    would need to keep in sync by hand. The write-side entry
    (``rt/lowcmd`` → ``LowCmd_``) is asserted separately below since
    the driver holds that in a ``ChannelPublisher`` rather than in
    ``_subscription_plan``.
    """
    from strands_robots.drivers.g1 import G1Driver

    # ``_subscription_plan`` is an instance method, but it references
    # only class-level constants, so a bare instance is enough - no
    # DDS init, no SDK load. Bypass ``__init__`` because it wires the
    # bus config a real driver needs.
    driver = G1Driver.__new__(G1Driver)
    plan = driver._subscription_plan()

    driver_read_types = {topic: {"idl_module": pair[0], "idl_class": pair[1]} for topic, pair, _decoder in plan}
    tool_read_types = {
        entry["topic"]: {
            "idl_module": entry["idl_module"],
            "idl_class": entry["idl_class"],
        }
        for entry in _DRIVER_TOPIC_IDL_TYPES
        if entry["direction"] == "read"
    }

    assert driver_read_types == tool_read_types, (
        f"The tool's read-side IDL-type snapshot has drifted from the "
        f"driver's ``_subscription_plan`` return. "
        f"Driver plan carries {driver_read_types}; tool snapshot "
        f"carries {tool_read_types}. Refs strands-labs/robots#358."
    )


def test_the_snapshot_carries_the_write_side_lowcmd_entry() -> None:
    """The write-side entry names ``rt/lowcmd`` decoded as ``LowCmd_``.

    ``G1Driver`` publishes low-level motor commands on ``rt/lowcmd``
    against a ``ChannelPublisher`` typed with
    ``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowCmd_``; the topic
    string is captured as ``_TOPIC_LOWCMD`` in the driver module. The
    snapshot carries the same pair on the write side, so a caller
    planning a mesh publish on the write topic sees the same IDL type
    the driver's own publisher was constructed with.
    """
    write_entries = [entry for entry in _DRIVER_TOPIC_IDL_TYPES if entry["direction"] == "write"]
    assert write_entries == [
        {
            "topic": "rt/lowcmd",
            "direction": "write",
            "role": "lowcmd",
            "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            "idl_class": "LowCmd_",
        }
    ], (
        f"The tool's write-side IDL-type snapshot must carry exactly one "
        f"entry (``rt/lowcmd`` → ``LowCmd_``) matching the driver's own "
        f"``ChannelPublisher`` construction. Snapshot carries "
        f"{write_entries}. Refs strands-labs/robots#358."
    )


def test_the_snapshot_carries_seven_entries() -> None:
    """The snapshot has exactly seven entries - six read, one write.

    The driver's ``connect_eagerly`` opens six subscribers and one
    publisher; the tool's snapshot has exactly seven entries. A drift
    on the count (five, eight) is the earliest signal a driver-side
    subscribe or publish was added or removed without the mirror
    change here.
    """
    assert len(_DRIVER_TOPIC_IDL_TYPES) == 7, (
        f"The tool's snapshot must carry seven entries "
        f"(six read + one write); got {len(_DRIVER_TOPIC_IDL_TYPES)}. "
        f"Refs strands-labs/robots#358."
    )
    reads = [e for e in _DRIVER_TOPIC_IDL_TYPES if e["direction"] == "read"]
    writes = [e for e in _DRIVER_TOPIC_IDL_TYPES if e["direction"] == "write"]
    assert len(reads) == 6, f"Expected 6 read entries; got {len(reads)}."
    assert len(writes) == 1, f"Expected 1 write entry; got {len(writes)}."


# ---------------------------------------------------------------------- #
# g1_list_dds_topic_idl_types                                            #
# ---------------------------------------------------------------------- #


def test_list_returns_every_snapshot_entry() -> None:
    """The list verb returns one descriptor per snapshot entry.

    Read-only, no filter, no arguments - the whole seven-entry table
    ships out. A caller who wanted to iterate the driver's plan
    without touching the driver itself can walk this list.
    """
    payload = _call(g1_list_dds_topic_idl_types)
    assert payload["status"] == "success", payload
    assert payload["count"] == len(_DRIVER_TOPIC_IDL_TYPES) == 7
    returned_topics = {entry["topic"] for entry in payload["topics"]}
    expected_topics = {entry["topic"] for entry in _DRIVER_TOPIC_IDL_TYPES}
    assert returned_topics == expected_topics, (
        f"The list verb returned {returned_topics}; snapshot carries {expected_topics}. Refs strands-labs/robots#358."
    )


def test_list_carries_every_field_per_entry() -> None:
    """Each returned descriptor names topic, direction, role, module, class.

    Every entry ships with the five keys the snapshot carries; a
    field drop surfaces here rather than at the caller. The values
    are the same strings the snapshot carries - the verb hands out
    dict copies of the snapshot entries, not paraphrases.
    """
    payload = _call(g1_list_dds_topic_idl_types)
    for entry in payload["topics"]:
        assert set(entry.keys()) == {
            "topic",
            "direction",
            "role",
            "idl_module",
            "idl_class",
        }, f"Entry {entry} is missing or has extra keys. Refs strands-labs/robots#358."


def test_list_returns_dict_copies_so_the_snapshot_is_immutable() -> None:
    """The verb hands out fresh dicts so a caller cannot mutate the snapshot.

    A caller who mutated a returned descriptor and then re-called the
    verb would otherwise see the mutation surface on the next call,
    because the snapshot is a module-level tuple of dicts. The verb
    guards against that by copying each entry before returning.
    """
    payload_a = _call(g1_list_dds_topic_idl_types)
    payload_a["topics"][0]["idl_class"] = "MutatedByCaller_"
    payload_b = _call(g1_list_dds_topic_idl_types)
    for entry in payload_b["topics"]:
        assert entry["idl_class"] != "MutatedByCaller_", (
            f"Mutation from a prior call surfaced on the next call: "
            f"{entry}. The verb must hand out dict copies. "
            f"Refs strands-labs/robots#358."
        )


# ---------------------------------------------------------------------- #
# g1_topic_idl_type - admit                                              #
# ---------------------------------------------------------------------- #


def test_admit_returns_the_driver_side_idl_type_for_a_known_topic() -> None:
    """A known topic admits with the snapshot's IDL identifier pair.

    ``rt/lowstate`` is decoded by the driver as ``LowState_`` in the
    ``unitree_sdk2py.idl.unitree_hg.msg.dds_`` module; the verb
    returns the same pair, plus the role and direction labels the
    caller sees on the list side too.
    """
    payload = _call(g1_topic_idl_type, topic="rt/lowstate")
    assert payload == {
        "status": "success",
        "topic": "rt/lowstate",
        "direction": "read",
        "role": "lowstate",
        "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "idl_class": "LowState_",
    }, payload


def test_admit_returns_the_write_side_idl_type_for_lowcmd() -> None:
    """``rt/lowcmd`` admits as the driver's write-side IDL type.

    The write topic is a ``ChannelPublisher`` on the driver side; the
    verb reads the same ``LowCmd_`` type off the snapshot so a caller
    planning a mesh publish sees the driver-side publisher type
    without a second lookup.
    """
    payload = _call(g1_topic_idl_type, topic="rt/lowcmd")
    assert payload == {
        "status": "success",
        "topic": "rt/lowcmd",
        "direction": "write",
        "role": "lowcmd",
        "idl_module": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "idl_class": "LowCmd_",
    }, payload


def test_admit_handles_every_snapshot_topic() -> None:
    """Every topic in the snapshot admits with its snapshot descriptor.

    Walks the snapshot and asserts each topic admits with the exact
    descriptor the snapshot carries. This is the sweep that catches a
    per-topic drift in ``idl_module`` or ``idl_class`` without needing
    seven per-topic tests.
    """
    for entry in _DRIVER_TOPIC_IDL_TYPES:
        payload = _call(g1_topic_idl_type, topic=entry["topic"])
        assert payload == {
            "status": "success",
            "topic": entry["topic"],
            "direction": entry["direction"],
            "role": entry["role"],
            "idl_module": entry["idl_module"],
            "idl_class": entry["idl_class"],
        }, (
            f"Topic {entry['topic']!r} admitted with unexpected shape: "
            f"{payload}. Snapshot carries {entry}. "
            f"Refs strands-labs/robots#358."
        )


# ---------------------------------------------------------------------- #
# g1_topic_idl_type - refuse                                             #
# ---------------------------------------------------------------------- #


def test_unknown_topic_refuses_and_names_the_known_set() -> None:
    """An off-set topic refuses with the known-topic list and a #358 ref.

    A caller who typoed the topic (``rt/lowstate2`` for
    ``rt/lowstate``) or named a topic the driver doesn't open
    (``rt/wirelesscontroller`` - the neon bundle carries it in the
    fuller catalog, but the driver itself does not subscribe it)
    reads the driver's actual subscription set off the refusal and
    can retry with a known name.
    """
    payload = _call(g1_topic_idl_type, topic="rt/lowstate2")
    assert payload["status"] == "error", payload
    assert "not in the driver's subscription set" in payload["message"]
    assert "strands-labs/robots#358" in payload["message"]
    assert "rt/lowstate" in payload["message"]
    # ``rt/wirelesscontroller`` sanity: neon carries it, driver doesn't.
    payload_off_catalog = _call(g1_topic_idl_type, topic="rt/wirelesscontroller")
    assert payload_off_catalog["status"] == "error", payload_off_catalog


def test_empty_string_topic_refuses_with_shape_error() -> None:
    """The empty string is refused as a shape error, not an off-set match.

    ``""`` is a ``str``, so the type gate admits it; the empty-string
    gate then refuses it with a message that names why - a caller who
    accidentally passed ``os.getenv("G1_TOPIC")`` on an unset env var
    reads the shape error rather than a confusing off-set list.
    """
    payload = _call(g1_topic_idl_type, topic="")
    assert payload["status"] == "error", payload
    assert "non-empty" in payload["message"], payload
    assert "strands-labs/robots#358" in payload["message"]


def test_non_string_topic_refuses_with_shape_error() -> None:
    """Non-string arguments refuse with a type-error message.

    ``None``, ``int``, ``list`` all refuse decidably rather than
    coerce through Python's equality rules. The refusal names the
    actual type the caller passed so the fix is obvious.
    """
    for bogus in (None, 42, ["rt/lowstate"], {"topic": "rt/lowstate"}):
        payload = _call(g1_topic_idl_type, topic=bogus)  # type: ignore[arg-type]
        assert payload["status"] == "error", (bogus, payload)
        assert "topic must be a str" in payload["message"], (bogus, payload)
        assert type(bogus).__name__ in payload["message"], (bogus, payload)
        assert "strands-labs/robots#358" in payload["message"]


def test_bool_topic_refuses_as_shape_error_not_off_set_match() -> None:
    """``True`` / ``False`` refuse as shape errors, not off-set names.

    Python's ``bool`` is a ``int`` subclass, and ``True == 1``; a
    naive membership test could confuse a caller who passed ``True``
    (say, from a truthiness check) as an integer that just happened
    not to match any topic string. The verb refuses ``bool``
    explicitly with the type name in the message.
    """
    for bogus in (True, False):
        payload = _call(g1_topic_idl_type, topic=bogus)  # type: ignore[arg-type]
        assert payload["status"] == "error", (bogus, payload)
        assert "bool" in payload["message"], (bogus, payload)
        assert "strands-labs/robots#358" in payload["message"]
