"""The DDS topic-description envelope tools name exactly what the neon catalog ships.

The neon bundle's ``_dds_engine.py`` (``cagataycali/neon-the-g1/tools/_dds_engine.py``)
carries a ``TOPIC_CATALOG`` dict whose per-topic value tuple's third
position is a plain-text description string; the neon generic
``g1_dds_list_topics`` verb returns those descriptors verbatim to a
caller. The :mod:`strands_robots.tools.g1.g1_dds_topic_descriptions`
module snapshots the twenty-two topic descriptions into a module-level
constant and exposes two agent-facing verbs -
:func:`g1_list_dds_topic_descriptions` (list the whole envelope) and
:func:`g1_dds_topic_description_admits` (decide one membership query) -
so a caller planning a bus-side read or write names what the topic
decodes at the wire decidably before a future driver-side wrapper
for the neon ``g1_dds_snapshot`` / ``g1_dds_subscribe`` verb lands.
The tests here fix that contract without pulling the SDK: the module
is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
constant surfaces here as a shape change rather than as a diverging
table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The neon bundle's own answer at wire time. The verbs answer against
  the module-level snapshot, not against a live import of the neon
  bundle's ``TOPIC_CATALOG`` value tuples (the whole point of the
  port is that the snapshot lets a headless host answer without
  pulling the neon module or the CycloneDDS bindings). A driver-side
  wrapper for the neon catalog read that lands later will
  re-validate against its own live table at wire time; testing the
  snapshot vs the live table is a driver-side test, not a
  lookup-side one.
* Per-topic IDL type or category. The verb answers only the description
  column of the neon catalog's per-topic tuple. Which IDL class decodes
  the payload is answered by
  :mod:`~strands_robots.tools.g1.g1_dds_topic_idl_types`, and which
  category partitions the topic is answered by
  :mod:`~strands_robots.tools.g1.g1_dds_topic_categories`. A caller
  wanting both fields dispatches to each verb separately; this test
  file does not cross that scope line.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_dds_topic_descriptions import (
    _DDS_TOPIC_DESCRIPTIONS,
    g1_dds_topic_description_admits,
    g1_list_dds_topic_descriptions,
)


def _call(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call a ``@tool``-decorated function and unwrap the payload.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on
    that: the wrapper's contract is that it returns the wrapped
    function's return value verbatim. This helper is where a shape
    drift would surface once, rather than at every call site.
    """
    return tool(**kwargs)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import
    time would break every headless CI runner and Thor before an
    office bring-up. The driver enforces the same rule against
    itself (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is
    the only path that loads the SDK); this cell holds the
    dds-topic-descriptions envelope verbs to it too (refs
    strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_dds_topic_descriptions")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_dds_topic_descriptions imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads only inside function bodies (refs "
        "strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_catalog() -> None:
    """The snapshot names every topic the neon catalog carries today.

    The neon bundle's ``_dds_engine.TOPIC_CATALOG`` ships twenty-two
    entries: nine ``state``, two ``lidar``, one ``joystick``, four
    ``control``, two ``hand``, three ``slam``, one ``config``. A drift
    on either side surfaces here: a driver-side generic-snapshot
    wrapper (when it lands) will validate the same catalog at wire
    time and its refusal string will quote the same membership test.
    The count is pinned rather than listed value-by-value so a
    caller widening the catalog on the neon side updates one number
    here rather than 22 assertions.
    """
    assert len(_DDS_TOPIC_DESCRIPTIONS) == 22, (
        f"expected 22 topic entries in the neon-observed snapshot, got "
        f"{len(_DDS_TOPIC_DESCRIPTIONS)}: {sorted(_DDS_TOPIC_DESCRIPTIONS)}. A "
        "neon-side widen or narrow would update this count; refs "
        "strands-labs/robots#358."
    )


def test_every_snapshot_topic_carries_a_non_empty_description() -> None:
    """Every topic key has a non-empty description string.

    :data:`_DDS_TOPIC_DESCRIPTIONS` is the table both verbs read when
    they build the returned descriptor. A neon-side widen must
    supply a non-empty description string; an empty description
    would leave a caller unable to name the intent of the topic at
    the surface, which is the whole point of surfacing the
    description column separately from the IDL-type and category
    columns.
    """
    for topic, description in _DDS_TOPIC_DESCRIPTIONS.items():
        assert isinstance(description, str) and description != "", (
            f"description for topic {topic!r} must be a non-empty str; "
            f"got {description!r}. Refs strands-labs/robots#358."
        )


def test_every_snapshot_topic_key_is_an_rt_prefixed_string() -> None:
    """Every topic name follows the neon catalog's ``rt/`` prefix convention.

    The neon ``_dds_engine.TOPIC_CATALOG`` names every G1 DDS topic
    with the ``rt/`` prefix Unitree's SDK ships. A key without that
    prefix would either be a caller mistake in the port or a neon
    widen that broke the convention; either way, the drift surfaces
    here so the assertion in a driver-side wrapper for the
    ``g1_dds_snapshot`` verb (when it lands) does not silently pass
    the un-prefixed key through.
    """
    for topic in _DDS_TOPIC_DESCRIPTIONS:
        assert topic.startswith("rt/"), (
            f"topic key {topic!r} does not start with 'rt/'; the neon "
            "TOPIC_CATALOG names every G1 DDS topic with that prefix. Refs "
            "strands-labs/robots#358."
        )


def test_the_control_topics_carry_the_dangerous_publish_marker() -> None:
    """Every ``rt/lowcmd``-family control topic description carries 🚨.

    The neon ``TOPIC_CATALOG`` marks every write-side low-level
    command topic with the 🚨 emoji in its description so a caller
    reading the raw catalog can see the dangerous-publish partition
    at a glance. A neon-side widen that dropped the marker on one
    of the four control topics would silently downgrade the caller's
    visible refusal; pinning the marker here surfaces the drift.
    Refs :mod:`~strands_robots.tools.g1.g1_dangerous_publish_topics`
    for the sibling snapshot of the same partition as a bare set.
    """
    control_topics = (
        "rt/lowcmd",
        "rt/armsdk",
        "rt/user_lowcmd",
        "rt/bmscmd",
    )
    for topic in control_topics:
        description = _DDS_TOPIC_DESCRIPTIONS[topic]
        assert "\U0001f6a8" in description, (
            f"description for control topic {topic!r} is {description!r}; "
            "the neon catalog marks every write-side low-level command "
            "topic with 🚨. Refs strands-labs/robots#358."
        )


def test_the_inspire_cmd_topic_carries_the_dangerous_publish_marker() -> None:
    """The Inspire hand ``cmd`` write topic carries 🚨 too.

    The neon ``DANGEROUS_PUB_TOPICS`` set is five entries: the four
    ``control`` category topics plus ``rt/inspire/cmd`` (the Inspire
    hand cmd, which sits in the ``hand`` category rather than
    ``control``). The description column marks it as 🚨 too, which
    a caller reading the raw catalog needs to see so the hand
    partition does not silently look safer than the low-level
    command partition.
    """
    description = _DDS_TOPIC_DESCRIPTIONS["rt/inspire/cmd"]
    assert "\U0001f6a8" in description, (
        f"description for rt/inspire/cmd is {description!r}; the neon "
        "catalog marks the Inspire hand cmd write topic with 🚨 too. "
        "Refs strands-labs/robots#358."
    )


def test_the_state_topics_do_not_carry_the_dangerous_publish_marker() -> None:
    """No read-side ``state`` topic description carries 🚨.

    The 🚨 marker names the write-side dangerous-publish partition
    only; a read-side topic with the marker would misrepresent the
    refusal surface to a caller scanning the catalog. This test
    pins the marker's meaning against the read partition.
    """
    state_topics = (
        "rt/lowstate",
        "rt/lf/lowstate",
        "rt/bmsstate",
        "rt/lf/bmsstate",
        "rt/mainboardstate",
        "rt/pressuresensorstate",
        "rt/lf/sportmodestate",
        "rt/lf/secondary_imu",
        "rt/multiplestate",
    )
    for topic in state_topics:
        description = _DDS_TOPIC_DESCRIPTIONS[topic]
        assert "\U0001f6a8" not in description, (
            f"description for read-side state topic {topic!r} is "
            f"{description!r}; the neon catalog marks only write-side "
            "topics with 🚨. Refs strands-labs/robots#358."
        )


def test_list_returns_every_topic_in_sorted_order() -> None:
    """The list verb returns descriptors sorted lexicographically by topic name.

    The order is fixed so a caller comparing two returned envelopes
    (before and after a widen, for instance) sees a stable diff
    rather than a permutation. Sorting is done at the verb boundary
    rather than in the snapshot so the underlying constant stays a
    ``dict`` (order-free at the shape level) and the verb-time order
    is a display concern.
    """
    payload = _call(g1_list_dds_topic_descriptions)
    assert payload["status"] == "success", payload
    topics = payload["topics"]
    assert topics == sorted(topics), (
        f"list verb returned topics {topics!r} out of sorted order; the "
        "verb's contract is a stable lexicographic sort so a caller sees a "
        "stable diff across widens. Refs strands-labs/robots#358."
    )
    descriptor_topics = [row["topic"] for row in payload["descriptions"]]
    assert descriptor_topics == topics, (
        f"descriptor list ordering {descriptor_topics!r} disagrees with "
        f"topics field ordering {topics!r}; both must present in the same "
        "sorted order. Refs strands-labs/robots#358."
    )


def test_list_names_every_snapshot_topic_and_no_others() -> None:
    """The list verb surfaces exactly the membership set.

    A drift here means the verb's returned envelope disagrees with
    the module-level constant it is supposed to snapshot; the whole
    point of the port is that the two agree by construction.
    """
    payload = _call(g1_list_dds_topic_descriptions)
    assert payload["count"] == len(_DDS_TOPIC_DESCRIPTIONS), (
        f"list verb count {payload['count']} disagrees with snapshot size "
        f"{len(_DDS_TOPIC_DESCRIPTIONS)}. Refs strands-labs/robots#358."
    )
    surfaced = set(payload["topics"])
    assert surfaced == set(_DDS_TOPIC_DESCRIPTIONS), (
        f"list verb topics {sorted(surfaced)} disagree with the snapshot "
        f"{sorted(_DDS_TOPIC_DESCRIPTIONS)}. A widen must update both "
        "together; refs strands-labs/robots#358."
    )


def test_list_surfaces_every_descriptor_with_the_snapshot_description() -> None:
    """Every descriptor carries the topic's snapshot description byte-for-byte.

    The ``description`` field on each returned descriptor must match
    the module's :data:`_DDS_TOPIC_DESCRIPTIONS` entry for that
    topic byte-for-byte; a re-wording that only landed on one side
    of the verb boundary would surface here.
    """
    payload = _call(g1_list_dds_topic_descriptions)
    for row in payload["descriptions"]:
        topic = row["topic"]
        expected = _DDS_TOPIC_DESCRIPTIONS[topic]
        assert row["description"] == expected, (
            f"descriptor row for topic {topic!r} carries description "
            f"{row['description']!r}; snapshot says {expected!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_returns_true_on_every_snapshot_topic() -> None:
    """Every topic key in the snapshot admits.

    Pins the round-trip: whatever the list verb returns as a valid
    topic, the admits verb agrees is a member of the catalog. A
    drift between the two would leave a caller unable to trust the
    two verbs' answers together.
    """
    for topic in sorted(_DDS_TOPIC_DESCRIPTIONS):
        payload = _call(g1_dds_topic_description_admits, topic=topic)
        assert payload["status"] == "success", payload
        assert payload["admitted"] is True, (
            f"admits verb refused {topic!r} but it is a member of the "
            f"catalog snapshot {sorted(_DDS_TOPIC_DESCRIPTIONS)}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["topic"] == topic, (
            f"admits verb target topic {payload['target']['topic']!r} "
            f"disagrees with the queried topic {topic!r}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["description"] == _DDS_TOPIC_DESCRIPTIONS[topic], (
            f"admits verb target description for {topic!r} is "
            f"{payload['target']['description']!r}; snapshot says "
            f"{_DDS_TOPIC_DESCRIPTIONS[topic]!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_refuses_an_off_catalog_topic_with_snapshot_note() -> None:
    """A topic outside the catalog returns ``admitted=False`` with the neon override note.

    A caller relying on the neon catalog's built-in decode resolves
    the miss decidably; the refusal names the neon-side override
    path (``type_module`` + ``type_class``) the generic
    ``g1_dds_snapshot`` verb accepts, so a caller reading the
    refusal knows how to reach the bus for an off-catalog topic
    without pretending the description snapshot carries it.
    """
    payload = _call(g1_dds_topic_description_admits, topic="rt/notacatalogtopic")
    assert payload["status"] == "success", payload
    assert payload["admitted"] is False, (
        "admits verb admitted an off-catalog topic; catalog snapshot "
        f"is {sorted(_DDS_TOPIC_DESCRIPTIONS)}. Refs "
        "strands-labs/robots#358."
    )
    assert "'rt/notacatalogtopic'" in payload["refusal_advice"], (
        f"admits verb refusal_advice {payload['refusal_advice']!r} must "
        "name the queried topic verbatim. Refs strands-labs/robots#358."
    )
    assert "type_module" in payload["refusal_advice"], (
        f"admits verb refusal_advice {payload['refusal_advice']!r} must "
        "name the type_module override path a caller uses to reach the "
        "bus for an off-catalog topic. Refs strands-labs/robots#358."
    )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """``bool`` is refused decidably, not resolved through ``str()`` coercion.

    ``True``/``False`` are not valid topic names under any DDS
    convention; a caller that passed a boolean is making a caller
    mistake, and the verb surfaces it as an error rather than
    resolving through Python's coercions.
    """
    for value in (True, False):
        payload = _call(g1_dds_topic_description_admits, topic=value)
        assert payload["status"] == "error", (
            f"admits verb accepted bool topic {value!r}; the argument must be str. Refs strands-labs/robots#358."
        )
        assert "bool" in payload["message"], (
            f"admits verb error message for bool topic {value!r} must "
            f"name the refused type; got {payload['message']!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_refuses_a_non_str_argument_as_a_shape_error() -> None:
    """Non-str inputs are refused decidably.

    ``int``, ``float``, ``None``, ``list``, ``tuple`` are all not
    valid topic names and the verb surfaces each as a shape error
    rather than resolving through Python's coercions.
    """
    for value in (1, 1.5, None, ["rt/lowstate"], ("rt/lowstate",)):
        payload = _call(g1_dds_topic_description_admits, topic=value)
        assert payload["status"] == "error", (
            f"admits verb accepted non-str topic {value!r}; the argument must be str. Refs strands-labs/robots#358."
        )
        assert type(value).__name__ in payload["message"], (
            f"admits verb error message for non-str topic {value!r} must "
            f"name the refused type; got {payload['message']!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_refuses_the_empty_string_as_a_shape_error() -> None:
    """The empty string is refused decidably.

    An empty topic name has no membership answer to compute; the
    verb refuses it as a shape error rather than returning
    ``admitted=False`` (which would let a caller silently pass the
    empty string through as an off-catalog topic).
    """
    payload = _call(g1_dds_topic_description_admits, topic="")
    assert payload["status"] == "error", (
        "admits verb accepted an empty topic string; the argument must be non-empty. Refs strands-labs/robots#358."
    )
    assert "non-empty" in payload["message"], (
        f"admits verb error message for the empty string must name the "
        f"non-empty contract; got {payload['message']!r}. Refs "
        "strands-labs/robots#358."
    )
