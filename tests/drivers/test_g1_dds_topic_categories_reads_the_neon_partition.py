"""The DDS topic-category envelope tools name exactly what the neon partition ships.

The neon bundle's ``_dds_engine.py`` (``cagataycali/neon-the-g1/tools/_dds_engine.py``)
carries a ``TOPIC_CATALOG`` dict whose per-topic value tuple ends in a
category label; the neon generic ``g1_dds_list_topics`` verb accepts
that label as a filter argument and returns the per-topic descriptors
partitioned by category. The
:mod:`strands_robots.tools.g1.g1_dds_topic_categories` module snapshots
the seven category labels into module-level constants and exposes two
agent-facing verbs -
:func:`g1_list_dds_topic_categories` (list the whole envelope) and
:func:`g1_dds_topic_category_admits` (decide one membership query) -
so a caller can name the category-scoped filter decidably before a
future driver-side wrapper for the neon ``g1_dds_list_topics`` verb
lands. The tests here fix that contract without pulling the SDK: the
module is loadable on a host without ``unitree_sdk2py`` (the same
SDK-load-hygiene rule every other file under
:mod:`strands_robots.tools.g1` carries, refs strands-labs/robots#358),
and every membership answer is read off the module's own snapshot
rather than restated in the tests, so a widen or narrow to the
constant surfaces here as a shape change rather than as a diverging
table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The neon bundle's own answer at wire time. The verbs answer against
  the module-level snapshot, not against a live import of the neon
  bundle's ``TOPIC_CATALOG`` value tuples (the whole point of the port
  is that the snapshot lets a headless host answer without pulling
  the neon module or the CycloneDDS bindings). A driver-side wrapper
  for the neon catalog read that lands later will re-validate against
  its own live table at wire time; testing the snapshot vs the live
  table is a driver-side test, not a lookup-side one.
* Per-topic membership by category. The verb answers the narrower
  question the neon partition answers: which category *names* the
  catalog partitions today. Which topics land in which category at
  any given firmware release is answered by
  :mod:`~strands_robots.tools.g1.g1_dds_topics` (the driver's own
  seven-topic subscription set, whose descriptors carry the same
  category labels) and by a future driver-side wrapper for the full
  22-topic neon catalog. A future test that pinned per-topic
  category membership would cross a scope line this file keeps.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_dds_topic_categories import (
    _DDS_TOPIC_CATEGORIES,
    _DDS_TOPIC_CATEGORY_COUNTS,
    _DDS_TOPIC_CATEGORY_DESCRIPTIONS,
    g1_dds_topic_category_admits,
    g1_list_dds_topic_categories,
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
    dds-topic-categories envelope verbs to it too (refs
    strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_dds_topic_categories")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_dds_topic_categories imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_partition() -> None:
    """The snapshot names every category label the neon catalog partitions today.

    The neon bundle's ``_dds_engine.TOPIC_CATALOG`` uses seven category
    labels (``state``, ``lidar``, ``joystick``, ``control``, ``hand``,
    ``slam``, ``config``). A drift on either side surfaces here: a
    driver-side generic-list wrapper (when it lands) will validate
    the same partition at wire time and its refusal string will quote
    the same membership test. The count is pinned rather than listed
    value-by-value so a caller widening the partition on the neon
    side updates one number here rather than 7 assertions.
    """
    assert len(_DDS_TOPIC_CATEGORIES) == 7, (
        f"expected 7 category labels in the neon-observed snapshot, got "
        f"{len(_DDS_TOPIC_CATEGORIES)}: {sorted(_DDS_TOPIC_CATEGORIES)}. A "
        "neon-side widen or narrow would update this count; refs "
        "strands-labs/robots#358."
    )


def test_every_snapshot_label_carries_a_description() -> None:
    """Every category label has a matching description string.

    :data:`_DDS_TOPIC_CATEGORY_DESCRIPTIONS` is the label table both
    verbs read when they build the returned descriptor. A drift
    between the membership set and the description table would
    surface here: a label in :data:`_DDS_TOPIC_CATEGORIES` without a
    matching key in :data:`_DDS_TOPIC_CATEGORY_DESCRIPTIONS` would
    raise ``KeyError`` on the ``_describe`` call at
    :func:`g1_list_dds_topic_categories` time, and a description
    key not in the membership set would be silently unreachable.
    Pinning both directions here catches the drift before wire.
    """
    assert set(_DDS_TOPIC_CATEGORY_DESCRIPTIONS) == set(_DDS_TOPIC_CATEGORIES), (
        f"description-table keys {sorted(_DDS_TOPIC_CATEGORY_DESCRIPTIONS)} "
        f"do not match the membership set {sorted(_DDS_TOPIC_CATEGORIES)}. "
        "A neon-side widen must update both together; refs "
        "strands-labs/robots#358."
    )
    for name, description in _DDS_TOPIC_CATEGORY_DESCRIPTIONS.items():
        assert isinstance(description, str) and description != "", (
            f"description for category {name!r} must be a non-empty str; "
            f"got {description!r}. Refs strands-labs/robots#358."
        )


def test_every_snapshot_label_carries_a_topic_count() -> None:
    """Every category label has a matching topic-count entry.

    :data:`_DDS_TOPIC_CATEGORY_COUNTS` snapshots how many neon-catalog
    topics carry each label; a drift between the membership set and
    the count table would surface here (a label without a count
    would raise ``KeyError`` at ``_describe`` time, a count without
    a matching label would be silently unreachable). The count
    totals to 22 - the size of the neon ``TOPIC_CATALOG`` today, so
    the sum is pinned as a shape invariant rather than as a
    per-category assertion.
    """
    assert set(_DDS_TOPIC_CATEGORY_COUNTS) == set(_DDS_TOPIC_CATEGORIES), (
        f"count-table keys {sorted(_DDS_TOPIC_CATEGORY_COUNTS)} do not "
        f"match the membership set {sorted(_DDS_TOPIC_CATEGORIES)}. A "
        "neon-side widen must update both together; refs "
        "strands-labs/robots#358."
    )
    for name, count in _DDS_TOPIC_CATEGORY_COUNTS.items():
        assert isinstance(count, int) and count >= 1, (
            f"topic count for category {name!r} must be a positive int; "
            f"got {count!r}. Every category the neon partition ships "
            "carries at least one topic. Refs strands-labs/robots#358."
        )
    assert sum(_DDS_TOPIC_CATEGORY_COUNTS.values()) == 22, (
        f"topic-count totals {sum(_DDS_TOPIC_CATEGORY_COUNTS.values())} "
        f"disagree with the neon TOPIC_CATALOG size of 22; per-category "
        f"counts: {dict(sorted(_DDS_TOPIC_CATEGORY_COUNTS.items()))}. A "
        "neon-side widen must update the counts together with the "
        "TOPIC_CATALOG; refs strands-labs/robots#358."
    )


def test_the_control_category_is_the_dangerous_publish_partition() -> None:
    """The ``control`` label is a member of the partition and its size matches.

    The neon bundle's ``DANGEROUS_PUB_TOPICS`` set is five entries but
    the ``TOPIC_CATALOG``'s ``control`` category is four - the ``hand``
    category carries the fifth entry (``rt/inspire/cmd``, the Inspire
    hand cmd) rather than ``control``. This test pins that
    interpretation so the count invariant between the two lookups
    does not silently collapse on a future widen (a caller that
    lifted the ``control`` count from the neon partition, expecting
    it to match the dangerous-publish set, would surface the drift
    here). Refs
    :mod:`~strands_robots.tools.g1.g1_dangerous_publish_topics` for
    the sibling snapshot.
    """
    assert "control" in _DDS_TOPIC_CATEGORIES, (
        "control is a neon TOPIC_CATALOG category label; it must be a "
        "member of the partition. Refs strands-labs/robots#358."
    )
    assert _DDS_TOPIC_CATEGORY_COUNTS["control"] == 4, (
        f"control category topic count {_DDS_TOPIC_CATEGORY_COUNTS['control']} "
        "disagrees with the neon TOPIC_CATALOG's 4-topic control partition "
        "(rt/lowcmd, rt/armsdk, rt/user_lowcmd, rt/bmscmd). The Inspire hand "
        "cmd is in the hand category, not control; refs "
        "strands-labs/robots#358."
    )


def test_list_returns_every_category_in_sorted_order() -> None:
    """The list verb returns descriptors sorted lexicographically by name.

    The order is fixed so a caller comparing two returned envelopes
    (before and after a widen, for instance) sees a stable diff
    rather than a permutation. Sorting is done at the verb boundary
    rather than in the snapshot so the underlying constant stays a
    ``frozenset`` (order-free) and the verb-time order is a display
    concern.
    """
    payload = _call(g1_list_dds_topic_categories)
    assert payload["status"] == "success", payload
    names = payload["names"]
    assert names == sorted(names), (
        f"list verb returned names {names!r} out of sorted order; the "
        "verb's contract is a stable lexicographic sort so a caller sees a "
        "stable diff across widens. Refs strands-labs/robots#358."
    )
    descriptor_names = [row["name"] for row in payload["categories"]]
    assert descriptor_names == names, (
        f"descriptor list ordering {descriptor_names!r} disagrees with "
        f"names field ordering {names!r}; both must present in the same "
        "sorted order. Refs strands-labs/robots#358."
    )


def test_list_names_every_snapshot_category_and_no_others() -> None:
    """The list verb surfaces exactly the membership set.

    A drift here means the verb's returned envelope disagrees with
    the module-level constant it is supposed to snapshot; the whole
    point of the port is that the two agree by construction.
    """
    payload = _call(g1_list_dds_topic_categories)
    assert payload["count"] == len(_DDS_TOPIC_CATEGORIES), (
        f"list verb count {payload['count']} disagrees with snapshot size "
        f"{len(_DDS_TOPIC_CATEGORIES)}. Refs strands-labs/robots#358."
    )
    surfaced = set(payload["names"])
    assert surfaced == set(_DDS_TOPIC_CATEGORIES), (
        f"list verb names {sorted(surfaced)} disagree with the snapshot "
        f"{sorted(_DDS_TOPIC_CATEGORIES)}. A widen must update both "
        "together; refs strands-labs/robots#358."
    )


def test_list_surfaces_every_descriptor_with_the_snapshot_topic_count() -> None:
    """Every descriptor carries the category's snapshot topic count.

    The ``topic_count`` field on each returned descriptor must match
    the module's :data:`_DDS_TOPIC_CATEGORY_COUNTS` entry for that
    label; a widen that updated the count table without also updating
    the verb boundary would surface here.
    """
    payload = _call(g1_list_dds_topic_categories)
    for row in payload["categories"]:
        name = row["name"]
        expected = _DDS_TOPIC_CATEGORY_COUNTS[name]
        assert row["topic_count"] == expected, (
            f"descriptor row for category {name!r} carries topic_count="
            f"{row['topic_count']}; snapshot says {expected}. Refs "
            "strands-labs/robots#358."
        )


def test_list_surfaces_every_descriptor_with_the_snapshot_description() -> None:
    """Every descriptor carries the category's snapshot description.

    The ``description`` field on each returned descriptor must match
    the module's :data:`_DDS_TOPIC_CATEGORY_DESCRIPTIONS` entry for
    that label byte-for-byte; a re-wording that only landed on one
    side of the verb boundary would surface here.
    """
    payload = _call(g1_list_dds_topic_categories)
    for row in payload["categories"]:
        name = row["name"]
        expected = _DDS_TOPIC_CATEGORY_DESCRIPTIONS[name]
        assert row["description"] == expected, (
            f"descriptor row for category {name!r} carries description "
            f"{row['description']!r}; snapshot says {expected!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_returns_true_on_every_snapshot_category() -> None:
    """Every label in the snapshot admits.

    Pins the round-trip: whatever the list verb returns as a valid
    category, the admits verb agrees is a member of the partition.
    A drift between the two would leave a caller unable to trust the
    two verbs' answers together.
    """
    for name in sorted(_DDS_TOPIC_CATEGORIES):
        payload = _call(g1_dds_topic_category_admits, name=name)
        assert payload["status"] == "success", payload
        assert payload["admitted"] is True, (
            f"admits verb refused {name!r} but it is a member of the "
            f"category snapshot {sorted(_DDS_TOPIC_CATEGORIES)}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["name"] == name, (
            f"admits verb target name {payload['target']['name']!r} "
            f"disagrees with the queried name {name!r}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["topic_count"] == _DDS_TOPIC_CATEGORY_COUNTS[name], (
            f"admits verb target for {name!r} carries topic_count="
            f"{payload['target']['topic_count']}; snapshot says "
            f"{_DDS_TOPIC_CATEGORY_COUNTS[name]}. Refs strands-labs/robots#358."
        )


def test_admits_refuses_an_off_partition_label_with_the_valid_labels_listed() -> None:
    """A label outside the partition returns ``admitted=False`` with the valid set listed.

    The neon ``g1_dds_list_topics`` filter's answer on an unknown
    category label is the empty topic list, which is
    indistinguishable from a category that happens to be empty at
    runtime; the verb's off-partition path surfaces the caller-side
    note so a caller reading the refusal does not silently pass an
    unknown label through. The refusal also lists the valid labels
    so the caller can resolve the drift without a follow-up call.
    """
    payload = _call(g1_dds_topic_category_admits, name="motion")
    assert payload["status"] == "success", payload
    assert payload["admitted"] is False, (
        "admits verb admitted 'motion' but it is not on the category "
        f"snapshot {sorted(_DDS_TOPIC_CATEGORIES)}. Refs strands-labs/robots#358."
    )
    assert "'motion'" in payload["refusal_advice"], (
        f"admits verb refusal_advice {payload['refusal_advice']!r} must "
        "name the queried label verbatim. Refs strands-labs/robots#358."
    )
    for known in sorted(_DDS_TOPIC_CATEGORIES):
        assert repr(known) in payload["refusal_advice"], (
            f"admits verb refusal_advice {payload['refusal_advice']!r} must "
            f"list every known category label; {known!r} is missing. Refs "
            "strands-labs/robots#358."
        )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """``bool`` is refused decidably, not resolved through ``str()`` coercion.

    ``True``/``False`` are not valid category labels under any DDS
    convention; a caller that passed a boolean is making a caller
    mistake, and the verb surfaces it as an error rather than
    resolving through Python's coercions.
    """
    for value in (True, False):
        payload = _call(g1_dds_topic_category_admits, name=value)
        assert payload["status"] == "error", (
            f"admits verb accepted bool name {value!r}; the argument must be str. Refs strands-labs/robots#358."
        )
        assert "bool" in payload["message"], (
            f"admits verb error message for bool name {value!r} must "
            f"name the refused type; got {payload['message']!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_refuses_a_non_str_argument_as_a_shape_error() -> None:
    """Non-str inputs are refused decidably.

    ``int``, ``float``, ``None``, ``list``, ``tuple`` are all not
    valid category labels and the verb surfaces each as a shape
    error rather than resolving through Python's coercions.
    """
    for value in (1, 1.5, None, ["state"], ("state",)):
        payload = _call(g1_dds_topic_category_admits, name=value)
        assert payload["status"] == "error", (
            f"admits verb accepted non-str name {value!r}; the argument must be str. Refs strands-labs/robots#358."
        )
        assert type(value).__name__ in payload["message"], (
            f"admits verb error message for non-str name {value!r} must "
            f"name the refused type; got {payload['message']!r}. Refs "
            "strands-labs/robots#358."
        )


def test_admits_refuses_the_empty_string_as_a_shape_error() -> None:
    """The empty string is refused decidably.

    An empty category label has no membership answer to compute; the
    verb refuses it as a shape error rather than returning
    ``admitted=False`` (which would let a caller silently pass the
    empty string through as an off-partition label).
    """
    payload = _call(g1_dds_topic_category_admits, name="")
    assert payload["status"] == "error", (
        "admits verb accepted an empty label string; the argument must be non-empty. Refs strands-labs/robots#358."
    )
    assert "non-empty" in payload["message"], (
        f"admits verb error message for the empty string must name the "
        f"non-empty contract; got {payload['message']!r}. Refs "
        "strands-labs/robots#358."
    )
