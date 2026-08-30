"""The dangerous-publish-topics envelope tools name exactly what the neon publish gate refuses.

The neon bundle's ``_dds_engine.py`` (``cagataycali/neon-the-g1/tools/_dds_engine.py``)
observes a fixed five-topic ``DANGEROUS_PUB_TOPICS`` set and its
generic ``g1_dds_publish`` verb refuses membership on it unless the
caller passes ``unsafe=True``. The
:mod:`strands_robots.tools.g1.g1_dangerous_publish_topics` module
snapshots that set into module-level constants and exposes two
agent-facing verbs -
:func:`g1_list_dangerous_publish_topics` (list the whole envelope) and
:func:`g1_dangerous_publish_topic_admits` (decide one query) - so a
caller can decide the caller-side refusal decidably before a future
driver-side publish wrapper fires. The tests here fix that contract
without pulling the SDK: the module is loadable on a host without
``unitree_sdk2py`` (the same SDK-load-hygiene rule every other file
under :mod:`strands_robots.tools.g1` carries, refs
strands-labs/robots#358), and every membership answer is read off the
module's own snapshot rather than restated in the tests, so a widen
or narrow to the constant surfaces here as a shape change rather than
as a diverging table this file would need to manually update.

Two things this file's cells deliberately do not pin:

* The neon bundle's own answer at wire time. The verbs answer against
  the module-level snapshot, not against a live import of the neon
  bundle's ``DANGEROUS_PUB_TOPICS`` set (the whole point of the port
  is that the snapshot lets a headless host answer without pulling
  the neon module or the CycloneDDS bindings). A driver-side wrapper
  for a generic DDS publish that lands later will re-validate against
  its own live constant at wire time; testing the snapshot vs the
  live table is a driver-side test, not a lookup-side one.
* Whether a topic outside the set is safe to publish. The verb only
  answers the narrower membership question the neon bundle's
  ``DANGEROUS_PUB_TOPICS`` set answers: is this topic on the observed
  known-dangerous list? A topic outside the set is *not*
  automatically safe (an unknown topic may still command the robot
  if the caller supplies a matching IDL type); a future test that
  pinned safe-topic membership would cross a scope line this file
  keeps.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_dangerous_publish_topics import (
    _DANGEROUS_PUBLISH_TOPICS,
    _REFUSAL_ADVICE,
    _TOPIC_DESCRIPTIONS,
    g1_dangerous_publish_topic_admits,
    g1_list_dangerous_publish_topics,
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
    dangerous-publish-topics envelope verbs to it too (refs
    strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_dangerous_publish_topics")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_dangerous_publish_topics imports pulled "
        f"SDK submodules: {leaked}. The rule for this package is that the "
        "SDK loads only inside function bodies (refs strands-labs/robots#358)."
    )


def test_the_snapshot_covers_the_neon_observed_set() -> None:
    """The snapshot names every topic the neon bundle marks as dangerous today.

    The neon bundle's ``_dds_engine.DANGEROUS_PUB_TOPICS`` has 5
    entries observed against the real robot (``rt/lowcmd``,
    ``rt/armsdk``, ``rt/user_lowcmd``, ``rt/inspire/cmd``,
    ``rt/bmscmd``). A drift on either side surfaces here: a
    driver-side generic-publish wrapper (when it lands) will
    validate the same set at wire time and its refusal string will
    quote the same membership test. The count is pinned rather than
    listed value-by-value so a caller widening the map on the neon
    side updates one number here rather than 5 assertions.
    """
    assert len(_DANGEROUS_PUBLISH_TOPICS) == 5, (
        f"expected 5 known-dangerous publish topics in the neon-observed "
        f"snapshot, got {len(_DANGEROUS_PUBLISH_TOPICS)}: "
        f"{sorted(_DANGEROUS_PUBLISH_TOPICS)}. A neon-side widen or narrow "
        "would update this count; refs strands-labs/robots#358."
    )


def test_every_snapshot_entry_carries_a_description() -> None:
    """Every dangerous topic has a matching description label.

    :data:`_TOPIC_DESCRIPTIONS` is the label table both verbs read
    when they build the returned descriptor. A drift between the
    membership set and the description table would surface here: a
    topic in :data:`_DANGEROUS_PUBLISH_TOPICS` without a matching
    key in :data:`_TOPIC_DESCRIPTIONS` would raise ``KeyError`` on
    the ``_describe`` call at :func:`g1_list_dangerous_publish_topics`
    time, and a description key not in the membership set would be
    silently unreachable. Pinning both directions here catches the
    drift before wire.
    """
    assert set(_TOPIC_DESCRIPTIONS) == set(_DANGEROUS_PUBLISH_TOPICS), (
        f"description-table keys {sorted(_TOPIC_DESCRIPTIONS)} do not match "
        f"the membership set {sorted(_DANGEROUS_PUBLISH_TOPICS)}. A neon-side "
        "widen must update both together; refs strands-labs/robots#358."
    )
    for topic, description in _TOPIC_DESCRIPTIONS.items():
        assert isinstance(description, str) and description != "", (
            f"description for topic {topic!r} must be a non-empty str; got "
            f"{description!r}. Refs strands-labs/robots#358."
        )


def test_the_rt_lowcmd_entry_matches_the_drivers_own_write_topic() -> None:
    """The ``rt/lowcmd`` string matches the driver's own write topic verbatim.

    :mod:`~strands_robots.tools.g1.g1_dds_topics` names the driver's
    own write side as ``rt/lowcmd``; the invariant a future widen
    must preserve is byte-for-byte identity between the two files'
    topic strings, so a driver-side rename lands on both surfaces
    or the ``rt/lowcmd`` refusal on one side and the same-named
    topic on the other silently diverge. Pinning it here catches
    the drift before wire.
    """
    assert "rt/lowcmd" in _DANGEROUS_PUBLISH_TOPICS, (
        "rt/lowcmd is the driver's own write topic; it must be a member of "
        "the dangerous-publish set. Refs strands-labs/robots#358."
    )


def test_the_refusal_advice_names_the_unsafe_override_argument() -> None:
    """The refusal advice mentions the ``unsafe=True`` override the caller-side verb takes.

    The neon bundle's ``g1_dds_publish`` verb takes an ``unsafe``
    argument the caller passes to override the danger refusal; the
    surfaced advice string names that argument verbatim so a caller
    reading the refusal sees the same knob the caller-side verb
    exposes. A re-wording that dropped the argument name would leave
    the caller reading the refusal without seeing how to override.
    """
    assert "unsafe=True" in _REFUSAL_ADVICE, (
        f"refusal advice must name the unsafe=True override argument "
        f"verbatim; got {_REFUSAL_ADVICE!r}. Refs strands-labs/robots#358."
    )


def test_list_returns_every_topic_in_sorted_order() -> None:
    """The list verb returns descriptors sorted lexicographically by topic string.

    The order is fixed so a caller comparing two returned envelopes
    (before and after a widen, for instance) sees a stable diff
    rather than a permutation. Sorting is done at the verb boundary
    rather than in the snapshot so the underlying constant stays a
    ``frozenset`` (order-free) and the verb-time order is a display
    concern.
    """
    payload = _call(g1_list_dangerous_publish_topics)
    assert payload["status"] == "success", payload
    topics = payload["topics"]
    assert topics == sorted(topics), (
        f"list verb returned topics {topics!r} out of sorted order; the "
        "verb's contract is a stable lexicographic sort so a caller sees a "
        "stable diff across widens. Refs strands-labs/robots#358."
    )
    descriptor_topics = [row["topic"] for row in payload["dangerous_topics"]]
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
    payload = _call(g1_list_dangerous_publish_topics)
    assert payload["count"] == len(_DANGEROUS_PUBLISH_TOPICS), (
        f"list verb count {payload['count']} disagrees with snapshot size "
        f"{len(_DANGEROUS_PUBLISH_TOPICS)}. Refs strands-labs/robots#358."
    )
    surfaced = set(payload["topics"])
    assert surfaced == set(_DANGEROUS_PUBLISH_TOPICS), (
        f"list verb topics {sorted(surfaced)} disagree with the snapshot "
        f"{sorted(_DANGEROUS_PUBLISH_TOPICS)}. A widen must update both "
        "together; refs strands-labs/robots#358."
    )


def test_list_surfaces_every_descriptor_with_dangerous_flag_true() -> None:
    """Every descriptor carries ``dangerous_to_publish=True``.

    The flag is always ``True`` on the returned envelope (membership
    *is* the danger contract); the verb surfaces it anyway so the
    returned shape mirrors the neon bundle's own ``list_topics``
    payload, which carries the same field on every row it returns.
    A widen that added a False-marked row here would be a shape
    change; the test catches the drift.
    """
    payload = _call(g1_list_dangerous_publish_topics)
    for row in payload["dangerous_topics"]:
        assert row["dangerous_to_publish"] is True, (
            f"descriptor row {row!r} must carry dangerous_to_publish=True; "
            "membership in the set is the danger contract. Refs "
            "strands-labs/robots#358."
        )


def test_list_surfaces_the_refusal_advice() -> None:
    """The list verb's envelope carries the refusal-advice string.

    A caller reading the envelope in one call sees both the
    membership set and the refusal a publish path would surface on
    it; a caller that only asked for the list would still see the
    advice string without a second call.
    """
    payload = _call(g1_list_dangerous_publish_topics)
    assert payload["refusal_advice"] == _REFUSAL_ADVICE, (
        f"list verb refusal_advice {payload['refusal_advice']!r} disagrees "
        f"with the module constant {_REFUSAL_ADVICE!r}. Refs "
        "strands-labs/robots#358."
    )


def test_admits_returns_true_on_every_snapshot_topic() -> None:
    """Every topic in the snapshot admits.

    Pins the round-trip: whatever the list verb returns as a
    dangerous topic, the admits verb agrees is a member of the
    dangerous set. A drift between the two would leave a caller
    unable to trust the two verbs' answers together.
    """
    for topic in sorted(_DANGEROUS_PUBLISH_TOPICS):
        payload = _call(g1_dangerous_publish_topic_admits, topic=topic)
        assert payload["status"] == "success", payload
        assert payload["admitted"] is True, (
            f"admits verb refused {topic!r} but it is a member of the "
            f"dangerous-publish snapshot {sorted(_DANGEROUS_PUBLISH_TOPICS)}. "
            "Refs strands-labs/robots#358."
        )
        assert payload["target"]["topic"] == topic, (
            f"admits verb target topic {payload['target']['topic']!r} "
            f"disagrees with the queried topic {topic!r}. Refs "
            "strands-labs/robots#358."
        )
        assert payload["target"]["dangerous_to_publish"] is True, (
            f"admits verb target for {topic!r} must carry dangerous_to_publish=True; refs strands-labs/robots#358."
        )


def test_admits_refuses_an_off_set_topic_with_a_cross_check_note() -> None:
    """A topic outside the set returns ``admitted=False`` with the caller-side advice.

    The neon bundle's own docstring notes that a topic outside
    ``DANGEROUS_PUB_TOPICS`` is not automatically safe (an unknown
    topic may still command the robot if the caller supplies a
    matching IDL type). The verb's off-set path surfaces the same
    note so a caller reading the refusal does not conclude an
    off-set topic is safe.
    """
    payload = _call(g1_dangerous_publish_topic_admits, topic="rt/lowstate")
    assert payload["status"] == "success", payload
    assert payload["admitted"] is False, (
        f"admits verb admitted rt/lowstate but it is not on the "
        f"dangerous-publish snapshot {sorted(_DANGEROUS_PUBLISH_TOPICS)}. "
        "Refs strands-labs/robots#358."
    )
    assert "rt/lowstate" in payload["refusal_advice"], (
        f"admits verb refusal_advice {payload['refusal_advice']!r} must "
        "name the queried topic verbatim. Refs strands-labs/robots#358."
    )
    assert "IDL knowledge" in payload["refusal_advice"], (
        f"admits verb refusal_advice {payload['refusal_advice']!r} must "
        "warn the caller that off-set topics are not automatically safe. "
        "Refs strands-labs/robots#358."
    )


def test_admits_refuses_a_bool_argument_as_a_shape_error() -> None:
    """``bool`` is refused decidably, not resolved through ``str()`` coercion.

    ``True``/``False`` are not valid topic strings under any DDS
    convention; a caller that passed a boolean is making a caller
    mistake, and the verb surfaces it as an error rather than
    resolving through Python's coercions.
    """
    for value in (True, False):
        payload = _call(g1_dangerous_publish_topic_admits, topic=value)
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
    valid topic strings and the verb surfaces each as a shape
    error rather than resolving through Python's coercions.
    """
    for value in (1, 1.5, None, ["rt/lowcmd"], ("rt/lowcmd",)):
        payload = _call(g1_dangerous_publish_topic_admits, topic=value)
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
    empty string through as an off-set topic).
    """
    payload = _call(g1_dangerous_publish_topic_admits, topic="")
    assert payload["status"] == "error", (
        "admits verb accepted an empty topic string; the argument must be non-empty. Refs strands-labs/robots#358."
    )
    assert "non-empty" in payload["message"], (
        f"admits verb error message for the empty string must name the "
        f"non-empty contract; got {payload['message']!r}. Refs "
        "strands-labs/robots#358."
    )
