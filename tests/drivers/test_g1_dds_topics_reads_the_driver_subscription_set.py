"""The DDS-topic tools name exactly the topics ``G1Driver`` opens on the bus.

``G1Driver`` holds its subscription set as seven module-level ``_TOPIC_*``
constants inside :mod:`strands_robots.drivers.g1`: six ``"read"``-side
topics the driver subscribes at ``connect_eagerly`` time (``rt/lowstate``,
``rt/lf/bmsstate``, ``rt/utlidar/lidar_state``,
``rt/utlidar/cloud_livox_mid360``, ``rt/mainboardstate``,
``rt/pressuresensorstate``), and one ``"write"``-side topic the driver
publishes on (``rt/lowcmd``). Two agent-facing tools -
:func:`g1_list_dds_topics` and :func:`g1_topic_role` - exist to surface
those seven topic names as a static snapshot before a caller decides
whether to open its own reader/publisher against the same wire.

The topic names the tool module carries are read here off the driver's
own constants rather than being restated in the tests, so a driver-side
widen or narrow of the subscription set (adding a new sensor topic, for
example) does not require also editing this file. What the tests do
restate is the *shape* of each returned record and the SDK-load-hygiene
contract every file under :mod:`strands_robots.tools.g1` carries:
importing the tool module must not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_dds_topics import (
    _DRIVER_TOPICS,
    _VALID_DIRECTIONS,
    g1_list_dds_topics,
    g1_topic_role,
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


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import time
    would break every headless CI runner and Thor before an office
    bring-up. The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the DDS-topic verbs to it
    too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_dds_topics")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_dds_topics imports pulled SDK "
        f"submodules: {leaked}. The rule for this package is that the "
        f"SDK loads only inside function bodies "
        f"(refs strands-labs/robots#358)."
    )


def test_the_snapshot_matches_the_driver_constant_set() -> None:
    """The tool's topic snapshot is the driver's own ``_TOPIC_*`` set.

    The driver holds seven module-level constants naming the topics it
    opens on the bus. The tool module's :data:`_DRIVER_TOPICS` names the
    same seven strings; a widen or narrow on the driver side surfaces
    here as a diff between the two sets, not as a diverging table the
    author would need to keep in sync by hand.
    """
    from strands_robots.drivers import g1 as driver_module

    driver_read_topics = {
        driver_module._TOPIC_LOWSTATE,
        driver_module._TOPIC_BMS,
        driver_module._TOPIC_LIDAR_STATE,
        driver_module._TOPIC_LIDAR_CLOUD,
        driver_module._TOPIC_MAINBOARD,
        driver_module._TOPIC_PRESSURE,
    }
    driver_write_topics = {driver_module._TOPIC_LOWCMD}

    tool_read_topics = {entry["topic"] for entry in _DRIVER_TOPICS if entry["direction"] == "read"}
    tool_write_topics = {entry["topic"] for entry in _DRIVER_TOPICS if entry["direction"] == "write"}

    assert tool_read_topics == driver_read_topics, (
        f"The tool module's read-side snapshot drifted from the driver's "
        f"``_TOPIC_*`` constants. Tool read topics: {sorted(tool_read_topics)}; "
        f"driver read topics: {sorted(driver_read_topics)}. Update "
        f"``_DRIVER_TOPICS`` in ``strands_robots.tools.g1.g1_dds_topics`` "
        f"to match, refs strands-labs/robots#358."
    )
    assert tool_write_topics == driver_write_topics, (
        f"The tool module's write-side snapshot drifted from the driver's "
        f"``_TOPIC_LOWCMD`` constant. Tool write topics: "
        f"{sorted(tool_write_topics)}; driver write topics: "
        f"{sorted(driver_write_topics)}. Update ``_DRIVER_TOPICS`` in "
        f"``strands_robots.tools.g1.g1_dds_topics`` to match, refs "
        f"strands-labs/robots#358."
    )


def test_the_snapshot_carries_the_documented_roles() -> None:
    """Each snapshot entry names a ``role`` matching the driver's cache.

    The driver decodes each subscribed topic into a named cache
    attribute; the snapshot's ``role`` field carries the same names as
    the driver's decode targets. A caller reading the snapshot sees the
    same label the driver's own callback wiring would emit.
    """
    expected_roles = {
        "rt/lowstate": "lowstate",
        "rt/lf/bmsstate": "battery",
        "rt/utlidar/lidar_state": "lidar_state",
        "rt/utlidar/cloud_livox_mid360": "lidar_cloud",
        "rt/mainboardstate": "mainboard",
        "rt/pressuresensorstate": "pressure",
        "rt/lowcmd": "lowcmd",
    }
    got_roles = {entry["topic"]: entry["role"] for entry in _DRIVER_TOPICS}
    assert got_roles == expected_roles, (
        f"role labels drifted from the driver's decode targets. "
        f"Got: {got_roles}; expected: {expected_roles}. "
        f"Refs strands-labs/robots#358."
    )


def test_valid_directions_are_read_and_write() -> None:
    """The tool admits exactly the two DDS wire directions.

    The driver subscribes and publishes; nothing else is a topic-side
    action. The tool refuses any other filter string with a refusal that
    quotes this same domain.
    """
    assert _VALID_DIRECTIONS == frozenset({"read", "write"}), (
        f"the valid-direction set drifted. Got {_VALID_DIRECTIONS!r}; "
        f"expected frozenset({{'read', 'write'}}). Refs strands-labs/robots#358."
    )


def test_g1_list_dds_topics_returns_the_whole_table() -> None:
    """No filter returns every topic the driver opens, sorted by insertion.

    The shape mirrors :func:`g1_list_motion_gates` and
    :func:`g1_list_balance_modes`: ``status``, ``count``, ``direction``
    (``None`` here), ``directions``, and a ``topics`` list.
    """
    result = _call(g1_list_dds_topics)
    assert result["status"] == "success"
    assert result["count"] == len(_DRIVER_TOPICS)
    assert result["direction"] is None
    assert result["directions"] == sorted(_VALID_DIRECTIONS)
    assert len(result["topics"]) == len(_DRIVER_TOPICS)
    got = [entry["topic"] for entry in result["topics"]]
    expected = [entry["topic"] for entry in _DRIVER_TOPICS]
    assert got == expected, f"the topic order drifted. Got {got}; expected {expected}. Refs strands-labs/robots#358."


def test_g1_list_dds_topics_filters_read_side() -> None:
    """A ``"read"`` filter surfaces only the driver's six subscribes.

    The driver's write-side ``rt/lowcmd`` is excluded from the filtered
    view; a caller planning a subscribe uses this filter to enumerate
    the read set the driver already carries.
    """
    result = _call(g1_list_dds_topics, direction="read")
    assert result["status"] == "success"
    assert result["direction"] == "read"
    assert result["count"] == 6
    directions_seen = {entry["direction"] for entry in result["topics"]}
    assert directions_seen == {"read"}, (
        f"``read`` filter admitted a non-read entry: {directions_seen}. Refs strands-labs/robots#358."
    )


def test_g1_list_dds_topics_filters_write_side() -> None:
    """A ``"write"`` filter surfaces only ``rt/lowcmd``, the driver's write.

    The G1 write path is a single-topic path (see the driver's own
    module-level docstring on ``_TOPIC_LOWCMD``); the filtered view
    reflects that.
    """
    result = _call(g1_list_dds_topics, direction="write")
    assert result["status"] == "success"
    assert result["direction"] == "write"
    assert result["count"] == 1
    assert result["topics"][0]["topic"] == "rt/lowcmd"
    assert result["topics"][0]["direction"] == "write"
    assert result["topics"][0]["role"] == "lowcmd"


def test_g1_list_dds_topics_refuses_an_unknown_direction() -> None:
    """An unknown filter is refused with the valid set in the message.

    A caller that mis-types ``"reader"`` or ``"subscribe"`` sees a
    refusal that quotes the valid domain and the ``#358`` reference,
    rather than silently returning an empty list that would read like a
    driver with no subscriptions.
    """
    result = _call(g1_list_dds_topics, direction="reader")
    assert result["status"] == "error"
    assert "reader" in result["message"]
    assert "read" in result["message"]
    assert "write" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_list_dds_topics_returns_fresh_containers() -> None:
    """Successive calls do not share a mutable container.

    A caller that mutates the returned ``topics`` list must not affect
    the module-level snapshot or a subsequent caller's read. The tool
    copies each entry with :func:`dict` and yields a fresh list.
    """
    first = _call(g1_list_dds_topics)
    second = _call(g1_list_dds_topics)
    assert first["topics"] is not second["topics"], (
        "successive calls shared a list reference; a caller mutating "
        "the returned ``topics`` would corrupt the snapshot. "
        "Refs strands-labs/robots#358."
    )
    first["topics"].append({"topic": "rt/injected", "direction": "read", "role": "x"})
    fresh = _call(g1_list_dds_topics)
    assert fresh["count"] == len(_DRIVER_TOPICS)
    assert all(entry["topic"] != "rt/injected" for entry in fresh["topics"])


def test_g1_topic_role_resolves_a_read_topic() -> None:
    """A known read-side topic returns the driver's cache role.

    ``rt/lf/bmsstate`` is the driver's battery subscribe; the tool
    answers ``direction="read"`` and ``role="battery"``, the same label
    :attr:`~strands_robots.drivers.g1.G1Driver._battery` carries in the
    cache.
    """
    result = _call(g1_topic_role, topic="rt/lf/bmsstate")
    assert result["status"] == "success"
    assert result["topic"] == "rt/lf/bmsstate"
    assert result["direction"] == "read"
    assert result["role"] == "battery"


def test_g1_topic_role_resolves_the_write_topic() -> None:
    """The write path ``rt/lowcmd`` resolves to ``direction="write"``.

    A caller planning a direct publish uses this to see that the topic
    is the driver's own write path; the correct call site is
    ``send_action`` / ``run_policy``, not a second publisher against the
    same wire.
    """
    result = _call(g1_topic_role, topic="rt/lowcmd")
    assert result["status"] == "success"
    assert result["topic"] == "rt/lowcmd"
    assert result["direction"] == "write"
    assert result["role"] == "lowcmd"


def test_g1_topic_role_refuses_an_unknown_topic() -> None:
    """A topic outside the driver's set is refused, not silently resolved.

    DDS topic names are exact strings on the wire; a mis-cased or
    trailing-slash variant is a different topic. The refusal names the
    known set so a caller can spot the typo.
    """
    result = _call(g1_topic_role, topic="rt/wirelesscontroller")
    assert result["status"] == "error"
    assert "rt/wirelesscontroller" in result["message"]
    assert "rt/lowcmd" in result["message"]
    assert "strands-labs/robots#358" in result["message"]


def test_g1_topic_role_refuses_a_case_variant() -> None:
    """A mis-cased topic name is not silently normalised to the canonical form.

    ``rt/LowState`` is not ``rt/lowstate``; the DDS wire is byte-exact
    on the topic string. The tool preserves that.
    """
    result = _call(g1_topic_role, topic="rt/LowState")
    assert result["status"] == "error"
    assert "rt/LowState" in result["message"]


def test_g1_topic_role_refuses_a_non_string_topic() -> None:
    """A caller that passes a non-str topic sees the type refusal.

    A dict-key confusion (passing an int or a bytes) is a caller
    mistake; the refusal names the received type so the mistake is
    visible instead of silently comparing bytes to str and returning a
    misleading "unknown topic".
    """
    for bad in (123, b"rt/lowstate", None, ["rt/lowstate"], {"topic": "rt/lowstate"}):
        result = _call(g1_topic_role, topic=bad)
        assert result["status"] == "error"
        assert "topic must be a str" in result["message"]
        assert "strands-labs/robots#358" in result["message"]
