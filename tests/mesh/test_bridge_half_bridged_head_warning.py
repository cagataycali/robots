"""A topic in the exact-match bridge list whose per-turn tails are not bridged warns once.

``STRANDS_MESH_BRIDGE_TOPICS`` is an exact-match list and
``STRANDS_MESH_BRIDGE_TOPICS_PREFIX`` a prefix list. An operator who adds a
head to the first and then sees ``<head>/<turn>`` traffic gets the bare topic
on MQTT and every per-turn message left on the LAN -- a change that looks like
it took while half of it silently did not. ``_should_bridge`` must still refuse
the tail (that is the documented contract) and say so, once per head, so the
misconfiguration is visible rather than a silent partial success.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.mesh.transport.bridge_transport import _should_bridge

EXACT_ONLY = frozenset({"telemetry"})
PREFIXES = frozenset({"response"})


def _prefix_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "STRANDS_MESH_BRIDGE_TOPICS_PREFIX" in r.getMessage()]


def test_tail_on_exact_only_topic_is_not_bridged_while_the_bare_topic_is() -> None:
    """The contract is unchanged: exact match bridges the bare topic, not its tails."""
    assert _should_bridge("strands/neon/telemetry/turn-9", EXACT_ONLY, PREFIXES) is False
    assert _should_bridge("strands/neon/telemetry", EXACT_ONLY, PREFIXES) is True


def test_half_bridged_head_warns_once_and_names_the_fix(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _should_bridge("strands/neon/telemetry/turn-9", EXACT_ONLY, PREFIXES)
        _should_bridge("strands/neon/telemetry/turn-10", EXACT_ONLY, PREFIXES)
    warnings = _prefix_warnings(caplog)
    assert len(warnings) == 1, f"expected one warning per head, got {len(warnings)}"
    assert "'telemetry'" in warnings[0]
    assert "STRANDS_MESH_BRIDGE_TOPICS_PREFIX" in warnings[0]


def test_each_half_bridged_head_warns_on_its_own(caplog: pytest.LogCaptureFixture) -> None:
    exact = frozenset({"telemetry", "diag"})
    with caplog.at_level(logging.WARNING):
        _should_bridge("strands/neon/telemetry/turn-1", exact, PREFIXES)
        _should_bridge("strands/neon/diag/turn-1", exact, PREFIXES)
        _should_bridge("strands/neon/diag/turn-2", exact, PREFIXES)
    warnings = _prefix_warnings(caplog)
    assert len(warnings) == 2
    assert any("'telemetry'" in w for w in warnings)
    assert any("'diag'" in w for w in warnings)


def test_a_head_in_both_lists_bridges_its_tails_and_stays_silent(caplog: pytest.LogCaptureFixture) -> None:
    """A head in BOTH lists is fully bridged, so there is no half to warn about.

    This is the overlap the guard must not misread: ``response`` is in the
    exact list *and* the prefix list, so its tails bridge and the warning has
    nothing to report. Pins the guard's position after the prefix-accept
    return -- hoisted above it, this head warns about traffic it does bridge.
    """
    both = frozenset({"telemetry", "response"})
    with caplog.at_level(logging.WARNING):
        assert _should_bridge("strands/neon/response/turn-9", both, PREFIXES) is True
    assert _prefix_warnings(caplog) == []


def test_no_warning_for_a_tail_whose_head_is_in_neither_list(caplog: pytest.LogCaptureFixture) -> None:
    """A head the operator never listed is not a half-bridge; it is simply unbridged."""
    with caplog.at_level(logging.WARNING):
        assert _should_bridge("strands/neon/camera/turn-9", EXACT_ONLY, PREFIXES) is False
    assert _prefix_warnings(caplog) == []
