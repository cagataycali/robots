#!/usr/bin/env python3
"""Edge-case contracts for the IoT MQTT transport wire-format helpers.

``tests/mesh/test_transport.py`` already pins the common-path translations and
QoS lookups. This module covers the *degenerate / fall-through* branches of the
two pure helper functions, which encode the contract the broker relies on:

  - ``_zenoh_to_mqtt_filter``: a ``**`` wildcard in a non-tail position is still
    translated to ``#`` (the broker will SUBACK-deny it) rather than raising.
  - ``_qos_and_retain_for``: every malformed / unknown topic shape resolves to a
    safe default of ``(0, False)`` instead of raising, so an unexpected topic can
    never crash the publish path or silently inherit another topic's policy.

All assertions are on return values (behaviour), never on internal state. These
helpers are pure and dependency-free, so no AWS IoT SDK is required.
"""

from strands_robots.mesh.transport.iot_transport import (
    _qos_and_retain_for,
    _zenoh_to_mqtt_filter,
)


class TestZenohToMqttFilterDegenerate:
    """Non-tail ``**`` and other unusual inputs translate without raising."""

    def test_double_wildcard_in_non_tail_position_becomes_hash(self):
        # '**' is only valid as the final MQTT segment ('#'); a mid-expression
        # occurrence is translated verbatim to '#' and left for the broker to
        # reject, rather than raising on the client.
        assert _zenoh_to_mqtt_filter("strands/**/cmd") == "strands/#/cmd"

    def test_leading_double_wildcard(self):
        assert _zenoh_to_mqtt_filter("**/state") == "#/state"

    def test_empty_string_passes_through(self):
        assert _zenoh_to_mqtt_filter("") == ""

    def test_mixed_single_and_double_wildcards(self):
        assert _zenoh_to_mqtt_filter("strands/*/foo/**") == "strands/+/foo/#"


class TestQosAndRetainFallThrough:
    """Malformed / unknown topics resolve to the safe ``(0, False)`` default."""

    def test_topic_without_strands_prefix(self):
        assert _qos_and_retain_for("foo/bar/baz") == (0, False)

    def test_bare_strands_prefix_with_trailing_slash(self):
        # 'strands/' -> empty remainder -> default, no IndexError.
        assert _qos_and_retain_for("strands/") == (0, False)

    def test_top_level_kind_with_no_matching_policy(self):
        # 'safety' is a reserved top-level kind (layout a), but this exact
        # suffix has no policy entry -> fall through to the default.
        assert _qos_and_retain_for("strands/safety/unknown/extra") == (0, False)

    def test_peer_topic_with_only_peer_segment(self):
        # Layout (b) needs at least peer_id + kind; a lone peer segment is
        # below the minimum and resolves to the default.
        assert _qos_and_retain_for("strands/peer-only") == (0, False)

    def test_peer_topic_with_unknown_kind(self):
        assert _qos_and_retain_for("strands/so100-01/totally-unknown") == (0, False)

    def test_top_level_broadcast_still_matches_policy(self):
        # Regression guard: the fall-through additions above must not perturb a
        # known top-level match.
        assert _qos_and_retain_for("strands/broadcast") == (1, False)
