"""A shadow-topic segment that is not a single MQTT topic level is refused.

``$aws/things/{thing}/shadow/name/{shadow}/update`` is a reserved 7-level AWS
IoT topic, so each interpolated name must be exactly ONE level.
:func:`~strands_robots.mesh.core.init_mesh` already refuses ``/``, ``+`` and
``#`` in a ``peer_id`` and says why - "these break MQTT topic structure and AWS
Thing-name rules" - but the shadow mirror, the one surface that actually puts
that identifier on a topic, interpolated it and ``shadow_name`` raw.

Nothing downstream reports the mistake: :meth:`ShadowMirror.update` is
best-effort and swallows whatever the transport says, so a malformed topic is
published, rejected by the broker, and logged at debug. The named shadow that
late-joining operators read for fleet state is then never written and every
call reports nothing wrong - which is why the refusal belongs where the topic is
built.

The domain is the shared mesh-identifier one, a SUPERSET of what ``init_mesh``
admits, so this cannot refuse an identifier the mesh already accepted.
"""

from __future__ import annotations

import string
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.iot.shadow import (
    ShadowMirror,
    shadow_get_topic,
    shadow_update_topic,
)
from strands_robots.mesh.security import MAX_PEER_ID_LEN, ValidationError

#: Every character class ``init_mesh``'s own refusal names as allowed in a
#: ``peer_id`` ("[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}"). A peer_id it admitted
#: reaches :func:`ShadowMirror` as ``thing_name``, so each must address a shadow.
INIT_MESH_ALLOWED_CHARS = string.ascii_letters + string.digits + "._-"

#: The three surfaces that turn a pair of names into a reserved topic. Each is
#: called with ``(thing_name, shadow_name)`` and must refuse the same domain
#: under its own name, so a refusal identifies the call the caller made.
SURFACES: tuple[tuple[str, Any], ...] = (
    ("shadow_update_topic", shadow_update_topic),
    ("shadow_get_topic", shadow_get_topic),
    ("ShadowMirror", ShadowMirror),
)

#: Segments that cannot be honored as written, with what each would have done.
BAD_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("fleet/rover-01", "a slash smuggles a level: the topic names Thing 'fleet'"),
    ("+", "the MQTT single-level wildcard, illegal in a publish topic"),
    ("#", "the MQTT multi-level wildcard, illegal in a publish topic"),
    ("", "an empty segment leaves an empty topic level"),
    ("rover 01", "whitespace"),
    ("rover\x0001", "a NUL"),
    ("*", "a Zenoh wildcard, which the bridge backend also routes by intersection"),
    ("a" * (MAX_PEER_ID_LEN + 1), "longer than one addressable segment"),
)


class TestASegmentThatIsNotOneTopicLevelIsRefused:
    @pytest.mark.parametrize("surface,build", SURFACES, ids=[s for s, _ in SURFACES])
    @pytest.mark.parametrize("segment,why", BAD_SEGMENTS, ids=[s[:12] or "empty" for s, _ in BAD_SEGMENTS])
    def test_thing_name_is_refused(self, surface: str, build: Any, segment: str, why: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            build(segment, "presence")
        assert f"{surface}.thing_name" in str(excinfo.value), why

    @pytest.mark.parametrize("surface,build", SURFACES, ids=[s for s, _ in SURFACES])
    @pytest.mark.parametrize("segment,why", BAD_SEGMENTS, ids=[s[:12] or "empty" for s, _ in BAD_SEGMENTS])
    def test_shadow_name_is_refused(self, surface: str, build: Any, segment: str, why: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            build("rover-01", segment)
        assert f"{surface}.shadow_name" in str(excinfo.value), why

    @pytest.mark.parametrize("surface,build", SURFACES, ids=[s for s, _ in SURFACES])
    def test_a_non_string_is_refused_before_it_is_formatted_into_a_topic(self, surface: str, build: Any) -> None:
        """``None`` interpolated raw addresses a Thing literally named 'None'."""
        with pytest.raises(ValidationError, match=f"{surface}.thing_name must be a string"):
            build(None, "presence")


class TestTheShippedTopicsAreUnchanged:
    """The refusal is the only new behaviour: every honorable pair still builds."""

    def test_update_and_get_topics(self) -> None:
        assert shadow_update_topic("rover-01") == "$aws/things/rover-01/shadow/name/presence/update"
        assert shadow_update_topic("rover-01", "task") == "$aws/things/rover-01/shadow/name/task/update"
        assert shadow_get_topic("rover-01") == "$aws/things/rover-01/shadow/name/presence/get"
        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=True)
        ShadowMirror("rover-01", "health").update(transport, {"connected": True})
        assert transport.put.call_args.args[0] == "$aws/things/rover-01/shadow/name/health/update"

    def test_a_valid_mirror_still_publishes_and_still_swallows_a_transport_error(self) -> None:
        """``update`` stays best-effort for a topic that IS addressable."""
        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=True)
        transport.put.side_effect = RuntimeError("network")
        ShadowMirror("rover-01").update(transport, {"connected": True})
        assert transport.put.call_count == 1


class TestNoIdentifierTheMeshAdmittedIsRefused:
    """The new domain is a superset of ``init_mesh``'s, so it cannot over-refuse."""

    @pytest.mark.parametrize("char", list(INIT_MESH_ALLOWED_CHARS), ids=lambda c: f"char-{ord(c)}")
    def test_every_character_init_mesh_allows_in_a_peer_id_addresses_a_shadow(self, char: str) -> None:
        thing = f"r{char}"
        assert shadow_update_topic(thing) == f"$aws/things/{thing}/shadow/name/presence/update"

    def test_a_dotted_peer_id_is_accepted_though_the_thing_name_rule_would_refuse_it(self) -> None:
        """The provisioning Thing-name rule rejects ``.`` for filesystem-path
        safety because it also names a certificate file. A topic segment is not
        a path component, and ``init_mesh`` admits a dotted ``peer_id``, so
        borrowing that stricter rule here would refuse a live peer.
        """
        assert ShadowMirror("rover.01").thing_name == "rover.01"

    def test_the_longest_peer_id_init_mesh_accepts_is_accepted(self) -> None:
        assert shadow_update_topic("a" * MAX_PEER_ID_LEN).count("/") == 6


class TestARefusedMirrorNeverExistsToPublishWith:
    def test_construction_refuses_before_any_topic_is_held(self) -> None:
        """A mirror holding an unaddressable topic would publish it once per
        heartbeat and swallow every rejection, so it must not be constructible.
        """
        transport = MagicMock()
        transport.is_alive = MagicMock(return_value=True)
        with pytest.raises(ValidationError):
            ShadowMirror("fleet/rover-01").update(transport, {"connected": True})
        transport.put.assert_not_called()
