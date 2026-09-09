"""``host`` and ``server_port`` are the two halves of one URI, held to one funnel.

:attr:`VeraConfig.server_uri` is ``ws://{host}:{port}``. The port half is held to
the shared TCP-port domain, for a reason recorded at length: three consumers read
it under three coercions, so an unusable value is not refused late but *applied*
as three different ports. The host half reaches those same three consumers - the
URI the client dials, the ``socket.create_connection`` the readiness probe makes,
and the ``--host`` on the server argv - and was held to nothing. A value that is
not a host was therefore resolved rather than refused:

* A URI delimiter re-cuts the URI, and the validated port is the component it
  takes. ``host="127.0.0.1/foo"`` parses as host ``127.0.0.1``, path
  ``/foo:8820`` and port **80**, so the client dials a port nobody configured;
  ``host="ws://127.0.0.1"`` - what a caller who pastes a URI supplies - parses as
  host ``ws`` on port 80. The port domain cannot see this: it is the host half
  that discards the port.
* ``""`` is the one unusable spelling the readiness probe *accepted*, because the
  probe maps a bind-only host to loopback. The runner logged "VERA server ready"
  and the client then could not build a URI at all, raising ``InvalidURI`` past
  the ``OSError`` channel that carries its actionable "could not reach the VERA
  policy server" hint.
* A non-string reached ``socket.getaddrinfo`` first and raised ``TypeError:
  getaddrinfo() argument 1 must be string or None`` out of ``start()``, past the
  ``TimeoutError``/``RuntimeError`` channel the runner documents.

These pin the domain, the shape of the harm against the real URI parser, the two
halves sharing one floor, the documented asymmetries between them, and the
over-reach control: every host a URI can carry is still named identically by all
three consumers - including ``"0.0.0.0"``, which the probe special-cases and
which the refusal for ``""`` names as the spelling that binds every interface.

Everything here is offline - no ``vera`` package, no server, and the one socket
is a loopback listener this file opens itself.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from strands_robots.policies.vera import VeraConfig, VeraPolicy, VeraServerRunner

# Hosts a URI cannot carry, grouped by what the value did instead of failing.
# ``':'`` outside brackets is here because the port follows it: a bare IPv6
# literal (``"::1"``) and a host that already carries a port both re-cut the URI.
RECUT_THE_URI: list[str] = [
    "127.0.0.1/foo",
    "ws://127.0.0.1",
    "127.0.0.1:9999",
    "::1",
    "user@127.0.0.1",
    "127.0.0.1?x=1",
    "127.0.0.1#frag",
]

# Spellings that resolve to no host at all.
NAME_NO_HOST: list[str] = ["", "[]"]

# Values that never reach a URI, because ``getaddrinfo`` refuses them first.
NOT_A_STRING: list[Any] = [None, 0, 8800, True, False, 3.5, ["127.0.0.1"], {}, b"127.0.0.1"]

# Hosts whose only defect is a character a resolver cannot be handed.
UNRESOLVABLE_SHAPE: list[str] = ["  ", "127.0.0.1 ", " 127.0.0.1", "127.0.0.1\x00", "my\thost"]

# Hosts every consumer can honor. ``0.0.0.0`` is the documented way to reach a
# server bound on every interface and ``[::1]`` is the bracketed IPv6 spelling a
# URI requires; both are over-reach controls for the delimiter rule.
USABLE_HOSTS: list[str] = ["127.0.0.1", "localhost", "0.0.0.0", "my-host.local", "[::1]", "vera-server"]

# The floor the two halves of ``ws://{host}:{port}`` share: neither a host nor a
# port can be one of these, so one funnel refuses both fields for the same value.
SHARED_FLOOR: list[Any] = [True, False, 3.5, ["8800"], {}]

# Where the two halves legitimately disagree, and why. A cell asserts every
# reason is stated, so a future asymmetry has to be argued rather than absorbed.
ASYMMETRIES: dict[str, str] = {
    "None": (
        "None is the port's documented 'apply the per-embodiment default' spelling; "
        "host has a concrete default and no table to look one up in, so None is a "
        "stated non-host rather than a request for the default"
    ),
    "'8800'": (
        "a decimal string is a legal DNS label, so it is a host the URI can carry; "
        "it is not a port, because the three port consumers coerce it three ways"
    ),
    "0": (
        "zero is refused as a port because the client cannot learn which ephemeral "
        "port the kernel handed the server; it is refused as a host only for being "
        "an int, and '0' as text would be accepted"
    ),
}


def _config(**kwargs: Any) -> VeraConfig:
    """Build a config through the funnel, splatted so off-type values reach it.

    mypy does not narrow a ``**dict[str, Any]`` splat, which is what lets one
    test hand a deliberately wrong type to a typed field without an ignore.
    """
    return VeraConfig(**kwargs)


def _refusal(**kwargs: Any) -> str | None:
    """The funnel's refusal for these keywords, read through the public door.

    Reading the message off the constructor rather than off the guard keeps this
    file pinned to the behaviour a caller sees, so the check can move or be
    renamed without a test edit.
    """
    try:
        _config(**kwargs)
    except ValueError as e:
        return str(e)
    return None


def _argv_value(cfg: VeraConfig, flag: str) -> str | None:
    """The value the subprocess argv carries for ``flag``, or ``None`` if absent."""
    argv = VeraServerRunner(cfg)._build_command()
    return next((v for a, v in zip(argv, [*argv[1:], ""], strict=True) if a == flag), None)


class TestAHostThatIsNotAHostIsRefused:
    @pytest.mark.parametrize("value", RECUT_THE_URI)
    def test_a_host_that_recuts_the_uri_is_refused_by_name(self, value):
        """The port half cannot catch this: the host half is what discards it."""
        message = _refusal(embodiment="pusht", host=value)
        assert message is not None
        assert "host" in message
        assert repr(f"ws://{value}:<port>") in message

    @pytest.mark.parametrize("value", NAME_NO_HOST)
    def test_a_spelling_that_names_no_host_is_refused_naming_the_one_that_works(self, value):
        """``""`` was reported ready by the probe and then had no URI to dial."""
        message = _refusal(embodiment="pusht", host=value)
        assert message is not None
        assert "host" in message
        assert "0.0.0.0" in message

    @pytest.mark.parametrize("value", NOT_A_STRING)
    def test_a_non_string_host_is_refused_rather_than_handed_to_getaddrinfo(self, value):
        message = _refusal(embodiment="pusht", host=value)
        assert message is not None
        assert "host" in message
        assert type(value).__name__ in message

    @pytest.mark.parametrize("value", UNRESOLVABLE_SHAPE)
    def test_a_host_carrying_whitespace_or_a_control_character_is_refused(self, value):
        message = _refusal(embodiment="pusht", host=value)
        assert message is not None
        assert "host" in message

    def test_a_bare_ipv6_literal_is_refused_naming_the_bracketed_spelling(self):
        """``"::1"`` is the mistake the bracket rule exists to answer."""
        message = _refusal(embodiment="pusht", host="::1")
        assert message is not None
        assert "[::1]" in message


class TestTheHarmAgainstTheRealUriParser:
    """The consumer decides what a re-cut URI means, so the consumer is asked."""

    @pytest.mark.parametrize("value", ["127.0.0.1/foo", "ws://127.0.0.1"])
    def test_a_refused_host_would_have_discarded_the_validated_port(self, value):
        parse_uri = pytest.importorskip("websockets.uri").parse_uri
        assert parse_uri(f"ws://{value}:8820").port == 80

    def test_the_empty_host_would_not_have_parsed_at_all(self):
        websockets_uri = pytest.importorskip("websockets.uri")
        with pytest.raises(websockets_uri.InvalidURI):
            websockets_uri.parse_uri("ws://:8820")

    @pytest.mark.parametrize("value", USABLE_HOSTS)
    def test_every_accepted_host_keeps_the_port_it_was_configured_with(self, value):
        """The property the refusals protect, stated positively."""
        parse_uri = pytest.importorskip("websockets.uri").parse_uri
        parsed = parse_uri(_config(embodiment="pusht", host=value).server_uri)
        assert (parsed.port, parsed.path) == (8820, "")


class TestEveryConsumerNamesTheSameHost:
    @pytest.mark.parametrize("value", USABLE_HOSTS)
    def test_client_uri_server_uri_and_argv_name_one_host(self, value):
        policy = VeraPolicy(embodiment="pusht", host=value, auto_launch_server=False)
        assert policy._client.uri == f"ws://{value}:8820"
        assert policy.config.server_uri == f"ws://{value}:8820"
        assert _argv_value(policy.config, "--host") == value

    def test_the_default_host_is_named_identically_too(self):
        policy = VeraPolicy(embodiment="pusht", auto_launch_server=False)
        assert policy._client.uri == "ws://127.0.0.1:8820"
        assert policy.config.server_uri == "ws://127.0.0.1:8820"
        assert _argv_value(policy.config, "--host") == "127.0.0.1"


class TestTheTwoHalvesOfTheUriShareOneFloor:
    @pytest.mark.parametrize("value", SHARED_FLOOR)
    def test_neither_half_accepts_what_the_other_refuses(self, value):
        """One expression, one funnel: the halves cannot drift on this set."""
        assert _refusal(embodiment="pusht", host=value) is not None
        assert _refusal(embodiment="pusht", server_port=value) is not None

    def test_every_documented_asymmetry_states_its_reason(self):
        assert all(reason.strip() for reason in ASYMMETRIES.values())

    def test_none_is_the_ports_default_spelling_and_not_a_host(self):
        assert _config(embodiment="pusht", server_port=None).server_port == 8820
        assert _refusal(embodiment="pusht", host=None) is not None

    def test_a_decimal_string_is_a_host_but_not_a_port(self):
        assert _config(embodiment="pusht", host="8800").host == "8800"
        assert _refusal(embodiment="pusht", server_port="8800") is not None


class TestTheReadinessProbeStillReusesABoundServer:
    """The probe's bind-address arm keeps the spelling that can also be dialed."""

    def test_a_server_listening_on_loopback_is_reused_for_host_0_0_0_0(self):
        with socket.create_server(("127.0.0.1", 0)) as listener:
            cfg = _config(embodiment="pusht", host="0.0.0.0", server_port=listener.getsockname()[1])
            runner = VeraServerRunner(cfg)
            runner.start()  # returns on the reuse path, without launching anything
            assert runner.is_running() is False

    def test_the_empty_spelling_can_no_longer_reach_that_arm(self):
        """It was the only unusable host the probe mapped to loopback."""
        assert _refusal(embodiment="pusht", host="") is not None
