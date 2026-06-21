"""Branch coverage for :func:`strands_robots.mesh.security.is_safe_server_address`.

``server_address`` is the operator-facing knob that points a robot at a remote
VLA policy server. The composite parser strips an optional ``scheme://`` and
``/path``, then resolves the host through three shapes: bracketed IPv6
(``[host]`` / ``[host]:port``), a single-colon ``host:port``, and a bare host
that may itself be an unbracketed IPv6 literal. Each shape has its own
port-validation and malformed-input rejection branches; those rejection paths
are the security boundary (a bad port or malformed bracket must fail closed,
never fall through to ``is_safe_policy_host`` with a half-parsed host).

These tests pin every accept/reject branch of the host-resolution ladder.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import security as sec


class TestBracketedIPv6:
    """``[host]`` / ``[host]:port`` parsing branch."""

    def test_loopback_no_port_allowed(self) -> None:
        assert sec.is_safe_server_address("[::1]") is True

    def test_loopback_valid_port_allowed(self) -> None:
        assert sec.is_safe_server_address("[::1]:8080") is True

    def test_missing_closing_bracket_rejected(self) -> None:
        # No ']' at all -> malformed bracketed address, fail closed.
        assert sec.is_safe_server_address("[::1") is False

    def test_trailing_garbage_after_bracket_rejected(self) -> None:
        # Remainder after ']' that is not ":port" is malformed.
        assert sec.is_safe_server_address("[::1]x") is False

    def test_empty_port_after_bracket_rejected(self) -> None:
        assert sec.is_safe_server_address("[::1]:") is False

    def test_non_digit_port_rejected(self) -> None:
        assert sec.is_safe_server_address("[::1]:abc") is False

    @pytest.mark.parametrize("port", ["0", "70000"])
    def test_out_of_range_port_rejected(self, port: str) -> None:
        assert sec.is_safe_server_address(f"[::1]:{port}") is False

    def test_unallowed_host_with_valid_port_rejected(self) -> None:
        # Valid bracket + valid port, but host not on the default loopback
        # allowlist -> rejected by is_safe_policy_host, not by the parser.
        assert sec.is_safe_server_address("[2001:db8::1]:8080") is False


class TestSingleColonHostPort:
    """``host:port`` parsing branch (exactly one colon)."""

    def test_loopback_valid_port_allowed(self) -> None:
        assert sec.is_safe_server_address("localhost:65535") is True

    def test_non_digit_port_rejected(self) -> None:
        assert sec.is_safe_server_address("localhost:abc") is False

    @pytest.mark.parametrize("port", ["0", "70000"])
    def test_out_of_range_port_rejected(self, port: str) -> None:
        assert sec.is_safe_server_address(f"localhost:{port}") is False


class TestUnbracketedHost:
    """Bare host: no colon, or two-plus colons (unbracketed IPv6 literal)."""

    def test_bare_loopback_hostname_allowed(self) -> None:
        assert sec.is_safe_server_address("localhost") is True

    def test_bare_loopback_ipv6_literal_allowed(self) -> None:
        # No brackets, multiple colons -> parsed as an IPv6 literal; ::1 is
        # on the default loopback allowlist.
        assert sec.is_safe_server_address("::1") is True

    def test_bare_unallowed_ipv6_literal_rejected(self) -> None:
        assert sec.is_safe_server_address("2001:db8::1") is False

    def test_multi_colon_non_ipv6_rejected(self) -> None:
        # Two-plus colons that are not a valid IPv6 literal must fail closed
        # rather than being mis-split into host/port.
        assert sec.is_safe_server_address("foo:bar:baz") is False


class TestSchemeAndPathStripping:
    """Scheme prefix and trailing path are stripped before host resolution."""

    def test_scheme_bracketed_ipv6_and_path_stripped(self) -> None:
        assert sec.is_safe_server_address("tcp://[::1]:5555/model") is True

    def test_empty_and_non_string_rejected(self) -> None:
        assert sec.is_safe_server_address("") is False
        assert sec.is_safe_server_address(None) is False  # type: ignore[arg-type]


class TestOperatorCidrAllowlist:
    """A CIDR entry in STRANDS_MESH_POLICY_HOST_ALLOW admits matching IPv6."""

    def test_cidr_admits_bare_and_bracketed_ipv6(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "2001:db8::/32")
        sec._clear_security_caches_for_tests()
        try:
            assert sec.is_safe_server_address("2001:db8::1") is True
            assert sec.is_safe_server_address("[2001:db8::1]:8080") is True
        finally:
            sec._clear_security_caches_for_tests()
