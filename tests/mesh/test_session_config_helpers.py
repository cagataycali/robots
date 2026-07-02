"""Regression coverage for mesh session config-resolution helpers.

Pins the observable behaviour of the pure-Python configuration helpers in
``strands_robots.mesh.session`` that resolve operator env vars and validate
endpoint schemes before a transport is ever opened:

- ``_max_peers`` clamps bad / non-positive ``STRANDS_MESH_MAX_PEERS`` to the
  default cap instead of propagating a parse error or a nonsensical bound.
- ``_validate_endpoint_schemes`` rejects TLS-incompatible endpoint schemes
  for the active auth mode, tolerates unknown auth modes (deferring the hard
  error to ``resolve_auth_mode``), and skips empty entries in the CSV list.
- ``PeerInfo.__repr__`` renders a stable, human-readable identity string.

All assertions are on returned values / raised types, never on internal
state, and none of these paths touch Zenoh, torch, or MuJoCo.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import session as _session
from strands_robots.mesh.session import (
    MAX_PEERS_DEFAULT,
    PeerInfo,
    _max_peers,
    _validate_endpoint_schemes,
)


class TestMaxPeers:
    """``_max_peers`` env resolution (STRANDS_MESH_MAX_PEERS)."""

    def test_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("STRANDS_MESH_MAX_PEERS", raising=False)
        assert _max_peers() == MAX_PEERS_DEFAULT

    def test_valid_positive_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", "42")
        assert _max_peers() == 42

    def test_non_integer_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A garbage value must not crash peer bookkeeping; it degrades to the cap.
        monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", "not-a-number")
        assert _max_peers() == MAX_PEERS_DEFAULT

    def test_float_string_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # int("10.5") raises ValueError -> fall back rather than truncate.
        monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", "10.5")
        assert _max_peers() == MAX_PEERS_DEFAULT

    @pytest.mark.parametrize("raw", ["0", "-1", "-1024"])
    def test_non_positive_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        # Zero / negative would disable or invert the flood bound; reject it.
        monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", raw)
        assert _max_peers() == MAX_PEERS_DEFAULT


class TestValidateEndpointSchemes:
    """``_validate_endpoint_schemes`` auth-mode / scheme gating."""

    def test_none_or_empty_endpoints_is_noop(self) -> None:
        # No endpoints configured -> nothing to validate, no raise.
        _validate_endpoint_schemes(None, "ZENOH_CONNECT", "mtls")
        _validate_endpoint_schemes("", "ZENOH_CONNECT", "mtls")

    def test_mtls_accepts_tls_bearing_scheme(self) -> None:
        _validate_endpoint_schemes("tls/127.0.0.1:7447", "ZENOH_CONNECT", "mtls")

    def test_mtls_rejects_plaintext_scheme(self) -> None:
        # tcp carries no TLS bytes; under mtls it must be rejected with an
        # actionable ValueError naming the env var and the offending endpoint.
        with pytest.raises(ValueError) as exc:
            _validate_endpoint_schemes("tcp/127.0.0.1:7447", "ZENOH_LISTEN", "mtls")
        msg = str(exc.value)
        assert "ZENOH_LISTEN" in msg
        assert "tcp" in msg

    def test_none_mode_accepts_plaintext_scheme(self) -> None:
        _validate_endpoint_schemes("tcp/0.0.0.0:7447", "ZENOH_LISTEN", "none")

    def test_none_mode_rejects_wss_scheme(self) -> None:
        with pytest.raises(ValueError):
            _validate_endpoint_schemes("wss/0.0.0.0:7447", "ZENOH_LISTEN", "none")

    def test_unknown_auth_mode_defers_without_raising(self) -> None:
        # An unrecognised auth mode is resolve_auth_mode's error to raise;
        # this helper must not pre-empt it, even with a would-be-bad scheme.
        _validate_endpoint_schemes("tcp/127.0.0.1:7447", "ZENOH_CONNECT", "bogus-mode")

    def test_empty_csv_entries_are_skipped(self) -> None:
        # Stray commas / whitespace-only segments must not be treated as a
        # zero-length endpoint (which has no scheme); they are skipped.
        _validate_endpoint_schemes("tcp/127.0.0.1:7447, ,udp/0.0.0.0:7448", "ZENOH_CONNECT", "none")

    def test_scheme_check_is_case_insensitive(self) -> None:
        # Upper-case schemes normalise before the membership test.
        _validate_endpoint_schemes("TLS/127.0.0.1:7447", "ZENOH_CONNECT", "mtls")


class TestPeerInfoRepr:
    """``PeerInfo.__repr__`` identity rendering."""

    def test_repr_contains_identity_fields(self) -> None:
        peer = PeerInfo(peer_id="so100-a1b2", peer_type="robot", hostname="thor")
        text = repr(peer)
        assert "so100-a1b2" in text
        assert "robot" in text
        # Age is rendered with one decimal place and a seconds suffix.
        assert "age=" in text and "s)" in text

    def test_repr_is_stable_for_default_peer_type(self) -> None:
        assert "type='robot'" in repr(PeerInfo(peer_id="x"))


def test_helpers_are_exported_from_module() -> None:
    # Guard against an accidental rename breaking these tests silently.
    assert callable(_session._max_peers)
    assert callable(_session._validate_endpoint_schemes)
