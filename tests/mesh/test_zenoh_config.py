"""Tests for :mod:`strands_robots.mesh._zenoh_config`.

These exercise the pure-function config builders (no Zenoh session
required). The integration smoke that the emitted JSON5 actually
parses cleanly through Zenoh's Rust ``Config`` validator lives in
``test_session_config.py``.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.mesh import _zenoh_config as zc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test runs without inherited STRANDS_MESH_* env vars."""
    for key in [
        "STRANDS_MESH_NAMESPACE",
        "STRANDS_MESH_MULTICAST",
        "STRANDS_MESH_MAX_SESSIONS",
        "STRANDS_MESH_MAX_CMD_BYTES",
        "STRANDS_MESH_MAX_CAMERA_BYTES",
        "STRANDS_MESH_CMD_RATE_HZ",
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_TLS_CA",
        "STRANDS_MESH_TLS_CERT",
        "STRANDS_MESH_TLS_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


# --- namespace ----------------------------------------------------------


class TestNamespace:
    def test_default(self):
        assert zc.resolve_namespace() == "strands"

    def test_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_NAMESPACE", "fleet_42")
        assert zc.resolve_namespace() == "fleet_42"

    def test_empty_falls_through_to_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_NAMESPACE", "   ")
        assert zc.resolve_namespace() == "strands"

    def test_namespace_block_returns_json5_string(self):
        path, value = zc.namespace_block()
        assert path == "namespace"
        assert json.loads(value) == "strands"


# --- auth mode ----------------------------------------------------------


class TestAuthMode:
    def test_default_is_mtls(self):
        assert zc.resolve_auth_mode() == "mtls"

    def test_explicit_none(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        assert zc.resolve_auth_mode() == "none"

    def test_typo_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtsl")
        with pytest.raises(ValueError, match="not supported"):
            zc.resolve_auth_mode()


# --- scouting -----------------------------------------------------------


class TestScouting:
    def test_default_is_multicast_off_gossip_on(self):
        out = dict(zc.scouting_block())
        assert out["scouting/multicast/enabled"] == "false"
        assert out["scouting/gossip/enabled"] == "true"

    def test_multicast_can_be_enabled(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
        out = dict(zc.scouting_block())
        assert out["scouting/multicast/enabled"] == "true"

    def test_invalid_bool_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "maybe")
        with pytest.raises(ValueError, match="boolean"):
            zc.scouting_block()


# --- transport caps -----------------------------------------------------


class TestTransportCaps:
    def test_default_max_sessions(self):
        out = dict(zc.transport_caps_block())
        assert out["transport/unicast/max_sessions"] == "256"

    def test_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "1024")
        out = dict(zc.transport_caps_block())
        assert out["transport/unicast/max_sessions"] == "1024"

    def test_oob_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "100000")
        with pytest.raises(ValueError, match="out of bounds"):
            zc.transport_caps_block()

    def test_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "0")
        with pytest.raises(ValueError):
            zc.transport_caps_block()


# --- downsampling -------------------------------------------------------


class TestDownsampling:
    def test_default_freq(self):
        path, value = zc.downsampling_block("ns")
        assert path == "downsampling"
        decoded = json.loads(value)
        assert decoded[0]["id"] == "strands_cmd_rate_cap"
        assert decoded[0]["messages"] == ["put"]
        assert decoded[0]["flows"] == ["ingress"]
        rules = {r["key_expr"]: r["freq"] for r in decoded[0]["rules"]}
        assert rules["**/cmd"] == zc.DEFAULT_CMD_RATE_HZ
        assert rules["**/broadcast"] == zc.DEFAULT_CMD_RATE_HZ

    def test_freq_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "5.0")
        _, value = zc.downsampling_block("ns")
        decoded = json.loads(value)
        rules = {r["key_expr"]: r["freq"] for r in decoded[0]["rules"]}
        assert rules["**/cmd"] == 5.0

    def test_freq_oob_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "9999999")
        with pytest.raises(ValueError):
            zc.downsampling_block("ns")


# --- low_pass_filter ----------------------------------------------------


class TestLowPassFilter:
    def test_default_caps(self):
        path, value = zc.low_pass_filter_block("ns")
        assert path == "low_pass_filter"
        decoded = json.loads(value)
        cmd, cam = decoded[0], decoded[1]
        assert cmd["id"] == "strands_cmd_size_cap"
        assert cmd["size_limit"] == zc.DEFAULT_MAX_CMD_BYTES
        assert "**/cmd" in cmd["key_exprs"]
        assert "**/broadcast" in cmd["key_exprs"]
        assert cam["id"] == "strands_camera_size_cap"
        assert cam["size_limit"] == zc.DEFAULT_MAX_CAMERA_BYTES
        assert "**/camera/**" in cam["key_exprs"]

    def test_override_cmd_bytes(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_CMD_BYTES", "8192")
        _, value = zc.low_pass_filter_block("ns")
        decoded = json.loads(value)
        assert decoded[0]["size_limit"] == 8192

    def test_oversize_cap_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_CMD_BYTES", "999999999999")
        with pytest.raises(ValueError):
            zc.low_pass_filter_block("ns")


# --- adminspace ---------------------------------------------------------


def test_adminspace_block_disabled():
    path, value = zc.adminspace_block()
    assert path == "adminspace"
    decoded = json.loads(value)
    assert decoded["enabled"] is False
    assert decoded["permissions"] == {"read": False, "write": False}


# --- mTLS ---------------------------------------------------------------


class TestTLSBlock:
    def test_missing_paths_raise(self):
        # No STRANDS_MESH_TLS_* env vars set.
        with pytest.raises(ValueError, match="STRANDS_MESH_TLS"):
            zc.tls_block()

    def test_nonexistent_files_raise(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(tmp_path / "missing.crt"))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(tmp_path / "missing.crt"))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(tmp_path / "missing.key"))
        with pytest.raises(FileNotFoundError):
            zc.tls_block()

    def test_valid_paths_emit_block(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.crt"
        cert = tmp_path / "peer.crt"
        key = tmp_path / "peer.key"
        for f in (ca, cert, key):
            f.write_text("dummy\n")

        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))

        path, value = zc.tls_block()
        assert path == "transport/link/tls"
        decoded = json.loads(value)
        assert decoded["enable_mtls"] is True
        assert decoded["verify_name_on_connect"] is True
        assert decoded["close_link_on_expiration"] is True
        assert decoded["root_ca_certificate"] == str(ca)
        assert decoded["listen_certificate"] == str(cert)
        assert decoded["connect_certificate"] == str(cert)
        assert decoded["listen_private_key"] == str(key)
        assert decoded["connect_private_key"] == str(key)


def test_link_protocols_block_restricts_to_tls():
    path, value = zc.link_protocols_block()
    assert path == "transport/link/protocols"
    assert json.loads(value) == ["tls"]
