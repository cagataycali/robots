"""End-to-end tests for :func:`strands_robots.mesh.session._build_config`.

These tests exercise the full chain from env var → ``_build_config`` →
``zenoh.Config`` round-trip. They require ``eclipse-zenoh`` because the
emitted JSON5 must validate against Zenoh's Rust ``Config`` parser. If
the wheel is unavailable the tests skip cleanly.
"""

from __future__ import annotations

import json

import pytest

zenoh = pytest.importorskip("zenoh")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
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
        "STRANDS_MESH_ACL_FILE",
        "ZENOH_CONNECT",
        "ZENOH_LISTEN",
    ]:
        monkeypatch.delenv(key, raising=False)


def _build():
    """Build a config in auth_mode=none (no TLS files needed)."""
    import os

    os.environ["STRANDS_MESH_AUTH_MODE"] = "none"
    from strands_robots.mesh.session import _build_config

    return _build_config()


def _build_mtls(tmp_path, monkeypatch):
    """Build a config in auth_mode=mtls with synthetic cert files."""
    ca = tmp_path / "ca.crt"
    cert = tmp_path / "peer.crt"
    key = tmp_path / "peer.key"
    for f in (ca, cert, key):
        f.write_text("dummy\n")
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
    from strands_robots.mesh.session import _build_config

    return _build_config()


# ─── Default (no env, auth_mode=none) ──────────────────────────────────


class TestDefaultBuild:
    def test_namespace_default_applied(self):
        cfg = _build()
        assert json.loads(cfg.get_json("namespace")) == "strands_robots"

    def test_multicast_disabled_by_default(self):
        cfg = _build()
        assert cfg.get_json("scouting/multicast/enabled") == "false"

    def test_gossip_enabled_by_default(self):
        cfg = _build()
        assert cfg.get_json("scouting/gossip/enabled") == "true"

    def test_max_sessions_default_256(self):
        cfg = _build()
        assert cfg.get_json("transport/unicast/max_sessions") == "256"

    def test_downsampling_present(self):
        cfg = _build()
        ds = json.loads(cfg.get_json("downsampling"))
        assert any(rule["id"] == "strands_cmd_rate_cap" for rule in ds)

    def test_low_pass_filter_present(self):
        cfg = _build()
        lpf = json.loads(cfg.get_json("low_pass_filter"))
        assert any(rule["id"] == "strands_cmd_size_cap" for rule in lpf)
        assert any(rule["id"] == "strands_camera_size_cap" for rule in lpf)

    def test_adminspace_disabled(self):
        cfg = _build()
        admin = json.loads(cfg.get_json("adminspace"))
        assert admin["enabled"] is False


# ─── Custom env overrides ───────────────────────────────────────────────


class TestEnvOverrides:
    def test_namespace_override_propagates(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_NAMESPACE", "fleet_42")
        cfg = _build()
        assert json.loads(cfg.get_json("namespace")) == "fleet_42"
        # Downsampling key_exprs must use the overridden namespace.
        ds = json.loads(cfg.get_json("downsampling"))
        rules = ds[0]["rules"]
        assert any(r["key_expr"].startswith("fleet_42/") for r in rules)

    def test_multicast_can_be_enabled(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
        cfg = _build()
        assert cfg.get_json("scouting/multicast/enabled") == "true"

    def test_max_sessions_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "1024")
        cfg = _build()
        assert cfg.get_json("transport/unicast/max_sessions") == "1024"


# ─── mTLS path ──────────────────────────────────────────────────────────


class TestMTLSBuild:
    def test_tls_block_present(self, tmp_path, monkeypatch):
        cfg = _build_mtls(tmp_path, monkeypatch)
        tls = json.loads(cfg.get_json("transport/link/tls"))
        assert tls["enable_mtls"] is True
        assert tls["verify_name_on_connect"] is True

    def test_link_protocols_restricted_to_tls(self, tmp_path, monkeypatch):
        cfg = _build_mtls(tmp_path, monkeypatch)
        protos = json.loads(cfg.get_json("transport/link/protocols"))
        assert protos == ["tls"]

    def test_acl_block_present_with_default_deny(self, tmp_path, monkeypatch):
        cfg = _build_mtls(tmp_path, monkeypatch)
        acl = json.loads(cfg.get_json("access_control"))
        assert acl["default_permission"] == "deny"
        assert {s["id"] for s in acl["subjects"]} == {"robot_peer", "operator_peer"}

    def test_mtls_missing_cert_files_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", "/nonexistent/ca.crt")
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", "/nonexistent/peer.crt")
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", "/nonexistent/peer.key")
        from strands_robots.mesh.session import _build_config

        with pytest.raises(FileNotFoundError):
            _build_config()

    def test_mtls_missing_env_vars_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        from strands_robots.mesh.session import _build_config

        with pytest.raises(ValueError, match="STRANDS_MESH_TLS"):
            _build_config()


# ─── Endpoint env vars ──────────────────────────────────────────────────


class TestEndpointEnvVars:
    def test_zenoh_connect_propagates(self, monkeypatch):
        monkeypatch.setenv("ZENOH_CONNECT", "tls/router.fleet.local:7447")
        cfg = _build()
        endpoints = json.loads(cfg.get_json("connect/endpoints"))
        assert endpoints == ["tls/router.fleet.local:7447"]

    def test_zenoh_listen_propagates(self, monkeypatch):
        monkeypatch.setenv("ZENOH_LISTEN", "tcp/0.0.0.0:7447")
        cfg = _build()
        endpoints = json.loads(cfg.get_json("listen/endpoints"))
        assert endpoints == ["tcp/0.0.0.0:7447"]


# ─── Auth-mode=none warning ─────────────────────────────────────────────


def test_auth_mode_none_logs_warning(caplog):
    import os

    os.environ["STRANDS_MESH_AUTH_MODE"] = "none"
    from strands_robots.mesh.session import _build_config

    with caplog.at_level("WARNING"):
        _build_config()
    assert any("authentication is OFF" in rec.message for rec in caplog.records)
