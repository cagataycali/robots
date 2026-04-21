"""Tests for strands_robots.mesh_session — Zenoh session singleton."""

from __future__ import annotations

import importlib
import threading
from unittest.mock import MagicMock, patch

import pytest

from strands_robots import mesh_session
from strands_robots.mesh_session import (
    DEFAULT_MESH_PORT,
    MeshConfig,
    get_session,
    release_session,
    session_info,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_session_state() -> None:
    """Reset module-level session state between tests."""
    mesh_session._SESSION = None
    mesh_session._SESSION_REFS = 0


@pytest.fixture(autouse=True)
def _clean_session():
    """Ensure each test starts and ends with a clean session."""
    _reset_session_state()
    yield
    _reset_session_state()


@pytest.fixture()
def mock_zenoh():
    """Provide a mock ``zenoh`` module with ``Config`` and ``open``."""
    mock_module = MagicMock()
    mock_config = MagicMock()
    mock_session = MagicMock()

    mock_module.Config.return_value = mock_config
    mock_module.open.return_value = mock_session

    with patch.object(importlib, "import_module", return_value=mock_module):
        yield {"module": mock_module, "config": mock_config, "session": mock_session}


# ---------------------------------------------------------------------------
# MeshConfig
# ---------------------------------------------------------------------------


class TestMeshConfig:
    """Tests for MeshConfig dataclass."""

    def test_defaults(self):
        cfg = MeshConfig()
        assert cfg.connect == ()
        assert cfg.listen == ()
        assert cfg.port == DEFAULT_MESH_PORT

    def test_frozen(self):
        cfg = MeshConfig()
        with pytest.raises(AttributeError):
            cfg.port = 9999  # type: ignore[misc]

    def test_from_env_connect(self, monkeypatch):
        monkeypatch.setenv("ZENOH_CONNECT", "tcp/10.0.0.1:7447,tcp/10.0.0.2:7447")
        monkeypatch.delenv("ZENOH_LISTEN", raising=False)
        monkeypatch.delenv("STRANDS_MESH_PORT", raising=False)

        cfg = MeshConfig.from_env()
        assert cfg.connect == ("tcp/10.0.0.1:7447", "tcp/10.0.0.2:7447")
        assert cfg.listen == ()
        assert cfg.port == DEFAULT_MESH_PORT

    def test_from_env_listen(self, monkeypatch):
        monkeypatch.delenv("ZENOH_CONNECT", raising=False)
        monkeypatch.setenv("ZENOH_LISTEN", "tcp/0.0.0.0:7447")
        monkeypatch.delenv("STRANDS_MESH_PORT", raising=False)

        cfg = MeshConfig.from_env()
        assert cfg.connect == ()
        assert cfg.listen == ("tcp/0.0.0.0:7447",)

    def test_from_env_port(self, monkeypatch):
        monkeypatch.delenv("ZENOH_CONNECT", raising=False)
        monkeypatch.delenv("ZENOH_LISTEN", raising=False)
        monkeypatch.setenv("STRANDS_MESH_PORT", "8888")

        cfg = MeshConfig.from_env()
        assert cfg.port == 8888

    def test_from_env_empty(self, monkeypatch):
        monkeypatch.delenv("ZENOH_CONNECT", raising=False)
        monkeypatch.delenv("ZENOH_LISTEN", raising=False)
        monkeypatch.delenv("STRANDS_MESH_PORT", raising=False)

        cfg = MeshConfig.from_env()
        assert cfg == MeshConfig()


# ---------------------------------------------------------------------------
# get_session / release_session
# ---------------------------------------------------------------------------


class TestGetSession:
    """Tests for session acquisition and ref-counting."""

    def test_returns_session(self, mock_zenoh):
        session = get_session()
        assert session is mock_zenoh["session"]

    def test_refcounting_same_session(self, mock_zenoh):
        s1 = get_session()
        s2 = get_session()
        assert s1 is s2
        # zenoh.open should only be called once
        assert mock_zenoh["module"].open.call_count == 1
        assert session_info()["refs"] == 2

    def test_release_does_not_close_above_zero(self, mock_zenoh):
        get_session()
        get_session()
        release_session()  # refs 2 → 1
        mock_zenoh["session"].close.assert_not_called()
        assert session_info()["active"] is True
        assert session_info()["refs"] == 1

    def test_release_closes_at_zero(self, mock_zenoh):
        get_session()
        release_session()  # refs 1 → 0
        mock_zenoh["session"].close.assert_called_once()
        assert session_info()["active"] is False
        assert session_info()["refs"] == 0

    def test_session_reopens_after_full_release(self, mock_zenoh):
        get_session()
        release_session()
        # Now get a fresh session
        s = get_session()
        assert s is mock_zenoh["session"]
        assert mock_zenoh["module"].open.call_count == 2

    def test_release_noop_when_no_session(self):
        # Should not raise
        release_session()
        assert session_info()["refs"] == 0


class TestGetSessionConfig:
    """Tests for configuration application."""

    def test_explicit_connect_config(self, mock_zenoh):
        cfg = MeshConfig(connect=("tcp/10.0.0.1:7447",))
        get_session(config=cfg)

        # Should have called insert_json5 with connect endpoints
        mock_zenoh["config"].insert_json5.assert_any_call("connect/endpoints", '["tcp/10.0.0.1:7447"]')

    def test_explicit_listen_config(self, mock_zenoh):
        cfg = MeshConfig(listen=("tcp/0.0.0.0:7447",))
        get_session(config=cfg)

        mock_zenoh["config"].insert_json5.assert_any_call("listen/endpoints", '["tcp/0.0.0.0:7447"]')

    def test_auto_mesh_first_process_listens(self, mock_zenoh, monkeypatch):
        monkeypatch.delenv("ZENOH_CONNECT", raising=False)
        monkeypatch.delenv("ZENOH_LISTEN", raising=False)

        get_session()

        # First call to zenoh.open should try listen+connect (auto-mesh)
        first_config = mock_zenoh["module"].Config.return_value
        first_config.insert_json5.assert_any_call("listen/endpoints", '["tcp/127.0.0.1:7447"]')

    def test_auto_mesh_client_fallback(self, mock_zenoh, monkeypatch):
        """When the first open (listen) fails, falls back to client mode."""
        monkeypatch.delenv("ZENOH_CONNECT", raising=False)
        monkeypatch.delenv("ZENOH_LISTEN", raising=False)

        # First open raises (port taken), second succeeds
        mock_zenoh["module"].open.side_effect = [OSError("Address in use"), mock_zenoh["session"]]
        # Need 2 Config instances for the 2 attempts
        cfg1, cfg2 = MagicMock(), MagicMock()
        mock_zenoh["module"].Config.side_effect = [cfg1, cfg2]

        session = get_session()
        assert session is mock_zenoh["session"]

        # Second config should be client mode
        cfg2.insert_json5.assert_any_call("mode", '"client"')


class TestGetSessionDisabled:
    """Tests for disabled/unavailable scenarios."""

    def test_global_kill_switch(self, monkeypatch, mock_zenoh):
        monkeypatch.setenv("STRANDS_MESH", "false")
        assert get_session() is None
        mock_zenoh["module"].open.assert_not_called()

    def test_global_kill_switch_case_insensitive(self, monkeypatch, mock_zenoh):
        monkeypatch.setenv("STRANDS_MESH", "False")
        assert get_session() is None

    def test_zenoh_not_installed(self):
        with patch.object(importlib, "import_module", side_effect=ImportError("no zenoh")):
            assert get_session() is None


class TestSessionInfo:
    """Tests for session_info diagnostic."""

    def test_inactive(self):
        info = session_info()
        assert info["active"] is False
        assert info["refs"] == 0

    def test_active(self, mock_zenoh):
        get_session()
        info = session_info()
        assert info["active"] is True
        assert info["refs"] == 1


class TestThreadSafety:
    """Basic thread-safety smoke tests."""

    def test_concurrent_get_and_release(self, mock_zenoh):
        """Multiple threads acquiring and releasing should not crash."""
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    get_session()
                for _ in range(50):
                    release_session()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # All refs should be released
        info = session_info()
        assert info["refs"] == 0
