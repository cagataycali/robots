#!/usr/bin/env python3
"""Tests for strands_robots.mesh_session — Zenoh session singleton.

All tests mock ``zenoh`` so no network or eclipse-zenoh installation
is required.  Tests verify ref-counting, fork-detection, environment
variable configuration, and thread safety.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh_session import MeshSession


@pytest.fixture(autouse=True)
def _clean_session():
    """Ensure MeshSession is reset before and after each test."""
    MeshSession._reset()
    yield
    MeshSession._reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_zenoh():
    """Build a mock ``zenoh`` module with open() and Config."""
    mock_zenoh = MagicMock()
    mock_session = MagicMock()
    mock_zenoh.open.return_value = mock_session
    mock_config = MagicMock()
    mock_zenoh.Config.return_value = mock_config
    return mock_zenoh, mock_session, mock_config


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Verify open / close / refcount behaviour."""

    @patch.dict(os.environ, {}, clear=False)
    def test_open_returns_session(self):
        mock_zenoh, mock_session, _ = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            session = MeshSession.open()

        assert session is mock_session
        assert MeshSession.is_open()
        assert MeshSession.refcount() == 1

    @patch.dict(os.environ, {}, clear=False)
    def test_second_open_reuses_session(self):
        mock_zenoh, mock_session, _ = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            s1 = MeshSession.open()
            s2 = MeshSession.open()

        assert s1 is s2
        assert MeshSession.refcount() == 2
        mock_zenoh.open.assert_called_once()  # only one real open

    @patch.dict(os.environ, {}, clear=False)
    def test_close_decrements_refcount(self):
        mock_zenoh, mock_session, _ = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open()
            MeshSession.open()

        MeshSession.close()
        assert MeshSession.refcount() == 1
        assert MeshSession.is_open()
        mock_session.close.assert_not_called()

    @patch.dict(os.environ, {}, clear=False)
    def test_close_at_zero_closes_session(self):
        mock_zenoh, mock_session, _ = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open()

        MeshSession.close()
        assert MeshSession.refcount() == 0
        assert not MeshSession.is_open()
        mock_session.close.assert_called_once()

    @patch.dict(os.environ, {}, clear=False)
    def test_close_when_already_zero_is_noop(self):
        MeshSession.close()  # should not raise
        assert MeshSession.refcount() == 0


# ---------------------------------------------------------------------------
# Import failure (zenoh not installed)
# ---------------------------------------------------------------------------


class TestZenohNotInstalled:
    """Verify graceful degradation when eclipse-zenoh is absent."""

    def test_returns_none_when_zenoh_missing(self):
        with patch.dict("sys.modules", {"zenoh": None}):
            # importlib.import_module will raise ImportError for None
            with patch("importlib.import_module", side_effect=ImportError("No module named 'zenoh'")):
                # We need to ensure zenoh isn't importable
                import sys

                original = sys.modules.get("zenoh")
                sys.modules["zenoh"] = None
                try:
                    session = MeshSession.open()
                    assert session is None
                    assert not MeshSession.is_open()
                finally:
                    if original is not None:
                        sys.modules["zenoh"] = original
                    else:
                        sys.modules.pop("zenoh", None)


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------


class TestEnvConfig:
    """Verify ZENOH_CONNECT and ZENOH_LISTEN are applied."""

    @patch.dict(os.environ, {"ZENOH_CONNECT": "tcp/10.0.0.1:7447"}, clear=False)
    def test_connect_env_applied(self):
        mock_zenoh, _, mock_config = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open()

        mock_config.insert_json5.assert_any_call("connect/endpoints", '["tcp/10.0.0.1:7447"]')

    @patch.dict(os.environ, {"ZENOH_LISTEN": "tcp/0.0.0.0:7448"}, clear=False)
    def test_listen_env_applied(self):
        mock_zenoh, _, mock_config = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open()

        mock_config.insert_json5.assert_any_call("listen/endpoints", '["tcp/0.0.0.0:7448"]')

    @patch.dict(
        os.environ,
        {"ZENOH_CONNECT": "tcp/a:1,tcp/b:2"},
        clear=False,
    )
    def test_multiple_connect_endpoints(self):
        mock_zenoh, _, mock_config = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open()

        mock_config.insert_json5.assert_any_call("connect/endpoints", '["tcp/a:1", "tcp/b:2"]')


# ---------------------------------------------------------------------------
# Programmatic config overrides
# ---------------------------------------------------------------------------


class TestConfigOverrides:
    """Verify programmatic config_overrides are applied."""

    @patch.dict(os.environ, {}, clear=False)
    def test_overrides_applied(self):
        mock_zenoh, _, mock_config = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open(config_overrides={"mode": "client"})

        mock_config.insert_json5.assert_any_call("mode", '"client"')


# ---------------------------------------------------------------------------
# Fork detection
# ---------------------------------------------------------------------------


class TestForkDetection:
    """Verify session re-init when PID changes (simulated fork)."""

    @patch.dict(os.environ, {}, clear=False)
    def test_pid_change_reinitialises(self):
        mock_zenoh, mock_session, _ = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open()
            assert mock_zenoh.open.call_count == 1

            # Simulate fork — PID changes
            MeshSession._pid = -1

            MeshSession.open()
            # Should have opened a new session
            assert mock_zenoh.open.call_count == 2
            assert MeshSession.refcount() == 1  # reset, not incremented


# ---------------------------------------------------------------------------
# Open failure (RuntimeError)
# ---------------------------------------------------------------------------


class TestOpenFailure:
    """Verify RuntimeError when zenoh.open() fails."""

    @patch.dict(os.environ, {}, clear=False)
    def test_raises_runtime_error_on_open_failure(self):
        mock_zenoh = MagicMock()
        mock_zenoh.open.side_effect = Exception("Connection refused")
        mock_zenoh.Config.return_value = MagicMock()

        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            with pytest.raises(RuntimeError, match="Failed to open Zenoh session"):
                MeshSession.open()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Verify concurrent open/close doesn't corrupt state."""

    @patch.dict(os.environ, {}, clear=False)
    def test_concurrent_open_close(self):
        mock_zenoh, mock_session, _ = _make_mock_zenoh()
        errors = []

        def opener():
            try:
                with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
                    MeshSession.open()
            except Exception as e:
                errors.append(e)

        def closer():
            try:
                MeshSession.close()
            except Exception as e:
                errors.append(e)

        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            # Open 10 times concurrently
            threads = [threading.Thread(target=opener) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert not errors, f"Errors in concurrent open: {errors}"
        assert MeshSession.refcount() == 10

        # Close 10 times concurrently
        threads = [threading.Thread(target=closer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors in concurrent close: {errors}"
        assert MeshSession.refcount() == 0
        assert not MeshSession.is_open()


# ---------------------------------------------------------------------------
# atexit cleanup
# ---------------------------------------------------------------------------


class TestAtexitCleanup:
    """Verify atexit hook cleans up properly."""

    @patch.dict(os.environ, {}, clear=False)
    def test_atexit_closes_session(self):
        mock_zenoh, mock_session, _ = _make_mock_zenoh()
        with patch.dict("sys.modules", {"zenoh": mock_zenoh}):
            MeshSession.open()

        MeshSession._atexit_cleanup()
        assert not MeshSession.is_open()
        assert MeshSession.refcount() == 0
        mock_session.close.assert_called_once()

    def test_atexit_noop_when_no_session(self):
        MeshSession._atexit_cleanup()  # should not raise
