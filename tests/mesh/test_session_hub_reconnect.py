"""Pin tests for BUGS.md #9 - a dead hub must not leave a dead session.

The first process on ``STRANDS_MESH_PORT`` becomes the listener ("hub");
every later process fails the bind and falls back. The pre-fix fallback
opened a plain CLIENT session with no reconnect: when the hub process
died, every fallen-back peer kept a dead session with no surfaced error
and stayed dark until its process was restarted (observed live twice -
orphaned Robot() children after a dashboard restart).

The fix (both ``get_session`` and its bridge-mode twin
``_get_zenoh_session_directly``) falls back to PEER mode on an ephemeral
listener with a background connect-retry to the hub endpoint, so a
restarted hub re-links automatically.

Proven live (2026-08-19, isolated port 7517, two sim peers + hub
kill/restart): both peers rejoined pub+sub within ~3.5-5s on both the
auto-fallback and explicit-ZENOH_CONNECT paths. These tests pin the
config shape that made that happen, because no integration test can run
zenoh in CI.
"""

from __future__ import annotations

import inspect

from strands_robots.mesh import session as session_mod

RETRY_KEY = "connect/retry"
EXIT_KEY = "connect/exit_on_failure"


def _fallback_block(src: str) -> str:
    """The code AFTER the auto-listener attempt (the fallback path)."""
    assert "if not connect_env and not listen_env:" in src
    tail = src.split("if not connect_env and not listen_env:")[1]
    # Everything after the listener's except-handler is the fallback.
    parts = tail.split("except zenoh_error_types() as exc:", 1)
    assert len(parts) == 2, "listener attempt must have a narrow except"
    return parts[1]


class TestGetSessionFallbackReconnects:
    """``get_session`` fallback keeps retrying the hub endpoint."""

    def test_fallback_sets_connect_retry(self) -> None:
        src = inspect.getsource(session_mod.get_session)
        block = _fallback_block(src)
        assert RETRY_KEY in block, "#9 regression - fallback session no longer retries the hub endpoint"

    def test_fallback_never_gives_up(self) -> None:
        src = inspect.getsource(session_mod.get_session)
        block = _fallback_block(src)
        assert EXIT_KEY in block and '"false"' in block.split(EXIT_KEY)[1].split(")")[0], (
            "#9 regression - fallback session exits on connect failure instead of retrying"
        )

    def test_fallback_is_peer_mode_not_client(self) -> None:
        # Client mode has no listener and (in our shape) no retry; the
        # fallback must keep an ephemeral listener so surviving peers can
        # also route to each other while the hub is down.
        src = inspect.getsource(session_mod.get_session)
        block = _fallback_block(src)
        client_marker = "insert_json5(" + '"mode", \'"client"\'' + ")"
        assert client_marker not in block, "#9 regression - fallback reverted to dead-end client mode"
        assert ":0" in block, "fallback must listen on an ephemeral port (scheme/127.0.0.1:0)"


class TestBridgeTwinFallbackReconnects:
    """``_get_zenoh_session_directly`` (bridge transport) mirrors the fix.

    This path had the pre-fix client-mode fallback until 2026-08-19: a
    BridgeTransport peer whose hub died stayed dark forever even after
    ``get_session`` was fixed.
    """

    def test_twin_fallback_sets_connect_retry(self) -> None:
        src = inspect.getsource(session_mod._get_zenoh_session_directly)
        block = _fallback_block(src)
        assert RETRY_KEY in block, "#9 regression - bridge twin's fallback no longer retries the hub endpoint"

    def test_twin_fallback_never_gives_up(self) -> None:
        src = inspect.getsource(session_mod._get_zenoh_session_directly)
        block = _fallback_block(src)
        assert EXIT_KEY in block

    def test_twin_fallback_is_peer_mode_not_client(self) -> None:
        src = inspect.getsource(session_mod._get_zenoh_session_directly)
        block = _fallback_block(src)
        client_marker = "insert_json5(" + '"mode", \'"client"\'' + ")"
        assert client_marker not in block, "#9 regression - bridge twin reverted to dead-end client mode"


class TestRetryPolicyShape:
    """The retry policy itself: gentle backoff, bounded period."""

    def test_backoff_parameters(self) -> None:
        for fn in (session_mod.get_session, session_mod._get_zenoh_session_directly):
            src = inspect.getsource(fn)
            block = _fallback_block(src)
            assert "period_init_ms" in block and "period_max_ms" in block, (
                f"{fn.__name__} fallback retry must bound its backoff period"
            )
