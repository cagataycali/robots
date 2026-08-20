"""A socket's death has to be visible, or a storm looks like popularity.

Q40 produced 63,906 `connection open` lines and ZERO closes, because ws_camera's
`except (WebSocketDisconnect, RuntimeError): pass` swallowed the end of every stream.
The question that would have solved it in one grep — did these sockets ever send a
frame? — had no answer on the machine.
"""

from __future__ import annotations

from strands_robots.dashboard.ws_observability import (
    CloseLogThrottle,
    close_line,
    close_verdict,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestVerdict:
    def test_a_stream_that_worked_says_so_with_numbers(self) -> None:
        v = close_verdict(frames_sent=1800, lifetime_s=120.0, publishing=True)
        # Q51 gave the lifetime a decimal (a socket that lived 0.4s used to read as "0s")
        assert "1800 frames" in v and "120.0s" in v and "15.0 fps" in v

    def test_the_q40_case_names_the_cause_and_not_the_symptom(self) -> None:
        # exactly the incident: accepted, nothing published, closed in milliseconds
        v = close_verdict(frames_sent=0, lifetime_s=0.05, publishing=False)
        assert "not publishing" in v
        assert "robot may not be running" in v, "the operator needs the next step, not a status"

    def test_a_client_that_hung_up_is_not_blamed_on_the_camera(self) -> None:
        v = close_verdict(frames_sent=0, lifetime_s=0.4, publishing=True)
        assert "client hung up" in v
        assert "retry loop with no backoff" in v, "name the bug this log is here to catch"

    def test_publishing_but_nothing_delivered_is_its_own_case(self) -> None:
        # the honest "I do not know why" bucket - a real state, not a guess
        v = close_verdict(frames_sent=0, lifetime_s=45.0, publishing=True)
        assert "not reaching this socket" in v

    def test_every_verdict_mentions_how_long_it_lived(self) -> None:
        for kwargs in (
            {"frames_sent": 5, "lifetime_s": 3.0, "publishing": True},
            {"frames_sent": 0, "lifetime_s": 3.0, "publishing": False},
            {"frames_sent": 0, "lifetime_s": 0.2, "publishing": True},
            {"frames_sent": 0, "lifetime_s": 99.0, "publishing": True},
        ):
            assert "s" in close_verdict(**kwargs)  # type: ignore[arg-type]


class TestThrottle:
    def test_the_first_close_is_always_logged(self) -> None:
        t = CloseLogThrottle(window_s=60.0, clock=_Clock())
        assert t.should_log("so101-arm-1/top") == (True, 0)

    def test_a_storm_is_collapsed_but_counted(self) -> None:
        clock = _Clock()
        t = CloseLogThrottle(window_s=60.0, clock=clock)
        t.should_log("a/top")
        for _ in range(4000):  # the incident's rate
            log_now, _ = t.should_log("a/top")
            assert log_now is False
        clock.t += 60.0
        log_now, suppressed = t.should_log("a/top")
        assert log_now is True
        assert suppressed == 4000, "the count is the whole point: silence would hide the storm"

    def test_the_counter_resets_after_it_is_reported(self) -> None:
        clock = _Clock()
        t = CloseLogThrottle(window_s=10.0, clock=clock)
        t.should_log("k")
        t.should_log("k")
        clock.t += 10.0
        assert t.should_log("k") == (True, 1)
        clock.t += 10.0
        assert t.should_log("k") == (True, 0)

    def test_cameras_are_throttled_independently(self) -> None:
        t = CloseLogThrottle(window_s=60.0, clock=_Clock())
        assert t.should_log("arm-1/top")[0] is True
        assert t.should_log("arm-1/wrist")[0] is True, "one noisy tile must not silence another"

    def test_the_key_set_is_bounded(self) -> None:
        t = CloseLogThrottle(window_s=60.0, clock=_Clock())
        for i in range(2000):  # a spawn loop can invent peer ids
            t.should_log(f"peer-{i}/top")
        assert len(t._seen) <= 256


class TestLine:
    def test_it_names_the_socket_and_the_verdict(self) -> None:
        line = close_line(peer_id="so101-arm-1", cam="top", verdict="sent nothing in 0.1s", suppressed=0)
        assert "so101-arm-1/top" in line and "sent nothing" in line
        assert "suppressed" not in line

    def test_a_storm_count_is_in_the_line_itself(self) -> None:
        line = close_line(peer_id="p", cam="c", verdict="v", suppressed=4000)
        assert "+4000 more closes suppressed" in line


# --- Q51: bytes, not just frames ------------------------------------------------
def test_verdict_carries_the_volume_a_socket_actually_moved():
    """A frame count cannot tell a buggy client from a link that cannot keep up."""
    line = close_verdict(frames_sent=42, lifetime_s=9.0, publishing=True, bytes_sent=4_075_761)
    assert "42 frames" in line and "3.9 MB" in line and "4.7 fps" in line and "0.43 MB/s" in line


def test_volume_is_optional_and_never_divides_by_a_zero_lifetime():
    """Callers that do not measure bytes, and the socket that closed instantly."""
    assert close_verdict(frames_sent=3, lifetime_s=2.0, publishing=True) == "streamed 3 frames over 2.0s (1.5 fps)"
    instant = close_verdict(frames_sent=1, lifetime_s=0.0, publishing=True, bytes_sent=1000)
    assert "streamed 1 frames" in instant and "fps" not in instant and "MB/s" not in instant


# --- Q52: a viewer may ask for fewer frames; it may never ask for more -------------
def test_a_cap_is_honoured_within_sane_bounds():
    from strands_robots.dashboard.ws_observability import MAX_CAP_FPS, MIN_CAP_FPS, fps_cap

    assert fps_cap("1") == 1.0
    assert fps_cap("0.5") == 0.5
    assert fps_cap("1000") == MAX_CAP_FPS, "an absurd number is clamped, not trusted"
    assert fps_cap("0.0001") == MIN_CAP_FPS, "a 'cap' that freezes the tile helps nobody"


def test_nonsense_never_becomes_a_request_for_more():
    """The failure mode that matters: a bad value must fall back to today's behaviour."""
    from strands_robots.dashboard.ws_observability import fps_cap

    for raw in (None, "", "abc", "-5", "0", "nan", "1e400x"):
        assert fps_cap(raw) is None, raw
    assert fps_cap("inf") == 30.0, "infinity clamps to the ceiling rather than dividing by nothing"


def test_the_verdict_says_which_rate_the_socket_agreed_to():
    from strands_robots.dashboard.ws_observability import cap_note

    assert cap_note(None) == ""
    assert "1 fps" in cap_note(1.0) and "2.5 fps" in cap_note(2.5)


class TestTheLineActuallyReachesTheLog:
    """MEASURED 2026-08-20 on the live dashboard: 75,489 `connection open` lines and
    ZERO `connection closed` lines in one process lifetime.

    The two strings both come from the `websockets` library (server.py logs the open,
    protocol.discard() logs the close), and the close only fires when the sans-io close
    path reaches EOF — which in this deployment it never did, not once in 75k sockets.
    That asymmetry is exactly why a storm burning 20.7 GB stayed invisible for 12 hours:
    the log recorded every socket's birth and no socket's death, so churn, lifetime and
    cause were all unanswerable from it.

    So our close verdict must NOT depend on the library's logging. It is emitted from the
    handler's own `finally`, and this test pins that — including the abrupt-disconnect case,
    which is the only case the live rig has ever actually produced.
    """

    def _app(self, monkeypatch, tmp_path):
        from strands_robots.dashboard import auth
        from strands_robots.dashboard import settings as dsettings
        from strands_robots.dashboard.server import create_app

        monkeypatch.setenv("STRANDS_MESH", "false")
        monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
        monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
        dsettings._cache = None
        auth._cache_key = None
        auth._cache = {}
        app = create_app()
        return app

    def test_an_abrupt_disconnect_still_logs_a_verdict(self, monkeypatch, tmp_path, caplog):
        import logging

        from fastapi.testclient import TestClient

        app = self._app(monkeypatch, tmp_path)
        # A camera that publishes nothing: the Q50 shape, and the one where the handler
        # has no failing send to notice the client left.
        monkeypatch.setattr(app.state.bridge, "latest_frame", lambda *a, **k: None)
        client = TestClient(app)
        with caplog.at_level(logging.INFO):
            with client.websocket_connect("/ws/camera/so101-arm-1/top"):
                pass  # closes immediately - the abrupt case
        lines = [r.getMessage() for r in caplog.records]
        assert any("camera socket so101-arm-1/top closed" in m for m in lines), lines[-6:]

    def test_a_socket_that_sent_nothing_is_logged_as_a_WARNING(self, monkeypatch, tmp_path, caplog):
        """Severity is the whole point: a storm of zero-frame sockets must be visible at
        the level an operator actually reads, not buried in INFO next to 75k opens.

        Uses a DIFFERENT camera than the test above on purpose: the per-identity rate
        limiter suppressed this close when both tests shared `arm-1/top`, which is the
        limiter doing exactly its job (one storm, one line) - and is worth stating, since
        a future reader will otherwise 'fix' the flake by weakening it.
        """
        import logging

        from fastapi.testclient import TestClient

        app = self._app(monkeypatch, tmp_path)
        monkeypatch.setattr(app.state.bridge, "latest_frame", lambda *a, **k: None)
        client = TestClient(app)
        with caplog.at_level(logging.INFO):
            with client.websocket_connect("/ws/camera/so101-arm-2/wrist"):
                pass
        closes = [r for r in caplog.records if "closed" in r.getMessage()]
        assert closes and closes[-1].levelno == logging.WARNING
