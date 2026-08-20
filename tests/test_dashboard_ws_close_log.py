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
