"""Q32 regression: a leaked session or non-daemon thread must be REPORTED."""

from tests.session_leak import leak_report


def test_clean_run_says_nothing() -> None:
    # Silence is the whole point: a clean suite must not grow noise.
    assert leak_report(session_open=False, threads=[]) == []
    assert leak_report(session_open=False, threads=["MainThread"]) == []


def test_open_session_is_named_with_its_consequence() -> None:
    out = "\n".join(leak_report(session_open=True, threads=[]))
    assert "Q32" in out
    assert "STILL OPEN" in out
    # the reason a reader should care, not just the fact
    assert "live fleet" in out and "real hardware" in out


def test_stuck_thread_is_counted_and_named() -> None:
    out = "\n".join(leak_report(session_open=False, threads=["mesh-heartbeat-x", "ThreadPoolExecutor-0_0"]))
    assert "2 non-daemon thread(s)" in out
    assert "ThreadPoolExecutor-0_0" in out
    assert "hangs the run forever" in out


def test_main_thread_is_not_a_leak() -> None:
    assert leak_report(session_open=False, threads=["MainThread", "MainThread"]) == []
    out = leak_report(session_open=False, threads=["MainThread", "worker"])
    assert any("1 non-daemon thread(s)" in ln for ln in out)


def test_both_problems_are_reported_together() -> None:
    out = "\n".join(leak_report(session_open=True, threads=["worker"]))
    assert "STILL OPEN" in out and "non-daemon thread" in out
