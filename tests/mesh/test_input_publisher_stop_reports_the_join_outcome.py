"""``InputPublisher.stop`` reports whether the publish loop actually stopped.

``Thread.join(timeout=...)`` returns ``None`` whether or not the thread
finished, so the liveness read after it is the only thing that distinguishes a
stopped loop from one that outlasted its budget. Before this suite ``stop()``
cleared the session flag, joined with a two-second budget, ignored the outcome
and logged "input publisher stopped" unconditionally - so a teleoperator whose
``get_action()`` was blocking left the loop free to put one more actuator frame
on the wire *after* the caller had been told the publisher stopped, with nothing
in the returned stats able to say so.

Every wedged cell parks the loop inside ``get_action()`` deterministically: the
double signals on entry and then blocks, so the caller knows the loop is in the
read rather than in the ticker before it calls ``stop()``. The budget is
shrunk for those cells so they cost milliseconds, and one cell drives the
shipped value so the budget itself stays graded.
"""

from __future__ import annotations

import ast
import inspect
import logging
import threading
import time
from typing import Any

import pytest

import strands_robots.mesh.input as input_mod
from strands_robots.mesh.input import InputPublisher

# Stated here rather than read off the module so a tree without the fix fails by
# name on the property under test instead of erroring on a missing attribute.
BUDGET_ATTR = "_INPUT_JOIN_TIMEOUT_S"
SHRUNK_BUDGET = 0.2


class _PubMesh:
    """Recording publish chokepoint, stamped so post-stop frames are visible."""

    peer_id = "leader-1"

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any], float]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload, time.monotonic()))


class _WedgingTeleop:
    """A leader whose second read blocks, like a serial read on a wedged bus.

    The first read answers immediately so a frame is published and the loop is
    provably running; the second signals ``entered`` and then blocks, which is
    what lets a test park the loop inside the read with no sleeps.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def get_action(self) -> dict[str, float]:
        self.calls += 1
        if self.calls > 1:
            self.entered.set()
            self.release.wait(30.0)
        return {"j0": 0.4}


class _HealthyTeleop:
    def get_action(self) -> dict[str, float]:
        return {"j0": 0.4}


def _parked(monkeypatch: pytest.MonkeyPatch) -> tuple[InputPublisher, _PubMesh, _WedgingTeleop]:
    """Return a publisher whose loop is parked inside ``get_action()``."""
    monkeypatch.setattr(input_mod, BUDGET_ATTR, SHRUNK_BUDGET, raising=False)
    mesh, tele = _PubMesh(), _WedgingTeleop()
    # peer_id + publish is the whole of Mesh this publisher touches.
    stand_in: Any = mesh
    pub = InputPublisher(stand_in, tele, device_name="leader", method="arm", hz=200.0)
    pub.start()
    assert tele.entered.wait(5.0), "the loop never reached the blocking read"
    return pub, mesh, tele


def _running(mesh: _PubMesh, tele: _HealthyTeleop) -> InputPublisher:
    stand_in: Any = mesh
    pub = InputPublisher(stand_in, tele, device_name="leader", method="arm", hz=200.0)
    pub.start()
    deadline = time.monotonic() + 2.0
    while not mesh.published and time.monotonic() < deadline:
        time.sleep(0.005)
    return pub


class TestAWedgedLoopIsReportedNotAnnounced:
    """The join outcome reaches the caller instead of being discarded."""

    def test_the_stats_report_the_thread_is_still_alive(self, monkeypatch):
        pub, _, tele = _parked(monkeypatch)
        try:
            assert pub.stop()["thread_alive"] is True
        finally:
            tele.release.set()

    def test_the_session_flag_still_reads_stopped(self, monkeypatch):
        # running is the flag stop() clears; thread_alive reads the thread. The
        # two differing is the window a caller needs to see, so running must
        # keep its meaning - the loop's own while reads it.
        pub, _, tele = _parked(monkeypatch)
        try:
            stats = pub.stop()
            assert (stats["running"], stats["thread_alive"]) == (False, True)
        finally:
            tele.release.set()

    def test_the_outcome_is_logged_at_warning(self, monkeypatch, caplog):
        pub, _, tele = _parked(monkeypatch)
        try:
            with caplog.at_level(logging.INFO, logger=input_mod.__name__):
                pub.stop()
            levels = {r.levelno for r in caplog.records}
            assert logging.WARNING in levels, [r.getMessage() for r in caplog.records]
        finally:
            tele.release.set()

    def test_the_warning_names_the_budget_it_waited(self, monkeypatch, caplog):
        pub, _, tele = _parked(monkeypatch)
        try:
            with caplog.at_level(logging.WARNING, logger=input_mod.__name__):
                pub.stop()
            budget = getattr(input_mod, BUDGET_ATTR)
            text = " ".join(r.getMessage() for r in caplog.records)
            assert f"{budget:.1f}s" in text, text
        finally:
            tele.release.set()

    def test_the_warning_names_the_topic_still_being_published_to(self, monkeypatch, caplog):
        pub, _, tele = _parked(monkeypatch)
        try:
            with caplog.at_level(logging.WARNING, logger=input_mod.__name__):
                pub.stop()
            text = " ".join(r.getMessage() for r in caplog.records)
            assert pub.topic in text, text
        finally:
            tele.release.set()

    def test_no_stop_is_announced_that_did_not_happen(self, monkeypatch, caplog):
        pub, _, tele = _parked(monkeypatch)
        try:
            with caplog.at_level(logging.INFO, logger=input_mod.__name__):
                pub.stop()
            stopped_claims = [r.getMessage() for r in caplog.records if "publisher stopped" in r.getMessage()]
            assert stopped_claims == [], stopped_claims
        finally:
            tele.release.set()

    def test_a_frame_reaching_the_wire_after_the_stop_is_accounted_for(self, monkeypatch):
        # The frame the loop publishes once released lands after stop() returned.
        # thread_alive is what lets a caller know that is still possible.
        pub, mesh, tele = _parked(monkeypatch)
        stats = pub.stop()
        returned_at = time.monotonic()
        tele.release.set()
        pub.stop()
        after = [entry for entry in mesh.published if entry[2] > returned_at]
        assert stats["thread_alive"] is True
        assert after, "expected the released loop to publish once more"


class TestAJoinThatTimedOutIsStillStoppable:
    """The live loop stays reachable through the same surface."""

    def test_a_second_stop_rejoins_the_live_loop(self, monkeypatch):
        pub, _, tele = _parked(monkeypatch)
        assert pub.stop()["thread_alive"] is True
        tele.release.set()
        assert pub.stop()["thread_alive"] is False


class TestTheCleanPathIsUnchanged:
    """Every expectation here is one the pre-fix code also met."""

    def test_a_clean_stop_reports_not_running(self):
        mesh = _PubMesh()
        pub = _running(mesh, _HealthyTeleop())
        stats = pub.stop()
        assert stats["running"] is False
        assert mesh.published

    def test_a_clean_stop_returns_promptly(self):
        pub = _running(_PubMesh(), _HealthyTeleop())
        start = time.monotonic()
        pub.stop()
        assert time.monotonic() - start < 1.0

    def test_a_clean_stop_logs_at_info(self, caplog):
        pub = _running(_PubMesh(), _HealthyTeleop())
        with caplog.at_level(logging.INFO, logger=input_mod.__name__):
            pub.stop()
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "publisher stopped" in text, text
        assert logging.WARNING not in {r.levelno for r in caplog.records}

    def test_a_never_started_publisher_returns_stats(self):
        pub = InputPublisher(_PubMesh(), _HealthyTeleop())
        stats = pub.stop()
        assert stats["running"] is False
        assert stats["frames"] == 0

    def test_the_counters_are_still_reported(self):
        mesh = _PubMesh()
        pub = _running(mesh, _HealthyTeleop())
        stats = pub.stop()
        assert set(stats) >= {
            "device",
            "method",
            "running",
            "frames",
            "errors",
            "event_read_errors",
            "hz_actual",
            "hz_target",
        }


class TestThePremises:
    """What the wedged cells rely on, asserted rather than assumed."""

    def test_a_timed_out_stop_leaves_the_flag_clear_and_the_thread_live(self, monkeypatch):
        # Holds either way: it is the state that makes the guard necessary. A
        # guard reading the session flag alone returns early here, leaving the
        # only handle to that live loop unreachable through stop().
        pub, _, tele = _parked(monkeypatch)
        try:
            pub.stop()
            assert pub._running is False
            assert pub._thread is not None and pub._thread.is_alive()
        finally:
            tele.release.set()
            pub.stop()

    def test_the_loop_parks_inside_the_device_read(self, monkeypatch):
        pub, _, tele = _parked(monkeypatch)
        try:
            assert tele.calls >= 2, "the double was not reached a second time"
            assert not tele.release.is_set()
        finally:
            tele.release.set()
            pub.stop()

    def test_the_shrunk_budget_is_shorter_than_the_wedge(self, monkeypatch):
        pub, _, tele = _parked(monkeypatch)
        try:
            start = time.monotonic()
            pub.stop()
            waited = time.monotonic() - start
            assert waited >= SHRUNK_BUDGET
            assert waited < 5.0, "stop() waited past the shrunk budget"
        finally:
            tele.release.set()
            pub.stop()


class TestTheBudgetIsNamedOnce:
    """The budget the docstring, the warning and the tests read is one value."""

    def test_the_join_timeout_is_a_name_not_a_literal(self):
        tree = ast.parse(inspect.getsource(InputPublisher.stop).lstrip())
        joins = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
        ]
        assert len(joins) == 1, ast.unparse(tree)
        timeouts = [kw.value for kw in joins[0].keywords if kw.arg == "timeout"]
        assert len(timeouts) == 1
        assert isinstance(timeouts[0], ast.Name), ast.unparse(timeouts[0])
        assert timeouts[0].id == BUDGET_ATTR

    def test_the_shipped_budget_is_what_an_unshrunk_stop_waits(self):
        mesh, tele = _PubMesh(), _WedgingTeleop()
        pub = InputPublisher(mesh, tele, device_name="leader", method="arm", hz=200.0)
        pub.start()
        assert tele.entered.wait(5.0)
        try:
            start = time.monotonic()
            stats = pub.stop()
            waited = time.monotonic() - start
            budget = getattr(input_mod, BUDGET_ATTR, None)
            assert budget is not None, f"{BUDGET_ATTR} is not named on the module"
            assert stats["thread_alive"] is True
            assert budget <= waited < budget + 2.0
        finally:
            tele.release.set()
            pub.stop()
