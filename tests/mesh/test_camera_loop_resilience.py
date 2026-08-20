"""Resilience + clean-shutdown contract for the mesh camera-publish loop.

``Mesh._camera_loop`` is the background thread that publishes camera frames on
the mesh at a fixed rate. Two guarantees matter and are pinned here:

  1. A transient error from a single ``_publish_cameras_once`` tick (a camera
     that momentarily fails to render or JPEG-encode) MUST NOT kill the loop -
     it is logged and the loop keeps publishing on the next tick.
  2. The loop shuts down promptly when ``_stop_event`` is signalled, and paces
     itself at ``period = 1 / hz`` (so stop is observed within an interval rather
     than after a full sleep).

The pacing MECHANISM changed (BUGS.md Q69): ``_stop_event.wait(period)`` is
inflated by ~145ms in a daemon-descended process tree, so a nominal 30fps camera
loop published at about 6fps and reported the 6 as the camera's own limit. The
loop now paces on :class:`strands_robots.mesh.pacing.Ticker`. These tests
therefore script the TICKER instead of the event -- the guarantees above are
unchanged, only the seam they are driven through moved. The period assertion is
kept, because "the loop asks for 1/hz" is the part that must not silently drift.

The loop only touches ``_running``, ``_publish_cameras_once``, ``_stop_event``
and ``peer_id``, so it is exercised on a bare instance built with
``Mesh.__new__`` (the same construction pattern used by the other mesh unit
tests) - no zenoh transport or live robot required.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from strands_robots.mesh.core import Mesh


class _ScriptedTicker:
    """Stands in for :class:`~strands_robots.mesh.pacing.Ticker`.

    ``wait()`` returns the scripted values (True == stop, the same sense the
    event had), so a test can step the loop tick by tick without spending real
    time. Records the period it was constructed with and whether the loop closed
    it -- a real ticker owns a pipe and a selector, so a loop that forgets to
    close it leaks two file descriptors per robot restart.
    """

    instances: list["_ScriptedTicker"] = []

    def __init__(self, period, stop_event=None, **_kw):
        self.period = period
        self.stop_event = stop_event
        self.waits: list[bool] = []
        self.closed = False
        _ScriptedTicker.instances.append(self)

    def wait(self) -> bool:
        value = bool(self._script.pop(0)) if self._script else True
        self.waits.append(value)
        return value

    def close(self) -> None:
        self.closed = True


def _bare_mesh(stop_waits, publish):
    """A Mesh with just the attributes ``_camera_loop`` reads.

    Args:
        stop_waits: return values for successive pacing waits; the loop breaks on
            the first truthy one.
        publish: the ``_publish_cameras_once`` callable (a mock).
    """
    mesh = Mesh.__new__(Mesh)
    mesh.peer_id = "test__arm"
    mesh._running = True
    mesh._publish_cameras_once = publish
    mesh._stop_event = threading.Event()
    _ScriptedTicker.instances.clear()
    _ScriptedTicker._script = list(stop_waits)
    return mesh


def test_camera_loop_publishes_each_tick_and_stops_on_event():
    publish = MagicMock()
    # Two ticks proceed, the third wait signals stop.
    mesh = _bare_mesh([False, False, True], publish)

    with patch("strands_robots.mesh.core.Ticker", _ScriptedTicker):
        mesh._camera_loop(10.0)

    assert publish.call_count == 3
    # Paces at period = 1 / hz so a stop is observed within one interval.
    ticker = _ScriptedTicker.instances[0]
    assert ticker.period == 0.1
    assert ticker.stop_event is mesh._stop_event, "the loop must still stop on the mesh's own event"
    assert ticker.closed, "the loop must close its ticker (it owns a pipe and a selector)"


def test_camera_loop_swallows_tick_error_and_keeps_going():
    # Every tick raises; the loop must log and continue rather than die on the
    # first failure. Stop is signalled after the second tick.
    publish = MagicMock(side_effect=RuntimeError("camera render blipped"))
    mesh = _bare_mesh([False, True], publish)

    # No exception escapes the loop.
    with patch("strands_robots.mesh.core.Ticker", _ScriptedTicker):
        mesh._camera_loop(20.0)

    # It kept publishing after the first error (resilience), then stopped.
    assert publish.call_count == 2
    ticker = _ScriptedTicker.instances[0]
    assert ticker.period == 0.05
    assert ticker.closed, "a loop that dies on an error path must still release its ticker"
