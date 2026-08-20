"""A rescued arm must not be reported dead by its own bookkeeping.

The camera-degrade recovery ends by recording ``{camera: reason}``. That write is
the last step of something that already worked - the camera was dropped, the
motors are up, the arm is usable - and it sits inside the caller's
``except Exception``, so any failure there is reported as a CONNECT failure and
erases the sentence naming the broken camera.
"""

from __future__ import annotations

from strands_robots.hardware_robot import Robot, _degraded_notes


def _bare_host() -> Robot:
    """A host built the way test harnesses build one: no ``__init__``."""
    return Robot.__new__(Robot)


def test_notes_are_usable_without_init() -> None:
    host = _bare_host()
    _degraded_notes(host).update({"top": "Failed to open top_cam"})
    assert _degraded_notes(host) == {"top": "Failed to open top_cam"}


def test_notes_are_per_instance() -> None:
    a, b = _bare_host(), _bare_host()
    _degraded_notes(a)["top"] = "gone"
    assert _degraded_notes(b) == {}, "a class-level default would share one dict across arms"


def test_notes_survive_a_field_that_is_not_a_dict() -> None:
    host = _bare_host()
    host.__dict__["_degraded_cameras"] = None  # an older field, or a partial restore
    _degraded_notes(host)["wrist"] = "busy"
    assert _degraded_notes(host) == {"wrist": "busy"}


def test_get_status_reports_the_same_book() -> None:
    host = _bare_host()
    _degraded_notes(host)["top"] = "Failed to open top_cam"
    assert host.__dict__["_degraded_cameras"] == {"top": "Failed to open top_cam"}


def test_a_borrowed_connect_path_gets_a_book_too() -> None:
    """Hosts that borrow the connect paths are not Robot instances (our own
    stubs are plain classes), so the book must be reachable for any object."""

    class Borrowed:
        pass

    host = Borrowed()
    _degraded_notes(host)["top"] = "Failed to open top_cam"
    assert _degraded_notes(host) == {"top": "Failed to open top_cam"}


def test_a_host_that_refuses_attributes_still_gets_a_book() -> None:
    class Slotted:
        __slots__ = ()

    assert _degraded_notes(Slotted()) == {}
