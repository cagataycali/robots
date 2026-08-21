"""The second liveness rail: an index the machine does not list at all.

Its sibling ``dead_cameras`` judges FRAME AGE, and cannot see a camera that never published a frame -
which is precisely a camera unplugged before the arm ever subscribed. These tests pin the independent
evidence (the OS enumeration) and, just as importantly, pin where it must stay silent: a failed scan
is not a camera-less machine, and an index list says nothing about a device path.
"""

from __future__ import annotations

from strands_robots.dashboard import camera_liveness

# ---------------------------------------------------------------------------
# The hole the frame-age rail admits: a camera that never published at all
# ---------------------------------------------------------------------------


_CFG = {"top": {"index_or_path": 0}, "wrist": {"index_or_path": 1}}


def test_an_index_the_machine_does_not_list_is_positive_evidence() -> None:
    """dead_cameras cannot catch this case, by design, and it is the worst one.

    A camera unplugged BEFORE the arm subscribed has no frame history, so it has no age and cannot
    be judged stale - the session would start with an image stream that never existed. The OS
    enumeration is independent evidence: if the index is not listed, nothing is there to open.
    """
    missing = camera_liveness.missing_cameras(_CFG, [0, 2])
    assert missing == [{"camera": "wrist", "index": 1}]


def test_an_empty_or_absent_scan_is_not_evidence_of_an_empty_machine() -> None:
    """A scan that found nothing is far more often a failed scan than a camera-less Mac.

    Refusing a session on that would make this gate the thing that blocks work, which is how a
    safety check gets switched off permanently.
    """
    assert camera_liveness.missing_cameras(_CFG, []) == []
    assert camera_liveness.missing_cameras(_CFG, None) == []


def test_a_camera_configured_by_path_is_not_judged_by_an_index_list() -> None:
    """Absence from an index list says nothing about /dev/video0 or a named device."""
    cfg = {"top": {"index_or_path": "/dev/video9"}, "side": {"index_or_path": True}}
    assert camera_liveness.missing_cameras(cfg, [0, 1]) == []


def test_the_refusal_names_the_renumbering_trap_not_just_the_gap() -> None:
    """"Put it back and press record" is the wrong fix and the tempting one.

    Removing a camera renumbers the rest, so after a replug the same index may be a different
    camera - recording then captures the wrong view while every surface looks healthy. The refusal
    has to say RESCAN, and it has to stay continuable like every other gate here.
    """
    text = camera_liveness.missing_refusal([{"camera": "wrist", "index": 1}], peer_id="so101-arm-1")
    assert "not listed by this machine at all" in text
    assert "wrist (index 1)" in text
    assert "RESCAN before recording" in text
    assert "renumbers" in text and "wrong view" in text
    assert "ignore_missing_cameras" in text, "a gate with no way past it gets disabled wholesale"


def test_both_rails_stay_independent() -> None:
    """A camera can be listed and dead, or unlisted and never-seen - one must not mask the other."""
    # t must be a real stamp: camera_age reads t<=0 as UNKNOWN, not as ancient - a zero is a missing
    # clock, and treating it as a two-day-old frame would refuse a session on a formatting quirk.
    listed_but_dead = camera_liveness.dead_cameras(_CFG, {"top": {"t": 1_000.0}}, now=100_000.0)
    assert [d["camera"] for d in listed_but_dead] == ["top"]
    assert camera_liveness.missing_cameras(_CFG, [0, 1]) == [], "top is listed: not this rail's fault"
