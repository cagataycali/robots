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


# ---------------------------------------------------------------------------
# The third rail: the index is listed, opens, streams - and is a different camera
# ---------------------------------------------------------------------------


_ROSTER = [
    {"listing_index": 0, "name": "Logi 4K Pro"},
    {"listing_index": 1, "name": "neon_net Camera"},
    {"listing_index": 2, "name": "USB2.0_CAM1"},
]


def test_a_renumbered_index_is_reported_with_where_the_camera_went() -> None:
    """The failure the other two rails cannot see, and the only one that produces a BAD dataset.

    Pull the camera at index 1 and the list closes up: index 1 now opens, streams and looks perfect
    while showing what used to be index 2. The remembered device name is the only thing that can
    say so - and the operator's real question is "where is my camera now", so the answer carries it.
    """
    drift = camera_liveness.identity_drift(
        {"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}, _ROSTER
    )
    assert drift == [
        {"camera": "wrist", "index": 1, "remembered": "USB2.0_CAM1", "now": "neon_net Camera", "moved_to": 2}
    ]


def test_two_cameras_with_ONE_name_is_admitted_as_a_guess_not_offered_as_a_fix() -> None:
    """This machine really has two devices both called USB2.0_CAM1.

    Names cannot tell them apart, so proposing an index would be a coin flip dressed as advice.
    """
    roster = [*_ROSTER, {"listing_index": 3, "name": "USB2.0_CAM1"}]
    drift = camera_liveness.identity_drift(
        {"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}, roster
    )
    assert drift[0]["ambiguous"] is True
    assert "moved_to" not in drift[0], "no guessed index"
    text = camera_liveness.drift_refusal(drift, peer_id="arm-1")
    assert "more than one camera is called USB2.0_CAM1" in text


def test_the_same_device_at_the_same_index_is_not_drift() -> None:
    cfg = {"top": {"index_or_path": 0, "device_name": "Logi 4K Pro"}}
    assert camera_liveness.identity_drift(cfg, _ROSTER) == []
    # Whitespace and case are formatting, not a different camera.
    cfg2 = {"top": {"index_or_path": 0, "device_name": "  logi  4K   pro "}}
    assert camera_liveness.identity_drift(cfg2, _ROSTER) == []


def test_no_remembered_name_means_nothing_to_compare() -> None:
    """Most profiles predate the field: a missing memory is not a change."""
    assert camera_liveness.identity_drift({"top": {"index_or_path": 0}}, _ROSTER) == []
    assert camera_liveness.identity_drift({"top": {"index_or_path": 0, "device_name": " "}}, _ROSTER) == []


def test_an_index_the_roster_does_not_list_belongs_to_the_OTHER_rail() -> None:
    """Absence is missing_cameras' verdict. Two rails must not both shout about one fact."""
    cfg = {"wrist": {"index_or_path": 9, "device_name": "USB2.0_CAM1"}}
    assert camera_liveness.identity_drift(cfg, _ROSTER) == []
    assert camera_liveness.missing_cameras(cfg, [0, 1, 2]) == [{"camera": "wrist", "index": 9}]


def test_an_empty_roster_or_nameless_entries_are_not_evidence() -> None:
    cfg = {"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}
    assert camera_liveness.identity_drift(cfg, []) == []
    assert camera_liveness.identity_drift(cfg, None) == []
    assert camera_liveness.identity_drift(cfg, [{"listing_index": 1, "name": "  "}]) == []


def test_the_drift_refusal_explains_why_a_working_camera_is_being_refused() -> None:
    """Refusing something that visibly works needs the reason stated, or it reads as a bug."""
    drift = camera_liveness.identity_drift(
        {"wrist": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}, _ROSTER
    )
    text = camera_liveness.drift_refusal(drift, peer_id="so101-arm-1")
    assert "changed hands" in text
    assert "still opens and still streams" in text and "wrong view" in text
    assert "look perfectly healthy and be unusable" in text
    assert "index 2 now" in text, "say where the camera went"
    assert "ignore_camera_identity" in text
