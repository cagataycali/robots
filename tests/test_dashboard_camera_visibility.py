"""A camera is never omitted just because it could not be opened (U14).

The defect these tests pin down was measured live: six real cameras on the
operator's Mac showed up in /api/devices as two bare indices, because
``scan_cameras`` dropped every index whose probe failed. "Held by a running
robot", "blocked by macOS privacy" and "unplugged" all rendered as an absence,
which is the one thing they are not.
"""

from __future__ import annotations

from strands_robots.dashboard import cameras as cam

# Verbatim from this machine, probing an index while the process lacked the
# macOS camera permission. The wording is the whole diagnosis and it only ever
# appears on stderr.
BLOCKED_STDERR = (
    "OpenCV: not authorized to capture video (status 0), requesting...\n"
    "OpenCV: camera failed to properly initialize!\n"
)
DEAD_STDERR = "OpenCV: camera failed to properly initialize!\n"
BUSY_STDERR = "VIDEOIO ERROR: V4L: can't open camera by index 0: Device or resource busy\n"

ROSTER = [
    {"listing_index": 0, "name": "USB2.0_CAM1"},
    {"listing_index": 1, "name": "USB2.0_CAM1"},
    {"listing_index": 2, "name": "Logi 4K Pro"},
    {"listing_index": 3, "name": "neon_net Camera"},
]


# ------------------------------------------------------------- classification


def test_permission_denial_is_not_a_missing_camera():
    state, reason, remedy = cam.classify_probe_stderr(BLOCKED_STDERR)
    assert state == "blocked"
    assert "camera access" in reason
    # The remedy has to name the actual place, or it is not a remedy.
    assert "Privacy" in remedy and "restart" in remedy


def test_busy_device_is_in_use_not_absent():
    state, reason, remedy = cam.classify_probe_stderr(BUSY_STDERR)
    assert state == "in_use"
    assert "holding the camera" in reason and remedy


def test_initialise_failure_is_unreadable():
    assert cam.classify_probe_stderr(DEAD_STDERR)[0] == "unreadable"


def test_silence_means_absent():
    state, reason, remedy = cam.classify_probe_stderr("")
    assert (state, remedy) == ("absent", None)
    assert "no camera answered" in reason


def test_classification_is_case_insensitive_and_none_safe():
    assert cam.classify_probe_stderr("NOT AUTHORIZED to capture")[0] == "blocked"
    assert cam.classify_probe_stderr(None)[0] == "absent"  # type: ignore[arg-type]


# --------------------------------------------------------------------- merge


def test_a_claimed_camera_keeps_the_size_we_measured_before():
    rows = cam.merge_cameras(
        probed=[],
        claimed={1: "so101-arm-1"},
        roster=ROSTER,
        remembered={1: {"index": 1, "width": 1920, "height": 1080, "fps": 30.0}},
    )
    (row,) = [r for r in rows if r["index"] == 1]
    assert row["state"] == "in_use" and row["claimed_by"] == "so101-arm-1"
    assert (row["width"], row["height"]) == (1920, 1080)
    # Honest about provenance: it is what we last saw, not a fresh measurement.
    assert row["geometry_from"] == "remembered"
    assert "despawn so101-arm-1" in row["remedy"]


def test_every_roster_camera_gets_a_row_even_when_all_probes_fail():
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        roster=ROSTER,
        failures={i: BLOCKED_STDERR for i in range(4)},
    )
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert {r["state"] for r in rows} == {"blocked"}
    assert all(r["available"] is False for r in rows)


def test_the_live_shape_of_this_machine_lists_four_not_two():
    """Regression for the reported symptom, with the real numbers."""
    rows = cam.merge_cameras(
        probed=[{"index": 0, "width": 640, "height": 480, "fps": 30.0}],
        claimed={1: "so101-arm-1", 2: "so101-arm-1"},
        roster=ROSTER,
        failures={3: DEAD_STDERR},
    )
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert [r["state"] for r in rows] == ["ready", "in_use", "in_use", "unreadable"]
    assert rows[0]["available"] is True and rows[0]["reason"].startswith("opened")


def test_a_name_is_labelled_as_a_guess():
    rows = cam.merge_cameras(probed=[{"index": 2}], claimed={}, roster=ROSTER)
    # Every roster index gets a row - that is the fix - so pick the one probed.
    (row,) = [r for r in rows if r["index"] == 2]
    assert row["name_hint"] == "Logi 4K Pro"
    # Listing order != OpenCV order, and this machine has two identical names.
    assert row["name_is_guess"] is True


def test_an_index_nobody_probed_says_unknown_not_absent():
    (row,) = cam.merge_cameras(probed=[], claimed={}, roster=[ROSTER[3]])
    assert row["state"] == "unknown" and "not probed" in row["reason"]


def test_claimed_indices_are_never_reported_as_failures():
    rows = cam.merge_cameras(
        probed=[], claimed={1: "arm"}, roster=ROSTER[:2], failures={1: BLOCKED_STDERR},
    )
    (row,) = [r for r in rows if r["index"] == 1]
    assert row["state"] == "in_use"  # ownership outranks a stale probe verdict


def test_rows_are_sorted_and_unique():
    rows = cam.merge_cameras(
        probed=[{"index": 2}], claimed={0: "a"}, roster=ROSTER, failures={2: ""},
        remembered={9: {"width": 1, "height": 1}},
    )
    indices = [r["index"] for r in rows]
    assert indices == sorted(set(indices)) and 9 in indices


def test_empty_everything_is_an_empty_list_not_a_crash():
    assert cam.merge_cameras(probed=[], claimed={}, roster=[]) == []


# ------------------------------------------------------------------- verdict


def test_blocked_verdict_speaks_once_when_the_machine_is_blocked():
    rows = cam.merge_cameras(
        probed=[], claimed={}, roster=ROSTER, failures={i: BLOCKED_STDERR for i in range(4)},
    )
    verdict = cam.blocked_verdict(rows)
    assert verdict and verdict["kind"] == "camera_permission"
    assert verdict["indices"] == [0, 1, 2, 3] and verdict["remedy"]


def test_one_working_camera_disproves_a_permission_problem():
    rows = cam.merge_cameras(
        probed=[{"index": 0}], claimed={}, roster=ROSTER, failures={1: BLOCKED_STDERR},
    )
    assert cam.blocked_verdict(rows) is None


def test_no_verdict_without_evidence():
    rows = cam.merge_cameras(probed=[], claimed={}, roster=ROSTER, failures={0: DEAD_STDERR})
    assert cam.blocked_verdict(rows) is None
    assert cam.blocked_verdict([]) is None


# ----------------------------------------------------- configured vs streaming


def test_a_configured_camera_with_no_frames_is_assigned_not_streaming():
    """Exactly the live state on this Mac: both arm cameras were configured and
    neither opened, so the arm dropped them and published nothing."""
    rows = cam.merge_cameras(
        probed=[], claimed={1: "so101-arm-1", 2: "so101-arm-1"}, roster=ROSTER,
        streaming=set(),
    )
    states = {r["index"]: r for r in rows if r["index"] in (1, 2)}
    assert [states[1]["state"], states[2]["state"]] == ["assigned", "assigned"]
    assert "no frames are arriving" in states[1]["reason"]
    # The remedy points at the evidence, not at a picture that will never load.
    assert "log" in states[1]["remedy"] and "so101-arm-1" in states[1]["remedy"]


def test_frames_arriving_still_reads_as_in_use():
    (row,) = [
        r for r in cam.merge_cameras(
            probed=[], claimed={1: "arm"}, roster=ROSTER[:2], streaming={1},
        ) if r["index"] == 1
    ]
    assert row["state"] == "in_use" and "streaming for arm" in row["reason"]


def test_no_streaming_evidence_keeps_the_kinder_reading():
    """None means nobody told us - absence of evidence is not evidence."""
    (row,) = [
        r for r in cam.merge_cameras(probed=[], claimed={1: "arm"}, roster=ROSTER[:2])
        if r["index"] == 1
    ]
    assert row["state"] == "in_use"
