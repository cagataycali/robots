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
    "OpenCV: not authorized to capture video (status 0), requesting...\nOpenCV: camera failed to properly initialize!\n"
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
    # The remedy has to name the actual place, or it is not a remedy - and it
    # has to be true for THIS case: a daemon-parented dashboard never gets a
    # prompt at all (tccd: "Policy disallows prompt"), so advice that waits for
    # one is worse than none.
    assert "Privacy" in remedy and "Terminal" in remedy
    assert "background daemon" in remedy


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
        probed=[],
        claimed={1: "arm"},
        roster=ROSTER[:2],
        failures={1: BLOCKED_STDERR},
    )
    (row,) = [r for r in rows if r["index"] == 1]
    assert row["state"] == "in_use"  # ownership outranks a stale probe verdict


def test_rows_are_sorted_and_unique():
    rows = cam.merge_cameras(
        probed=[{"index": 2}],
        claimed={0: "a"},
        roster=ROSTER,
        failures={2: ""},
        remembered={9: {"width": 1, "height": 1}},
    )
    indices = [r["index"] for r in rows]
    assert indices == sorted(set(indices)) and 9 in indices


def test_empty_everything_is_an_empty_list_not_a_crash():
    assert cam.merge_cameras(probed=[], claimed={}, roster=[]) == []


# ------------------------------------------------------------------- verdict


def test_blocked_verdict_speaks_once_when_the_machine_is_blocked():
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        roster=ROSTER,
        failures={i: BLOCKED_STDERR for i in range(4)},
    )
    verdict = cam.blocked_verdict(rows)
    assert verdict and verdict["kind"] == "camera_permission"
    assert verdict["indices"] == [0, 1, 2, 3] and verdict["remedy"]


def test_one_working_camera_disproves_a_permission_problem():
    rows = cam.merge_cameras(
        probed=[{"index": 0}],
        claimed={},
        roster=ROSTER,
        failures={1: BLOCKED_STDERR},
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
        probed=[],
        claimed={1: "so101-arm-1", 2: "so101-arm-1"},
        roster=ROSTER,
        streaming=set(),
    )
    states = {r["index"]: r for r in rows if r["index"] in (1, 2)}
    assert [states[1]["state"], states[2]["state"]] == ["assigned", "assigned"]
    assert "no frames are arriving" in states[1]["reason"]
    # The remedy points at the evidence, not at a picture that will never load.
    assert "log" in states[1]["remedy"] and "so101-arm-1" in states[1]["remedy"]


def test_frames_arriving_still_reads_as_in_use():
    (row,) = [
        r
        for r in cam.merge_cameras(
            probed=[],
            claimed={1: "arm"},
            roster=ROSTER[:2],
            streaming={1},
        )
        if r["index"] == 1
    ]
    assert row["state"] == "in_use" and "streaming for arm" in row["reason"]


def test_no_streaming_evidence_keeps_the_kinder_reading():
    """None means nobody told us - absence of evidence is not evidence."""
    (row,) = [r for r in cam.merge_cameras(probed=[], claimed={1: "arm"}, roster=ROSTER[:2]) if r["index"] == 1]
    assert row["state"] == "in_use"


# ---------------------------------------------------------------------------
# A camera that GOES AWAY mid-session is not an index that was always empty
# ---------------------------------------------------------------------------


def test_an_index_we_measured_before_reports_vanished_not_absent() -> None:
    """ "no camera answered at this index" is true here and useless.

    An index that was always empty needs no action. An index where we measured a real camera and now
    get nothing is an EVENT - unplugged, asleep, or dropped by its hub - and it was reading as the
    harmless case while quietly carrying its remembered 1920x1080, so a gap looked like a healthy
    camera with no advice attached.
    """
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        remembered={1: {"width": 1920, "height": 1080, "fps": 30}},
    )
    row = next(r for r in rows if r["index"] == 1)
    assert row["state"] == "vanished"
    assert "answered at this index earlier" in row["reason"]
    assert "unplugged" in row["reason"] and "hub" in row["reason"]
    assert row["available"] is False, "a vanished camera is not available"
    assert row["geometry_from"] == "remembered", "keep the measurement, keep the tag that dates it"


def test_the_vanished_remedy_names_the_index_shift_danger() -> None:
    """The consequence an operator cannot deduce, and must not learn from a ruined dataset.

    macOS camera indices are positions in a list that closes up when a device is removed, so pulling
    one camera can hand its number to another. A remembered resolution is then not proof of identity -
    it is the last thing seen at that POSITION.
    """
    rows = cam.merge_cameras(probed=[], claimed={}, remembered={2: {"width": 640, "height": 480}})
    remedy = next(r for r in rows if r["index"] == 2)["remedy"]
    assert "replug" in remedy
    assert "renumbers" in remedy, "say WHY the index cannot be trusted afterwards"
    assert "preview an index before assigning" in remedy, "give the check that makes it safe"
    assert "wrong view while everything on screen looks healthy" in remedy


def test_an_index_never_measured_still_reports_absent() -> None:
    """The distinction only exists where there is evidence: no memory, no event, no escalation."""
    rows = cam.merge_cameras(probed=[], claimed={}, failures={3: "nothing there"}, max_index=3)
    row = next(r for r in rows if r["index"] == 3)
    assert row["state"] == "absent"
    assert "no camera answered" in row["reason"]
    assert "remedy" not in row, "an empty index asks nothing of the operator"


def test_a_remembered_index_that_opens_now_is_simply_ready() -> None:
    """Memory must not haunt a working camera: a frame right now beats any history."""
    rows = cam.merge_cameras(
        probed=[{"index": 1, "width": 1280, "height": 720, "fps": 30}],
        claimed={},
        remembered={1: {"width": 1920, "height": 1080}},
    )
    row = next(r for r in rows if r["index"] == 1)
    assert row["state"] == "ready" and row["available"] is True
    assert row.get("geometry_from") != "remembered"
    assert row["width"] == 1280, "the fresh measurement wins"


def test_a_remembered_index_that_is_BLOCKED_keeps_the_permission_verdict() -> None:
    """A stderr that explains itself always beats an inference from memory.

    Permission denial has a cure that has nothing to do with cables; calling it "vanished" would send
    the operator to replug a camera that is plugged in and working.
    """
    rows = cam.merge_cameras(
        probed=[],
        claimed={},
        remembered={0: {"width": 1920, "height": 1080}},
        failures={0: "OpenCV: not authorized to capture video (status 0)"},
    )
    row = next(r for r in rows if r["index"] == 0)
    assert row["state"] == "blocked"
    assert "not granted camera access" in row["reason"]
