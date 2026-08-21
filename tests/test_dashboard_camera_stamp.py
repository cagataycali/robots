"""The WRITE side of camera identity: remembering which device an index was.

``identity_drift`` (rail 3) can only speak if something recorded the roster name an index carried
when the operator picked it. These tests pin that write -- and above all the rule that makes the
whole arc work rather than silently defeating it: a stamp is never overwritten, because a memory
that re-agrees with the present on every respawn is not a memory.

They also pin the strip. The dashboard's own bookkeeping key must never reach a robot process:
``hardware_robot._build_camera_config`` refuses any key ``OpenCVCameraConfig`` does not declare, so
an unstripped annotation kills EVERY camera on the arm instead of degrading one.
"""

from __future__ import annotations

from strands_robots.dashboard import camera_liveness

_ROSTER = [
    {"listing_index": 0, "name": "USB2.0_CAM1"},
    {"listing_index": 1, "name": "Logi 4K Pro"},
]


def test_a_numeric_index_remembers_the_device_that_answered_it() -> None:
    out = camera_liveness.stamp_device_names({"top": {"index_or_path": 1, "fps": 30}}, _ROSTER)
    assert out["top"] == {"index_or_path": 1, "fps": 30, "device_name": "Logi 4K Pro"}


def test_an_existing_stamp_is_never_overwritten() -> None:
    """The load-bearing rule: a respawn after a renumber must not erase the evidence.

    If the stamp refreshed itself, index 1 would come back describing whichever camera slid into
    that slot, ``identity_drift`` would find nothing to compare, and the wrong view would be
    recorded with every surface healthy -- the exact failure rail 3 exists to catch.
    """
    configured = {"top": {"index_or_path": 1, "device_name": "USB2.0_CAM1"}}
    out = camera_liveness.stamp_device_names(configured, _ROSTER)
    assert out["top"]["device_name"] == "USB2.0_CAM1"
    # and the stamp it refused to write is exactly what rail 3 then reports
    assert camera_liveness.identity_drift(out, _ROSTER) == [
        {"camera": "top", "index": 1, "remembered": "USB2.0_CAM1", "now": "Logi 4K Pro", "moved_to": 0}
    ]


def test_no_roster_means_no_stamp_rather_than_a_guess() -> None:
    """An empty roster is far more often a FAILED SCAN than a camera-less Mac."""
    configured = {"top": {"index_or_path": 0}}
    for roster in (None, (), [{"listing_index": 0}], [{"listing_index": 0, "name": "  "}], ["junk"]):
        assert camera_liveness.stamp_device_names(configured, roster) is configured


def test_what_cannot_be_named_by_an_index_is_left_alone() -> None:
    configured = {
        "path": {"index_or_path": "/dev/video9"},   # absence from an index list says nothing
        "boolish": {"index_or_path": True},          # True == 1 must not inherit index 1's name
        "unlisted": {"index_or_path": 7},            # nothing at 7 to remember
        "broken": "not-a-mapping",
    }
    assert camera_liveness.stamp_device_names(configured, _ROSTER) is configured


def test_the_input_is_never_mutated_and_untouched_entries_are_shared() -> None:
    cfg = {"index_or_path": 0}
    configured = {"top": cfg, "path": {"index_or_path": "/dev/video9"}}
    out = camera_liveness.stamp_device_names(configured, _ROSTER)
    assert cfg == {"index_or_path": 0}, "a profile often holds this very dict"
    assert out is not configured
    assert out["path"] is configured["path"]


def test_nothing_to_stamp_returns_the_same_object() -> None:
    for configured in (None, {}, "junk", {"top": {"index_or_path": 0, "device_name": "USB2.0_CAM1"}}):
        assert camera_liveness.stamp_device_names(configured, _ROSTER) is configured


def test_the_annotation_is_stripped_before_a_camera_config_reaches_a_robot() -> None:
    stamped = {"top": {"index_or_path": 1, "fps": 30, "device_name": "Logi 4K Pro"}}
    assert camera_liveness.without_annotations(stamped) == {"top": {"index_or_path": 1, "fps": 30}}
    # the driver's own keys survive untouched, including type
    keep = {"top": {"index_or_path": 1, "width": 640, "height": 480, "type": "opencv"}}
    assert camera_liveness.without_annotations(keep) is keep


def test_stripping_leaves_non_mapping_entries_and_empties_alone() -> None:
    for cameras in (None, {}, "junk"):
        assert camera_liveness.without_annotations(cameras) is cameras
    mixed = {"top": {"index_or_path": 0, "device_name": "USB2.0_CAM1"}, "broken": 3}
    assert camera_liveness.without_annotations(mixed) == {"top": {"index_or_path": 0}, "broken": 3}


def test_a_stripped_config_carries_only_keys_the_camera_driver_declares() -> None:
    """The reason the strip exists, asserted against the real refusal rule.

    ``_build_camera_config`` accepts the declared fields of ``OpenCVCameraConfig`` plus ``type``;
    anything else raises for the whole robot. So: whatever this dashboard adds for its own memory
    must be listed in ANNOTATION_KEYS and removed here.
    """
    stamped = camera_liveness.stamp_device_names({"top": {"index_or_path": 0}}, _ROSTER)
    child = camera_liveness.without_annotations(stamped)
    assert not (set(child["top"]) & set(camera_liveness.ANNOTATION_KEYS))
    assert set(child["top"]) == {"index_or_path"}
