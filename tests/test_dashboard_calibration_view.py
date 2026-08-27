"""Q16: the calibration detail route could never succeed.

``lerobot_calibrate("view", name, device_type)`` passes positionally into
``(action, device_type, device_model, device_id, ...)``, so the calibration's
name became the device_type, the query parameter became the device_model,
device_id stayed None, and the tool answered "**view** action requires:
device_type, device_model, and device_id" for every possible input. The drawer
rendered that sentence as if it were calibration data.

Underneath it, the real design bug: a calibration NAME IS NOT AN IDENTITY. On the
machine this was found on, ``leader_arm`` exists three times.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from strands_robots.dashboard import calibration as calib


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A calibration tree shaped like the real one, ambiguity included."""
    files = {
        "robots/so101_follower/follower_arm.json": {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -2048, "range_min": 700, "range_max": 3300},
            "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 1024, "range_min": 800, "range_max": 3000},
            "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 0, "range_min": 900, "range_max": 3100},
        },
        "robots/so101_follower/leader_arm.json": {"shoulder_pan": {"id": 1}},
        "robots/so_follower/leader_arm.json": {"shoulder_pan": {"id": 1}},
        "teleoperators/so101_leader/leader_arm.json": {"shoulder_pan": {"id": 1}},
    }
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
    return tmp_path


# --------------------------------------------------------------------------
# candidates: a name is not an identity
# --------------------------------------------------------------------------


def test_a_unique_name_resolves_to_exactly_one_calibration(root: Path):
    found = calib.candidates("follower_arm", root=root)
    assert len(found) == 1
    assert found[0]["device_type"] == "robots"
    assert found[0]["device_model"] == "so101_follower"
    assert found[0]["device_id"] == "follower_arm"
    assert found[0]["path"].endswith("robots/so101_follower/follower_arm.json")


def test_an_ambiguous_name_returns_every_match_so_the_caller_can_choose(root: Path):
    """The bug this makes visible: picking one silently shows the wrong arm."""
    found = calib.candidates("leader_arm", root=root)
    assert [(f["device_type"], f["device_model"]) for f in found] == [
        ("robots", "so101_follower"),
        ("robots", "so_follower"),
        ("teleoperators", "so101_leader"),
    ]


def test_the_order_is_stable_because_a_directory_listing_is_not(root: Path):
    assert calib.candidates("leader_arm", root=root) == calib.candidates("leader_arm", root=root)


def test_filters_narrow_an_ambiguous_name_to_one(root: Path):
    found = calib.candidates("leader_arm", root=root, device_type="teleoperators")
    assert len(found) == 1 and found[0]["device_model"] == "so101_leader"

    found = calib.candidates("leader_arm", root=root, device_model="so_follower")
    assert len(found) == 1 and found[0]["device_type"] == "robots"


def test_a_filter_that_matches_nothing_finds_nothing(root: Path):
    assert calib.candidates("follower_arm", root=root, device_model="so_follower") == []


def test_an_unknown_name_is_empty_not_an_error(root: Path):
    assert calib.candidates("no_such_arm", root=root) == []


@pytest.mark.parametrize("name", ["../../etc/passwd", "robots/so101_follower/leader_arm", ".hidden", ""])
def test_a_device_id_is_one_path_segment_never_a_traversal(root: Path, name: str):
    assert calib.candidates(name, root=root) == []


def test_a_missing_root_is_empty_not_a_crash(tmp_path: Path):
    assert calib.candidates("anything", root=tmp_path / "nope") == []


# --------------------------------------------------------------------------
# payload / motors: the structured data that was always there
# --------------------------------------------------------------------------


def test_motors_keep_the_file_order_not_alphabetical_order():
    """Dict order is the order of joints on the arm; sorting invents an arm."""
    rows = calib.motors(
        {
            "shoulder_pan": {"id": 1},
            "shoulder_lift": {"id": 2},
            "elbow_flex": {"id": 3},
            "wrist_flex": {"id": 4},
        }
    )
    assert [r["name"] for r in rows] == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


def test_every_motor_field_survives_including_unknown_ones():
    rows = calib.motors({"shoulder_pan": {"id": 1, "drive_mode": 0, "future_field": "keep me"}})
    assert rows[0] == {"name": "shoulder_pan", "id": 1, "drive_mode": 0, "future_field": "keep me"}


def test_a_scalar_motor_entry_is_kept_rather_than_dropped():
    assert calib.motors({"gripper": 512}) == [{"name": "gripper", "value": 512}]


@pytest.mark.parametrize("data", [None, [], "not a mapping", 7])
def test_a_calibration_without_motor_data_yields_no_rows(data):
    assert calib.motors(data) == []


def test_the_datetime_is_converted_explicitly_not_by_default_str():
    info = {
        "device_type": "robots",
        "device_model": "so101_follower",
        "device_id": "follower_arm",
        "path": "/tmp/x.json",
        "size_bytes": 1234,
        "motor_count": 6,
        "modified_time": datetime(2026, 8, 19, 2, 47, 11),
        "data": {"shoulder_pan": {"id": 1}},
    }

    out = calib.payload(info)

    assert out["modified"] == "2026-08-19T02:47:11"
    assert out["modified_epoch"] == datetime(2026, 8, 19, 2, 47, 11).timestamp()
    json.dumps(out)  # the whole payload is serialisable without default=str


def test_the_file_s_own_motor_count_is_reported_even_when_it_disagrees():
    """A mismatch is evidence of a truncated file; smoothing it hides that."""
    out = calib.payload({"motor_count": 6, "data": {"shoulder_pan": {"id": 1}}})
    assert out["motor_count"] == 6
    assert len(out["motors"]) == 1


def test_payload_of_an_empty_info_is_still_shaped():
    out = calib.payload({})
    assert out["motors"] == [] and out["device_id"] is None
    json.dumps(out)
