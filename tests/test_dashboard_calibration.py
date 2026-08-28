"""Calibration surface of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_calibration_advice.py, test_dashboard_calibration_run.py, test_dashboard_calibration_view.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import pytest

from strands_robots.dashboard import calibration as calib
from strands_robots.dashboard import calibration_run as cr
from strands_robots.dashboard import joint_silence

# ============================================================================
# from tests/test_dashboard_calibration_advice.py
# Q89: an 'uncalibrated' arm whose calibration file EXISTS must not be sent to recalibrate.
# ============================================================================

LEADER_LOG = [
    "03:59:01 hardware connected",
    "03:59:02 WARNING state probe 'hw_joints' failed, omitted (further failures logged at debug): "
    "RuntimeError: FeetechMotorsBus(Port '/dev/cu.usbmodem5AB01818061', 6x sts3215) has no calibration registered.",
]
ON_DISK = {
    "robots/so101_follower": ["follower", "follower_arm", "leader_arm"],
    "teleoperators/so101_leader": ["leader", "leader_arm"],
}


def test_files_exist_so_the_remedy_is_an_id_fix_not_hardware_work():
    v = joint_silence.classify(LEADER_LOG, ON_DISK)
    assert v["kind"] == "uncalibrated"
    r = v["remedy"]
    assert "id/path mismatch" in r
    assert "teleoperators/so101_leader/leader.json" in r, "name the file that DOES exist"
    assert "robots/so101_follower/follower.json" in r
    # The dangerous sentence must be gone: re-teaching a calibrated arm is physical work on hardware.
    assert "Calibrate this arm" not in r
    assert "BEFORE recalibrating" in r


def test_nothing_on_disk_keeps_the_original_remedy():
    v = joint_silence.classify(LEADER_LOG, {})
    assert "Calibrate this arm" in v["remedy"], "with no evidence, the generic advice is the right advice"
    assert joint_silence.classify(LEADER_LOG)["remedy"].startswith("Calibrate this arm")
    assert joint_silence.calibration_advice(None) is None
    assert joint_silence.calibration_advice({"robots/x": []}) is None, "an empty listing is not evidence"


def test_the_hint_touches_no_other_verdict():
    port = [
        "03:59:02 WARNING state probe 'hw_joints' failed: ConnectionError: Failed to sync read "
        "'Present_Position' on ids=[1,2,3,4,5,6] after 3 tries. [TxRxResult] Port is in use!",
    ]
    v = joint_silence.classify(port, ON_DISK)
    assert v["kind"] == "port_in_use"
    assert "mismatch" not in v["remedy"], "a contended port is not a calibration story"


def test_long_listings_are_summarised_not_dumped():
    big = {f"robots/kind{i}": [f"id{i}"] for i in range(12)}
    r = joint_silence.calibration_advice(big)
    assert "+6 more" in r and len(r) < 800, "a badge is not a directory dump"


# ============================================================================
# from tests/test_dashboard_calibration_run.py
# The calibration wizard's backend (dashboard/calibration_run.py).
# ============================================================================


def test_follower_is_robot_prefix():
    args = cr.cli_args("follower", "so101_follower", "leader_arm", "/dev/cu.usbmodem1")
    assert args == [
        "--robot.type=so101_follower",
        "--robot.id=leader_arm",
        "--robot.port=/dev/cu.usbmodem1",
    ]


def test_leader_is_teleop_prefix():
    args = cr.cli_args("leader", "so101_leader", "leader", "/dev/cu.usbmodem2")
    assert args[0] == "--teleop.type=so101_leader"
    assert all(a.startswith("--teleop.") for a in args)


@pytest.mark.parametrize(
    "role,model,device_id,port",
    [
        ("unpowered", "so101_follower", "x", "/dev/cu.usbmodem1"),  # role is a measurement, not a guess
        ("follower", "", "x", "/dev/cu.usbmodem1"),
        ("follower", "so101_follower", "../escape", "/dev/cu.usbmodem1"),  # id is a FILE NAME
        ("follower", "so101_follower", "", "/dev/cu.usbmodem1"),
        ("follower", "so101_follower", "x", "not-a-device"),
        ("follower", "so101_follower", "x", "/dev/cu.usb modem"),
    ],
)
def test_bad_input_is_refused_with_a_sentence(role, model, device_id, port):
    with pytest.raises(ValueError):
        cr.cli_args(role, model, device_id, port)


# ---------------------------------------------------------------------------
# wizard_step — every phase of the real flow, in the real output's shape
# ---------------------------------------------------------------------------


def step(text, alive=True, returncode=None):
    return cr.wizard_step(text, alive=alive, returncode=returncode)


def test_blank_output_is_starting():
    assert step("")["step"] == "starting"
    assert step("INFO 2026-08-22 loading config\n")["step"] == "starting"


def test_reuse_prompt_is_its_own_step():
    s = step(
        "Press ENTER to use provided calibration file associated with the id leader_arm, or type 'c' and press ENTER to run calibration: "
    )
    assert s["step"] == "reuse"
    assert s["waiting"] is True


def test_middle_prompt():
    s = step(
        "Running calibration of so101_follower\nMove so101_follower to the middle of its range of motion and press ENTER...."
    )
    assert s["step"] == "middle"
    assert s["waiting"] is True
    assert "limp" in s["prompt"]  # the one physical fact the operator must know


def test_recording_parses_the_LAST_table():
    # The table is redrawn in place with cursor-up; the parser must return the newest values.
    text = textwrap.dedent("""\
        Move all joints except 'wrist_roll' sequentially through their entire ranges of motion.
        Recording positions. Press ENTER to stop...

        -------------------------------------------
        NAME            |    MIN |    POS |    MAX
        shoulder_pan    |   2000 |   2000 |   2000
        shoulder_lift   |   1800 |   1800 |   1800
        \x1b[3A
        -------------------------------------------
        NAME            |    MIN |    POS |    MAX
        shoulder_pan    |   1500 |   2100 |   2600
        shoulder_lift   |   1700 |   1900 |   2200
    """)
    s = step(text)
    assert s["step"] == "recording"
    assert s["motors"] == [
        {"name": "shoulder_pan", "min": 1500, "pos": 2100, "max": 2600},
        {"name": "shoulder_lift", "min": 1700, "pos": 1900, "max": 2200},
    ]


def test_saved_wins_over_everything_before_it():
    text = (
        "Move so101_follower to the middle of its range of motion and press ENTER....\n"
        "Recording positions. Press ENTER to stop...\n"
        "Calibration saved to ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/leader_arm.json\n"
    )
    s = step(text, alive=False, returncode=0)
    assert s["step"] == "saved"
    assert s["path"].endswith("leader_arm.json")


def test_usage_screen_is_named_as_our_bug_not_the_arm():
    s = step(
        "usage: lerobot-calibrate [-h] ...\nlerobot-calibrate: error: unrecognized arguments\n",
        alive=False,
        returncode=2,
    )
    assert s["step"] == "failed"
    assert "bug" in s["reason"]


def test_death_without_save_reports_the_last_error_line():
    text = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1\n'
        "ConnectionError: Could not connect on port '/dev/cu.usbmodem1'\n"
    )
    s = step(text, alive=False, returncode=1)
    assert s["step"] == "failed"
    assert "Could not connect" in s["reason"]


def test_one_point_range_refusal_surfaces():
    text = "Recording positions. Press ENTER to stop...\nValueError: Some motors have the same min and max values:\n['gripper']\n"
    s = step(text, alive=False, returncode=1)
    assert s["step"] == "failed"
    # the ValueError sentence is what the operator needs (which motor was skipped)
    assert "same min and max" in s["reason"] or "gripper" in s["reason"]


# ---------------------------------------------------------------------------
# the live session, against a fake CLI speaking the real protocol
# ---------------------------------------------------------------------------

FAKE = textwrap.dedent("""\
    import sys, time
    args = sys.argv[1:]
    assert any(a.startswith("--robot.type=") or a.startswith("--teleop.type=") for a in args), args
    input("Move fake to the middle of its range of motion and press ENTER....")
    print("Recording positions. Press ENTER to stop...")
    print("-------------------------------------------")
    print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
    print(f"{'shoulder_pan':<15} | {1500:>6} | {2100:>6} | {2600:>6}")
    input()
    print("Calibration saved to /tmp/fake/robots/so101_follower/test_arm.json")
""")


def wait_for(run, want, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = run.status()
        if s["step"] == want:
            return s
        if s["step"] == "failed":
            raise AssertionError(f"run failed while waiting for {want}: {s['reason']}\ntail: {s['tail']}")
        time.sleep(0.05)
    raise AssertionError(f"never reached step {want}; last: {run.status()}")


def test_full_session_walks_the_protocol(tmp_path):
    fake = tmp_path / "fake_calibrate.py"
    fake.write_text(FAKE)
    cr.runs.clear()
    run = cr.start(
        role="follower",
        model="so101_follower",
        device_id="test_arm",
        port="/dev/cu.usbmodemFAKE",
        argv=[sys.executable, str(fake)],
    )
    try:
        wait_for(run, "middle")
        run.press("enter")
        s = wait_for(run, "recording")
        assert s["motors"][0]["name"] == "shoulder_pan"
        run.press("enter")
        s = wait_for(run, "saved")
        assert s["path"].endswith("test_arm.json")
        assert s["alive"] is False
    finally:
        run.close()
        cr.runs.clear()


def test_second_wizard_on_same_port_is_refused(tmp_path):
    fake = tmp_path / "fake_calibrate.py"
    fake.write_text(FAKE)
    cr.runs.clear()
    run = cr.start(
        role="follower",
        model="so101_follower",
        device_id="test_arm",
        port="/dev/cu.usbmodemFAKE",
        argv=[sys.executable, str(fake)],
    )
    try:
        with pytest.raises(RuntimeError, match="already running on /dev/cu.usbmodemFAKE"):
            cr.start(
                role="follower",
                model="so101_follower",
                device_id="other",
                port="/dev/cu.usbmodemFAKE",
                argv=[sys.executable, str(fake)],
            )
    finally:
        run.close()
        cr.runs.clear()


def test_cancel_kills_the_child(tmp_path):
    fake = tmp_path / "fake_calibrate.py"
    fake.write_text(FAKE)
    cr.runs.clear()
    run = cr.start(
        role="follower",
        model="so101_follower",
        device_id="test_arm",
        port="/dev/cu.usbmodemFAKE",
        argv=[sys.executable, str(fake)],
    )
    try:
        wait_for(run, "middle")
        run.cancel()
        assert run.alive() is False
        s = run.status()
        assert s["step"] == "failed"  # died without the saved line — named, not a spinner
    finally:
        run.close()
        cr.runs.clear()


# ---------------------------------------------------------------------------
# the routes (guarded server surface) — refusals and the full walk over HTTP
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from strands_robots.dashboard import auth
    from strands_robots.dashboard import server as srv
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}

    class _StubBridge:
        peers: dict = {}

        def snapshot(self):
            return {"peers": {}}

        def start(self, *a, **k):
            pass

        def stop(self, *a, **k):
            pass

    with TestClient(srv.create_app(bridge=_StubBridge())) as c:
        yield c


def test_route_refuses_a_port_held_by_a_live_robot(client, monkeypatch):
    from strands_robots.dashboard.device_manager import DeviceManager

    monkeypatch.setattr(DeviceManager, "port_owner", lambda self, port: "so101-follower")
    r = client.post(
        "/api/calibration/run",
        json={
            "role": "follower",
            "model": "so101_follower",
            "device_id": "x",
            "port": "/dev/cu.usbmodem1",
        },
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "Port is in use" in detail["error"]
    assert "despawn so101-follower" in detail["remedy"]


def test_route_walks_the_whole_protocol(client, monkeypatch, tmp_path):
    fake = tmp_path / "fake_calibrate.py"
    fake.write_text(FAKE)
    monkeypatch.setattr(cr, "_calibrate_argv", lambda: [sys.executable, str(fake)])
    cr.runs.clear()
    try:
        r = client.post(
            "/api/calibration/run",
            json={
                "role": "follower",
                "model": "so101_follower",
                "device_id": "test_arm",
                "port": "/dev/cu.usbmodemFAKE",
            },
        )
        assert r.status_code == 200, r.text
        sid = r.json()["id"]

        deadline = time.time() + 8
        while time.time() < deadline:
            s = client.get(f"/api/calibration/run/{sid}").json()
            if s["step"] == "middle":
                break
            time.sleep(0.05)
        assert s["step"] == "middle", s

        s = client.post(f"/api/calibration/run/{sid}/key", json={"key": "enter"}).json()
        deadline = time.time() + 8
        while time.time() < deadline:
            s = client.get(f"/api/calibration/run/{sid}").json()
            if s["step"] == "saved":
                break
            time.sleep(0.05)
            if s["step"] == "recording":
                client.post(f"/api/calibration/run/{sid}/key", json={"key": "enter"})
        assert s["step"] == "saved", s
        assert s["path"].endswith("test_arm.json")
    finally:
        for run in list(cr.runs.values()):
            run.close()
        cr.runs.clear()


def test_route_422_on_unmeasured_role(client):
    r = client.post(
        "/api/calibration/run",
        json={
            "role": "unknown",
            "model": "so101_follower",
            "device_id": "x",
            "port": "/dev/cu.usbmodem1",
        },
    )
    assert r.status_code == 422
    assert "cannot be guessed" in r.json()["detail"]


def test_route_404_names_the_supersession(client):
    r = client.get("/api/calibration/run/nope")
    assert r.status_code == 404
    assert "superseded" in r.json()["detail"]


# ============================================================================
# from tests/test_dashboard_calibration_view.py
# Q16: the calibration detail route could never succeed.
# ============================================================================


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
