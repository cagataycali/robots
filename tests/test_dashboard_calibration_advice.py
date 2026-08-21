"""Q89: an 'uncalibrated' arm whose calibration file EXISTS must not be sent to recalibrate."""
from strands_robots.dashboard import joint_silence

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
    assert joint_silence.classify(LEADER_LOG) ["remedy"].startswith("Calibrate this arm")
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
