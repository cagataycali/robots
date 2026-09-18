"""``look_at`` is one verb: pixel geometry resolved privately (reads only), then one bounded goto."""

import pytest

from strands_robots.drivers.reachy import ReachyDriver


def test_look_at_is_the_only_pixel_verb_and_the_planner_is_private():
    # One verb, one meaning: ``look_at`` moves the head. The geometry step is a
    # private helper - no public ``plan_look_at`` verb or ``dry_run`` switch for
    # an agent to confuse with motion.
    schema = ReachyDriver().tool_spec["inputSchema"]["json"]
    assert "look_at" in schema["properties"]["action"]["enum"]
    assert "plan_look_at" not in schema["properties"]["action"]["enum"]
    assert "dry_run" not in schema["properties"]
    assert schema["properties"]["frame_width"]["type"] == "integer"
    assert not hasattr(ReachyDriver, "plan_look_at")
    plan = ReachyDriver._look_at_pose.__doc__ or ""
    assert "never command the head" in plan


def test_plan_requires_connection():
    result = ReachyDriver()._look_at_pose(640, 360, 1280, 720)
    assert result["status"] == "error"
    assert "not connected" in result["content"][0]["text"]


@pytest.mark.parametrize(
    "coords", [(True, 10, 1280, 720), (10, 20, None, 720), (-1, 10, 1280, 720), (1280, 10, 1280, 720)]
)
def test_plan_domain_precedes_any_network(coords, monkeypatch):
    driver = ReachyDriver()
    driver._connected = True
    monkeypatch.setattr(driver, "_daemon_get", lambda *a: pytest.fail("invalid input hit network"))
    assert driver._look_at_pose(*coords)["status"] == "error"
    assert driver.look_at(*coords)["status"] == "error"


@pytest.fixture
def geometry():
    import numpy as np

    specs = {
        "name": "wireless",
        "available_resolutions": [{"width": 1280, "height": 720, "crop_factor": 1.0}],
        "K": [[3840.0, 0.0, 1920.0], [0.0, 2592.0, 1296.0], [0.0, 0.0, 1.0]],
        "D": [0.0] * 5,
    }
    return specs, {"m": np.eye(4).flatten().tolist()}


def test_principal_pixel_points_forward_and_never_moves(geometry):
    import numpy as np

    from strands_robots.drivers.reachy_look_at import _pixel_plan

    plan = _pixel_plan(*geometry, 640, 360, 1280, 720)
    assert np.allclose(plan["head_pose"], np.eye(4))
    assert "motion_executed" not in plan
    assert plan["safety_validated"] is False
    assert plan["frame_pose_synchronized"] is False


@pytest.mark.parametrize("u,v,axis,sign", [(740, 360, 1, -1), (540, 360, 1, 1), (640, 460, 2, -1), (640, 260, 2, 1)])
def test_optical_pixel_axes_map_to_head_axes(geometry, u, v, axis, sign):
    import numpy as np

    from strands_robots.drivers.reachy_look_at import _pixel_plan

    plan = _pixel_plan(*geometry, u, v, 1280, 720)
    rotation = np.array(plan["head_pose"])[:3, :3]
    ray = rotation[:, 0]
    assert ray[axis] * sign > 0
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_crop_is_applied_and_translation_policy_is_explicit(geometry):
    import numpy as np

    from strands_robots.drivers.reachy_look_at import _pixel_plan

    specs, pose = geometry
    pose["m"][3] = 0.02
    first = _pixel_plan(specs, pose, 740, 360, 1280, 720)
    specs["available_resolutions"][0]["crop_factor"] = 2
    zoomed = _pixel_plan(specs, pose, 740, 360, 1280, 720)
    assert abs(zoomed["head_pose"][1][0]) < abs(first["head_pose"][1][0])
    assert zoomed["reference_head_pose"][0][3] == 0.02
    assert np.array(zoomed["head_pose"])[:3, 3].tolist() == [0, 0, 0]
    assert zoomed["translation_policy"] == "recenter_at_origin"


def test_current_head_rotation_is_used(geometry):
    import numpy as np

    from strands_robots.drivers.reachy_look_at import _pixel_plan

    specs, pose = geometry
    # 90 degrees around world Z: head forward is now world left.
    matrix = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    pose["m"] = matrix.flatten().tolist()
    result = _pixel_plan(specs, pose, 640, 360, 1280, 720)
    assert np.allclose(np.array(result["head_pose"])[:3, 0], [0, 1, 0])


@pytest.mark.parametrize(
    "fault", ["camera", "resolution", "ambiguous", "crop", "focal", "nan", "bool", "distortion", "pose", "reflection"]
)
def test_invalid_calibration_or_pose_is_not_guessed(geometry, fault):
    from strands_robots.drivers.reachy_look_at import _pixel_plan

    specs, pose = geometry
    if fault == "camera":
        specs["name"] = "unknown"
    elif fault == "resolution":
        specs["available_resolutions"] = []
    elif fault == "ambiguous":
        specs["available_resolutions"].append({"width": 1280, "height": 720, "crop_factor": 2})
    elif fault == "crop":
        specs["available_resolutions"][0]["crop_factor"] = float("inf")
    elif fault == "focal":
        specs["K"][0][0] = 0
    elif fault == "nan":
        specs["K"][0][0] = float("nan")
    elif fault == "bool":
        specs["K"][0][0] = True
    elif fault == "distortion":
        specs["D"] = [0, 0, 0]
    elif fault == "pose":
        pose["m"] = [0] * 16
    elif fault == "reflection":
        pose["m"][0] = -1
    with pytest.raises(ValueError):
        _pixel_plan(specs, pose, 640, 360, 1280, 720)


def test_agent_look_at_reads_geometry_then_sends_exactly_one_goto(geometry, monkeypatch):
    from strands import Agent

    driver = ReachyDriver()
    driver._connected = True
    reads = []
    posts = []

    def get(path):
        reads.append(path)
        return geometry[0] if path == "/api/camera/specs" else geometry[1]

    def post(path, body=None, *a, **k):
        posts.append((path, body))
        return {"uuid": "move-1"}

    monkeypatch.setattr(driver, "_daemon_get", get)
    monkeypatch.setattr(driver, "_daemon_post", post)
    monkeypatch.setattr(driver, "_send_cmd", lambda *a, **k: pytest.fail("look_at used the real-time link"))
    agent = Agent(tools=[driver], callback_handler=None)
    result = agent.tool.reachy_mini(action="look_at", u=640, v=360, frame_width=1280, frame_height=720)
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    # The principal pixel is straight ahead: the bounded target is the level pose.
    assert payload["target_deg"] == {"roll": pytest.approx(0.0), "pitch": pytest.approx(0.0), "yaw": pytest.approx(0.0)}
    assert payload["motion_verified"] is False
    assert "dry_run" not in payload
    assert reads == ["/api/camera/specs", "/api/state/present_head_pose?use_pose_matrix=true"]
    assert [p for p, _ in posts] == ["/api/move/goto"]
    assert not hasattr(driver, "look_at_image")


def test_a_pixel_beyond_the_pitch_envelope_is_refused_before_any_goto(geometry, monkeypatch):
    driver = ReachyDriver()
    driver._connected = True
    specs, pose = geometry
    monkeypatch.setattr(driver, "_daemon_get", lambda path: specs if path == "/api/camera/specs" else pose)
    monkeypatch.setattr(driver, "_daemon_post", lambda *a, **k: pytest.fail("an out-of-envelope look_at was sent"))
    # v far above the frame centre with a short focal length asks for far more
    # pitch than the platform has (the fixture's K is a 3840 px focal length at
    # 1280 wide, so the crop-scaled focal length is 1280 px: v=0 -> ~15 deg,
    # within limits; shrink the focal length so the same pixel asks for >40 deg).
    specs["K"][1][1] = 300.0
    result = driver.look_at(640, 0, 1280, 720)
    assert result["status"] == "error"
    assert "pitch" in result["content"][0]["text"]


def test_small_fk_matrix_error_is_reported_not_silently_projected(geometry):
    from strands_robots.drivers.reachy_look_at import _pixel_plan

    specs, pose = geometry
    pose["m"][0] = 1.0002
    plan = _pixel_plan(specs, pose, 640, 360, 1280, 720)
    assert plan["reference_head_pose"][0][0] == 1.0002
    assert 0 < plan["reference_rotation_orthogonality_error"] < 0.001
    assert plan["safety_validated"] is False


def test_gross_fk_scale_error_still_refuses(geometry):
    from strands_robots.drivers.reachy_look_at import _pixel_plan

    specs, pose = geometry
    pose["m"][0] = 1.02
    with pytest.raises(ValueError, match="transform"):
        _pixel_plan(specs, pose, 640, 360, 1280, 720)


def test_antipodal_ray_has_a_proper_rotation(geometry):
    import numpy as np

    from strands_robots.drivers.reachy_look_at import _pixel_plan

    specs, pose = geometry
    pose["m"][0] = pose["m"][5] = -1
    result = _pixel_plan(specs, pose, 640, 360, 1280, 720)
    rotation = np.array(result["head_pose"])[:3, :3]
    assert np.allclose(rotation[:, 0], [-1, 0, 0])
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1)


@pytest.mark.parametrize("xy", [[10.0, 10.0], [float("nan"), 0.0]])
def test_lens_inversion_failure_is_not_a_pinhole_fallback(geometry, monkeypatch, xy):
    import cv2
    import numpy as np

    from strands_robots.drivers.reachy_look_at import _pixel_plan

    monkeypatch.setattr(cv2, "undistortPointsIter", lambda *a: np.array([[xy]]))
    with pytest.raises(ValueError, match="no pinhole fallback"):
        _pixel_plan(*geometry, 640, 360, 1280, 720)


def test_missing_cv_dependency_becomes_driver_refusal(geometry, monkeypatch):
    from strands_robots.drivers import reachy_look_at

    d = ReachyDriver()
    d._connected = True
    monkeypatch.setattr(d, "_daemon_get", lambda path: geometry[0] if path == "/api/camera/specs" else geometry[1])

    original = reachy_look_at.require_optional

    def missing(name, *a, **k):
        if name == "cv2":
            raise ImportError("cv2 is required for pixel planning")
        return original(name, *a, **k)

    monkeypatch.setattr(reachy_look_at, "require_optional", missing)
    result = d.look_at(640, 360, 1280, 720)
    assert result["status"] == "error"
    assert "cv2" in result["content"][0]["text"]


def test_playback_and_volume_refuse_before_any_daemon_touch_when_disconnected():
    from strands_robots.tools.reachy.reachy_actions import reachy_play_sound, reachy_volume

    d = ReachyDriver()
    sound = reachy_play_sound(d, "impatient1.wav")
    volume = reachy_volume(d, 50, allow_test_sound=True)
    assert sound["status"] == volume["status"] == "error"
    assert "not connected" in sound["content"][0]["text"]
    assert "not connected" in volume["content"][0]["text"]


def test_a_volume_write_names_the_test_sound_it_would_play():
    from strands_robots.tools.reachy.reachy_actions import reachy_volume

    d = ReachyDriver()
    d._connected = True
    refused = reachy_volume(d, 50)
    assert refused["status"] == "error"
    assert "test sound" in refused["content"][0]["text"]
    assert "allow_test_sound" in refused["content"][0]["text"]
