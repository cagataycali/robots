"""U19 v1: changing a robot's cameras is a respawn - one named, refuse-first operation.

Peers take cameras only at spawn. Before this, "change the wrist camera's fps"
meant despawn + re-type the whole spawn form; and a camera entry of ``3`` instead
of ``{"index_or_path": 3}`` was first judged by the CHILD - a ValueError after
the route had answered 200 + pid (cagatay hit exactly this live). The law here:
an invalid config is refused BEFORE the running peer is touched - a refusal must
never cost the operator the process they already had.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard.device_manager import DeviceManager, validate_cameras


class TestValidateCameras:
    def test_none_detaches_everything_and_is_legal(self) -> None:
        assert validate_cameras(None) is None

    def test_a_lerobot_shaped_config_passes(self) -> None:
        assert validate_cameras({
            "top": {"index_or_path": 0, "fps": 30, "width": 1280, "height": 720},
            "wrist": {"index_or_path": "/dev/video1"},
        }) is None

    def test_the_live_crash_shape_is_refused_with_the_example(self) -> None:
        # The exact config that killed a child after 200+pid: a bare int.
        bad = validate_cameras({"main": 3})
        assert bad is not None
        assert "mapping" in bad["error"] and "index_or_path" in bad["error"]

    def test_a_non_dict_config_is_refused(self) -> None:
        assert validate_cameras([0, 1]) is not None

    def test_missing_index_or_path_is_refused_naming_the_camera(self) -> None:
        bad = validate_cameras({"top": {"fps": 30}})
        assert bad is not None and "top" in bad["error"] and "index_or_path" in bad["error"]

    @pytest.mark.parametrize("iop", [True, -1, 1.5, None])
    def test_index_or_path_must_be_an_index_or_a_path(self, iop: object) -> None:
        assert validate_cameras({"top": {"index_or_path": iop}}) is not None

    @pytest.mark.parametrize(
        ("field", "value"),
        [("fps", 0), ("fps", 241), ("fps", "30"), ("fps", True),
         ("width", 8), ("width", 100000), ("height", 5000)],
    )
    def test_fantasy_settings_are_refused_by_bound_not_by_driver(self, field: str, value: object) -> None:
        bad = validate_cameras({"top": {"index_or_path": 0, field: value}})
        assert bad is not None and field in bad["error"]

    def test_omitted_fields_mean_driver_defaults(self) -> None:
        assert validate_cameras({"top": {"index_or_path": 0}}) is None


class TestSpawnRefusesBadCamerasBeforePopen:
    def test_the_child_never_sees_the_bare_int_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        import strands_robots.dashboard.device_manager as mod

        monkeypatch.setattr(
            mod.subprocess, "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("Popen reached")),
        )
        result = dm.spawn("so101", "sim", cameras={"main": 3})
        assert "error" in result and "pid" not in result
        assert dm.robots == {}


class TestReconfigureCameras:
    def _fake_managed(self, dm: DeviceManager, peer_id: str = "so101-sim-1") -> None:
        import strands_robots.dashboard.device_manager as mod

        m = mod.ManagedRobot(peer_id=peer_id, robot_name="so101", mode="sim")
        dm.robots[peer_id] = m

    def test_an_unknown_peer_is_a_refusal_not_a_spawn(self, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        result = dm.reconfigure_cameras("ghost-1", {"top": {"index_or_path": 0}})
        assert "error" in result and "unknown managed peer" in result["error"]

    def test_an_invalid_config_never_touches_the_running_peer(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # THE law of this feature: refusal before destruction.
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        self._fake_managed(dm)
        monkeypatch.setattr(
            dm, "despawn",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("despawn reached for an invalid config")),
        )
        result = dm.reconfigure_cameras("so101-sim-1", {"main": 3})
        assert "error" in result
        assert "so101-sim-1" in dm.robots, "the running peer must survive a refused reconfigure"

    def test_a_replay_job_is_not_a_respawnable_robot(self, tmp_path) -> None:
        import strands_robots.dashboard.device_manager as mod

        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        dm.robots["replay-1"] = mod.ManagedRobot(peer_id="replay-1", robot_name="so101", mode="replay")
        result = dm.reconfigure_cameras("replay-1", None)
        assert "error" in result and "replay" in result["error"]

    def test_a_valid_reconfigure_despawns_then_respawns_under_the_same_id(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        dm = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
        self._fake_managed(dm)
        calls: list[tuple] = []

        def fake_despawn(peer_id):
            calls.append(("despawn", peer_id))
            dm.robots.pop(peer_id, None)
            return {"peer_id": peer_id, "stopped": True}

        def fake_spawn(robot_name, mode, peer_id=None, port=None, cameras=None, robot_id=None, remember=True):
            calls.append(("spawn", robot_name, mode, peer_id, cameras, remember))
            return {"peer_id": peer_id, "pid": 4242, "mode": mode}

        monkeypatch.setattr(dm, "despawn", fake_despawn)
        monkeypatch.setattr(dm, "spawn", fake_spawn)
        new_cams = {"wrist": {"index_or_path": 1, "fps": 60}}
        result = dm.reconfigure_cameras("so101-sim-1", new_cams)
        assert result.get("reconfigured") is True and result.get("pid") == 4242
        assert calls[0] == ("despawn", "so101-sim-1")
        assert calls[1] == ("spawn", "so101", "sim", "so101-sim-1", new_cams, True), (
            "the identity of the spawn must be the OLD peer's, only the cameras change, "
            "and remember=True so the profile keeps the change across replugs"
        )


class TestUnknownOptionsAreRefusedBeforeAnythingStops:
    """An unknown camera option cost the operator a WORKING arm (U19 backend verify, 2026-08-20).

    validate_cameras bounds-checked index_or_path/fps/width/height and let every other key through.
    hardware_robot._build_camera_config refuses unknown keys (deliberately — a silently dropped option
    reports success while the camera streams at the default), but it only speaks inside the CHILD, and
    reconfigure_cameras despawns the running robot BEFORE spawning the replacement. So "framerate" instead
    of "fps" meant: arm killed, respawn dead with a ValueError in a log ring, and a 200 from the route.

    This class pins the promise validate_cameras' own docstring makes: everything the child would refuse
    is refused here, before a process exists.
    """

    def test_a_wrong_option_name_is_refused_and_named(self) -> None:
        bad = validate_cameras({"wrist": {"index_or_path": 1, "framerate": 60}})
        assert bad is not None
        assert "framerate" in bad["error"], "name the option the operator actually typed"
        assert "fps" in bad["error"], "and the accepted set, which is what tells them the right word"
        assert "despawn" in bad["error"], "say why refusing early matters: a reconfigure stops the robot first"
        # Deliberately NOT asserting a "did you mean 'fps'" here: difflib does not consider 'framerate'
        # close to 'fps' (measured), and writing the assertion first is what caught me claiming a
        # suggestion the code cannot make. The accepted list carries the answer instead.

    def test_a_near_miss_does_get_a_suggestion(self) -> None:
        """Where difflib CAN help, it should — a one-character slip is the common case."""
        bad = validate_cameras({"wrist": {"index_or_path": 1, "widht": 640}})
        assert bad is not None and "Did you mean" in bad["error"] and "'width'" in bad["error"]

    def test_an_unknown_option_with_no_near_match_still_names_the_accepted_set(self) -> None:
        bad = validate_cameras({"top": {"index_or_path": 0, "zoom_factor": 3}})
        assert bad is not None and "zoom_factor" in bad["error"]
        for field in ("fps", "width", "height", "index_or_path"):
            assert field in bad["error"]

    def test_every_real_lerobot_option_is_accepted(self) -> None:
        """The refusal must not become a whitelist that fights the driver it wraps."""
        full = {
            "top": {
                "index_or_path": 0, "fps": 30, "width": 640, "height": 480,
                "color_mode": "rgb", "rotation": 90, "warmup_s": 1, "backend": "any", "type": "opencv",
            }
        }
        assert validate_cameras(full) is None

    def test_the_frozen_fallback_field_list_matches_lerobot(self) -> None:
        """The fallback exists for a machine with no robot stack; a stale list would refuse a legal option."""
        dataclasses = pytest.importorskip("dataclasses")
        cfgmod = pytest.importorskip("lerobot.cameras.opencv.configuration_opencv")
        real = tuple(sorted(f.name for f in dataclasses.fields(cfgmod.OpenCVCameraConfig)))
        from strands_robots.dashboard.device_manager import _CAMERA_OPTION_FIELDS

        assert tuple(sorted(_CAMERA_OPTION_FIELDS)) == real, (
            "lerobot's camera options changed: update _CAMERA_OPTION_FIELDS, the list used when "
            "lerobot is not importable"
        )
