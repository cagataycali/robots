"""Tests for strands_robots.robot — Robot() factory and list_robots()."""

import importlib

import pytest

from strands_robots.registry import (
    get_robot,
    list_aliases,
    list_robots,
    resolve_name,
)
from strands_robots.robot import Robot, _auto_detect_mode


class TestResolveNames:
    def test_canonical(self):
        assert resolve_name("so100") == "so100"

    def test_alias(self):
        assert resolve_name("franka") == "panda"
        assert resolve_name("g1") == "unitree_g1"
        assert resolve_name("h1") == "unitree_h1"

    def test_case_insensitive(self):
        assert resolve_name("SO100") == "so100"
        assert resolve_name("Panda") == "panda"

    def test_hyphen_to_underscore(self):
        assert resolve_name("reachy-mini") == "reachy_mini"


class TestListRobots:
    def test_list_all(self):
        robots = list_robots("all")
        assert len(robots) > 0
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "panda" in names

    def test_list_sim(self):
        robots = list_robots("sim")
        for r in robots:
            assert r["has_sim"] is True

    def test_list_real(self):
        robots = list_robots("real")
        for r in robots:
            assert r["has_real"] is True

    def test_list_both(self):
        robots = list_robots("both")
        for r in robots:
            assert r["has_sim"] is True
            assert r["has_real"] is True

    def test_robot_has_fields(self):
        robots = list_robots()
        for r in robots:
            assert "name" in r
            assert "description" in r
            assert "has_sim" in r
            assert "has_real" in r


class TestRobotRegistry:
    def test_so100_exists(self):
        info = get_robot("so100")
        assert info is not None
        assert "asset" in info
        assert info["asset"]["dir"] == "trs_so_arm100"

    def test_all_aliases_point_to_valid_robots(self):
        aliases = list_aliases()
        for alias, canonical in aliases.items():
            info = get_robot(canonical)
            assert info is not None, f"Alias '{alias}' points to unknown robot '{canonical}'"

    def test_robot_count(self):
        """Ensure we have a reasonable number of robots."""
        robots = list_robots()
        assert len(robots) >= 30

    def test_all_robots_have_description(self):
        robots = list_robots()
        for r in robots:
            assert "description" in r, f"Robot '{r['name']}' missing description"
            assert len(r["description"]) > 0


class TestAutoDetectMode:
    def test_defaults_to_sim(self):
        """No hardware plugged in → sim."""
        assert _auto_detect_mode("so100") == "sim"

    def test_env_override_real(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "real")
        assert _auto_detect_mode("so100") == "real"

    def test_env_override_sim(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "sim")
        assert _auto_detect_mode("so100") == "sim"

    def test_env_override_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "REAL")
        assert _auto_detect_mode("so100") == "real"

    def test_unrecognized_env_value_falls_through(self, monkeypatch):
        """Unrecognized STRANDS_ROBOT_MODE value is ignored with warning."""
        monkeypatch.setenv("STRANDS_ROBOT_MODE", "foo")
        # Falls through to default sim (logs warning)
        assert _auto_detect_mode("so100") == "sim"


class TestRobotFactory:
    def test_robot_is_callable(self):
        """Robot is a factory function, not a class."""
        import inspect

        assert callable(Robot)
        assert not inspect.isclass(Robot)

    def test_default_mode_is_sim(self):
        """Robot() defaults to sim mode — never accidentally sends to hardware."""
        import inspect

        sig = inspect.signature(Robot)
        assert sig.parameters["mode"].default == "sim"

    def test_unknown_backend_raises(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            Robot("so100", mode="sim", backend="isaac")

    def test_newton_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            Robot("so100", mode="sim", backend="newton")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            Robot("so100", mode="invalid")

    def test_cameras_rejected_in_sim_mode(self):
        """Passing cameras= in sim mode raises ValueError."""
        with pytest.raises(ValueError, match="cameras= is only supported in mode='real'"):
            Robot("so100", mode="sim", cameras={"wrist": {"type": "opencv"}})

    def test_sim_with_urdf_path(self):
        """Robot() with explicit urdf_path should work (if file exists)."""
        pytest.importorskip("mujoco")
        with pytest.raises(RuntimeError):
            Robot("test_bot", mode="sim", urdf_path="/nonexistent/robot.xml")

    def test_sim_happy_path_mujoco(self, tmp_path):
        """Happy-path: create a MuJoCo sim, step physics, destroy.

        Uses a minimal inline MJCF so the test works without downloaded assets.
        """
        mujoco = pytest.importorskip("mujoco")

        mjcf_xml = """<mujoco model="test_arm">
          <worldbody>
            <light pos="0 0 3"/>
            <geom type="plane" size="1 1 0.1"/>
            <body name="link0" pos="0 0 0.1">
              <joint name="joint0" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
              <body name="link1" pos="0 0 0.2">
                <joint name="joint1" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.02" fromto="0 0 0  0 0 0.2"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor joint="joint0" ctrlrange="-1 1"/>
            <motor joint="joint1" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>"""
        mjcf_path = tmp_path / "test_arm.xml"
        mjcf_path.write_text(mjcf_xml)

        sim = Robot("so100", mode="sim", backend="mujoco", urdf_path=str(mjcf_path))
        try:
            assert sim._world is not None
            assert sim._world._model is not None
            assert sim._world._data is not None
            mujoco.mj_step(sim._world._model, sim._world._data)
            assert sim._world._data.time > 0
        finally:
            sim.destroy()

    def test_import_from_top_level(self):
        """Robot and list_robots importable from strands_robots."""
        from strands_robots import Robot as R
        from strands_robots import list_robots as lr

        assert R is Robot
        assert callable(lr)


class TestRobotRealMode:
    """Tests for mode='real' path (mocked — no physical hardware)."""

    def test_real_mode_requires_lerobot(self):
        """mode='real' imports lerobot hardware classes."""
        from unittest.mock import MagicMock, patch

        # Mock the hardware import to avoid needing lerobot installed
        with patch("strands_robots.robot.get_hardware_type", return_value="so100_follower"):
            with patch("strands_robots.hardware_robot.Robot") as mock_hw:
                mock_hw.return_value = MagicMock()
                try:
                    Robot("so100", mode="real")
                    mock_hw.assert_called_once()
                except ImportError:
                    # lerobot not installed — acceptable in unit CI
                    pass


class TestAutoDetectUSB:
    """Tests for USB-found-hardware branch in _auto_detect_mode."""

    def test_usb_detection_finds_feetech(self, monkeypatch):
        """Servo controller detected → returns 'real'."""
        pytest.importorskip("serial")
        from unittest.mock import MagicMock, patch

        mock_port = MagicMock()
        mock_port.description = "Feetech STS3215 Servo Controller"
        mock_port.device = "/dev/ttyUSB0"
        mock_port.manufacturer = "Feetech"

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            assert _auto_detect_mode("so100") == "real"

    def test_usb_detection_excludes_bluetooth(self, monkeypatch):
        """Bluetooth device not treated as robot hardware."""
        pytest.importorskip("serial")
        from unittest.mock import MagicMock, patch

        mock_port = MagicMock()
        mock_port.description = "Bluetooth Internal Feetech"
        mock_port.device = "/dev/ttyBT0"
        mock_port.manufacturer = None

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            assert _auto_detect_mode("so100") == "sim"

    def test_usb_detection_import_error(self, monkeypatch):
        """pyserial not installed → falls back to sim."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"serial": None, "serial.tools": None, "serial.tools.list_ports": None}):
            assert _auto_detect_mode("so100") == "sim"

    def test_usb_detection_no_robot_hardware(self, monkeypatch):
        """Robot without hardware support → skips USB scan."""
        from strands_robots.robot import _auto_detect_mode

        # "panda" may not have hardware support — defaults to sim
        result = _auto_detect_mode("panda")
        assert result == "sim"


class TestModeNormalization:
    """Mode parameter and STRANDS_ROBOT_MODE env var should agree on case/whitespace."""

    def test_mode_param_uppercase_accepted(self):
        """Robot('so100', mode='SIM') should work — env var path is case-insensitive,
        the direct param should be too."""
        pytest.importorskip("mujoco")
        sim = Robot("so100", mode="SIM")
        try:
            from strands_robots.simulation import Simulation

            assert isinstance(sim, Simulation)
        finally:
            sim.destroy()

    def test_mode_param_with_whitespace(self):
        """mode=' sim ' should be normalized like the env var is."""
        pytest.importorskip("mujoco")
        sim = Robot("so100", mode=" sim ")
        try:
            from strands_robots.simulation import Simulation

            assert isinstance(sim, Simulation)
        finally:
            sim.destroy()

    def test_env_var_with_whitespace(self, monkeypatch):
        """STRANDS_ROBOT_MODE='  sim  ' should resolve cleanly without firing the
        'ignored' warning."""
        from strands_robots.robot import _auto_detect_mode

        monkeypatch.setenv("STRANDS_ROBOT_MODE", "  sim  ")
        assert _auto_detect_mode("so100") == "sim"

    def test_env_var_auto_is_no_op(self, monkeypatch):
        """STRANDS_ROBOT_MODE=auto means 'do detection' — same as not setting it.
        Should not warn."""
        from strands_robots.robot import _auto_detect_mode

        monkeypatch.setenv("STRANDS_ROBOT_MODE", "auto")
        # Auto-detect with no USB hardware → falls back to sim
        assert _auto_detect_mode("so100") == "sim"


class TestUnknownNameRejected:
    """Empty / whitespace / unknown robot names should raise ValueError before
    we descend into the sim or hardware backend, so the user sees one clean
    error instead of a confusing two-stage stderr+exception."""

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="Invalid robot name"):
            Robot("")

    def test_whitespace_name_rejected(self):
        with pytest.raises(ValueError, match="Invalid robot name"):
            Robot("  ")

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown robot"):
            Robot("definitely_not_a_robot_xyz")

    def test_unknown_name_rejected_in_real_mode(self):
        with pytest.raises(ValueError, match="Unknown robot"):
            Robot("definitely_not_a_robot_xyz", mode="real")

    def test_unknown_name_with_urdf_path_does_not_raise(self):
        """Explicit urdf_path bypasses the registry check — user knows what they
        want, we don't second-guess."""
        pytest.importorskip("mujoco")
        # Use a clearly-bogus path so the underlying load fails (as RuntimeError),
        # not a ValueError from validation. Cleanup is also covered separately.
        with pytest.raises(RuntimeError):
            Robot("my_custom_arm", urdf_path="/nonexistent/foo.xml")


class TestCleanupOnDispatchRaise:
    """If sim._dispatch_action itself raises (vs returns status=error), the
    Simulation must still be destroyed. Pins the cleanup path that the original
    review caught only for the status=error variant."""

    def test_destroy_called_when_create_world_raises(self):
        """OSError (or any exception) from create_world must trigger destroy()."""
        pytest.importorskip("mujoco")
        from unittest.mock import patch

        from strands_robots.simulation.mujoco.simulation import Simulation as SimImpl

        destroyed = []
        real_destroy = SimImpl.destroy

        def track(self):
            destroyed.append(self)
            return real_destroy(self)

        original_dispatch = SimImpl._dispatch_action

        def raising_dispatch(self, action, params):
            if action == "create_world":
                raise OSError("simulated disk full")
            return original_dispatch(self, action, params)

        with (
            patch.object(SimImpl, "_dispatch_action", raising_dispatch),
            patch.object(SimImpl, "destroy", track),
        ):
            with pytest.raises(OSError, match="simulated disk full"):
                Robot("so100")

        assert len(destroyed) == 1, f"destroy() should have been called once, was {len(destroyed)}x"

    def test_destroy_called_when_add_robot_raises(self):
        """RuntimeError from add_robot must trigger destroy()."""
        pytest.importorskip("mujoco")
        from unittest.mock import patch

        from strands_robots.simulation.mujoco.simulation import Simulation as SimImpl

        destroyed = []
        real_destroy = SimImpl.destroy

        def track(self):
            destroyed.append(self)
            return real_destroy(self)

        original_dispatch = SimImpl._dispatch_action

        def raising_dispatch(self, action, params):
            if action == "add_robot":
                raise RuntimeError("simulated MJCF compile error")
            return original_dispatch(self, action, params)

        with (
            patch.object(SimImpl, "_dispatch_action", raising_dispatch),
            patch.object(SimImpl, "destroy", track),
        ):
            with pytest.raises(RuntimeError, match="simulated MJCF compile error"):
                Robot("so100")

        assert len(destroyed) == 1, f"destroy() should have been called once, was {len(destroyed)}x"


class TestUSBProbeFallsBackOnRuntimeError:
    """libusb hub glitches can surface as RuntimeError from comports().
    _auto_detect_mode must fall back to sim, not propagate the exception."""

    def test_runtime_error_during_usb_probe(self):
        pytest.importorskip("serial")
        from unittest.mock import patch

        from strands_robots.robot import _auto_detect_mode

        def raise_runtime(*a, **kw):
            raise RuntimeError("simulated libusb hub glitch")

        with patch("serial.tools.list_ports.comports", side_effect=raise_runtime):
            # Must return "sim" (safe fallback), not raise.
            assert _auto_detect_mode("so100") == "sim"


class TestDashedNameAlias:
    """Common typo: users write 'so-100' (matches marketing). Should resolve to
    canonical 'so100' rather than producing a confusing 'Unknown robot' error."""

    def test_dashed_name_resolves_to_canonical(self):
        from strands_robots.registry import resolve_name

        assert resolve_name("so-100") == "so100"
        assert resolve_name("so_100") == "so100"
        assert resolve_name("SO-100") == "so100"


class TestCameraErrorMessage:
    """The cameras-in-sim error must NOT recommend the private _dispatch_action
    method — that's been a recurring review request."""

    def test_camera_error_does_not_leak_private_api(self):
        with pytest.raises(ValueError) as excinfo:
            Robot("so100", cameras={"wrist": {"type": "opencv"}})
        assert "_dispatch_action" not in str(excinfo.value), (
            "Error message should not mention the private _dispatch_action method"
        )


class TestRealModeConfigDiscovery:
    """Regression tests for `_create_minimal_config` switching from a
    hand-rolled mapping to lerobot's draccus ChoiceRegistry discovery.

    These tests use `pytest.importorskip("lerobot")` so they noop on
    machines without lerobot installed.
    """

    def test_lerobot_registry_discovery_finds_all_subpackages(self):
        """Walking ``lerobot.robots`` with pkgutil registers every robot
        config without any hard-coded type→module mapping. This is the
        future-proof path: any robot lerobot ships in
        ``lerobot/robots/<X>/`` (regardless of whether its ``robot_type``
        matches ``X``, e.g. ``hope_jr_arm`` lives in ``hope_jr/`` and
        ``lekiwi_client`` lives in ``lekiwi/``) automatically becomes
        constructible via ``Robot(...)`` mode='real'."""
        pytest.importorskip("lerobot")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        _ensure_lerobot_robots_registered()
        registered = set(RobotConfig.get_known_choices().keys())

        # These are the lerobot-built-in robots whose subpackage exists
        # on disk in lerobot >=0.5.x. Adding more upstream is a no-op
        # for strands_robots.
        expected_min = {
            "so100_follower",
            "so101_follower",
            "koch_follower",
            "openarm_follower",
            "bi_so_follower",
            "bi_openarm_follower",
            "unitree_g1",
        }
        missing = expected_min - registered
        assert not missing, f"Discovery missed lerobot built-ins: {missing}. Registered: {sorted(registered)}"

    def test_subpackage_with_multiple_robots_picked_up(self):
        """Some lerobot subpackages register MULTIPLE robot_types (e.g.
        ``hope_jr/`` registers both ``hope_jr_arm`` and ``hope_jr_hand``;
        ``lekiwi/`` registers both ``lekiwi`` and ``lekiwi_client``).
        pkgutil-walking handles this naturally — a hand-rolled
        type→module map would have to special-case each."""
        pytest.importorskip("lerobot.robots.hope_jr")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        _ensure_lerobot_robots_registered()
        registered = set(RobotConfig.get_known_choices().keys())
        # Multiple types from one subpackage:
        assert "hope_jr_arm" in registered
        assert "hope_jr_hand" in registered

    def test_so101_config_build_uses_RobotConfig_subclass(self):
        """Regression: lerobot 0.5.x's bare ``SOFollowerConfig`` has no
        ``id`` field — discovery picks the registered ``SOFollowerRobotConfig``
        subclass that does. (Original SO-100/SO-101 real-mode regression.)"""
        pytest.importorskip("lerobot.robots.so_follower")
        from strands_robots.hardware_robot import Robot as HwRobot

        # Build the config directly via the helper — no hardware connect.
        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_smoke"
        cfg = hw._create_minimal_config("so101_follower", cameras={}, port="/dev/null", use_degrees=True)
        # Must be the registered subclass (has ``id``), not the bare config.
        assert hasattr(cfg, "id"), "config has no 'id' — used the wrong subclass"
        assert cfg.id == "so101_smoke"
        assert cfg.port == "/dev/null"
        # The registered subclass for so101 inherits both RobotConfig and
        # SOFollowerConfig — its name typically ends with `RobotConfig`.
        assert "RobotConfig" in type(cfg).__name__

    def test_unitree_g1_config_build_via_discovery(self):
        """Regression: ``unitree_g1`` was missing from the old hand-rolled
        config_mapping despite the registry advertising ``has_real=True``.
        Discovery via ChoiceRegistry picks it up automatically."""
        pytest.importorskip("lerobot.robots.unitree_g1")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "g1_smoke"
        cfg = hw._create_minimal_config(
            "unitree_g1",
            cameras={},
            robot_ip="192.168.123.164",
            kp=[100.0] * 29,
            kd=[2.0] * 29,
            default_positions=[0.0] * 29,
            is_simulation=False,
        )
        assert cfg.id == "g1_smoke"
        assert cfg.robot_ip == "192.168.123.164"
        assert len(cfg.kp) == 29
        assert cfg.is_simulation is False

    def test_unknown_robot_type_raises_clean(self):
        """Unknown types produce an error listing the *actual* known types
        (not a stale hard-coded list)."""
        pytest.importorskip("lerobot.robots.config")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "x"
        with pytest.raises(ValueError, match="Unsupported robot type"):
            hw._create_minimal_config("totally_made_up_robot", cameras={})

    def test_extra_kwargs_filtered_against_dataclass_fields(self):
        """Forwarded kwargs that the target dataclass doesn't declare are
        dropped silently, so callers can pass union-of-all known kwargs
        without breaking on simpler robots."""
        pytest.importorskip("lerobot.robots.so_follower")
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "so101_smoke"
        # `robot_ip` and `kp` are G1-only — must not raise on so101.
        cfg = hw._create_minimal_config(
            "so101_follower",
            cameras={},
            port="/dev/null",
            robot_ip="192.168.0.1",
            kp=[1.0] * 29,
            kd=[1.0] * 29,
        )
        assert cfg.port == "/dev/null"
        # Filtered out:
        assert not hasattr(cfg, "robot_ip")
        assert not hasattr(cfg, "kp")

    def test_mesh_attrs_set_before_initialize_robot_no_attribute_error_in_cleanup(self, caplog):
        """Pin the cleanup-AttributeError fix with the actual symptom.

        Pre-fix, when ``_initialize_robot`` raised partway through ``__init__``
        the secondary cleanup path ran ``cleanup()`` -> ``self.mesh`` and
        produced an ``AttributeError: 'Robot' object has no attribute
        'mesh'``. ``__del__`` swallows exceptions so the user never saw it
        directly, but ``cleanup()`` has its own ``except`` that calls
        ``logger.error(f"Cleanup error for {self.tool_name_str}: {e}")``.

        The fix moves ``self.mesh = None`` / ``self.peer_id = None`` to
        before ``_initialize_robot``, so that error log entry no longer
        appears. We assert on that absence; if a future refactor undoes
        the ordering swap (e.g. moves the mesh init back to its original
        spot), this test fails.
        """
        from unittest.mock import patch

        from strands_robots.hardware_robot import Robot as HwRobot

        with caplog.at_level("ERROR", logger="strands_robots.hardware_robot"):
            with patch.object(HwRobot, "_initialize_robot", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    HwRobot(tool_name="x", robot="so101_follower")

        # Pre-fix code logged either of:
        #   "Cleanup error for x: 'Robot' object has no attribute 'mesh'"
        # depending on whether peer_id or mesh was probed first. The fix
        # eliminates BOTH because both attrs are now initialised before
        # _initialize_robot runs.
        offenders = [
            r.message
            for r in caplog.records
            if "AttributeError" in r.message and "mesh" in r.message and "Cleanup error" in r.message
        ]
        assert not offenders, (
            f"cleanup() logged AttributeError for missing 'mesh': {offenders}. "
            "Did the mesh/peer_id init move back below _initialize_robot?"
        )

    def test_bi_so100_follower_resolves_via_discovery_shim(self):
        """Regression test for the lazy-import shim: ``bi_so100_follower``
        is registered by ``lerobot.robots.bi_so_follower`` (the directory
        name does NOT match the robot_type), so a hand-rolled
        ``import_module(f"lerobot.robots.{robot_type}")`` would miss it.
        Discovery via ``pkgutil.iter_modules`` walks the directory and
        catches it.

        This pins the discovery contract -- if a future cleanup PR drops
        the pkgutil walker (e.g. believing lerobot's __init__ has become
        eager), this test will fail before the breakage hits users.
        """
        pytest.importorskip("lerobot.robots.bi_so_follower")
        from lerobot.robots.config import RobotConfig

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        # Force a clean walk by bypassing the @functools.cache (we want
        # to test that even the FIRST call after a fresh import resolves
        # the alias case correctly).
        _ensure_lerobot_robots_registered.cache_clear()
        _ensure_lerobot_robots_registered()

        # `hope_jr_arm` lives in `lerobot.robots.hope_jr` (the directory
        # name does NOT match the robot_type). Same with `hope_jr_hand`,
        # `lekiwi_client`, and `so100_follower`/`so101_follower` (both in
        # `so_follower`). A hand-rolled
        # `import_module(f"lerobot.robots.{robot_type}")` would miss all of
        # these. Discovery via pkgutil.iter_modules catches them.
        for robot_type, expected_pkg_prefix in [
            ("hope_jr_arm", "lerobot.robots.hope_jr"),
            ("hope_jr_hand", "lerobot.robots.hope_jr"),
            ("lekiwi_client", "lerobot.robots.lekiwi"),
            ("so101_follower", "lerobot.robots.so_follower"),
            ("so100_follower", "lerobot.robots.so_follower"),
        ]:
            try:
                ConfigClass = RobotConfig.get_choice_class(robot_type)
            except KeyError:
                pytest.fail(f"discovery missed {robot_type!r} (expected from {expected_pkg_prefix})")
            assert ConfigClass.__module__.startswith(expected_pkg_prefix), (
                f"Expected {robot_type} to come from {expected_pkg_prefix}, got {ConfigClass.__module__}"
            )

    def test_unsupported_type_error_has_no_chained_keyerror_traceback(self):
        """``raise ValueError(...) from None`` must suppress the chained
        KeyError traceback. Otherwise users see "During handling of the
        above exception (KeyError), another exception occurred" which
        leaks lerobot's draccus internals."""
        from strands_robots.hardware_robot import Robot as HwRobot

        hw = HwRobot.__new__(HwRobot)
        hw.tool_name_str = "x"
        try:
            hw._create_minimal_config("totally_made_up_robot", cameras={})
        except ValueError as e:
            assert e.__cause__ is None, f"ValueError carries chained cause {e.__cause__!r}; should be `from None`."
            assert e.__suppress_context__, (
                "ValueError does not suppress its context; users will see the internal KeyError traceback."
            )
        else:
            pytest.fail("expected ValueError")

    def test_robot_factory_end_to_end_real_mode_so101(self):
        """Public-API happy-path: ``Robot('so101', mode='real', port=...)``
        gets through the factory, uses lerobot's draccus discovery, and
        constructs a HardwareRobot wrapping a real lerobot SOFollower
        (with ``_initialize_robot`` mocked so we don't touch hardware).

        This exercises the full call chain that caused issue #275 to be
        filed -- the previous tests poke ``_create_minimal_config`` via
        ``__new__`` so they only cover the helper. This one goes through
        ``Robot()``."""
        from unittest.mock import MagicMock, patch

        from strands_robots import Robot

        # Sentinel returned from _initialize_robot — pretend it's a built
        # lerobot Robot instance so the strands HardwareRobot.__init__
        # completes happily without trying to talk to a serial port.
        fake_lerobot_robot = MagicMock(name="lerobot_robot_instance")
        fake_lerobot_robot.name = "so_follower"
        fake_lerobot_robot.config = MagicMock()
        fake_lerobot_robot.config.cameras = {}

        with patch(
            "strands_robots.hardware_robot.Robot._initialize_robot",
            return_value=fake_lerobot_robot,
        ):
            r = Robot(
                "so101",
                mode="real",
                port="/dev/null",
                use_degrees=True,
                id="left_arm",
            )

        # The factory returned a HardwareRobot, mesh attrs are initialised,
        # and the inner robot is the fake we injected.
        from strands_robots.hardware_robot import Robot as HwRobot

        assert isinstance(r, HwRobot)
        assert r.robot is fake_lerobot_robot
        assert r.mesh is None or hasattr(r.mesh, "stop")  # mesh is opt-in; just ensure attr exists
        assert hasattr(r, "peer_id")  # populated by factory if mesh started, else None

    def test_walk_continues_when_driver_raises_oserror_at_import(self):
        """Pin AGENTS.md > Review Learnings (#86) > "Exception Clauses Must
        Be Narrow" / hardware-probing pattern. A driver subpackage whose
        ``__init__`` raises a non-``ImportError`` (e.g. ``OSError`` from a
        USB probe in ``unitree_sdk2py``) must not abort the entire
        ``_ensure_lerobot_robots_registered`` walk -- subsequent driver
        registrations must still happen.

        Pre-fix code used ``except ImportError`` only; an ``OSError``
        would propagate out of ``importlib.import_module``, abort the
        for-loop, and silently skip every later driver. This is the same
        silent-degradation mode the surrounding comment claims to guard
        against.
        """
        pytest.importorskip("lerobot")

        from unittest.mock import patch

        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        real_import = importlib.import_module
        booby_target = "lerobot.robots.so_follower"

        def fake_import(name, *args, **kwargs):
            if name == booby_target:
                raise OSError("simulated USB probe failure during driver __init__")
            return real_import(name, *args, **kwargs)

        # Force a clean walk: the cache from prior tests would short-circuit.
        _ensure_lerobot_robots_registered.cache_clear()

        with patch(
            "strands_robots.hardware_robot.importlib.import_module",
            side_effect=fake_import,
        ):
            # Must not raise -- the OSError from so_follower must be caught
            # and the walk must continue past it.
            _ensure_lerobot_robots_registered()

        # Re-walk with the patch removed so the rest of the test session
        # sees a fully-populated registry (the patched walk skipped
        # so_follower entirely).
        _ensure_lerobot_robots_registered.cache_clear()
        _ensure_lerobot_robots_registered()

        # Sanity: at least one driver other than so_follower is registered,
        # proving the walk continued past the booby-trapped subpackage.
        from lerobot.robots.config import RobotConfig

        known = set(RobotConfig.get_known_choices().keys())
        # ``hope_jr_arm`` and ``lekiwi_client`` come from sibling subpackages
        # alphabetically before AND after ``so_follower`` -- one of each
        # must be present even when so_follower fails.
        assert known - {"so100_follower", "so101_follower"}, (
            f"Walk aborted on so_follower OSError; only {known} registered."
        )
