"""Tests for strands_robots.robot — Robot() factory and list_robots()."""

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
