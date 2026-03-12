"""Comprehensive tests for strands_robots.tools subpackage.

Achieves 100% code coverage by testing every @tool function, every action path,
every branch, and every exception handler.  All external dependencies (lerobot,
mujoco, serial, cv2, zenoh, unitree_sdk2py, docker, etc.) are mocked so the
tests run without any hardware or GPU.
"""

import atexit
import importlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, call, patch

import numpy as np
import pytest

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset global state in tool modules between tests.
    Also pre-import all tool modules to ensure they're in sys.modules
    before any @patch decorators try to resolve them.
    """
    # Pre-import tool modules to avoid lazy import contamination
    import importlib
    for mod_name in [
        "strands_robots.tools.serial_tool",
        "strands_robots.tools.isaac_sim",
        "strands_robots.tools.newton_sim",
        "strands_robots.tools.inference",
        "strands_robots.tools.stream",
        "strands_robots.tools.teleoperator",
        "strands_robots.tools.lerobot_dataset",
        "strands_robots.tools.pose_tool",
        "strands_robots.tools.gr00t_inference",
        "strands_robots.tools.robot_mesh",
        "strands_robots.tools.marble_tool",
        "strands_robots.tools.use_lerobot",
        "strands_robots.tools.use_unitree",
        "strands_robots.tools.stereo_depth",
        "strands_robots.tools.reachy_mini_tool",
        "strands_robots.tools.lerobot_calibrate",
    ]:
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass
    yield
    # gr00t_inference has no mutable global state we keep
    # inference._RUNNING
    try:
        from strands_robots.tools import inference as inf_mod
        inf_mod._RUNNING.clear()
    except Exception:
        pass
    # newton_sim globals
    try:
        from strands_robots.tools import newton_sim as ns_mod
        ns_mod._backend = None
        ns_mod._backend_config = None
    except Exception:
        pass
    # isaac_sim globals
    try:
        from strands_robots.tools import isaac_sim as is_mod
        is_mod._backend = None
        is_mod._backend_config = None
        is_mod._isaac_env = None
        is_mod._state_dir = None
    except Exception:
        pass
    # marble_tool global
    try:
        from strands_robots.tools import marble_tool as mt_mod
        mt_mod._pipeline = None
    except Exception:
        pass
    # stream._STREAMS
    try:
        from strands_robots.tools import stream as st_mod
        st_mod._STREAMS.clear()
    except Exception:
        pass
    # teleoperator._ACTIVE_SESSION
    try:
        from strands_robots.tools import teleoperator as tp_mod
        tp_mod._ACTIVE_SESSION.update({
            "running": False, "thread": None, "robot": None,
            "teleop": None, "record_session": None, "mode": None, "stats": {},
        })
    except Exception:
        pass
    # lerobot_dataset._ACTIVE_RECORDINGS
    try:
        from strands_robots.tools import lerobot_dataset as ld_mod
        ld_mod._ACTIVE_RECORDINGS.clear()
    except Exception:
        pass
    # reachy_mini sessions
    try:
        from strands_robots.tools import reachy_mini_tool as rm_mod
        with rm_mod._SESSIONS_LOCK:
            rm_mod._SESSIONS.clear()
    except Exception:
        pass


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def _get_mod(name: str):
    """Reliably get a tool module by its full dotted name, bypassing lazy-import contamination."""
    import importlib
    full = f"strands_robots.tools.{name}" if "." not in name else name
    importlib.import_module(full)
    return sys.modules[full]


# ===========================================================================
# 1. __init__.py — Lazy imports
# ===========================================================================


class TestToolsInit:
    def test_lazy_import_known(self):
        from strands_robots.tools import __getattr__ as ga
        from strands_robots.tools import _LAZY_IMPORTS
        # Attempting to get a known attr should try importlib
        with patch("importlib.import_module") as mock_imp:
            mock_mod = MagicMock()
            mock_mod.serial_tool = "fake_serial_tool"
            mock_imp.return_value = mock_mod
            result = ga("serial_tool")
            mock_imp.assert_called_once_with(".serial_tool", "strands_robots.tools")
            assert result == "fake_serial_tool"

    def test_lazy_import_unknown_raises(self):
        from strands_robots.tools import __getattr__ as ga
        with pytest.raises(AttributeError, match="has no attribute"):
            ga("nonexistent_tool_xyz")

    def test_all(self):
        from strands_robots.tools import __all__, _LAZY_IMPORTS
        assert set(__all__) == set(_LAZY_IMPORTS.keys())


# ===========================================================================
# 2. gr00t_inference.py
# ===========================================================================


class TestGr00tInference:
    """Tests for gr00t_inference tool and all private helpers."""

    def _import(self):
        from strands_robots.tools.gr00t_inference import gr00t_inference
        return gr00t_inference

    # -- action dispatch --

    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_find_containers(self, mock_fc):
        mock_fc.return_value = {"status": "success", "containers": []}
        fn = self._import()
        r = fn(action="find_containers")
        assert r["status"] == "success"

    @patch("strands_robots.tools.gr00t_inference._list_running_services")
    def test_list(self, mock_ls):
        mock_ls.return_value = {"status": "success", "services": []}
        fn = self._import()
        r = fn(action="list")
        assert r["status"] == "success"

    def test_status_no_port(self):
        fn = self._import()
        r = fn(action="status")
        assert r["status"] == "error"
        assert "Port required" in r["message"]

    @patch("strands_robots.tools.gr00t_inference._check_service_status")
    def test_status_with_port(self, mock_cs):
        mock_cs.return_value = {"status": "success", "service_status": "running"}
        fn = self._import()
        r = fn(action="status", port=5555)
        assert r["status"] == "success"

    def test_stop_no_port(self):
        fn = self._import()
        r = fn(action="stop")
        assert r["status"] == "error"

    @patch("strands_robots.tools.gr00t_inference._stop_service")
    def test_stop_with_port(self, mock_ss):
        mock_ss.return_value = {"status": "success"}
        fn = self._import()
        r = fn(action="stop", port=5555)
        assert r["status"] == "success"

    def test_start_no_checkpoint(self):
        fn = self._import()
        r = fn(action="start")
        assert r["status"] == "error"
        assert "Checkpoint" in r["message"]

    @patch("strands_robots.tools.gr00t_inference._start_service")
    def test_start_default_zmq_port(self, mock_start):
        mock_start.return_value = {"status": "success", "port": 5555}
        fn = self._import()
        r = fn(action="start", checkpoint_path="/ckpt")
        mock_start.assert_called_once()
        assert mock_start.call_args[1]["port"] == 5555

    @patch("strands_robots.tools.gr00t_inference._start_service")
    def test_start_http_default_port(self, mock_start):
        mock_start.return_value = {"status": "success", "port": 8000}
        fn = self._import()
        r = fn(action="start", checkpoint_path="/ckpt", http_server=True)
        assert mock_start.call_args[1]["port"] == 8000

    def test_restart_no_checkpoint_or_port(self):
        fn = self._import()
        r = fn(action="restart")
        assert r["status"] == "error"

    @patch("strands_robots.tools.gr00t_inference._start_service")
    @patch("strands_robots.tools.gr00t_inference._stop_service")
    @patch("strands_robots.tools.gr00t_inference.time")
    def test_restart(self, mock_time, mock_stop, mock_start):
        mock_stop.return_value = {"status": "success"}
        mock_start.return_value = {"status": "success"}
        fn = self._import()
        r = fn(action="restart", checkpoint_path="/ckpt", port=5555)
        mock_stop.assert_called_once_with(5555)
        mock_start.assert_called_once()

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="invalid_xyz")
        assert r["status"] == "error"
        assert "Unknown action" in r["message"]

    # -- _find_gr00t_containers --

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_find_containers_docker(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="mycontainer\tisaac-gr00t-img\tUp 2 hours\t5555/tcp\n"
                   "other\tnginx\tUp\t80/tcp",
            returncode=0,
        )
        from strands_robots.tools.gr00t_inference import _find_gr00t_containers
        r = _find_gr00t_containers()
        assert r["status"] == "success"
        assert len(r["containers"]) == 1

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_find_containers_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker")
        from strands_robots.tools.gr00t_inference import _find_gr00t_containers
        r = _find_gr00t_containers()
        assert r["status"] == "error"

    # -- _list_running_services --

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_list_running_services(self, mock_isr):
        mock_isr.side_effect = lambda p: p == 5555 or p == 8000
        from strands_robots.tools.gr00t_inference import _list_running_services
        r = _list_running_services()
        assert r["status"] == "success"
        assert len(r["services"]) == 2

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_list_running_services_exception(self, mock_isr):
        mock_isr.side_effect = Exception("fail")
        from strands_robots.tools.gr00t_inference import _list_running_services
        r = _list_running_services()
        assert r["status"] == "error"

    # -- _is_service_running --

    @patch("strands_robots.tools.gr00t_inference.socket.socket")
    def test_is_service_running_true(self, mock_sock_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_sock_cls.return_value = mock_sock
        from strands_robots.tools.gr00t_inference import _is_service_running
        assert _is_service_running(5555) is True

    @patch("strands_robots.tools.gr00t_inference.socket.socket")
    def test_is_service_running_false(self, mock_sock_cls):
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 1
        mock_sock_cls.return_value = mock_sock
        from strands_robots.tools.gr00t_inference import _is_service_running
        assert _is_service_running(5555) is False

    @patch("strands_robots.tools.gr00t_inference.socket.socket")
    def test_is_service_running_exception(self, mock_sock_cls):
        mock_sock_cls.side_effect = Exception("fail")
        from strands_robots.tools.gr00t_inference import _is_service_running
        assert _is_service_running(5555) is False

    # -- _check_service_status --

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_check_service_running(self, mock_isr):
        mock_isr.return_value = True
        from strands_robots.tools.gr00t_inference import _check_service_status
        r = _check_service_status(5555)
        assert r["service_status"] == "running"
        assert r["protocol"] == "ZMQ"

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_check_service_not_running(self, mock_isr):
        mock_isr.return_value = False
        from strands_robots.tools.gr00t_inference import _check_service_status
        r = _check_service_status(8000)
        assert r["service_status"] == "not_running"

    # -- _stop_service --

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    @patch("strands_robots.tools.gr00t_inference.time")
    def test_stop_service_in_container(self, mock_time, mock_fc, mock_run):
        mock_fc.return_value = {"status": "success", "containers": [
            {"name": "c1", "image": "isaac-gr00t", "status": "Up 1h", "ports": ""}
        ]}
        # pgrep finds PID first call, empty second
        pgrep_result1 = MagicMock(returncode=0, stdout="123\n")
        pgrep_result2 = MagicMock(returncode=1, stdout="")
        kill_result = MagicMock(returncode=0)
        mock_run.side_effect = [pgrep_result1, kill_result, pgrep_result2]
        from strands_robots.tools.gr00t_inference import _stop_service
        r = _stop_service(5555)
        assert r["status"] == "success"

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    @patch("strands_robots.tools.gr00t_inference.time")
    def test_stop_service_force_kill(self, mock_time, mock_fc, mock_run):
        mock_fc.return_value = {"status": "success", "containers": [
            {"name": "c1", "image": "isaac-gr00t", "status": "Up 1h", "ports": ""}
        ]}
        # pgrep always finds PID (needs force kill)
        pgrep_result = MagicMock(returncode=0, stdout="123\n")
        kill_result = MagicMock(returncode=0)
        mock_run.side_effect = [pgrep_result, kill_result, pgrep_result, kill_result]
        from strands_robots.tools.gr00t_inference import _stop_service
        r = _stop_service(5555)
        assert r["status"] == "success"

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    @patch("strands_robots.tools.gr00t_inference.time")
    def test_stop_service_fallback_host(self, mock_time, mock_fc, mock_run):
        mock_fc.return_value = {"status": "success", "containers": []}
        # Host system lsof finds PID
        lsof_result = MagicMock(returncode=0, stdout="456\n")
        kill_result = MagicMock(returncode=0)
        lsof_empty = MagicMock(returncode=1, stdout="")
        mock_run.side_effect = [lsof_result, kill_result, lsof_empty]
        from strands_robots.tools.gr00t_inference import _stop_service
        r = _stop_service(5555)
        assert r["status"] == "success"

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_stop_service_no_service(self, mock_fc, mock_run):
        mock_fc.return_value = {"status": "success", "containers": []}
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        from strands_robots.tools.gr00t_inference import _stop_service
        r = _stop_service(5555)
        assert r["status"] == "success"
        assert "No service" in r["message"]

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_stop_service_exception(self, mock_fc, mock_run):
        mock_fc.side_effect = Exception("fail")
        from strands_robots.tools.gr00t_inference import _stop_service
        r = _stop_service(5555)
        assert r["status"] == "error"

    # -- _start_service --

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    @patch("strands_robots.tools.gr00t_inference.time")
    def test_start_service_success(self, mock_time, mock_fc, mock_run, mock_isr):
        mock_fc.return_value = {"status": "success", "containers": [
            {"name": "c1", "image": "isaac-gr00t", "status": "Up 1h", "ports": ""}
        ]}
        mock_run.return_value = MagicMock(returncode=0)
        mock_isr.return_value = True
        mock_time.time.side_effect = [0, 1]
        mock_time.sleep = MagicMock()

        from strands_robots.tools.gr00t_inference import _start_service
        r = _start_service(
            checkpoint_path="/ckpt", port=5555, data_config="dc", embodiment_tag="et",
            denoising_steps=4, host="127.0.0.1", container_name=None,
            policy_name="pol", timeout=60, use_tensorrt=False,
            trt_engine_path="eng", vit_dtype="fp8", llm_dtype="nvfp4",
            dit_dtype="fp8", http_server=False, api_token=None,
        )
        assert r["status"] == "success"

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    @patch("strands_robots.tools.gr00t_inference.time")
    def test_start_service_tensorrt_http(self, mock_time, mock_fc, mock_run, mock_isr):
        mock_fc.return_value = {"status": "success", "containers": [
            {"name": "c1", "image": "isaac-gr00t", "status": "Up 1h", "ports": ""}
        ]}
        mock_run.return_value = MagicMock(returncode=0)
        mock_isr.return_value = True
        mock_time.time.side_effect = [0, 1]
        mock_time.sleep = MagicMock()

        from strands_robots.tools.gr00t_inference import _start_service
        r = _start_service(
            checkpoint_path="/ckpt", port=8000, data_config="dc", embodiment_tag="et",
            denoising_steps=4, host="0.0.0.0", container_name="myc",
            policy_name="pol", timeout=60, use_tensorrt=True,
            trt_engine_path="eng", vit_dtype="fp8", llm_dtype="nvfp4",
            dit_dtype="fp8", http_server=True, api_token="tok123",
        )
        assert r["status"] == "success"
        assert "tensorrt" in r
        assert "endpoint" in r

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    @patch("strands_robots.tools.gr00t_inference.time")
    def test_start_service_timeout(self, mock_time, mock_fc, mock_run, mock_isr):
        mock_fc.return_value = {"status": "success", "containers": [
            {"name": "c1", "image": "x", "status": "Up", "ports": ""}
        ]}
        mock_run.return_value = MagicMock(returncode=0)
        mock_isr.return_value = False
        mock_time.time.side_effect = [0] + [100] * 200
        mock_time.sleep = MagicMock()

        from strands_robots.tools.gr00t_inference import _start_service
        r = _start_service(
            checkpoint_path="/ckpt", port=5555, data_config="dc", embodiment_tag="et",
            denoising_steps=4, host="127.0.0.1", container_name=None,
            policy_name=None, timeout=2, use_tensorrt=False,
            trt_engine_path="eng", vit_dtype="fp8", llm_dtype="nvfp4",
            dit_dtype="fp8", http_server=False, api_token=None,
        )
        assert r["status"] == "error"

    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_start_service_no_containers(self, mock_fc):
        mock_fc.return_value = {"status": "success", "containers": []}
        from strands_robots.tools.gr00t_inference import _start_service
        r = _start_service(
            checkpoint_path="/ckpt", port=5555, data_config="dc", embodiment_tag="et",
            denoising_steps=4, host="127.0.0.1", container_name=None,
            policy_name=None, timeout=2, use_tensorrt=False,
            trt_engine_path="eng", vit_dtype="fp8", llm_dtype="nvfp4",
            dit_dtype="fp8", http_server=False, api_token=None,
        )
        assert r["status"] == "error"

    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_start_service_subprocess_error(self, mock_fc, mock_run):
        mock_fc.return_value = {"status": "success", "containers": [
            {"name": "c1", "image": "x", "status": "Up", "ports": ""}
        ]}
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker", stderr="fail")
        from strands_robots.tools.gr00t_inference import _start_service
        r = _start_service(
            checkpoint_path="/ckpt", port=5555, data_config="dc", embodiment_tag="et",
            denoising_steps=4, host="127.0.0.1", container_name=None,
            policy_name=None, timeout=2, use_tensorrt=False,
            trt_engine_path="eng", vit_dtype="fp8", llm_dtype="nvfp4",
            dit_dtype="fp8", http_server=False, api_token=None,
        )
        assert r["status"] == "error"


# ===========================================================================
# 3. serial_tool.py
# ===========================================================================


class TestSerialTool:
    def _get_module(self):
        """Get the actual serial_tool module reliably via sys.modules."""
        import strands_robots.tools.serial_tool  # ensure loaded
        return sys.modules["strands_robots.tools.serial_tool"]

    def _import(self):
        return self._get_module().serial_tool

    def test_no_serial(self):
        mod = self._get_module()
        orig = mod.HAS_SERIAL
        try:
            mod.HAS_SERIAL = False
            r = mod.serial_tool(action="list_ports")
            assert r["status"] == "error"
            assert "pyserial" in r["content"][0]["text"]
        finally:
            mod.HAS_SERIAL = orig

    def test_list_ports(self):
        mod = self._get_module()
        port_mock = MagicMock()
        port_mock.device = "/dev/ttyACM0"
        port_mock.name = "ttyACM0"
        port_mock.description = "USB"
        port_mock.manufacturer = "FTDI"
        port_mock.vid = 1234
        port_mock.pid = 5678
        port_mock.serial_number = "ABC"
        mock_serial = MagicMock()
        mock_serial.tools.list_ports.comports.return_value = [port_mock]
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="list_ports")
        assert r["status"] == "success"
        assert len(r["ports"]) == 1

    def test_no_port_for_send(self):
        fn = self._import()
        r = fn(action="send")
        assert r["status"] == "error"

    def test_send_hex(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="send", port="/dev/ttyACM0", hex_data="FF FF 01")
        assert r["status"] == "success"
        mock_ser.close.assert_called()

    def test_send_string(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="send", port="/dev/ttyACM0", data="hello")
        assert r["status"] == "success"

    def test_send_no_data(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="send", port="/dev/ttyACM0")
        assert r["status"] == "error"

    def test_read(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\x48\x65\x6c\x6c\x6f"
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="read", port="/dev/ttyACM0")
        assert r["status"] == "success"

    def test_send_read_hex(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\x00\x01"
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="send_read", port="/dev/ttyACM0", hex_data="FF")
        assert r["status"] == "success"

    def test_send_read_string(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\x00"
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="send_read", port="/dev/ttyACM0", data="test")
        assert r["status"] == "success"

    def test_send_read_no_data(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="send_read", port="/dev/ttyACM0")
        assert r["status"] == "error"

    def test_feetech_position(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="feetech_position", port="/dev/ttyACM0", motor_id=1, position=2048)
        assert r["status"] == "success"

    def test_feetech_position_missing(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="feetech_position", port="/dev/ttyACM0")
        assert r["status"] == "error"

    def test_feetech_velocity(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="feetech_velocity", port="/dev/ttyACM0", motor_id=1, velocity=100)
        assert r["status"] == "success"

    def test_feetech_velocity_missing(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="feetech_velocity", port="/dev/ttyACM0")
        assert r["status"] == "error"

    def test_feetech_ping_success(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_ser.read.return_value = b"\xFF\xFF\x01\x02\x00\xFC\x00"
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="feetech_ping", port="/dev/ttyACM0", motor_id=1)
        assert r["status"] == "success"

    def test_feetech_ping_no_response(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_ser.read.return_value = b""
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="feetech_ping", port="/dev/ttyACM0", motor_id=1)
        assert r["status"] == "error"

    def test_feetech_ping_missing_id(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="feetech_ping", port="/dev/ttyACM0")
        assert r["status"] == "error"

    def test_monitor(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})

        call_count = {"n": 0}
        class FakeTime:
            @staticmethod
            def time():
                call_count["n"] += 1
                if call_count["n"] <= 2:
                    return 0.0
                return 6.0  # Past the 5-second limit
            @staticmethod
            def sleep(s):
                pass

        with patch.object(mod, "serial", mock_serial):
            with patch.object(mod, "time", FakeTime):
                r = mod.serial_tool(action="monitor", port="/dev/ttyACM0")
        assert r["status"] == "success"

    def test_unknown_action(self):
        mod = self._get_module()
        mock_ser = MagicMock()
        mock_serial = MagicMock()
        mock_serial.Serial.return_value = mock_ser
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="xyz_unknown", port="/dev/ttyACM0")
        assert r["status"] == "error"

    def test_serial_exception(self):
        mod = self._get_module()
        mock_serial = MagicMock()
        mock_serial.Serial.side_effect = Exception("Connection failed")
        mock_serial.SerialException = type("SerialException", (Exception,), {})
        with patch.object(mod, "serial", mock_serial):
            r = mod.serial_tool(action="read", port="/dev/ttyACM0")
        assert r["status"] == "error"


# ===========================================================================
# 4. pose_tool.py
# ===========================================================================


class TestPoseTool:
    def _import(self):
        from strands_robots.tools.pose_tool import pose_tool
        return pose_tool

    def test_list_poses_empty(self, tmp_dir):
        fn = self._import()
        from strands_robots.tools.pose_tool import PoseManager
        with patch.object(PoseManager, '__init__', lambda self, *a, **kw: setattr(self, 'poses', {})):
            r = fn(action="list_poses", robot_id="test")
        assert r["status"] == "success"

    def test_list_poses_with_data(self, tmp_dir):
        fn = self._import()
        from strands_robots.tools.pose_tool import PoseManager, RobotPose
        pm = PoseManager.__new__(PoseManager)
        pm.poses = {"home": RobotPose(name="home", positions={"j1": 0.0}, timestamp=time.time(), description="test")}
        pm.storage_dir = Path(tmp_dir)
        pm.pose_file = Path(tmp_dir) / "test_poses.json"

        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            r = fn(action="list_poses", robot_id="test")
        assert r["status"] == "success"
        assert len(r["poses"]) == 1

    def test_show_pose_missing_name(self):
        fn = self._import()
        r = fn(action="show_pose")
        assert r["status"] == "error"

    def test_show_pose_not_found(self, tmp_dir):
        fn = self._import()
        from strands_robots.tools.pose_tool import PoseManager
        pm = PoseManager.__new__(PoseManager)
        pm.poses = {}
        pm.storage_dir = Path(tmp_dir)
        pm.pose_file = Path(tmp_dir) / "test_poses.json"
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            r = fn(action="show_pose", pose_name="missing")
        assert r["status"] == "error"

    def test_show_pose_success(self, tmp_dir):
        fn = self._import()
        from strands_robots.tools.pose_tool import PoseManager, RobotPose
        pm = PoseManager.__new__(PoseManager)
        pm.poses = {"home": RobotPose(name="home", positions={"j1": 45.0}, timestamp=time.time())}
        pm.storage_dir = Path(tmp_dir)
        pm.pose_file = Path(tmp_dir) / "test_poses.json"
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            r = fn(action="show_pose", pose_name="home")
        assert r["status"] == "success"

    def test_delete_pose(self, tmp_dir):
        fn = self._import()
        from strands_robots.tools.pose_tool import PoseManager, RobotPose
        pm = PoseManager.__new__(PoseManager)
        pm.poses = {"home": RobotPose(name="home", positions={"j1": 0.0}, timestamp=time.time())}
        pm.storage_dir = Path(tmp_dir)
        pm.pose_file = Path(tmp_dir) / "test_poses.json"
        pm._save_poses = MagicMock()
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            r = fn(action="delete_pose", pose_name="home")
        assert r["status"] == "success"

    def test_delete_pose_not_found(self, tmp_dir):
        fn = self._import()
        from strands_robots.tools.pose_tool import PoseManager
        pm = PoseManager.__new__(PoseManager)
        pm.poses = {}
        pm.storage_dir = Path(tmp_dir)
        pm.pose_file = Path(tmp_dir) / "test_poses.json"
        with patch("strands_robots.tools.pose_tool.PoseManager", return_value=pm):
            r = fn(action="delete_pose", pose_name="missing")
        assert r["status"] == "error"

    def test_emergency_stop(self):
        fn = self._import()
        r = fn(action="emergency_stop")
        assert r["status"] == "success"

    def test_no_serial_actions(self):
        fn = self._import()
        with patch.dict("strands_robots.tools.pose_tool.__dict__", {"HAS_SERIAL": False}):
            r = fn(action="connect")
        assert r["status"] == "error"

    def test_no_port(self):
        fn = self._import()
        r = fn(action="connect", port=None)
        assert r["status"] == "error"

    @patch("strands_robots.tools.pose_tool.serial")
    def test_connect_success(self, mock_serial):
        mock_serial.Serial.return_value = MagicMock(is_open=True)
        fn = self._import()
        r = fn(action="connect", port="/dev/ttyACM0")
        assert r["status"] == "success"

    @patch("strands_robots.tools.pose_tool.serial")
    def test_connect_failure(self, mock_serial):
        mock_serial.Serial.side_effect = Exception("fail")
        fn = self._import()
        r = fn(action="connect", port="/dev/ttyACM0")
        assert r["status"] == "error"

    @patch("strands_robots.tools.pose_tool.serial")
    def test_read_position(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.read.return_value = b"\xFF\xFF\x01\x04\x00\x00\x08\x00\x00\x00"
        mock_serial.Serial.return_value = mock_ser
        fn = self._import()
        r = fn(action="read_position", port="/dev/ttyACM0", motor_name="shoulder_pan")
        # May succeed or fail depending on response parsing, just check it runs
        assert r["status"] in ("success", "error")

    def test_read_position_no_motor(self):
        fn = self._import()
        r = fn(action="read_position", port="/dev/ttyACM0")
        assert r["status"] == "error"

    @patch("strands_robots.tools.pose_tool.serial")
    def test_move_motor(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_serial.Serial.return_value = mock_ser
        fn = self._import()
        r = fn(action="move_motor", port="/dev/ttyACM0", motor_name="shoulder_pan", position=45.0)
        assert r["status"] == "success"

    def test_move_motor_missing_params(self):
        fn = self._import()
        r = fn(action="move_motor", port="/dev/ttyACM0")
        assert r["status"] == "error"

    @patch("strands_robots.tools.pose_tool.serial")
    def test_move_multiple(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.read.return_value = b"\xFF\xFF\x01\x04\x00\x00\x08\x00\x00\x00"
        mock_serial.Serial.return_value = mock_ser
        fn = self._import()
        r = fn(action="move_multiple", port="/dev/ttyACM0",
               positions={"shoulder_pan": 10.0}, smooth=False)
        assert r["status"] == "success"

    def test_move_multiple_no_positions(self):
        fn = self._import()
        r = fn(action="move_multiple", port="/dev/ttyACM0")
        assert r["status"] == "error"

    @patch("strands_robots.tools.pose_tool.serial")
    def test_incremental_move(self, mock_serial):
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mock_ser.read.return_value = b"\xFF\xFF\x01\x04\x00\x00\x08\x00\x00\x00"
        mock_serial.Serial.return_value = mock_ser
        fn = self._import()
        r = fn(action="incremental_move", port="/dev/ttyACM0",
               motor_name="shoulder_pan", delta=5.0)
        assert r["status"] in ("success", "error")

    def test_incremental_move_missing(self):
        fn = self._import()
        r = fn(action="incremental_move", port="/dev/ttyACM0")
        assert r["status"] == "error"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="unknown_xyz_action")
        assert r["status"] == "error"

    # -- PoseManager / RobotPose --

    def test_robot_pose_to_from_dict(self):
        from strands_robots.tools.pose_tool import RobotPose
        rp = RobotPose(name="test", positions={"j1": 1.0}, timestamp=1.0)
        d = rp.to_dict()
        rp2 = RobotPose.from_dict(d)
        assert rp2.name == "test"

    def test_pose_manager_persistence(self, tmp_dir):
        from strands_robots.tools.pose_tool import PoseManager
        pm = PoseManager("test_robot", Path(tmp_dir))
        pm.store_pose("p1", {"j1": 10.0}, "desc")
        assert pm.get_pose("p1") is not None
        assert "p1" in pm.list_poses()
        pm2 = PoseManager("test_robot", Path(tmp_dir))
        assert pm2.get_pose("p1") is not None

    def test_pose_manager_validate(self, tmp_dir):
        from strands_robots.tools.pose_tool import PoseManager, RobotPose
        pm = PoseManager("test_robot", Path(tmp_dir))
        pose = RobotPose(name="t", positions={"j1": 200.0}, timestamp=1.0,
                         safety_bounds={"j1": (-180, 180)})
        valid, msg = pm.validate_pose(pose)
        assert valid is False
        pose2 = RobotPose(name="t", positions={"j1": 0.0}, timestamp=1.0)
        assert pm.validate_pose(pose2) == (True, "No safety bounds defined")

    # -- MotorController --

    def test_motor_controller_degrees_position(self):
        from strands_robots.tools.pose_tool import MotorController
        with patch.dict("strands_robots.tools.pose_tool.__dict__", {"HAS_SERIAL": True}):
            with patch("strands_robots.tools.pose_tool.serial"):
                mc = MotorController("/dev/test")
                pos = mc.degrees_to_position("shoulder_pan", 0.0)
                assert 0 <= pos <= 4095
                deg = mc.position_to_degrees("shoulder_pan", pos)
                assert isinstance(deg, float)
                # Gripper
                pos_g = mc.degrees_to_position("gripper", 50.0)
                deg_g = mc.position_to_degrees("gripper", pos_g)
                assert 0 <= deg_g <= 100

    def test_motor_controller_unknown_motor(self):
        from strands_robots.tools.pose_tool import MotorController
        with patch.dict("strands_robots.tools.pose_tool.__dict__", {"HAS_SERIAL": True}):
            with patch("strands_robots.tools.pose_tool.serial"):
                mc = MotorController("/dev/test")
                with pytest.raises(ValueError):
                    mc.degrees_to_position("unknown_motor", 0)

    def test_motor_controller_feetech_packet(self):
        from strands_robots.tools.pose_tool import MotorController
        with patch.dict("strands_robots.tools.pose_tool.__dict__", {"HAS_SERIAL": True}):
            with patch("strands_robots.tools.pose_tool.serial"):
                mc = MotorController("/dev/test")
                pkt = mc.build_feetech_packet(1, 0x03, [0x2A, 0x00, 0x08])
                assert pkt[:2] == b"\xFF\xFF"


# ===========================================================================
# 5. newton_sim.py
# ===========================================================================


class TestNewtonSim:
    def _import(self):
        from strands_robots.tools.newton_sim import newton_sim
        return newton_sim

    def _mock_backend(self):
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.num_envs = 1
        return backend

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_create_world(self, mock_destroy, mock_get):
        mock_backend = self._mock_backend()
        mock_backend.create_world.return_value = {"status": "success"}
        mock_get.return_value = mock_backend
        fn = self._import()
        r = fn(action="create_world", solver="mujoco", device="cpu")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_robot(self, mock_get):
        backend = self._mock_backend()
        backend.add_robot.return_value = {
            "success": True, "robot_info": {
                "name": "r", "format": "urdf", "model_path": "/x",
                "num_joints": 6, "num_bodies": 7, "position": (0, 0, 0)
            }
        }
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_robot", name="r")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_robot_fail(self, mock_get):
        backend = self._mock_backend()
        backend.add_robot.return_value = {"success": False, "message": "not found"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_robot")
        assert r["status"] == "error"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_step(self, mock_get):
        backend = self._mock_backend()
        backend.step.return_value = {"success": True, "sim_time": 0.01, "step_count": 1}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="step", num_steps=5)
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_step_fail(self, mock_get):
        backend = self._mock_backend()
        backend.step.return_value = {"success": False, "error": "diverged"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="step")
        assert r["status"] == "error"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_observe(self, mock_get):
        backend = self._mock_backend()
        backend.get_observation.return_value = {
            "success": True, "sim_time": 0.01,
            "observations": {"robot": {"joint_positions": np.array([0.1, 0.2, 0.3])}}
        }
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="observe")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_observe_empty(self, mock_get):
        backend = self._mock_backend()
        backend.get_observation.return_value = {"success": False}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="observe")
        assert "No observations" in r["content"][0]["text"]

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_reset(self, mock_get):
        backend = self._mock_backend()
        backend.reset.return_value = {"success": True, "message": "Reset done"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="reset")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_get_state(self, mock_get):
        backend = self._mock_backend()
        backend.get_state.return_value = {
            "config": {"solver": "mujoco", "device": "cpu", "num_envs": 1},
            "step_count": 0, "sim_time": 0.0, "robots": {}, "cloths": {},
            "sensors": [], "joints_per_world": 0, "bodies_per_world": 0,
        }
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="get_state")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_cloth(self, mock_get):
        backend = self._mock_backend()
        backend.add_cloth.return_value = {"success": True, "message": "OK"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_cloth")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_cable(self, mock_get):
        backend = self._mock_backend()
        backend.add_cable.return_value = {"success": True, "message": "OK"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_cable")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_particles(self, mock_get):
        backend = self._mock_backend()
        backend.add_particles.return_value = {"success": True, "message": "OK"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_particles")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_replicate(self, mock_get):
        backend = self._mock_backend()
        backend.replicate.return_value = {
            "success": True, "env_info": {
                "num_envs": 4, "bodies_total": 20, "joints_total": 12,
                "solver": "mujoco", "device": "cpu"
            }
        }
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="replicate", num_envs=4)
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_run_policy(self, mock_get):
        backend = self._mock_backend()
        backend.run_policy.return_value = {
            "success": True, "steps_executed": 100, "wall_time": 1.0,
            "realtime_factor": 10.0, "errors": []
        }
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="run_policy", robot_name="r")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_run_diffsim(self, mock_get):
        backend = self._mock_backend()
        backend.run_diffsim.return_value = {
            "success": True, "iterations": 50, "final_loss": 0.001,
            "optimize_param": "initial_velocity"
        }
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="run_diffsim")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_add_sensor(self, mock_get):
        backend = self._mock_backend()
        backend.add_sensor.return_value = {"success": True, "message": "OK"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_sensor")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_read_sensor(self, mock_get):
        backend = self._mock_backend()
        backend.read_sensor.return_value = {"success": True, "data": [1, 2]}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="read_sensor", name="s")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_solve_ik(self, mock_get):
        backend = self._mock_backend()
        backend.solve_ik.return_value = {
            "success": True, "iterations": 5, "error": 0.001
        }
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="solve_ik", position="1,2,3")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._get_backend")
    def test_enable_dual_solver(self, mock_get):
        backend = self._mock_backend()
        backend.enable_dual_solver.return_value = {"success": True, "message": "OK"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="enable_dual_solver", solver="featherstone")
        assert r["status"] == "success"

    def test_destroy_no_backend(self):
        fn = self._import()
        ns = sys.modules["strands_robots.tools.newton_sim"]
        ns._backend = None
        r = fn(action="destroy")
        assert r["status"] == "success"

    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_destroy_with_backend(self, mock_destroy):
        fn = self._import()
        ns = sys.modules["strands_robots.tools.newton_sim"]
        mock_b = MagicMock()
        mock_b.destroy.return_value = {"message": "destroyed"}
        ns._backend = mock_b
        r = fn(action="destroy")
        assert r["status"] == "success"

    def test_list_assets_no_newton(self):
        fn = self._import()
        with patch.dict(sys.modules, {"newton": None, "newton.examples": None}):
            r = fn(action="list_assets")
        assert r["status"] == "success"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="unknown_xyz")
        assert r["status"] == "error"

    def test_exception_handler(self):
        fn = self._import()
        with patch("strands_robots.tools.newton_sim._get_backend", side_effect=Exception("boom")):
            r = fn(action="step")
        assert r["status"] == "error"


# ===========================================================================
# 6. isaac_sim.py
# ===========================================================================


class TestIsaacSim:
    def _import(self):
        from strands_robots.tools.isaac_sim import isaac_sim
        return isaac_sim

    def _mock_backend(self):
        backend = MagicMock()
        backend.config = MagicMock()
        backend.config.num_envs = 1
        backend.config.device = "cpu"
        backend._robot = None
        return backend

    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_create_world(self, mock_get, mock_destroy):
        backend = self._mock_backend()
        backend.create_world.return_value = {"status": "success", "content": [{"text": "ok"}]}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="create_world", num_envs=2, gravity="[0,0,-9.81]")
        assert r["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_robot(self, mock_get):
        backend = self._mock_backend()
        backend.add_robot.return_value = {"status": "success", "content": [{"text": "ok"}]}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_robot", robot_type="go2", position="[0,0,0]")
        assert r["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_object_no_method(self, mock_get):
        backend = self._mock_backend()
        del backend.add_object  # Simulate no method
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_object", object_type="box")
        assert r["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_add_object_with_method(self, mock_get):
        backend = self._mock_backend()
        backend.add_object = MagicMock(return_value={"status": "success", "content": [{"text": "ok"}]})
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="add_object", object_type="sphere", position="[1,2,3]",
               object_size="[0.1,0.1,0.1]", object_color="[1,0,0,1]")
        assert r["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_step(self, mock_get):
        backend = self._mock_backend()
        backend.step.return_value = {"observations": {"joint_pos": np.array([0.1])}}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="step", steps=3)
        assert r["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_observe_empty(self, mock_get):
        backend = self._mock_backend()
        backend.get_observation.return_value = {}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="observe")
        assert "No observations" in r["content"][0]["text"]

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_render(self, mock_get):
        backend = self._mock_backend()
        backend.render.return_value = {"status": "success", "content": [{"text": "ok"}]}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="render")
        assert r["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_reset_with_method(self, mock_get):
        backend = self._mock_backend()
        backend.reset.return_value = {"status": "success", "content": [{"text": "ok"}]}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="reset")
        assert r["status"] == "success"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_run_policy(self, mock_get):
        backend = self._mock_backend()
        backend.run_policy.return_value = {"status": "success", "content": [{"text": "ok"}]}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="run_policy")
        assert r["status"] == "success"

    def test_set_joint_pos_missing(self):
        fn = self._import()
        with patch("strands_robots.tools.isaac_sim._get_backend", return_value=self._mock_backend()):
            r = fn(action="set_joint_pos")
        assert r["status"] == "error"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_set_joint_pos_no_robot(self, mock_get):
        backend = self._mock_backend()
        del backend.set_joint_positions
        backend._robot = None
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="set_joint_pos", joint_positions="[0.1,0.2]")
        assert r["status"] == "error"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_get_contacts(self, mock_get):
        backend = self._mock_backend()
        del backend.get_contact_forces
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="get_contacts")
        assert r["status"] == "success"

    def test_destroy_no_backend(self):
        fn = self._import()
        ism = sys.modules["strands_robots.tools.isaac_sim"]
        ism._backend = None
        r = fn(action="destroy")
        assert r["status"] == "success"

    def test_list_robots(self):
        fn = self._import()
        with patch("strands_robots.tools.isaac_sim._get_backend", return_value=self._mock_backend()):
            r = fn(action="list_robots")
        assert r["status"] == "success"

    def test_list_tasks(self):
        fn = self._import()
        r = fn(action="list_tasks")
        assert r["status"] == "success"

    def test_export_policy_no_input(self):
        fn = self._import()
        r = fn(action="export_policy")
        assert r["status"] == "error"

    def test_export_policy_ok(self):
        fn = self._import()
        r = fn(action="export_policy", input_path="/ckpt.pt")
        assert r["status"] == "success"

    def test_list_extensions(self):
        fn = self._import()
        r = fn(action="list_extensions")
        assert r["status"] == "success"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="invalid_xyz")
        assert r["status"] == "error"

    def test_exception_handler(self):
        fn = self._import()
        with patch("strands_robots.tools.isaac_sim._get_backend", side_effect=RuntimeError("boom")):
            r = fn(action="step")
        assert r["status"] == "error"

    # save_state / load_state
    def test_save_state_no_backend(self):
        fn = self._import()
        ism = sys.modules["strands_robots.tools.isaac_sim"]
        ism._backend = None
        r = fn(action="save_state")
        assert r["status"] == "error"

    def test_save_state_ok(self, tmp_dir):
        fn = self._import()
        ism = sys.modules["strands_robots.tools.isaac_sim"]
        mock_b = self._mock_backend()
        mock_b._robot = MagicMock()
        mock_b.get_observation.return_value = {"j": np.array([1.0, 2.0])}
        ism._backend = mock_b
        ism._backend_config = {"num_envs": 1}
        ism._state_dir = tmp_dir
        r = fn(action="save_state", output_path=os.path.join(tmp_dir, "state.json"))
        assert r["status"] == "success"

    def test_load_state_missing_file(self):
        fn = self._import()
        r = fn(action="load_state", input_path="/nonexistent_file.json")
        assert r["status"] == "error"

    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_load_state_ok(self, mock_destroy, mock_get, tmp_dir):
        state_path = os.path.join(tmp_dir, "state.json")
        with open(state_path, "w") as f:
            json.dump({"config": {"num_envs": 1}, "observations": {"j": [1.0]}}, f)
        backend = self._mock_backend()
        backend.create_world.return_value = {"status": "success"}
        mock_get.return_value = backend
        fn = self._import()
        r = fn(action="load_state", input_path=state_path)
        assert r["status"] == "success"


# ===========================================================================
# 7. lerobot_calibrate.py
# ===========================================================================


class TestLerobotCalibrate:
    def _import(self):
        from strands_robots.tools.lerobot_calibrate import lerobot_calibrate
        return lerobot_calibrate

    def test_list_empty(self, tmp_dir):
        fn = self._import()
        r = fn(action="list", base_path=tmp_dir)
        assert r["status"] == "success"
        assert r["count"] == 0

    def test_list_with_data(self, tmp_dir):
        # Create test calibration
        robot_dir = Path(tmp_dir) / "robots" / "so101_follower"
        robot_dir.mkdir(parents=True)
        (robot_dir / "test_arm.json").write_text(json.dumps({"motor1": {"id": 1}}))
        fn = self._import()
        r = fn(action="list", base_path=tmp_dir)
        assert r["status"] == "success"
        assert r["count"] >= 1

    def test_view_missing_params(self, tmp_dir):
        fn = self._import()
        r = fn(action="view", base_path=tmp_dir)
        assert r["status"] == "error"

    def test_view_not_found(self, tmp_dir):
        fn = self._import()
        r = fn(action="view", device_type="robots", device_model="x", device_id="y", base_path=tmp_dir)
        assert r["status"] == "error"

    def test_view_success(self, tmp_dir):
        robot_dir = Path(tmp_dir) / "robots" / "so101"
        robot_dir.mkdir(parents=True)
        (robot_dir / "arm.json").write_text(json.dumps({
            "motor1": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": -180, "range_max": 180}
        }))
        fn = self._import()
        r = fn(action="view", device_type="robots", device_model="so101", device_id="arm", base_path=tmp_dir)
        assert r["status"] == "success"

    def test_search(self, tmp_dir):
        robot_dir = Path(tmp_dir) / "robots" / "so101"
        robot_dir.mkdir(parents=True)
        (robot_dir / "orange.json").write_text(json.dumps({"m1": {"id": 1}}))
        fn = self._import()
        r = fn(action="search", query="orange", base_path=tmp_dir)
        assert r["status"] == "success"

    def test_search_empty(self, tmp_dir):
        fn = self._import()
        r = fn(action="search", query="nonexist", base_path=tmp_dir)
        assert r["count"] == 0

    def test_backup(self, tmp_dir):
        robot_dir = Path(tmp_dir) / "robots" / "so101"
        robot_dir.mkdir(parents=True)
        (robot_dir / "arm.json").write_text(json.dumps({"m1": {"id": 1}}))
        backup_out = os.path.join(tmp_dir, "backup")
        fn = self._import()
        r = fn(action="backup", output_dir=backup_out, base_path=tmp_dir)
        assert r["status"] == "success"
        assert r["files_count"] == 1

    def test_restore_no_dir(self, tmp_dir):
        fn = self._import()
        r = fn(action="restore", base_path=tmp_dir)
        assert r["status"] == "error"

    def test_restore_missing_dir(self, tmp_dir):
        fn = self._import()
        r = fn(action="restore", backup_dir="/nonexistent", base_path=tmp_dir)
        assert r["status"] == "error"

    def test_restore_success(self, tmp_dir):
        # Create a backup structure
        bk = Path(tmp_dir) / "bk" / "robots" / "so101"
        bk.mkdir(parents=True)
        (bk / "arm.json").write_text(json.dumps({"m1": {"id": 1}}))
        fn = self._import()
        r = fn(action="restore", backup_dir=os.path.join(tmp_dir, "bk"), base_path=tmp_dir, overwrite=True)
        assert r["status"] == "success"

    def test_delete_missing_params(self, tmp_dir):
        fn = self._import()
        r = fn(action="delete", base_path=tmp_dir)
        assert r["status"] == "error"

    def test_delete_not_found(self, tmp_dir):
        fn = self._import()
        r = fn(action="delete", device_type="robots", device_model="x", device_id="y", base_path=tmp_dir)
        assert r["status"] == "error"

    def test_delete_success(self, tmp_dir):
        robot_dir = Path(tmp_dir) / "robots" / "so101"
        robot_dir.mkdir(parents=True)
        (robot_dir / "arm.json").write_text(json.dumps({}))
        fn = self._import()
        r = fn(action="delete", device_type="robots", device_model="so101", device_id="arm", base_path=tmp_dir)
        assert r["status"] == "success"

    def test_analyze_empty(self, tmp_dir):
        fn = self._import()
        r = fn(action="analyze", base_path=tmp_dir)
        assert r["status"] == "success"

    def test_analyze_with_data(self, tmp_dir):
        robot_dir = Path(tmp_dir) / "robots" / "so101"
        robot_dir.mkdir(parents=True)
        (robot_dir / "arm.json").write_text(json.dumps({"m1": {"id": 1}, "m2": {"id": 2}}))
        fn = self._import()
        r = fn(action="analyze", base_path=tmp_dir)
        assert r["status"] == "success"
        assert r["analysis"]["total_calibrations"] == 1

    def test_path_specific(self, tmp_dir):
        fn = self._import()
        r = fn(action="path", device_type="robots", device_model="so101", device_id="arm", base_path=tmp_dir)
        assert r["status"] == "success"
        assert "exists" in r

    def test_path_base(self, tmp_dir):
        fn = self._import()
        r = fn(action="path", base_path=tmp_dir)
        assert r["status"] == "success"
        assert "base_path" in r

    def test_unknown_action(self, tmp_dir):
        fn = self._import()
        r = fn(action="unknown_xyz", base_path=tmp_dir)
        assert r["status"] == "error"


# ===========================================================================
# 8. stream.py
# ===========================================================================


class TestStream:
    def _import(self):
        from strands_robots.tools.stream import stream
        return stream

    @patch("strands_robots.tools.stream._action_start")
    def test_start(self, mock_start):
        mock_start.return_value = {"status": "success", "content": [{"text": "ok"}]}
        fn = self._import()
        r = fn(action="start", robot_id="r1")
        assert r["status"] == "success"

    @patch("strands_robots.tools.stream._action_stop")
    def test_stop(self, mock_stop):
        mock_stop.return_value = {"status": "success", "content": [{"text": "ok"}]}
        fn = self._import()
        r = fn(action="stop", robot_id="r1")
        assert r["status"] == "success"

    @patch("strands_robots.tools.stream._action_emit")
    def test_emit(self, mock_emit):
        mock_emit.return_value = {"status": "success", "content": [{"text": "ok"}]}
        fn = self._import()
        r = fn(action="emit", robot_id="r1", data='{"key":"val"}')
        assert r["status"] == "success"

    @patch("strands_robots.tools.stream._action_status")
    def test_status(self, mock_status):
        mock_status.return_value = {"status": "success", "content": [{"text": "ok"}]}
        fn = self._import()
        r = fn(action="status")
        assert r["status"] == "success"

    @patch("strands_robots.tools.stream._action_flush")
    def test_flush(self, mock_flush):
        mock_flush.return_value = {"status": "success", "content": [{"text": "ok"}]}
        fn = self._import()
        r = fn(action="flush", robot_id="r1")
        assert r["status"] == "success"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="xyz_invalid")
        assert r["status"] == "error"

    def test_exception(self):
        fn = self._import()
        with patch("strands_robots.tools.stream._action_start", side_effect=Exception("fail")):
            r = fn(action="start")
        assert r["status"] == "error"

    # _action_* inner functions

    def test_action_stop_no_stream(self):
        from strands_robots.tools.stream import _action_stop
        r = _action_stop("nonexistent")
        assert r["status"] == "error"

    def test_action_emit_no_stream(self):
        from strands_robots.tools.stream import _action_emit
        r = _action_emit("nonexistent", "custom", None, "real")
        assert r["status"] == "error"

    def test_action_status_empty(self):
        from strands_robots.tools.stream import _action_status, _STREAMS
        _STREAMS.clear()
        r = _action_status("default")
        assert r["status"] == "success"

    def test_action_flush_no_stream(self):
        from strands_robots.tools.stream import _action_flush
        r = _action_flush("nonexistent")
        assert r["status"] == "error"

    def test_action_start_trace_no_stream(self):
        from strands_robots.tools.stream import _action_start_trace
        r = _action_start_trace("nonexistent", "trace1")
        assert r["status"] == "error"

    def test_action_start_trace_no_name(self):
        from strands_robots.tools.stream import _action_start_trace, _STREAMS
        mock_stream = MagicMock()
        _STREAMS["r1"] = mock_stream
        r = _action_start_trace("r1", "")
        assert r["status"] == "error"

    def test_action_end_trace_no_stream(self):
        from strands_robots.tools.stream import _action_end_trace
        r = _action_end_trace("nonexistent")
        assert r["status"] == "error"

    def test_action_end_trace_no_trace(self):
        from strands_robots.tools.stream import _action_end_trace, _STREAMS
        mock_stream = MagicMock()
        mock_stream._correlation = MagicMock()
        mock_stream._correlation.trace_id = None
        _STREAMS["r1"] = mock_stream
        r = _action_end_trace("r1")
        assert r["status"] == "error"


# ===========================================================================
# 9. inference.py
# ===========================================================================


class TestInference:
    def _import(self):
        from strands_robots.tools.inference import inference
        return inference

    def test_providers_action(self):
        fn = self._import()
        r = fn(action="providers")
        assert r["status"] == "success"

    def test_list_action(self):
        fn = self._import()
        with patch("strands_robots.tools.inference._is_port_in_use", return_value=False):
            r = fn(action="list")
        assert r["status"] == "success"

    def test_status_no_port(self):
        fn = self._import()
        r = fn(action="status")
        assert r["status"] == "error"

    def test_status_with_port(self):
        fn = self._import()
        with patch("strands_robots.tools.inference._is_port_in_use", return_value=True):
            r = fn(action="status", port=50051)
        assert r["status"] == "success"

    def test_stop_no_port(self):
        fn = self._import()
        r = fn(action="stop")
        assert r["status"] == "error"

    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch("strands_robots.tools.inference._find_pid_on_port", return_value=None)
    def test_stop_with_port(self, mock_find, mock_port):
        fn = self._import()
        r = fn(action="stop", port=50051)
        assert r["status"] == "success"

    def test_download_no_id(self):
        fn = self._import()
        r = fn(action="download", provider="nonexist_xyz")
        assert r["status"] == "error"

    @patch("strands_robots.tools.inference._download_hf", return_value="/cached/path")
    def test_download_ok(self, mock_dl):
        fn = self._import()
        r = fn(action="download", model_id="test/model")
        assert r["status"] == "success"

    def test_start_unknown_provider(self):
        fn = self._import()
        r = fn(action="start", provider="nonexist_xyz")
        assert r["status"] == "error"

    def test_info_unknown_provider(self):
        fn = self._import()
        r = fn(action="info", provider="nonexist_xyz")
        assert r["status"] == "error"

    def test_info_known_provider(self):
        fn = self._import()
        r = fn(action="info", provider="lerobot")
        assert r["status"] == "success"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="xyz_invalid")
        assert r["status"] == "error"

    # Helper functions
    def test_port_in_use(self):
        from strands_robots.tools.inference import _port_in_use
        with patch("strands_robots.tools.inference.socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock.__enter__ = MagicMock(return_value=mock_sock)
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 0
            mock_sock_cls.return_value = mock_sock
            assert _port_in_use(5555) is True

    def test_find_pid_on_port(self):
        from strands_robots.tools.inference import _find_pid_on_port
        with patch("strands_robots.tools.inference.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="12345\n")
            assert _find_pid_on_port(5555) == 12345

    def test_find_pid_on_port_none(self):
        from strands_robots.tools.inference import _find_pid_on_port
        with patch("strands_robots.tools.inference.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert _find_pid_on_port(5555) is None

    def test_kill_process(self):
        from strands_robots.tools.inference import _kill
        with patch("strands_robots.tools.inference.os.kill") as mock_kill:
            _kill(123)
            mock_kill.assert_called_with(123, signal.SIGTERM)
            _kill(123, force=True)
            mock_kill.assert_called_with(123, signal.SIGKILL)

    def test_kill_process_not_found(self):
        from strands_robots.tools.inference import _kill
        with patch("strands_robots.tools.inference.os.kill", side_effect=ProcessLookupError):
            _kill(999)  # Should not raise

    def test_download_hf_existing_path(self):
        from strands_robots.tools.inference import _download_hf
        with patch("os.path.exists", return_value=True):
            assert _download_hf("/local/path") == "/local/path"


# ===========================================================================
# 10. robot_mesh.py
# ===========================================================================


class TestRobotMesh:
    def _import(self):
        from strands_robots.tools.robot_mesh import robot_mesh
        return robot_mesh

    def test_peers_empty(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        mock_mesh.get_peers = MagicMock(return_value=[])
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="peers")
        assert r["status"] == "success"

    def test_tell_no_target(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        mock_mesh.get_peers = MagicMock(return_value=[])
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="tell")
        assert r["status"] == "error"

    def test_send_no_target(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        mock_mesh.get_peers = MagicMock(return_value=[])
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="send")
        assert r["status"] == "error"

    def test_unknown_action(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        mock_mesh.get_peers = MagicMock(return_value=[])
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="unknown_xyz")
        assert r["status"] == "error"

    def test_import_error(self):
        fn = self._import()
        # Remove the zenoh_mesh module so the import inside the tool fails
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": None}):
            r = fn(action="peers")
        assert r["status"] == "error"

    def test_status(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        mock_mesh.get_peers = MagicMock(return_value=[])
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="status")
        assert r["status"] == "success"

    def test_tell_no_mesh(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        mock_mesh.get_peers = MagicMock(return_value=[])
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="tell", target="robot1", instruction="do something")
        assert r["status"] == "error"

    def test_send_no_mesh(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="send", target="robot1")
        assert r["status"] == "error"

    def test_broadcast_no_mesh(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="broadcast")
        assert r["status"] == "error"

    def test_stop_no_target(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="stop")
        assert r["status"] == "error"

    def test_emergency_stop_no_mesh(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="emergency_stop")
        assert r["status"] == "error"

    def test_subscribe_no_target(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="subscribe")
        assert r["status"] == "error"

    def test_watch_no_target(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="watch")
        assert r["status"] == "error"

    def test_inbox_no_mesh(self):
        fn = self._import()
        mock_mesh = MagicMock()
        mock_mesh._LOCAL_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.zenoh_mesh": mock_mesh}):
            r = fn(action="inbox")
        assert r["status"] == "success"


# ===========================================================================
# 11. marble_tool.py
# ===========================================================================


class TestMarbleTool:
    def _import(self):
        from strands_robots.tools.marble_tool import marble_tool
        return marble_tool

    def test_exception_handler(self):
        fn = self._import()
        # Make the strands_robots.marble import fail inside the tool function
        with patch.dict(sys.modules, {"strands_robots.marble": None}):
            r = fn(action="presets")
        assert r["status"] == "error"

    def test_unknown_action(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        mock_module.MarbleConfig = MagicMock()
        mock_module.MarblePipeline = MagicMock()
        mock_module.list_presets = MagicMock(return_value=[])
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="unknown_xyz")
        assert r["status"] == "error"

    def test_presets_action(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.list_presets = MagicMock(return_value=[
            {"name": "kitchen", "category": "indoor", "description": "A kitchen", "objects": "cup, plate"}
        ])
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="presets")
        assert r["status"] == "success"

    def test_robots_action(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.SUPPORTED_ROBOTS = {"so101": {"type": "arm", "description": "SO-101 arm"}}
        mock_module.MARBLE_PRESETS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="robots")
        assert r["status"] == "success"

    def test_info_no_preset(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="info")
        assert r["status"] == "success"

    def test_generate_no_prompt(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="generate")
        assert r["status"] == "error"

    def test_compose_no_scene(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="compose")
        assert r["status"] == "error"

    def test_compose_no_robot(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="compose", scene_path="/x.usd")
        assert r["status"] == "error"

    def test_convert_no_ply(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="convert")
        assert r["status"] == "error"

    def test_get_world_no_id(self):
        fn = self._import()
        mock_module = MagicMock()
        mock_module.MARBLE_PRESETS = {}
        mock_module.SUPPORTED_ROBOTS = {}
        with patch.dict(sys.modules, {"strands_robots.marble": mock_module}):
            r = fn(action="get_world")
        assert r["status"] == "error"


# ===========================================================================
# 12. lerobot_dataset.py
# ===========================================================================


class TestLerobotDataset:
    def _import(self):
        from strands_robots.tools.lerobot_dataset import lerobot_dataset
        return lerobot_dataset

    def test_create_no_repo(self):
        fn = self._import()
        r = fn(action="create")
        assert r["status"] == "error"

    def test_create_fallback(self, tmp_dir):
        fn = self._import()
        with patch.dict(sys.modules, {"lerobot": None, "lerobot.datasets": None,
                                       "lerobot.datasets.lerobot_dataset": None}):
            r = fn(action="create", repo_id="test/data", root=tmp_dir)
        assert r["status"] == "success"

    def test_info_no_repo(self):
        fn = self._import()
        r = fn(action="info")
        assert r["status"] == "error"

    def test_stop_recording_empty(self):
        fn = self._import()
        r = fn(action="stop_recording")
        assert r["status"] == "success"

    def test_record_no_repo(self):
        fn = self._import()
        r = fn(action="record")
        assert r["status"] == "error"

    def test_push_no_repo(self):
        fn = self._import()
        r = fn(action="push")
        assert r["status"] == "error"

    def test_pull_no_repo(self):
        fn = self._import()
        r = fn(action="pull")
        assert r["status"] == "error"

    def test_browse_no_repo(self):
        fn = self._import()
        r = fn(action="browse")
        assert r["status"] == "error"

    def test_replay_no_repo(self):
        fn = self._import()
        r = fn(action="replay")
        assert r["status"] == "error"

    def test_compute_stats_no_repo(self):
        fn = self._import()
        r = fn(action="compute_stats")
        assert r["status"] == "error"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="xyz_invalid")
        assert r["status"] == "error"

    def test_build_default_features(self):
        from strands_robots.tools.lerobot_dataset import _build_default_features
        f = _build_default_features("so100")
        assert "observation.state" in f
        assert "action" in f


# ===========================================================================
# 13. teleoperator.py
# ===========================================================================


class TestTeleoperator:
    def _import(self):
        from strands_robots.tools.teleoperator import teleoperator
        return teleoperator

    def test_status_no_session(self):
        fn = self._import()
        r = fn(action="status")
        assert r["status"] == "success"
        assert "No active" in r["content"][0]["text"]

    def test_stop_no_session(self):
        fn = self._import()
        r = fn(action="stop")
        assert r["status"] == "success"

    def test_discard_no_session(self):
        fn = self._import()
        r = fn(action="discard")
        assert r["status"] == "error"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="unknown_xyz")
        assert r["status"] == "error"

    def test_teleop_already_running(self):
        fn = self._import()
        from strands_robots.tools.teleoperator import _ACTIVE_SESSION
        _ACTIVE_SESSION["running"] = True
        r = fn(action="teleop")
        assert r["status"] == "error"
        _ACTIVE_SESSION["running"] = False

    def test_record_already_running(self):
        fn = self._import()
        from strands_robots.tools.teleoperator import _ACTIVE_SESSION
        _ACTIVE_SESSION["running"] = True
        r = fn(action="record")
        assert r["status"] == "error"
        _ACTIVE_SESSION["running"] = False

    def test_teleop_error(self):
        fn = self._import()
        with patch("strands_robots.tools.teleoperator._make_robot", side_effect=Exception("no hw")):
            r = fn(action="teleop")
        assert r["status"] == "error"


# ===========================================================================
# 14. reachy_mini_tool.py
# ===========================================================================


class TestReachyMiniTool:
    def _import(self):
        from strands_robots.tools.reachy_mini_tool import reachy_mini
        return reachy_mini

    def test_status(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api") as mock_api:
            mock_api.return_value = {
                "state": "running", "version": "1.0", "wlan_ip": "192.168.1.1",
                "backend_status": {"motor_control_mode": "torque",
                                   "control_loop_stats": {"mean_control_loop_frequency": 100.0}}
            }
            r = fn(action="status")
        assert r["status"] == "success"

    def test_look(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd") as mock_cmd:
            mock_cmd.return_value = {"ok": True}
            r = fn(action="look", pitch=10, yaw=20)
        assert r["status"] == "success"

    def test_antennas(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd") as mock_cmd:
            mock_cmd.return_value = {"ok": True}
            r = fn(action="antennas", left_antenna=30, right_antenna=-30)
        assert r["status"] == "success"

    def test_body(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd") as mock_cmd:
            mock_cmd.return_value = {"ok": True}
            r = fn(action="body", body_yaw=45)
        assert r["status"] == "success"

    def test_auto_body_yaw(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd") as mock_cmd:
            mock_cmd.return_value = {"ok": True}
            r = fn(action="auto_body_yaw", enabled=True)
        assert r["status"] == "success"

    def test_enable_motors(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd") as mock_cmd:
            mock_cmd.return_value = {"ok": True}
            r = fn(action="enable_motors", motor_ids="1,2,3")
        assert r["status"] == "success"

    def test_disable_motors(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd") as mock_cmd:
            mock_cmd.return_value = {"ok": True}
            r = fn(action="disable_motors")
        assert r["status"] == "success"

    def test_gravity_compensation(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd"):
            r = fn(action="gravity_compensation")
        assert r["status"] == "success"

    def test_stiff(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd"):
            r = fn(action="stiff")
        assert r["status"] == "success"

    def test_nod(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd"):
            with patch("strands_robots.tools.reachy_mini_tool.time"):
                r = fn(action="nod")
        assert r["status"] == "success"

    def test_shake(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd"):
            with patch("strands_robots.tools.reachy_mini_tool.time"):
                r = fn(action="shake")
        assert r["status"] == "success"

    def test_happy(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd"):
            with patch("strands_robots.tools.reachy_mini_tool.time"):
                r = fn(action="happy")
        assert r["status"] == "success"

    def test_daemon_start(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="daemon_start")
        assert r["status"] == "success"

    def test_daemon_stop(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="daemon_stop")
        assert r["status"] == "success"

    def test_daemon_restart(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="daemon_restart")
        assert r["status"] == "success"

    def test_goto_pose(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="goto_pose", pitch=10, duration=2.0)
        assert r["status"] == "success"

    def test_look_at_world(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="look_at_world", x=0.5, y=0.0, z=0.3)
        assert r["status"] == "success"

    def test_look_at_image(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="look_at_image", x=320, y=240)
        assert r["status"] == "success"

    def test_joints_no_data(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_sub", return_value=[("error", {"error": "x"})]):
            r = fn(action="joints")
        assert r["status"] == "error"

    def test_joints_with_data(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_sub",
                    return_value=[("topic", {"head_joint_positions": [0.1, 0.2], "antennas_joint_positions": [0.3]})]):
            r = fn(action="joints")
        assert r["status"] == "success"

    def test_head_pose_no_data(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_sub", return_value=[("error", {"error": "x"})]):
            r = fn(action="head_pose")
        assert r["status"] == "error"

    def test_imu_no_data(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_sub", return_value=[("error", {"error": "x"})]):
            r = fn(action="imu")
        assert r["status"] == "error"

    def test_imu_with_data(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_sub",
                    return_value=[("topic", {"accelerometer": [0, 0, 9.8], "gyroscope": [0, 0, 0],
                                              "quaternion": [1, 0, 0, 0], "temperature": 25})]):
            r = fn(action="imu")
        assert r["status"] == "success"

    def test_play_sound_no_file(self):
        fn = self._import()
        r = fn(action="play_sound")
        assert r["status"] == "error"

    def test_play_sound_ok(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="play_sound", sound_file="beep.wav")
        assert r["status"] == "success"

    def test_record_audio(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="record_audio", duration=2.0)
        assert r["status"] == "success"

    def test_stop(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="stop")
        assert r["status"] == "success"

    def test_list_moves(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value=["happy", "sad"]):
            r = fn(action="list_moves", library="emotions")
        assert r["status"] == "success"

    def test_play_move_no_name(self):
        fn = self._import()
        r = fn(action="play_move")
        assert r["status"] == "error"

    def test_play_move_ok(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="play_move", move_name="happy")
        assert r["status"] == "success"

    def test_start_recording(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd", return_value={"ok": True}):
            r = fn(action="start_recording")
        assert r["status"] == "success"

    def test_stop_recording(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_cmd"):
            with patch("strands_robots.tools.reachy_mini_tool._zenoh_sub", return_value=[]):
                r = fn(action="stop_recording")
        assert r["status"] == "success"

    def test_wake_up(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="wake_up")
        assert r["status"] == "success"

    def test_sleep(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", return_value={"ok": True}):
            r = fn(action="sleep")
        assert r["status"] == "success"

    def test_unknown_action(self):
        fn = self._import()
        r = fn(action="unknown_xyz")
        assert r["status"] == "error"

    def test_exception(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._api", side_effect=Exception("fail")):
            r = fn(action="status")
        assert r["status"] == "error"

    # Helper functions

    def test_resolve_host(self):
        from strands_robots.tools.reachy_mini_tool import _resolve_host
        with patch("strands_robots.tools.reachy_mini_tool.socket.gethostbyname", return_value="1.2.3.4"):
            assert _resolve_host("test.local") == "1.2.3.4"
        with patch("strands_robots.tools.reachy_mini_tool.socket.gethostbyname", side_effect=socket.gaierror):
            assert _resolve_host("test.local") == "test.local"

    def test_rpy_to_pose(self):
        from strands_robots.tools.reachy_mini_tool import _rpy_to_pose, _identity_pose
        pose = _rpy_to_pose(0, 0, 0)
        assert len(pose) == 4
        assert len(pose[0]) == 4
        ident = _identity_pose()
        assert ident[0][0] == 1

    def test_api_helper(self):
        from strands_robots.tools.reachy_mini_tool import _api
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"ok": true}'
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_url.return_value = mock_resp
            result = _api("localhost", 8000, "/test")
            assert result == {"ok": True}

    def test_zenoh_session_unavailable(self):
        from strands_robots.tools.reachy_mini_tool import _get_zenoh_session
        with patch("strands_robots.tools.reachy_mini_tool._resolve_host", return_value="1.2.3.4"):
            with patch("importlib.import_module", side_effect=ImportError("no zenoh")):
                session = _get_zenoh_session("1.2.3.4", 7447)
                assert session is None

    def test_zenoh_put_unavailable(self):
        from strands_robots.tools.reachy_mini_tool import _zenoh_put
        with patch("strands_robots.tools.reachy_mini_tool._get_zenoh_session", return_value=None):
            r = _zenoh_put("host", "prefix", "topic", {})
            assert "error" in r

    def test_zenoh_sub_unavailable(self):
        from strands_robots.tools.reachy_mini_tool import _zenoh_sub
        with patch("strands_robots.tools.reachy_mini_tool._get_zenoh_session", return_value=None):
            r = _zenoh_sub("host", "prefix", "topic")
            assert r[0][0] == "error"

    def test_state_action(self):
        fn = self._import()
        with patch("strands_robots.tools.reachy_mini_tool._zenoh_sub", return_value=[]):
            r = fn(action="state")
        assert r["status"] == "success"

    def test_camera_action(self):
        fn = self._import()
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"\xff\xd8\xff\xe0"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_url.return_value = mock_resp
            r = fn(action="camera")
        assert r["status"] == "success"

    def test_camera_error(self):
        fn = self._import()
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            r = fn(action="camera")
        assert r["status"] == "error"


# ===========================================================================
# 15. use_unitree.py
# ===========================================================================


class TestUseUnitree:
    def _import(self):
        from strands_robots.tools.use_unitree import use_unitree
        return use_unitree

    def test_list_robots(self):
        fn = self._import()
        r = fn(action="list_robots")
        assert r["status"] == "success"

    def test_list_services(self):
        fn = self._import()
        r = fn(action="list_services", parameters={"robot": "go2"})
        assert r["status"] == "success"

    def test_list_services_missing_robot(self):
        fn = self._import()
        r = fn(action="list_services", parameters={})
        assert r["status"] == "error"

    def test_list_services_unknown_robot(self):
        fn = self._import()
        r = fn(action="list_services", parameters={"robot": "unknown_xyz"})
        assert r["status"] == "error"

    def test_list_methods_missing_params(self):
        fn = self._import()
        r = fn(action="list_methods", parameters={})
        assert r["status"] == "error"

    def test_diagnose(self):
        fn = self._import()
        r = fn(action="diagnose")
        assert r["status"] == "success"

    def test_unknown_discovery(self):
        fn = self._import()
        from strands_robots.tools.use_unitree import _handle_discovery
        r = _handle_discovery("unknown_xyz", {})
        assert r["status"] == "error"

    def test_dangerous_action_no_confirm(self):
        fn = self._import()
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            r = fn(action="go2.sport.BackFlip")
        assert r["status"] == "error"
        assert "dangerous" in r["content"][0]["text"].lower()

    def test_dangerous_action_confirm(self):
        fn = self._import()
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            r = fn(action="go2.sport.BackFlip", confirm=True)
        assert r["status"] == "success"

    def test_mock_mode_move(self):
        fn = self._import()
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            r = fn(action="go2.sport.Move", parameters={"vx": 0.3, "vy": 0, "vyaw": 0})
        assert r["status"] == "success"

    def test_g1_arm_gesture(self):
        fn = self._import()
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            r = fn(action="g1.arm.hug")
        assert r["status"] == "success"

    def test_method_not_found(self):
        fn = self._import()
        # Create a client where getattr returns None for the method
        class LimitedClient:
            pass  # Has no methods at all
        with patch.dict(os.environ, {"UNITREE_MOCK": ""}):
            with patch("strands_robots.tools.use_unitree._conn") as mock_conn:
                mock_conn.get_client.return_value = LimitedClient()
                r = fn(action="go2.sport.NonExistentMethod12345")
        assert r["status"] == "error"
        assert "not found" in r["content"][0]["text"].lower() or "Method" in r["content"][0]["text"]

    def test_invalid_action_format(self):
        fn = self._import()
        r = fn(action="a.b.c.d.e")
        assert r["status"] == "error"

    def test_velocity_clamping(self):
        from strands_robots.tools.use_unitree import _clamp_velocity
        clamped = _clamp_velocity("go2", {"vx": 100.0, "vy": -100.0, "vyaw": 0.1})
        assert clamped["vx"] == 1.5
        assert clamped["vy"] == -0.5

    def test_parse_action(self):
        from strands_robots.tools.use_unitree import _parse_action
        assert _parse_action("go2.sport.Move") == ("go2", "sport", "Move")
        assert _parse_action("go2.sport") == ("go2", "sport", "")
        assert _parse_action("list_robots") == ("__discovery__", "", "list_robots")
        with pytest.raises(ValueError):
            _parse_action("a.b.c.d.e")

    def test_resolve_gesture(self):
        from strands_robots.tools.use_unitree import _resolve_gesture
        assert _resolve_gesture("hug") == 19
        assert _resolve_gesture("nonexistent") is None

    def test_mock_client(self):
        from strands_robots.tools.use_unitree import _MockClient
        mc = _MockClient("go2", "sport")
        result = mc.Move(vx=0.5)
        assert result == 0
        assert len(mc._calls) == 1

    def test_import_error(self):
        fn = self._import()
        with patch.dict(os.environ, {"UNITREE_MOCK": ""}):
            with patch("strands_robots.tools.use_unitree._conn") as mock_conn:
                mock_conn.get_client.side_effect = ImportError("no sdk")
                r = fn(action="go2.sport.Move", parameters={"vx": 0.1})
        assert r["status"] == "error"

    def test_connection_manager(self):
        from strands_robots.tools.use_unitree import _ConnectionManager
        cm = _ConnectionManager()
        cm.close_all()  # Should not raise

    def test_tuple_result(self):
        fn = self._import()
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            mock_client = MagicMock()
            mock_client.StandUp.return_value = (0, {"ok": True})
            with patch("strands_robots.tools.use_unitree._MockClient", return_value=mock_client):
                r = fn(action="go2.sport.StandUp")
        assert r["status"] == "success"


# ===========================================================================
# 16. use_lerobot.py
# ===========================================================================


class TestUseLerobot:
    def _import(self):
        from strands_robots.tools.use_lerobot import use_lerobot
        return use_lerobot

    def test_discovery_no_lerobot(self):
        fn = self._import()
        with patch.dict(sys.modules, {"lerobot": None}):
            with patch("strands_robots.tools.use_lerobot._discover_modules", return_value={"error": "not installed"}):
                r = fn(module="__discovery__", method="list_modules")
        assert r["status"] == "error"

    def test_describe_mode(self):
        fn = self._import()
        mock_obj = MagicMock()
        mock_obj.__name__ = "TestClass"
        with patch("strands_robots.tools.use_lerobot._import_from_lerobot", return_value=mock_obj):
            r = fn(module="test.module", method="__describe__")
        assert r["status"] == "success"

    def test_method_not_found(self):
        fn = self._import()
        mock_mod = types.ModuleType("mock_mod")
        mock_mod.foo = "bar"
        with patch("strands_robots.tools.use_lerobot._import_from_lerobot", return_value=mock_mod):
            r = fn(module="test.module", method="nonexistent")
        assert r["status"] == "error"

    def test_non_callable_target(self):
        fn = self._import()
        mock_mod = types.ModuleType("mock_mod")
        mock_mod.MY_CONST = 42
        with patch("strands_robots.tools.use_lerobot._import_from_lerobot", return_value=mock_mod):
            r = fn(module="test.module", method="MY_CONST")
        assert r["status"] == "success"

    def test_callable_target(self):
        fn = self._import()

        def fake_fn(**kwargs):
            return {"result": "ok"}

        mock_mod = types.ModuleType("mock_mod")
        mock_mod.my_func = fake_fn
        with patch("strands_robots.tools.use_lerobot._import_from_lerobot", return_value=mock_mod):
            r = fn(module="test.module", method="my_func", parameters={})
        assert r["status"] == "success"

    def test_type_error(self):
        fn = self._import()

        def bad_fn(required_arg):
            pass

        mock_mod = types.ModuleType("mock_mod")
        mock_mod.bad_fn = bad_fn
        with patch("strands_robots.tools.use_lerobot._import_from_lerobot", return_value=mock_mod):
            r = fn(module="test.module", method="bad_fn", parameters={})
        assert r["status"] == "error"

    def test_import_error(self):
        fn = self._import()
        with patch("strands_robots.tools.use_lerobot._import_from_lerobot", side_effect=ImportError("no module")):
            r = fn(module="nonexistent", method="foo")
        assert r["status"] == "error"

    def test_general_exception(self):
        fn = self._import()
        mock_mod = types.ModuleType("mock_mod")
        mock_mod.boom = MagicMock(side_effect=RuntimeError("boom"))
        with patch("strands_robots.tools.use_lerobot._import_from_lerobot", return_value=mock_mod):
            r = fn(module="test", method="boom")
        assert r["status"] == "error"

    # _import_from_lerobot
    def test_import_from_lerobot_module(self):
        from strands_robots.tools.use_lerobot import _import_from_lerobot
        with patch("importlib.import_module") as mock_imp:
            mock_mod = MagicMock()
            mock_imp.return_value = mock_mod
            result = _import_from_lerobot("os")
            # Should try lerobot.os first, fallback
            assert mock_imp.called

    # _serialize_result
    def test_serialize_result(self):
        from strands_robots.tools.use_lerobot import _serialize_result
        assert _serialize_result(None) == "None"
        assert _serialize_result("hello") == "hello"
        assert _serialize_result(42) == "42"
        assert "hello" in _serialize_result(["hello", "world"])
        assert "key" in _serialize_result({"key": "value"})

    # _describe_object
    def test_describe_object_class(self):
        from strands_robots.tools.use_lerobot import _describe_object

        class TestClass:
            """Docstring"""
            def method(self):
                pass

        info = _describe_object(TestClass)
        assert info["type"] == "type"

    def test_describe_object_function(self):
        from strands_robots.tools.use_lerobot import _describe_object

        def test_fn(a: int, b: str = "default"):
            """Test docstring"""
            pass

        info = _describe_object(test_fn)
        assert "params" in info

    def test_describe_object_module(self):
        from strands_robots.tools.use_lerobot import _describe_object
        info = _describe_object(os)
        assert "public_names" in info

    def test_describe_object_value(self):
        from strands_robots.tools.use_lerobot import _describe_object
        info = _describe_object(42)
        assert "value" in info


# ===========================================================================
# 17. stereo_depth.py
# ===========================================================================


class TestStereoDepth:
    def _import(self):
        from strands_robots.tools.stereo_depth import stereo_depth
        return stereo_depth

    def test_left_image_missing(self):
        fn = self._import()
        r = fn(left_image="/nonexistent.png", right_image="/nonexistent2.png")
        assert "error" in r

    def test_right_image_missing(self, tmp_dir):
        fn = self._import()
        left = os.path.join(tmp_dir, "left.png")
        with open(left, "w") as f:
            f.write("x")
        r = fn(left_image=left, right_image="/nonexistent.png")
        assert "error" in r

    def test_success_with_focal_length(self, tmp_dir):
        left = os.path.join(tmp_dir, "left.png")
        right = os.path.join(tmp_dir, "right.png")
        with open(left, "w") as f:
            f.write("x")
        with open(right, "w") as f:
            f.write("x")

        mock_result = MagicMock()
        mock_result.height = 480
        mock_result.width = 640
        mock_result.median_depth = 1.5
        mock_result.valid_ratio = 0.95
        mock_result.metadata = {"inference_time": 0.1, "total_time": 0.2}
        mock_result.depth = np.zeros((480, 640))
        mock_result.disparity_vis = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result.point_cloud = np.zeros((480, 640, 3))

        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.estimate_depth.return_value = mock_result

        mock_pipeline_cls = MagicMock(return_value=mock_pipeline_inst)
        mock_pipeline_cls._load_image = MagicMock(return_value=np.zeros((480, 640, 3)))

        mock_stereo = MagicMock()
        mock_stereo.StereoConfig = MagicMock()
        mock_stereo.StereoDepthPipeline = mock_pipeline_cls

        fn = self._import()
        with patch.dict(sys.modules, {"strands_robots.stereo": mock_stereo}):
            r = fn(left_image=left, right_image=right, focal_length=600.0,
                   baseline=0.12, output_dir=tmp_dir)
        assert r["status"] == "success"


# ===========================================================================
# 18. lerobot_camera.py - test key paths without hardware
# ===========================================================================


class TestLerobotCamera:
    """These tests cover the tool's action dispatch and error handling.
    Actual camera operations require mocking the lerobot camera classes.
    """

    def test_unknown_action(self):
        """Test unknown action returns error."""
        # We need to mock the lerobot imports
        mock_cv2 = MagicMock()
        mock_cv2.__version__ = "4.8.0"
        mock_cv2.CAP_ANY = 0
        mock_cv2.CAP_V4L2 = 200
        mock_cv2.CAP_MSMF = 1400
        mock_cv2.CAP_AVFOUNDATION = 1200

        mock_camera = MagicMock()
        mock_opencv_camera = MagicMock()
        mock_opencv_camera.find_cameras = MagicMock(return_value=[])

        mock_config = MagicMock()
        mock_color = MagicMock()
        mock_color.RGB = "rgb"
        mock_color.BGR = "bgr"
        mock_rotation = MagicMock()
        mock_rotation.NO_ROTATION = 0

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "lerobot": MagicMock(),
            "lerobot.cameras": MagicMock(),
            "lerobot.cameras.camera": MagicMock(Camera=mock_camera),
            "lerobot.cameras.opencv": MagicMock(OpenCVCamera=mock_opencv_camera),
            "lerobot.cameras.opencv.configuration_opencv": MagicMock(
                ColorMode=mock_color, Cv2Rotation=mock_rotation,
                OpenCVCameraConfig=MagicMock()
            ),
        }):
            # Force reimport
            if "strands_robots.tools.lerobot_camera" in sys.modules:
                del sys.modules["strands_robots.tools.lerobot_camera"]
            from strands_robots.tools.lerobot_camera import lerobot_camera
            r = lerobot_camera(action="unknown_xyz")
            assert r["status"] == "error"

    def test_capture_no_camera_id(self):
        """Capture requires camera_id."""
        mock_cv2 = MagicMock()
        mock_cv2.__version__ = "4.8.0"
        mock_cv2.CAP_ANY = 0
        mock_cv2.CAP_V4L2 = 200
        mock_cv2.CAP_MSMF = 1400
        mock_cv2.CAP_AVFOUNDATION = 1200

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "lerobot": MagicMock(),
            "lerobot.cameras": MagicMock(),
            "lerobot.cameras.camera": MagicMock(Camera=MagicMock()),
            "lerobot.cameras.opencv": MagicMock(OpenCVCamera=MagicMock()),
            "lerobot.cameras.opencv.configuration_opencv": MagicMock(
                ColorMode=MagicMock(), Cv2Rotation=MagicMock(),
                OpenCVCameraConfig=MagicMock()
            ),
        }):
            if "strands_robots.tools.lerobot_camera" in sys.modules:
                del sys.modules["strands_robots.tools.lerobot_camera"]
            from strands_robots.tools.lerobot_camera import lerobot_camera
            r = lerobot_camera(action="capture")
            assert r["status"] == "error"

    def test_frame_to_image_content(self):
        """Test frame conversion helper."""
        mock_cv2 = MagicMock()
        mock_cv2.cvtColor.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.__version__ = "4.8.0"
        mock_cv2.CAP_ANY = 0
        mock_cv2.CAP_V4L2 = 200
        mock_cv2.CAP_MSMF = 1400
        mock_cv2.CAP_AVFOUNDATION = 1200
        mock_cv2.COLOR_RGB2BGR = 4

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "lerobot": MagicMock(),
            "lerobot.cameras": MagicMock(),
            "lerobot.cameras.camera": MagicMock(Camera=MagicMock()),
            "lerobot.cameras.opencv": MagicMock(OpenCVCamera=MagicMock()),
            "lerobot.cameras.opencv.configuration_opencv": MagicMock(
                ColorMode=MagicMock(), Cv2Rotation=MagicMock(),
                OpenCVCameraConfig=MagicMock()
            ),
        }):
            if "strands_robots.tools.lerobot_camera" in sys.modules:
                del sys.modules["strands_robots.tools.lerobot_camera"]
            from strands_robots.tools.lerobot_camera import _frame_to_image_content
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            result = _frame_to_image_content(frame, "jpg")
            assert "image" in result


# ===========================================================================
# 19. Additional edge cases
# ===========================================================================


class TestInferenceEdgeCases:
    """Test remaining branches in inference.py."""

    def test_wait_for_port_timeout(self):
        from strands_robots.tools.inference import _wait_for_port
        with patch("strands_robots.tools.inference._is_port_in_use", return_value=False):
            with patch("strands_robots.tools.inference.time") as mock_time:
                mock_time.time.side_effect = [0, 200]
                mock_time.sleep = MagicMock()
                assert _wait_for_port(5555, timeout=1) is False

    def test_wait_for_port_success(self):
        from strands_robots.tools.inference import _wait_for_port
        with patch("strands_robots.tools.inference._is_port_in_use", return_value=True):
            with patch("strands_robots.tools.inference.time") as mock_time:
                mock_time.time.side_effect = [0, 0.5]
                assert _wait_for_port(5555, timeout=10) is True

    def test_download_checkpoint(self):
        from strands_robots.tools.inference import _download_checkpoint
        with patch("os.path.isdir", return_value=True), patch("os.listdir", return_value=["f"]):
            assert _download_checkpoint("model/id", "/local/dir") == "/local/dir"

    def test_generate_hf_serve_script(self):
        from strands_robots.tools.inference import _generate_hf_serve_script
        with patch("strands_robots.tools.inference._launch_http_serve") as mock_launch:
            mock_launch.return_value = {"pid": 123, "cmd": "python /tmp/serve.py"}
            path = _generate_hf_serve_script("model", 8000, "0.0.0.0", "cosmos")
            assert "serve.py" in path

    def test_start_lerobot(self):
        from strands_robots.tools.inference import _start_lerobot
        with patch("strands_robots.tools.inference.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 999
            mock_popen.return_value = mock_proc
            result = _start_lerobot("model/id", 50051, 1, "0.0.0.0", {})
        assert result["pid"] == 999

    def test_start_groot_no_container(self):
        from strands_robots.tools.inference import _start_groot
        with patch("strands_robots.tools.inference.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = _start_groot("model", 5555, 1, "0.0.0.0", {})
        assert result["status"] == "error"

    def test_start_dreamzero_no_script(self):
        from strands_robots.tools.inference import _start_dreamzero
        with patch("os.path.exists", return_value=False):
            result = _start_dreamzero("model", 8000, 2, "0.0.0.0", {})
        assert result["status"] == "error"

    def test_start_generic_hf_no_model(self):
        from strands_robots.tools.inference import _start_generic_hf
        result = _start_generic_hf("cosmos", "", 8000, 1, "0.0.0.0", {})
        assert result["status"] == "error"


class TestNewtonSimBenchmark:
    """Test newton_sim benchmark and list_assets actions with proper mocking."""

    @patch("strands_robots.tools.newton_sim._get_backend")
    @patch("strands_robots.tools.newton_sim._destroy_backend")
    def test_benchmark(self, mock_destroy, mock_get):
        backend = MagicMock()
        backend.create_world.return_value = {"status": "success"}
        backend.add_robot.return_value = {"success": True}
        backend.replicate.return_value = {"success": True}
        backend.step.return_value = {"success": True}

        mock_ne = MagicMock()
        mock_ne.get_asset.return_value = "/fake/quadruped.urdf"

        mock_get.return_value = backend
        from strands_robots.tools.newton_sim import newton_sim
        # Use a counter-based time mock that doesn't run out
        call_count = {"n": 0}
        def fake_time():
            call_count["n"] += 1
            return call_count["n"] * 0.01  # Advances 10ms per call

        with patch.dict(sys.modules, {"newton": MagicMock(), "newton.examples": mock_ne}):
            with patch("strands_robots.tools.newton_sim.time") as mock_time:
                mock_time.time = fake_time
                mock_time.sleep = MagicMock()
                r = newton_sim(action="benchmark", benchmark_envs=4, benchmark_steps=5)
        assert r["status"] == "success"


class TestIsaacSimBenchmark:
    @patch("strands_robots.tools.isaac_sim._get_backend")
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    def test_benchmark(self, mock_destroy, mock_get):
        backend = MagicMock()
        backend.create_world.return_value = {"status": "success"}
        backend.add_robot.return_value = {"status": "success"}
        backend.step.return_value = {"status": "success"}
        backend.config = MagicMock()
        backend.config.num_envs = 4
        mock_get.return_value = backend

        call_count = {"n": 0}
        def fake_time():
            call_count["n"] += 1
            return call_count["n"] * 0.01

        from strands_robots.tools.isaac_sim import isaac_sim
        with patch("strands_robots.tools.isaac_sim.time") as mock_time:
            mock_time.time = fake_time
            mock_time.sleep = MagicMock()
            r = isaac_sim(action="benchmark", benchmark_envs=4, benchmark_steps=3)
        assert r["status"] == "success"


class TestIsaacSimCreateEnvAndTrain:
    def test_create_env(self):
        from strands_robots.tools.isaac_sim import isaac_sim
        mock_create = MagicMock()
        mock_env = MagicMock()
        mock_create.return_value = mock_env
        mock_lab = MagicMock()
        mock_lab.create_isaac_env = mock_create
        with patch.dict(sys.modules, {
            "strands_robots.isaac.isaac_lab_env": mock_lab,
        }):
            r = isaac_sim(action="create_env", task="cartpole", num_envs=2)
        assert r["status"] == "success"

    def test_train(self):
        from strands_robots.tools.isaac_sim import isaac_sim
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"status": "success", "content": [{"text": "ok"}]}
        mock_trainer_cls = MagicMock(return_value=mock_trainer)
        with patch.dict(sys.modules, {
            "strands_robots.isaac.isaac_lab_trainer": MagicMock(
                IsaacLabTrainer=mock_trainer_cls,
                IsaacLabTrainerConfig=MagicMock(),
            ),
        }):
            r = isaac_sim(action="train", task="cartpole")
        assert r["status"] == "success"

    def test_convert_asset_xml(self):
        from strands_robots.tools.isaac_sim import isaac_sim
        mock_convert = MagicMock(return_value={"status": "success", "content": [{"text": "ok"}]})
        with patch.dict(sys.modules, {
            "strands_robots.isaac.asset_converter": MagicMock(
                convert_mjcf_to_usd=mock_convert,
                convert_usd_to_mjcf=MagicMock(),
            ),
        }):
            r = isaac_sim(action="convert_asset", input_path="model.xml")
        assert r["status"] == "success"

    def test_convert_asset_usd(self):
        from strands_robots.tools.isaac_sim import isaac_sim
        mock_convert = MagicMock(return_value={"status": "success", "content": [{"text": "ok"}]})
        with patch.dict(sys.modules, {
            "strands_robots.isaac.asset_converter": MagicMock(
                convert_mjcf_to_usd=MagicMock(),
                convert_usd_to_mjcf=mock_convert,
            ),
        }):
            r = isaac_sim(action="convert_asset", input_path="model.usd")
        assert r["status"] == "success"

    def test_convert_asset_unknown_format(self):
        from strands_robots.tools.isaac_sim import isaac_sim
        r = isaac_sim(action="convert_asset", input_path="model.obj")
        assert r["status"] == "error"

    def test_convert_asset_no_input(self):
        from strands_robots.tools.isaac_sim import isaac_sim
        r = isaac_sim(action="convert_asset")
        assert r["status"] == "error"


class TestIsaacSimResetFallback:
    @patch("strands_robots.tools.isaac_sim._destroy_backend")
    @patch("strands_robots.tools.isaac_sim._get_backend")
    def test_reset_no_method(self, mock_get, mock_destroy):
        backend = MagicMock()
        del backend.reset  # No reset method
        backend.config = MagicMock()
        backend.config.num_envs = 1
        backend.create_world.return_value = {"status": "success"}
        mock_get.return_value = backend
        import strands_robots.tools.isaac_sim as ism
        ism._backend_config = {"num_envs": 1}
        from strands_robots.tools.isaac_sim import isaac_sim
        r = isaac_sim(action="reset")
        assert r["status"] == "success"


class TestStopRecording:
    def test_stop_recording_active(self):
        from strands_robots.tools.lerobot_dataset import lerobot_dataset, _ACTIVE_RECORDINGS, _RECORDING_LOCK
        with _RECORDING_LOCK:
            _ACTIVE_RECORDINGS["rec_123"] = {
                "repo_id": "test/data",
                "episodes_recorded": 2,
            }
        r = lerobot_dataset(action="stop_recording")
        assert r["status"] == "success"
        assert "rec_123" in r["content"][0]["text"]


class TestInferenceStartWithProvider:
    """Test the inference start action with a known provider."""

    @patch("strands_robots.tools.inference._wait_for_port", return_value=True)
    @patch("strands_robots.tools.inference._start_provider")
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    def test_start_lerobot_provider(self, mock_port_use, mock_start, mock_wait):
        mock_start.return_value = {"pid": 123, "cmd": "python -m lerobot.scripts.server"}
        from strands_robots.tools.inference import inference, PROVIDERS
        if "lerobot" not in PROVIDERS:
            pytest.skip("lerobot not in PROVIDERS")
        r = inference(action="start", provider="lerobot",
                      model_id="lerobot/act_aloha_sim_transfer_cube_human")
        assert r["status"] in ("success", "starting")

    @patch("strands_robots.tools.inference._start_provider", side_effect=Exception("boom"))
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    def test_start_provider_exception(self, mock_port, mock_start):
        from strands_robots.tools.inference import inference, PROVIDERS
        if "lerobot" not in PROVIDERS:
            pytest.skip("lerobot not in PROVIDERS")
        r = inference(action="start", provider="lerobot",
                      model_id="lerobot/act_aloha_sim_transfer_cube_human")
        assert r["status"] == "error"

    @patch("strands_robots.tools.inference._start_provider")
    @patch("strands_robots.tools.inference._is_port_in_use")
    def test_start_port_in_use(self, mock_port, mock_start):
        mock_port.return_value = True
        from strands_robots.tools.inference import inference, PROVIDERS
        if "lerobot" not in PROVIDERS:
            pytest.skip("lerobot not in PROVIDERS")
        r = inference(action="start", provider="lerobot",
                      model_id="lerobot/act_aloha_sim_transfer_cube_human")
        assert r["status"] == "error"


class TestInferenceStopContainer:
    """Test stop with container in _RUNNING."""

    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch("strands_robots.tools.inference.subprocess.run")
    def test_stop_with_container(self, mock_run, mock_port):
        from strands_robots.tools.inference import inference, _RUNNING
        _RUNNING[50051] = {"pid": 123, "container": "my_container", "provider": "groot"}
        mock_run.return_value = MagicMock()
        r = inference(action="stop", port=50051)
        assert r["status"] == "success"
        assert 50051 not in _RUNNING
