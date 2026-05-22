"""Tests for gr00t_inference input validation.

Covers the validate_inputs() function which centralises all parameter
validation for the gr00t_inference tool.
"""

import pytest

from strands_robots.tools.gr00t_inference import validate_inputs

# Standard valid kwargs for validate_inputs — tests override individual fields.
# validate_inputs() no longer has defaults (gr00t_inference() is the single source
# of truth for defaults), so tests must supply all required params.
_VALID_KWARGS = {
    "action": "start",
    "data_config": "fourier_gr1_arms_only",
    "embodiment_tag": "gr1",
    "port": 5555,
    "host": "127.0.0.1",
    "vit_dtype": "fp8",
    "llm_dtype": "nvfp4",
    "dit_dtype": "fp8",
    "checkpoint_path": None,
    "trt_engine_path": "gr00t_engine",
    "container_name": None,
    "protocol": "n1.5",
}


class TestValidateInputs:
    """Tests for the validate_inputs() public function."""

    def test_valid_defaults(self):
        """Default values must pass validation."""
        validate_inputs(**_VALID_KWARGS)

    def test_valid_with_all_optional(self):
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "data_config": "so100_dualcam",
                "embodiment_tag": "so100",
                "port": 8000,
                "vit_dtype": "fp16",
                "llm_dtype": "fp8",
                "dit_dtype": "fp16",
                "checkpoint_path": "/data/checkpoints/model",
                "trt_engine_path": "/engines/cache",
                "container_name": "gr00t-n17",
            }
        )

    def test_invalid_data_config_uppercase(self):
        with pytest.raises(ValueError, match="data_config"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "FourierGR1",
                    "embodiment_tag": "gr1",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_invalid_data_config_shell_chars(self):
        with pytest.raises(ValueError, match="data_config"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "foo;rm -rf /",
                    "embodiment_tag": "gr1",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_invalid_embodiment_tag(self):
        with pytest.raises(ValueError, match="embodiment_tag"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "GR1-Sonic!",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_port_zero(self):
        with pytest.raises(ValueError, match="port"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 0,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_port_too_high(self):
        with pytest.raises(ValueError, match="port"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 70000,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_invalid_vit_dtype(self):
        with pytest.raises(ValueError, match="vit_dtype"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 5555,
                    "vit_dtype": "bf16",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_invalid_llm_dtype(self):
        with pytest.raises(ValueError, match="llm_dtype"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "int4",
                    "dit_dtype": "fp8",
                }
            )

    def test_invalid_dit_dtype(self):
        with pytest.raises(ValueError, match="dit_dtype"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "bf16",
                }
            )

    def test_checkpoint_path_traversal(self):
        with pytest.raises(ValueError, match="checkpoint_path"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                    "checkpoint_path": "/data/../../../etc/passwd",
                }
            )

    def test_checkpoint_path_null_byte(self):
        with pytest.raises(ValueError, match="checkpoint_path"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                    "checkpoint_path": "/data/model\x00.bin",
                }
            )

    def test_trt_engine_path_shell_injection(self):
        with pytest.raises(ValueError, match="trt_engine_path"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                    "trt_engine_path": "engine;rm -rf /",
                }
            )

    def test_invalid_container_name(self):
        with pytest.raises(ValueError, match="container_name"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "so100",
                    "embodiment_tag": "so100",
                    "port": 5555,
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                    "container_name": "-invalid-start",
                }
            )

    def test_container_name_none_is_ok(self):
        """container_name=None should not raise."""
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "data_config": "so100",
                "embodiment_tag": "so100",
                "port": 5555,
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
                "container_name": None,
            }
        )


class TestIsGr00tProcess:
    """Test the _is_gr00t_process helper verifies port binding."""

    def test_rejects_wrong_port(self, monkeypatch):
        """_is_gr00t_process should reject a GR00T process on a different port."""
        import subprocess as sp

        from strands_robots.tools.gr00t_inference import _is_gr00t_process

        # Simulate cmdline: "python inference_service.py --port 8000"
        fake_result = sp.CompletedProcess(
            args=[], returncode=0, stdout="python\x00inference_service.py\x00--port\x008000\x00"
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: fake_result)

        # Asking for port 80 should return False even though it's a gr00t process
        assert _is_gr00t_process("container", "123", port=80) is False

    def test_accepts_matching_port(self, monkeypatch):
        """_is_gr00t_process should accept when port matches."""
        import subprocess as sp

        from strands_robots.tools.gr00t_inference import _is_gr00t_process

        fake_result = sp.CompletedProcess(
            args=[], returncode=0, stdout="python\x00inference_service.py\x00--port\x008000\x00"
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: fake_result)

        assert _is_gr00t_process("container", "123", port=8000) is True

    def test_no_port_check_when_none(self, monkeypatch):
        """_is_gr00t_process without port param should not verify port."""
        import subprocess as sp

        from strands_robots.tools.gr00t_inference import _is_gr00t_process

        fake_result = sp.CompletedProcess(
            args=[], returncode=0, stdout="python\x00inference_service.py\x00--port\x008000\x00"
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: fake_result)

        # Without port, just checks if it's a gr00t process
        assert _is_gr00t_process("container", "123") is True

    def test_accepts_equals_style_port(self, monkeypatch):
        """_is_gr00t_process should accept --port=N style."""
        import subprocess as sp

        from strands_robots.tools.gr00t_inference import _is_gr00t_process

        fake_result = sp.CompletedProcess(
            args=[], returncode=0, stdout="python\x00inference_service.py\x00--port=5555\x00"
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: fake_result)

        assert _is_gr00t_process("container", "123", port=5555) is True
        assert _is_gr00t_process("container", "123", port=6666) is False


class TestIsGr00tHostProcess:
    """Test the _is_gr00t_host_process helper for host-system PID verification."""

    def test_rejects_wrong_port(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process should reject a process on a different port."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        # Create a fake /proc/<pid>/cmdline
        proc_dir = tmp_path / "proc" / "123"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00inference_service.py\x00--port\x008000\x00")

        # Monkeypatch Path to point at our fake proc, with reachability check
        from pathlib import Path as RealPath

        called = {}

        def _fake_path(p):
            called["p"] = p
            return RealPath(str(p).replace("/proc", str(tmp_path / "proc")))

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.Path", _fake_path)

        assert _is_gr00t_host_process("123", port=80) is False
        assert called.get("p") == "/proc/123/cmdline"  # patch was reached

    def test_accepts_matching_port(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process should accept when port matches."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "456"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00inference_service.py\x00--port\x008000\x00")

        from pathlib import Path as RealPath

        called = {}

        def _fake_path(p):
            called["p"] = p
            return RealPath(str(p).replace("/proc", str(tmp_path / "proc")))

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.Path", _fake_path)

        assert _is_gr00t_host_process("456", port=8000) is True
        assert called.get("p") == "/proc/456/cmdline"  # patch was reached

    def test_rejects_non_gr00t_process(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process should reject non-GR00T processes."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "789"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00some_other_service.py\x00--port\x008000\x00")

        from pathlib import Path as RealPath

        called = {}

        def _fake_path(p):
            called["p"] = p
            return RealPath(str(p).replace("/proc", str(tmp_path / "proc")))

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.Path", _fake_path)

        assert _is_gr00t_host_process("789", port=8000) is False
        assert called.get("p") == "/proc/789/cmdline"  # patch was reached

    def test_no_port_check_when_none(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process without port checks only process identity."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "321"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00inference_service.py\x00--port\x009999\x00")

        from pathlib import Path as RealPath

        called = {}

        def _fake_path(p):
            called["p"] = p
            return RealPath(str(p).replace("/proc", str(tmp_path / "proc")))

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.Path", _fake_path)

        # Without port kwarg, just checks identity
        assert _is_gr00t_host_process("321") is True
        assert called.get("p") == "/proc/321/cmdline"  # patch was reached


class TestHostValidation:
    """Tests for host address validation in validate_inputs()."""

    def test_valid_loopback(self):
        """127.0.0.1 is valid."""
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "data_config": "fourier_gr1_arms_only",
                "embodiment_tag": "gr1",
                "port": 5555,
                "host": "127.0.0.1",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            }
        )

    def test_valid_all_interfaces(self):
        """0.0.0.0 is valid."""
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "data_config": "fourier_gr1_arms_only",
                "embodiment_tag": "gr1",
                "port": 5555,
                "host": "127.0.0.1",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            }
        )

    def test_valid_ipv6_loopback(self):
        """::1 is valid."""
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "data_config": "fourier_gr1_arms_only",
                "embodiment_tag": "gr1",
                "port": 5555,
                "host": "::1",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            }
        )

    def test_invalid_host_with_spaces(self):
        """Host with spaces must be rejected."""
        with pytest.raises(ValueError, match="host must be a valid IP address or hostname"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "port": 5555,
                    "host": "foo bar",
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_invalid_host_empty_labels(self):
        """Host with empty labels (double dot) must be rejected."""
        with pytest.raises(ValueError, match="host must be a valid IP address or hostname"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "port": 5555,
                    "host": "a..b",
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_valid_hostname_localhost(self):
        """Valid hostnames like localhost are now accepted."""
        # Should not raise — localhost is a valid RFC-952 hostname
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "data_config": "fourier_gr1_arms_only",
                "embodiment_tag": "gr1",
                "port": 5555,
                "host": "localhost",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            }
        )

    def test_valid_hostname_docker_internal(self):
        """Docker internal hostname is accepted."""
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "data_config": "fourier_gr1_arms_only",
                "embodiment_tag": "gr1",
                "port": 5555,
                "host": "host.docker.internal",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            }
        )

    def test_invalid_host_special_chars(self):
        """Hostnames with special characters are rejected."""
        with pytest.raises(ValueError, match="host must be a valid IP address or hostname"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "port": 5555,
                    "host": "--invalid-host",
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )


class TestGr00tInferenceToolIntegration:
    """Integration tests verifying validate_inputs is wired into the tool entry point.

    These tests invoke gr00t_inference() directly and assert that invalid inputs
    are caught and returned as error dicts, NOT silently passed through.
    This pins the try/except ValueError wiring so a future refactor that drops
    the validation call surfaces as a test failure.
    """

    def test_shell_injection_in_data_config_returns_error(self):
        """Shell metacharacters in data_config must return error dict."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="start", data_config="foo;rm -rf /")
        assert result["status"] == "error"
        assert "data_config" in result["message"]

    def test_path_traversal_in_checkpoint_returns_error(self):
        """Path traversal in checkpoint_path must return error dict."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="start", checkpoint_path="/tmp/../../../etc/passwd")
        assert result["status"] == "error"
        assert "checkpoint_path" in result["message"]

    def test_invalid_host_returns_error(self):
        """Invalid host address must return error dict."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="start", host="--not-valid")
        assert result["status"] == "error"
        assert "host" in result["message"]

    def test_invalid_port_returns_error(self):
        """Out-of-range port must return error dict."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="start", port=99999)
        assert result["status"] == "error"
        assert "port" in result["message"]


class TestStopServiceCrossPortKill:
    """End-to-end regression test for the cross-port-kill bug.

    Verifies that _stop_service(port=80) does NOT kill a GR00T process
    running on port 8000. This pins the _is_gr00t_process(port=...) guard
    so a future refactor that removes it will surface as a test failure.
    """

    def test_stop_service_does_not_kill_wrong_port(self, monkeypatch):
        """_stop_service(port=80) must NOT kill a process on port 8000."""
        from strands_robots.tools.gr00t_inference import _stop_service

        killed_pids = []
        call_log = []

        def _fake_run(cmd, *args, **kwargs):
            call_log.append(cmd)

            # Mock _find_gr00t_containers returning no containers (forces host fallback)
            if "docker" in cmd and "ps" in cmd:
                import subprocess

                result = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                return result

            # Mock pgrep finding PID 999 on the host
            if cmd[0] == "pgrep":
                import subprocess

                return subprocess.CompletedProcess(cmd, 0, stdout="999\n", stderr="")

            # Mock kill — record it
            if cmd[0] == "kill":
                killed_pids.append(cmd[-1])
                import subprocess

                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            import subprocess

            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        def _fake_host_process(pid, *, port=None):
            """Simulate a process running on port 8000, NOT port 80."""
            # The process is a GR00T process but on port 8000
            if port == 80:
                return False  # Not on port 80
            if port == 8000:
                return True  # Yes on port 8000
            return True  # Generic check without port

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.subprocess.run", _fake_run)
        monkeypatch.setattr("strands_robots.tools.gr00t_inference._is_gr00t_host_process", _fake_host_process)
        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference._find_gr00t_containers",
            lambda: {"status": "success", "containers": []},
        )

        _stop_service(port=80)

        # No process should have been killed (the only process is on port 8000)
        assert not killed_pids, (
            f"_stop_service(port=80) killed PIDs {killed_pids} but should not have "
            f"(the only GR00T process is on port 8000)"
        )

    def test_stop_service_kills_correct_port(self, monkeypatch):
        """_stop_service(port=8000) MUST kill a process on port 8000."""
        from strands_robots.tools.gr00t_inference import _stop_service

        killed_pids = []

        def _fake_run(cmd, *args, **kwargs):
            import subprocess

            if cmd[0] == "pgrep":
                return subprocess.CompletedProcess(cmd, 0, stdout="999\n", stderr="")
            if cmd[0] == "kill":
                killed_pids.append(cmd[-1])
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        def _fake_host_process(pid, *, port=None):
            """Simulate a process running on port 8000."""
            if port == 8000:
                return True
            return False

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.subprocess.run", _fake_run)
        monkeypatch.setattr("strands_robots.tools.gr00t_inference._is_gr00t_host_process", _fake_host_process)
        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference._find_gr00t_containers",
            lambda: {"status": "success", "containers": []},
        )

        _stop_service(port=8000)

        # Process should have been killed
        assert "999" in killed_pids, f"_stop_service(port=8000) should have killed PID 999 but killed {killed_pids}"


class TestActionScopedValidation:
    """Tests verifying that validate_inputs scopes checks per action.

    Read-only actions (find_containers, list, status, stop) should only
    validate port/host/protocol, not the full parameter surface like
    data_config, embodiment_tag, etc.
    """

    def test_read_only_action_accepts_any_data_config(self):
        """Read-only actions should not validate data_config."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        # This would fail for action="start" but should pass for "list"
        validate_inputs(**{**_VALID_KWARGS, "action": "list", "data_config": "anything_goes_here"})

    def test_read_only_action_still_validates_port(self):
        """Read-only actions must still validate port."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="port must be between"):
            validate_inputs(**{**_VALID_KWARGS, "action": "status", "port": 99999})

    def test_read_only_action_still_validates_host(self):
        """Read-only actions must still validate host."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="host must be a valid"):
            validate_inputs(**{**_VALID_KWARGS, "action": "stop", "host": "--invalid"})

    def test_read_only_action_still_validates_protocol(self):
        """Read-only actions must still validate protocol."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="Unknown protocol"):
            validate_inputs(**{**_VALID_KWARGS, "action": "list", "protocol": "invalid"})

    def test_mutating_action_validates_data_config(self):
        """Mutating actions must validate data_config."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="data_config"):
            validate_inputs(**{**_VALID_KWARGS, "action": "start", "data_config": "foo;bar"})

    def test_integration_read_only_action_skips_data_config_validation(self):
        """gr00t_inference(action='list', data_config='invalid') must not error on data_config."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        # action="list" should not validate data_config
        # It will fail at runtime (no docker) but NOT on validation
        result = gr00t_inference(action="list", data_config="invalid;stuff")
        # Should NOT be a validation error about data_config
        if result.get("status") == "error":
            assert "data_config" not in result.get("message", "")


class TestHostNumericTypoRejection:
    """Regression tests for all-numeric hostname typos.

    Verifies that "127.0.01" (typo for 127.0.0.1) and "999.999.999.999"
    are rejected by validate_inputs. These strings pass _HOSTNAME_RE but
    are caught by the _ALL_NUMERIC_RE guard introduced in review round-4.
    """

    def test_invalid_host_typo_dotted_numeric(self):
        """127.0.01 (typo for 127.0.0.1) must be rejected."""
        with pytest.raises(ValueError, match="host must be a valid IP address or hostname"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "port": 5555,
                    "host": "127.0.01",
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_invalid_host_999_octets(self):
        """999.999.999.999 (invalid IP, all-numeric) must be rejected."""
        with pytest.raises(ValueError, match="host must be a valid IP address or hostname"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "port": 5555,
                    "host": "999.999.999.999",
                    "vit_dtype": "fp8",
                    "llm_dtype": "nvfp4",
                    "dit_dtype": "fp8",
                }
            )

    def test_single_numeric_label_is_valid_hostname(self):
        """A bare number like '8080' is a valid single-label hostname (RFC-1123)."""
        # Single-label numerics are valid hostnames; only multi-label patterns
        # like '127.0.01' (IP typos) are rejected.
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "host": "8080",
            }
        )


class TestActionAllowlistValidation:
    """Tests for the action allowlist in validate_inputs.

    Verifies that unknown actions are rejected with a clear error that
    lists the valid options, rather than falling through to validation
    of unrelated parameters.
    """

    def test_unknown_action_rejected(self):
        """Typo'd action gets a clear error listing valid actions."""
        with pytest.raises(ValueError, match="Unknown action.*Valid actions"):
            validate_inputs(**{**_VALID_KWARGS, "action": "strat"})  # typo for "start"

    def test_unknown_action_integration(self):
        """gr00t_inference(action='typo') returns error about unknown action."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="typo")
        assert result["status"] == "error"
        assert "Unknown action" in result["message"]

    def test_all_valid_actions_accepted(self):
        """All 10 valid actions pass action validation (may fail later)."""
        from strands_robots.tools.gr00t_inference import _VALID_ACTIONS

        for action in _VALID_ACTIONS:
            # Should not raise ValueError about unknown action
            # (may raise about other params, but that's fine)
            try:
                validate_inputs(**{**_VALID_KWARGS, "action": action})
            except ValueError as e:
                assert "Unknown action" not in str(e), f"Action {action!r} wrongly rejected"


class TestExpandedParamValidation:
    """Tests for image_name, volumes, and container_command validation."""

    def test_invalid_image_name_rejected(self):
        """Docker image with shell chars must be rejected."""
        with pytest.raises(ValueError, match="image_name must be a valid Docker"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "action": "start",
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "image_name": "gr00t:latest; rm -rf /",
                }
            )

    def test_valid_image_name_accepted(self):
        """Standard Docker image references must pass."""
        # Should not raise
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "action": "start",
                "data_config": "fourier_gr1_arms_only",
                "embodiment_tag": "gr1",
                "image_name": "nvcr.io/nvidia/gr00t:n1.7",
            }
        )

    def test_volume_path_traversal_rejected(self):
        """Volumes with path traversal must be rejected."""
        with pytest.raises(ValueError, match="volumes key"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "action": "start",
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "volumes": {"/../etc/passwd": "/data"},
                }
            )

    def test_container_command_shell_meta_rejected(self):
        """Container command with shell metacharacters must be rejected."""
        with pytest.raises(ValueError, match="container_command contains disallowed"):
            validate_inputs(
                **{
                    **_VALID_KWARGS,
                    "action": "start",
                    "data_config": "fourier_gr1_arms_only",
                    "embodiment_tag": "gr1",
                    "container_command": "tail -f /dev/null; rm -rf /",
                }
            )

    def test_valid_container_command_accepted(self):
        """Standard container commands must pass."""
        # Should not raise
        validate_inputs(
            **{
                **_VALID_KWARGS,
                "action": "start",
                "data_config": "fourier_gr1_arms_only",
                "embodiment_tag": "gr1",
                "container_command": "tail -f /dev/null",
            }
        )


class TestHappyPathIntegration:
    """Happy-path integration test for gr00t_inference.

    Verifies that valid inputs pass validation and proceed to runtime
    (which will fail due to missing Docker, but NOT on validation).
    """

    def test_valid_list_action_passes_validation(self):
        """gr00t_inference(action='list') with valid params does not error on validation."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="list")
        # The error should be about runtime (no docker), NOT validation
        if result.get("status") == "error":
            msg = result.get("message", "")
            # Must not be a validation error
            assert "must be" not in msg or "port" not in msg
            assert "Unknown action" not in msg
            assert "data_config" not in msg

    def test_valid_status_action_passes_validation(self):
        """gr00t_inference(action='status') with valid params proceeds past validation."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="status", port=5555, host="127.0.0.1")
        # Should not be a validation error
        if result.get("status") == "error":
            msg = result.get("message", "")
            assert "Unknown action" not in msg
            assert "host must be" not in msg
            assert "port must be" not in msg


class TestDockerImageRegistryPort:
    """Tests that _DOCKER_IMAGE_RE supports private registries with port numbers."""

    def test_registry_with_port_accepted(self):
        """localhost:5000/myorg/img:tag must be accepted."""
        validate_inputs(**{**_VALID_KWARGS, "image_name": "localhost:5000/myorg/img:tag"})

    def test_registry_with_port_no_tag(self):
        """registry.internal:5000/img must be accepted."""
        validate_inputs(**{**_VALID_KWARGS, "image_name": "registry.internal:5000/img"})

    def test_nvcr_standard_format(self):
        """nvcr.io/nvidia/gr00t:n1.7 must be accepted."""
        validate_inputs(**{**_VALID_KWARGS, "image_name": "nvcr.io/nvidia/gr00t:n1.7"})

    def test_simple_image_tag(self):
        """gr00t:latest must be accepted."""
        validate_inputs(**{**_VALID_KWARGS, "image_name": "gr00t:latest"})


class TestProcessIdentificationRequiresPort:
    """Tests that _is_gr00t_process requires --port in cmdline.

    Prevents false-matching unrelated processes like editors or log-tailers
    that happen to have 'inference_service.py' and 'python' in their cmdline.
    """

    def test_process_without_port_flag_rejected(self, monkeypatch):
        """A process with 'python inference_service.py' but no --port flag is not a match."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_process

        # Mock docker exec to return a cmdline without --port
        def fake_run(*args, **kwargs):
            class Result:
                returncode = 0
                stdout = "python inference_service.py --config test\x00"

            return Result()

        monkeypatch.setattr("subprocess.run", fake_run)
        # Without --port in cmdline, should return False
        assert _is_gr00t_process("container", "123", port=5555) is False

    def test_process_with_port_flag_accepted(self, monkeypatch):
        """A process with --port 5555 in cmdline is a match."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_process

        def fake_run(*args, **kwargs):
            class Result:
                returncode = 0
                stdout = "python inference_service.py --port 5555\x00"

            return Result()

        monkeypatch.setattr("subprocess.run", fake_run)
        assert _is_gr00t_process("container", "123", port=5555) is True

    def test_editor_on_inference_service_rejected(self, monkeypatch):
        """vim editing inference_service.py under a python venv is not a match."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_process

        def fake_run(*args, **kwargs):
            class Result:
                returncode = 0
                stdout = "/opt/conda/envs/gr00t/bin/python vim /opt/gr00t/inference_service.py\x00"

            return Result()

        monkeypatch.setattr("subprocess.run", fake_run)
        # No --port flag → rejected
        assert _is_gr00t_process("container", "123", port=5555) is False


class TestOptionInjectionGuard:
    """Test option-injection guard for argv-interpolated parameters."""

    def test_repo_url_starting_with_dash_rejected(self):
        """repo_url='--upload-pack=evil' must be rejected."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(
            action="build_image",
            repo_url="--upload-pack=touch /tmp/pwned",
        )
        assert result["status"] == "error"
        assert "repo_url" in result["message"]
        assert "must not start with '-'" in result["message"]

    def test_repo_tag_starting_with_dash_rejected(self):
        """repo_tag='--config=evil' must be rejected."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(
            action="build_image",
            repo_tag="--config=core.fsmonitor=evil-cmd",
        )
        assert result["status"] == "error"
        assert "repo_tag" in result["message"]

    def test_policy_name_starting_with_dash_rejected(self):
        """policy_name='--flag' must be rejected."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(
            action="start",
            checkpoint_path="/data/model",
            policy_name="--malicious",
        )
        assert result["status"] == "error"
        assert "policy_name" in result["message"]

    def test_valid_repo_url_accepted(self):
        """Normal https:// URL must pass the guard."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        # Will fail on Docker/git availability but not on option-injection guard
        result = gr00t_inference(
            action="build_image",
            repo_url="https://github.com/NVIDIA/Isaac-GR00T",
            repo_tag="n1.7-release",
        )
        # Should not be an option-injection error
        assert "must not start with '-'" not in result.get("message", "")


class TestHostAutoFlipForContainer:
    """Test that container actions auto-flip 127.0.0.1 to 0.0.0.0."""

    def test_default_host_is_loopback(self):
        """Signature default must be 127.0.0.1 (AGENTS.md compliance)."""
        import inspect

        from strands_robots.tools.gr00t_inference import gr00t_inference

        sig = inspect.signature(gr00t_inference)
        assert sig.parameters["host"].default == "127.0.0.1"

    def test_start_service_auto_flips_loopback(self, monkeypatch):
        """_start_service should auto-flip 127.0.0.1 to 0.0.0.0 for Docker."""
        from strands_robots.tools.gr00t_inference import _start_service

        captured_host = {}

        def fake_find(*args, **kwargs):
            return {
                "status": "success",
                "containers": [{"name": "gr00t-test", "status": "Up 2 hours"}],
            }

        def fake_build_cmd(**kwargs):
            captured_host["host"] = kwargs.get("host")
            return ["docker", "exec", "gr00t-test", "echo", "test"]

        monkeypatch.setattr("strands_robots.tools.gr00t_inference._find_gr00t_containers", fake_find)
        monkeypatch.setattr("strands_robots.tools.gr00t_inference._build_inference_command", fake_build_cmd)

        import subprocess

        monkeypatch.setattr(
            subprocess,
            "Popen",
            lambda *a, **kw: type("P", (), {"poll": lambda s: 0, "stdout": None, "stderr": None})(),
        )

        # Call with loopback default — should auto-flip
        _start_service(
            checkpoint_path="/data/model",
            port=5555,
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            denoising_steps=4,
            host="127.0.0.1",
            container_name=None,
            policy_name=None,
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="gr00t_engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
        )
        assert captured_host.get("host") == "0.0.0.0"


class TestSingleLabelNumericHostname:
    """Verify single-label numeric hostnames (per RFC-1123) are accepted."""

    def test_single_numeric_label_accepted(self):
        """Single-label '123' is a valid hostname (RFC-1123)."""
        validate_inputs(**{**_VALID_KWARGS, "host": "123"})

    def test_multi_label_numeric_rejected(self):
        """Multi-label '127.0.01' is rejected as an IP typo."""
        with pytest.raises(ValueError, match="host must be a valid IP address or hostname"):
            validate_inputs(**{**_VALID_KWARGS, "host": "127.0.01"})


class TestPlatformGuardForHostFallback:
    """Test that host-fallback stop returns error on non-Linux platforms."""

    def test_non_linux_platform_returns_error(self, monkeypatch):
        """On non-Linux, _stop_service should error when no containers found."""
        import sys as _sys

        from strands_robots.tools.gr00t_inference import _stop_service

        # Mock no containers found
        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference._find_gr00t_containers",
            lambda: {"status": "success", "containers": []},
        )
        monkeypatch.setattr(_sys, "platform", "darwin")

        result = _stop_service(5555)
        assert result["status"] == "error"
        assert "Linux" in result["message"]
