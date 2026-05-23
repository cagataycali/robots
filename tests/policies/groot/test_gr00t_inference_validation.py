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
                "host": "0.0.0.0",
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

        # This has hyphens/caps which WOULD fail for action="start" but passes for "list"
        validate_inputs(**{**_VALID_KWARGS, "action": "list", "data_config": "Has-Hyphens-And-Caps"})

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

    def test_valid_repo_url_accepted(self, monkeypatch):
        """Normal https:// URL must pass the guard."""
        from strands_robots.tools import gr00t_inference as gi_mod

        # Mock _build_image to avoid actual git/docker operations
        monkeypatch.setattr(
            gi_mod,
            "_build_image",
            lambda **kwargs: {"status": "success", "message": "mocked"},
        )

        result = gi_mod.gr00t_inference(
            action="build_image",
            repo_url="https://github.com/NVIDIA/Isaac-GR00T",
            repo_tag="n1.7-release",
        )
        # Should not be an option-injection error
        assert "must not start with '-'" not in result.get("message", "")


class TestHostBindingHonoursUserChoice:
    """R1 pin tests — host kwarg controls docker -p binding; NO auto-flip.

    Pre-R1: ``_start_service`` rewrote ``host="127.0.0.1"`` to ``"0.0.0.0"``
    when ``host_was_explicit=False``, silently widening the bind to all
    interfaces. R1 drops the rewrite. The host kwarg now flows verbatim into
    ``docker -p HOST:port:port``, so:
        - host="127.0.0.1" (default)  → docker binds loopback only
        - host="0.0.0.0" (explicit)   → docker binds all interfaces
    """

    def test_default_host_is_loopback_sentinel(self):
        """Signature default must be None (resolves to 127.0.0.1) — AGENTS.md compliance."""
        import inspect

        from strands_robots.tools.gr00t_inference import gr00t_inference

        sig = inspect.signature(gr00t_inference)
        assert sig.parameters["host"].default is None, (
            "host signature default must remain None sentinel (resolves to 127.0.0.1)"
        )

    def test_start_service_does_not_flip_default_loopback(self, monkeypatch):
        """R1+R4 pin: _start_service must NOT auto-flip user's host kwarg.

        Pre-R1 _start_service rewrote host=127.0.0.1 to 0.0.0.0 when
        host_was_explicit=False. Post-R1 the rewrite is gone; post-R4
        host is no longer passed into _build_inference_command at all
        (the inside-container --host is hardcoded to 0.0.0.0). Either
        way, the user's host kwarg must reach _start_service unchanged.
        """
        import inspect

        from strands_robots.tools.gr00t_inference import _start_service

        # Static check: _start_service still takes host kwarg (controls docker -p
        # via _start_container, not the inside-container --host which is now hardcoded).
        sig = inspect.signature(_start_service)
        assert "host" in sig.parameters, "_start_service must keep host kwarg for the docker -p host-side bind"

        # Behavioural check: invoking with host=127.0.0.1 must not raise (no
        # auto-flip side-effects), and the inference cmd argv inside the
        # container must bind 0.0.0.0 (R4 contract).
        from strands_robots.tools.gr00t_inference import _build_inference_command

        argv = _build_inference_command(
            container_name="gr00t-test",
            checkpoint_path="/data/model",
            port=5555,
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            denoising_steps=4,
            http_server=False,
            use_tensorrt=False,
            trt_engine_path="gr00t_engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            api_token=None,
            protocol="n1.5",
            use_sim_policy_wrapper=False,
        )
        host_idx = argv.index("--host")
        assert argv[host_idx + 1] == "0.0.0.0", (
            "R4 contract: inference server must always bind 0.0.0.0 inside container"
        )

    def test_explicit_zero_zero_zero_zero_passes_through(self):
        """R1+R4 pin: explicit host='0.0.0.0' is honoured for the docker -p bind.

        Verified via the public tool signature: host kwarg is still accepted
        (R1: no auto-flip) and is destined for the docker host-side bind in
        _start_container, not the inside-container --host which R4 hardcodes.
        """
        import inspect

        from strands_robots.tools.gr00t_inference import gr00t_inference

        sig = inspect.signature(gr00t_inference)
        assert "host" in sig.parameters, "host kwarg must remain on public tool"
        # Default sentinel still None (resolves to 127.0.0.1 internally).
        assert sig.parameters["host"].default is None


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


class TestN17ProcessIdentification:
    """Regression tests for N1.7 process identification — GH review thread.

    N1.7 services are started via `python -m gr00t.eval.run_gr00t_server` which
    doesn't contain `inference_service.py` in cmdline. These tests ensure the
    stop/status path can identify N1.7 services.
    """

    def test_n17_cmdline_detected_by_host_process_check(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process detects N1.7 server cmdline."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        # Simulate N1.7 cmdline: python -m gr00t.eval.run_gr00t_server --port 5555
        proc_dir = tmp_path / "proc" / "999"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00-m\x00gr00t.eval.run_gr00t_server\x00--port\x005555\x00")

        called = {}
        from pathlib import Path as RealPath

        def _fake_path(p):
            called["p"] = p
            return RealPath(str(p).replace("/proc", str(tmp_path / "proc")))

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.Path", _fake_path)

        assert _is_gr00t_host_process("999", port=5555) is True
        assert called.get("p") == "/proc/999/cmdline"

    def test_n17_cmdline_wrong_port_rejected(self, tmp_path, monkeypatch):
        """N1.7 server on wrong port is not killed."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "999"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00-m\x00gr00t.eval.run_gr00t_server\x00--port\x008000\x00")

        called = {}
        from pathlib import Path as RealPath

        def _fake_path(p):
            called["p"] = p
            return RealPath(str(p).replace("/proc", str(tmp_path / "proc")))

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.Path", _fake_path)

        # Request port 80 — should not match 8000
        assert _is_gr00t_host_process("999", port=80) is False
        assert called.get("p") == "/proc/999/cmdline"

    def test_n15_cmdline_still_detected(self, tmp_path, monkeypatch):
        """N1.5/N1.6 cmdline (inference_service.py) still works after N1.7 support."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "123"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00inference_service.py\x00--port\x005555\x00")

        from pathlib import Path as RealPath

        def _fake_path(p):
            return RealPath(str(p).replace("/proc", str(tmp_path / "proc")))

        monkeypatch.setattr("strands_robots.tools.gr00t_inference.Path", _fake_path)

        assert _is_gr00t_host_process("123", port=5555) is True


class TestExpandedParamValidationExtended:
    """Extended tests for image_name, volumes, and container_command — covers happy paths."""

    def test_valid_image_name(self):
        from strands_robots.tools.gr00t_inference import validate_inputs

        # Should not raise
        validate_inputs(
            action="start",
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            port=5555,
            host="127.0.0.1",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            checkpoint_path="/tmp/ckpt",
            trt_engine_path="gr00t_engine",
            container_name="gr00t",
            protocol="n1.5",
            image_name="localhost:5000/myorg/img:tag",
        )

    def test_invalid_image_name_shell_meta(self):
        import pytest

        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="image_name"):
            validate_inputs(
                action="start",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path="/tmp/ckpt",
                trt_engine_path="gr00t_engine",
                container_name="gr00t",
                protocol="n1.5",
                image_name="evil;rm -rf /",
            )

    def test_volumes_path_traversal_rejected(self):
        import pytest

        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="volumes"):
            validate_inputs(
                action="start",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path="/tmp/ckpt",
                trt_engine_path="gr00t_engine",
                container_name="gr00t",
                protocol="n1.5",
                volumes={"../../etc/passwd": "/data"},
            )

    def test_container_command_shell_meta_rejected(self):
        import pytest

        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="container_command"):
            validate_inputs(
                action="start",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path="/tmp/ckpt",
                trt_engine_path="gr00t_engine",
                container_name="gr00t",
                protocol="n1.5",
                container_command="tail -f /dev/null; rm -rf /",
            )

    def test_valid_container_command(self):
        from strands_robots.tools.gr00t_inference import validate_inputs

        # Should not raise - legitimate container commands without shell metas
        validate_inputs(
            action="start",
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            port=5555,
            host="127.0.0.1",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            checkpoint_path="/tmp/ckpt",
            trt_engine_path="gr00t_engine",
            container_name="gr00t",
            protocol="n1.5",
            container_command="tail -f /dev/null",
        )

    def test_valid_volumes(self):
        from strands_robots.tools.gr00t_inference import validate_inputs

        # Should not raise
        validate_inputs(
            action="start",
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            port=5555,
            host="127.0.0.1",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            checkpoint_path="/tmp/ckpt",
            trt_engine_path="gr00t_engine",
            container_name="gr00t",
            protocol="n1.5",
            volumes={"/tmp/checkpoints": "/data/checkpoints"},
        )


class TestHostKwargNotPlumbed:
    """R2 pin tests -- ``host_was_explicit`` kwarg is no longer plumbed.

    Pre-R2: ``gr00t_inference()`` set ``_host_was_explicit = host is not None``
    and threaded it through ``_lifecycle`` and the ``start`` / ``restart``
    dispatch into ``_start_service``, where it was unused (``# noqa: ARG001``).
    The auto-flip the flag once gated was removed in R1 (commit ecf5f0f), so
    the kwarg became dead plumbing.

    R2: removed per AGENTS.md > Key Conventions #10 ('No dead code'). These
    pins assert the removal so a future refactor that re-introduces the flag
    re-introduces a meaningless plumbing chain.
    """

    def test_start_service_signature_has_no_host_was_explicit(self):
        """``_start_service`` signature must not contain ``host_was_explicit``."""
        import inspect

        from strands_robots.tools.gr00t_inference import _start_service

        params = inspect.signature(_start_service).parameters
        assert "host_was_explicit" not in params, (
            "Dead kwarg `host_was_explicit` reintroduced into _start_service signature"
        )

    def test_lifecycle_signature_has_no_host_was_explicit(self):
        """``_lifecycle`` signature must not contain ``host_was_explicit``."""
        import inspect

        from strands_robots.tools.gr00t_inference import _lifecycle

        params = inspect.signature(_lifecycle).parameters
        assert "host_was_explicit" not in params, (
            "Dead kwarg `host_was_explicit` reintroduced into _lifecycle signature"
        )

    def test_start_dispatch_does_not_pass_host_was_explicit(self, monkeypatch):
        """``gr00t_inference(action='start')`` must not forward ``host_was_explicit``."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        captured = {}

        def _mock_start_service(**kwargs):
            captured.update(kwargs)
            return {"status": "error", "message": "mocked"}

        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference._start_service",
            _mock_start_service,
        )
        gr00t_inference(action="start", checkpoint_path="/data/model", host="127.0.0.1")
        assert "host_was_explicit" not in captured, (
            "`start` dispatch passed dead `host_was_explicit` kwarg to _start_service"
        )

    def test_restart_dispatch_does_not_pass_host_was_explicit(self, monkeypatch):
        """``gr00t_inference(action='restart')`` must not forward ``host_was_explicit``."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        captured = {}

        def _mock_start_service(**kwargs):
            captured.update(kwargs)
            return {"status": "success", "message": "mocked"}

        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference._start_service",
            _mock_start_service,
        )
        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference._stop_service",
            lambda port: None,
        )
        monkeypatch.setattr("time.sleep", lambda _: None)

        gr00t_inference(action="restart", checkpoint_path="/data/model")
        assert "host_was_explicit" not in captured, (
            "`restart` dispatch passed dead `host_was_explicit` kwarg to _start_service"
        )


class TestReviewRound8Fixes:
    """Regression tests for review round-8 fixes (2026-05-22 21:44 UTC).

    Covers:
    - restart path forwarding host_was_explicit
    - colon rejection in volume paths (docker -v mount-redirect)
    - digest-pinned image references
    - TypeError handling in validation wrapper
    - dash-prefix rejection in volume paths
    """

    def test_volume_path_colon_rejected(self):
        """Volume paths containing ':' must be rejected (docker -v mount-redirect)."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(
            action="start_container",
            image_name="gr00t:latest",
            volumes={"/legit/dir:rw,nosuid": "/container/path"},
        )
        assert result["status"] == "error"
        assert ":" in result["message"] or "colon" in result["message"].lower()

    def test_volume_value_colon_rejected(self):
        """Container-side volume paths containing ':' must also be rejected."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(
            action="start_container",
            image_name="gr00t:latest",
            volumes={"/host/path": "/container:path"},
        )
        assert result["status"] == "error"
        assert ":" in result["message"]

    def test_volume_path_dash_prefix_rejected(self):
        """Volume paths starting with '-' must be rejected (option injection)."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(
            action="start_container",
            image_name="gr00t:latest",
            volumes={"--privileged=foo": "/bar"},
        )
        assert result["status"] == "error"
        assert "'-'" in result["message"] or "start with" in result["message"]

    def test_digest_pinned_image_accepted(self):
        """Digest-pinned image refs (registry/path@sha256:hex) must be accepted."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        # Should NOT fail validation on image_name (may fail later on docker ops)
        result = gr00t_inference(
            action="start_container",
            image_name="nvcr.io/nvidia/gr00t@sha256:" + "a" * 64,
        )
        # If it fails, it should NOT be an image_name validation error
        if result["status"] == "error":
            assert "valid Docker image" not in result["message"]

    def test_type_error_returns_structured_error(self):
        """TypeError from bad parameter types must return structured error, not raise."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        # port="5555" (str instead of int) -> TypeError on `1 <= port <= 65535`
        result = gr00t_inference(action="start", checkpoint_path="/data/model", port="5555")
        assert result["status"] == "error"
        # Must not propagate as unhandled exception - returns dict

    def test_end_to_end_bogus_action_returns_error_dict(self):
        """Bogus action returns structured error dict (not raw exception)."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        result = gr00t_inference(action="bogus_action")
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "Unknown action" in result["message"] or "bogus_action" in result["message"]


class TestImageOnlyBranchValidation:
    """Tests for validation on image-only actions (build_image, download_checkpoint, start_container)."""

    def test_container_name_validated_on_start_container(self):
        """container_name must be validated on start_container (image-only branch)."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="container_name"):
            validate_inputs(
                action="start_container",
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="/opt/engine",
                container_name="--privileged",
                protocol="n1.5",
                image_name=None,
                volumes=None,
                container_command=None,
                repo_url=None,
                repo_tag=None,
                policy_name=None,
            )

    def test_policy_name_dash_rejected_on_start_container(self):
        """policy_name starting with '-' must be rejected on image-only actions."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="policy_name"):
            validate_inputs(
                action="start_container",
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="/opt/engine",
                container_name=None,
                protocol="n1.5",
                image_name=None,
                volumes=None,
                container_command=None,
                repo_url=None,
                repo_tag=None,
                policy_name="--malicious",
            )

    def test_valid_container_name_accepted_on_start_container(self):
        """Valid container_name passes on start_container."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        # Should not raise
        validate_inputs(
            action="start_container",
            data_config="so100",
            embodiment_tag="so100",
            port=5555,
            host="127.0.0.1",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            checkpoint_path=None,
            trt_engine_path="/opt/engine",
            container_name="my-gr00t-container",
            protocol="n1.5",
            image_name=None,
            volumes=None,
            container_command=None,
            repo_url=None,
            repo_tag=None,
            policy_name=None,
        )


class TestRegexBugFixesR4:
    """R4 pin tests for the 4 regex bugs raised in PR #90 review.

    Each test fails on pre-R4 code and passes after R4 lands.
    """

    # === Bug 1: _DOCKER_IMAGE_RE registry port range ===

    def test_registry_port_99999_rejected(self):
        """Pre-R4 the regex accepted :99999 (>65535). Post-R4 we range-check."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert not _is_valid_docker_image_ref("localhost:99999/myorg/img:tag"), (
            "R4 regression: registry port 99999 must be rejected (TCP max is 65535)"
        )

    def test_registry_port_65535_accepted(self):
        """65535 is the max valid TCP port — must be accepted."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert _is_valid_docker_image_ref("localhost:65535/myorg/img:tag")

    def test_registry_port_5000_accepted(self):
        """Common private-registry port — sanity check."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert _is_valid_docker_image_ref("localhost:5000/myorg/img:tag")

    def test_registry_port_zero_rejected(self):
        """Port 0 is not a valid bind target."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert not _is_valid_docker_image_ref("localhost:0/myorg/img:tag")

    def test_no_port_still_accepted(self):
        """Image refs without a registry port must still match."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert _is_valid_docker_image_ref("nvcr.io/nvidia/gr00t:n1.7")
        assert _is_valid_docker_image_ref("gr00t:latest")

    def test_digest_pinned_image_accepted(self):
        """Digest-pinned refs (@sha256:...) must continue to match."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        digest = "a" * 64
        assert _is_valid_docker_image_ref(f"nvcr.io/nvidia/gr00t@sha256:{digest}")

    # === Bug 3: _HOSTNAME_RE trailing-dot FQDN ===

    def test_trailing_dot_fqdn_accepted(self):
        """RFC 1034 §3.1: FQDNs may end with a dot to disambiguate.

        Pre-R4 the regex required the last label to end with [a-zA-Z0-9],
        which rejected legitimate FQDNs like 'host.example.com.'.
        """
        from strands_robots.tools.gr00t_inference import _HOSTNAME_RE

        assert _HOSTNAME_RE.match("host.example.com."), (
            "R4 regression: trailing-dot FQDN must be accepted per RFC 1034 §3.1"
        )
        # Without trailing dot still accepted
        assert _HOSTNAME_RE.match("host.example.com")

    def test_single_dot_alone_rejected(self):
        """A bare '.' is not a valid hostname."""
        from strands_robots.tools.gr00t_inference import _HOSTNAME_RE

        assert not _HOSTNAME_RE.match(".")

    # === Bug 4: hostname total length cap (already implemented; pin it) ===

    def test_host_validation_rejects_oversize(self):
        """RFC 1035 §2.3.4: hostname must not exceed 253 octets total."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        oversize = ".".join(["a" * 60] * 5)  # 60*5 + 4 dots = 304 > 253
        assert len(oversize) > 253
        try:
            validate_inputs(
                action="start",
                port=5555,
                host=oversize,
                protocol="n1.5",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                container_name=None,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="gr00t_engine",
                image_name=None,
                volumes=None,
                container_command="tail -f /dev/null",
                policy_name=None,
            )
            raise AssertionError("Expected ValueError for hostname > 253 octets")
        except ValueError as e:
            assert "253" in str(e), f"Error should mention RFC 1035 limit; got: {e}"

    # === Bug 2 / IPv4 typo regression — explicit '127.0.01' typo case ===

    def test_host_typo_127_0_01_rejected(self):
        """The typo called out in the PR description must be rejected.

        '127.0.01' looks like an IPv4 attempt to a human but is not a valid
        IPv4 string under ipaddress.ip_address. Without the _ALL_NUMERIC_RE
        guard it would be accepted as a hostname (it matches RFC-952), which
        would then fail at runtime with a confusing connection error.
        """
        from strands_robots.tools.gr00t_inference import validate_inputs

        try:
            validate_inputs(
                action="start",
                port=5555,
                host="127.0.01",
                protocol="n1.5",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                container_name=None,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="gr00t_engine",
                image_name=None,
                volumes=None,
                container_command="tail -f /dev/null",
                policy_name=None,
            )
            raise AssertionError("Expected ValueError for '127.0.01' IP typo")
        except ValueError as e:
            assert "host" in str(e).lower()


class TestDockerImageNumericTagRegression:
    """Pin: numeric-only Docker tags must not be falsely rejected.

    Pre-fix, the port-capture group in _DOCKER_IMAGE_RE greedily matched
    :digits even without a following / path component, causing the port-range
    check to reject valid name:tag refs where the tag was purely numeric.

    Reproducer (pre-fix):
        >>> _is_valid_docker_image_ref('myimage:0')       # False (should be True)
        >>> _is_valid_docker_image_ref('myimage:65536')   # False (should be True)
        >>> _is_valid_docker_image_ref('myimage:99999')   # False (should be True)

    Fix: lookahead (?=/) on the port group so :digits is only interpreted as a
    registry port when followed by a path component.
    """

    def test_numeric_tag_zero_accepted(self):
        """Tag ':0' is valid -- common in dev builds."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert _is_valid_docker_image_ref("myimage:0"), (
            "Regression: numeric tag ':0' must not be rejected as an invalid port"
        )

    def test_numeric_tag_above_port_range_accepted(self):
        """Tag ':65536' is valid -- it is a tag, not a port."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert _is_valid_docker_image_ref("myimage:65536"), (
            "Regression: numeric tag ':65536' must not be rejected as an invalid port"
        )

    def test_numeric_tag_99999_accepted(self):
        """Tag ':99999' is valid -- common date-style build IDs."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert _is_valid_docker_image_ref("myimage:99999"), (
            "Regression: numeric tag ':99999' must not be rejected as an invalid port"
        )

    def test_numeric_tag_five_digit_accepted(self):
        """Tag ':23456' is valid -- common sequential build numbers."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        assert _is_valid_docker_image_ref("gr00t:23456"), (
            "Regression: numeric tag ':23456' must not be rejected as an invalid port"
        )

    def test_registry_port_with_path_still_range_checked(self):
        """Port in host:port/path form must still be range-checked."""
        from strands_robots.tools.gr00t_inference import _is_valid_docker_image_ref

        # Valid port with path -- accepted
        assert _is_valid_docker_image_ref("localhost:5000/myorg/img:tag")
        assert _is_valid_docker_image_ref("localhost:65535/myorg/img:tag")

        # Invalid port (>65535) with path -- rejected
        assert not _is_valid_docker_image_ref("localhost:99999/myorg/img:tag"), (
            "Registry port 99999 with /path must still be rejected (TCP max is 65535)"
        )
        assert not _is_valid_docker_image_ref("localhost:100000/img:tag"), (
            "Registry port 100000 with /path must still be rejected"
        )


class TestPathTraversalPosixBackslash:
    """R2 pin tests -- _validate_path must not over-reject POSIX paths containing
    literal backslashes.

    Pre-R2: ``_validate_path`` split on both ``/`` and ``\\`` via
    ``re.split(r"[/\\\\]", value)``. On POSIX, ``\\`` is a legal filename byte
    (only ``/`` and NUL are forbidden), so a path like ``a\\..\\b`` -- a single
    legitimate filename containing literal backslashes -- was wrongly flagged
    as ``..`` traversal because the splitter isolated ``..`` between the
    backslash bytes.

    R2: ``_validate_path`` splits on ``/`` only. docker -v interprets just ``/``
    as a separator on Linux (the only platform this tool supports), so this
    matches the executor's contract. Real ``..`` traversal between ``/``
    separators (e.g. ``/foo/../etc``) remains rejected.
    """

    def test_posix_backslash_in_filename_accepted(self):
        """POSIX path with literal backslash bytes is not traversal."""
        # checkpoint_path is one of the validated path kwargs.
        validate_inputs(**{**_VALID_KWARGS, "checkpoint_path": "/data/odd\\..\\name"})

    def test_real_traversal_still_rejected(self):
        """Genuine '..' between '/' separators must still be rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_inputs(**{**_VALID_KWARGS, "checkpoint_path": "/data/../etc/passwd"})

    def test_real_traversal_relative_still_rejected(self):
        """Genuine relative '..' traversal must still be rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_inputs(**{**_VALID_KWARGS, "checkpoint_path": "../etc/passwd"})

    def test_backslash_dotdot_backslash_filename_accepted(self):
        """Filename with embedded \\..\\ as literal bytes (no '/' separator) accepted."""
        # "a\..\b" as a single-component filename -- legal on POSIX.
        validate_inputs(**{**_VALID_KWARGS, "trt_engine_path": "a\\..\\b"})


class TestHfPathTraversalValidation:
    """Pin tests for hf_repo/hf_subfolder/hf_local_dir validation (R3).

    These fail on pre-fix code where hf_subfolder flows unvalidated into
    docker --model-path argv via _lifecycle(). See AGENTS.md > Review
    Learnings (#92) > 'LLM Input Safety > Validate before subprocess
    interpolation'.
    """

    def test_hf_subfolder_traversal_rejected(self):
        """hf_subfolder='../../etc' must be rejected by validate_inputs."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="hf_subfolder"):
            validate_inputs(
                action="lifecycle",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="gr00t_engine",
                container_name=None,
                protocol="n1.5",
                hf_repo="nvidia/GR00T-N1.7-LIBERO",
                hf_subfolder="../../etc/passwd",
                lifecycle="full",
            )

    def test_hf_subfolder_shell_meta_rejected(self):
        """hf_subfolder with shell metacharacters must be rejected."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="hf_subfolder"):
            validate_inputs(
                action="lifecycle",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="gr00t_engine",
                container_name=None,
                protocol="n1.5",
                hf_repo="nvidia/GR00T-N1.7-LIBERO",
                hf_subfolder="libero;rm -rf /",
                lifecycle="full",
            )

    def test_hf_repo_malformed_rejected(self):
        """hf_repo must be org/name format."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="hf_repo"):
            validate_inputs(
                action="lifecycle",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="gr00t_engine",
                container_name=None,
                protocol="n1.5",
                hf_repo="../../etc/shadow",
                lifecycle="full",
            )

    def test_hf_local_dir_traversal_rejected(self):
        """hf_local_dir with traversal must be rejected."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="hf_local_dir"):
            validate_inputs(
                action="lifecycle",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="gr00t_engine",
                container_name=None,
                protocol="n1.5",
                hf_repo="nvidia/GR00T-N1.7-LIBERO",
                hf_local_dir="/data/../../../etc",
                lifecycle="full",
            )

    def test_lifecycle_invalid_phase_rejected(self):
        """lifecycle phase must be 'full' or 'teardown'."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match="lifecycle"):
            validate_inputs(
                action="lifecycle",
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                port=5555,
                host="127.0.0.1",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path=None,
                trt_engine_path="gr00t_engine",
                container_name=None,
                protocol="n1.5",
                lifecycle="exec_shell",
            )

    def test_valid_hf_params_pass(self):
        """Valid hf_repo/hf_subfolder/lifecycle should not raise."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        # Should not raise
        validate_inputs(
            action="lifecycle",
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            port=5555,
            host="127.0.0.1",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            checkpoint_path=None,
            trt_engine_path="gr00t_engine",
            container_name=None,
            protocol="n1.5",
            hf_repo="nvidia/GR00T-N1.7-LIBERO",
            hf_subfolder="libero_spatial",
            hf_local_dir="/data/checkpoints/libero",
            lifecycle="full",
        )


class TestHfValidationOnDownloadCheckpoint:
    """Pin: hf_* validation must run for action='download_checkpoint'.

    Regression: in R3, hf_repo/hf_subfolder/hf_local_dir validation was
    placed AFTER the `_image_only_actions` early-return inside
    validate_inputs(). Since 'download_checkpoint' is in that early-return
    set, the hf_* checks were never reached when called via that action -
    silently bypassing the path-traversal guard that the docstring
    advertises. R4 hoists the hf_*/lifecycle validation BEFORE the
    action-specific gates so it applies regardless of action.

    These tests fail on pre-R4 code (hf_* checks bypassed for
    'download_checkpoint') and pass on post-R4 code.
    """

    _COMMON_KWARGS = dict(
        data_config="fourier_gr1_arms_only",
        embodiment_tag="gr1",
        port=5555,
        host="127.0.0.1",
        vit_dtype="fp8",
        llm_dtype="nvfp4",
        dit_dtype="fp8",
        checkpoint_path=None,
        trt_engine_path="gr00t_engine",
        container_name=None,
        protocol="n1.5",
    )

    def test_hf_subfolder_traversal_rejected_on_download_checkpoint(self):
        """hf_subfolder='../../etc' must be rejected on 'download_checkpoint'."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match=r"hf_subfolder.*\.\."):
            validate_inputs(
                action="download_checkpoint",
                hf_subfolder="../../etc/passwd",
                **self._COMMON_KWARGS,
            )

    def test_hf_local_dir_traversal_rejected_on_download_checkpoint(self):
        """hf_local_dir with '..' must be rejected on 'download_checkpoint'."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match=r"hf_local_dir.*\.\."):
            validate_inputs(
                action="download_checkpoint",
                hf_local_dir="../../etc",
                **self._COMMON_KWARGS,
            )

    def test_hf_repo_invalid_format_rejected_on_download_checkpoint(self):
        """hf_repo='--evil/x' (option-injection-like) must be rejected."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match=r"hf_repo.*org/name"):
            validate_inputs(
                action="download_checkpoint",
                hf_repo="--evil/x",
                **self._COMMON_KWARGS,
            )

    def test_hf_subfolder_traversal_rejected_on_lifecycle_too(self):
        """Sanity: lifecycle path still validates hf_subfolder (no regression)."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        with pytest.raises(ValueError, match=r"hf_subfolder.*\.\."):
            validate_inputs(
                action="lifecycle",
                hf_subfolder="../escape",
                lifecycle="full",
                **self._COMMON_KWARGS,
            )

    def test_valid_hf_params_pass_on_download_checkpoint(self):
        """Sanity: legitimate hf_* values must not raise on 'download_checkpoint'."""
        from strands_robots.tools.gr00t_inference import validate_inputs

        validate_inputs(
            action="download_checkpoint",
            hf_repo="nvidia/GR00T-N1.7-LIBERO",
            hf_subfolder="libero_spatial",
            hf_local_dir="/data/checkpoints/libero",
            **self._COMMON_KWARGS,
        )


class TestInferenceServerBindsAllInterfaces:
    """Pin: the inference server inside the container always binds 0.0.0.0.

    Regression: pre-R4 the `host` kwarg flowed verbatim into BOTH the
    docker `-p HOST:port:port` host-side bind AND the inference server's
    `--host` flag inside the container. With the new default
    `host="127.0.0.1"`, the service bound to container-loopback and the
    docker port-publish forwarded to nothing -- the headline contract
    ("loopback default is reachable") was broken end-to-end.

    R4 hardcodes `--host 0.0.0.0` for the inference server inside the
    container; the `host` kwarg now exclusively controls the host-side
    bind. Tests fail on pre-R4 code and pass on post-R4.
    """

    def _build_argv(self, **overrides):
        from strands_robots.tools.gr00t_inference import _build_inference_command

        defaults = dict(
            container_name="gr00t-test",
            checkpoint_path="/data/checkpoints/x",
            port=5555,
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            denoising_steps=4,
            http_server=False,
            use_tensorrt=False,
            trt_engine_path="gr00t_engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            api_token=None,
            protocol="n1.5",
            use_sim_policy_wrapper=False,
        )
        defaults.update(overrides)
        return _build_inference_command(**defaults)

    def test_n15_inference_server_binds_all_interfaces(self):
        """N1.5 protocol must include '--host 0.0.0.0' regardless of caller."""
        argv = self._build_argv(protocol="n1.5")
        # Find --host flag and assert its value is 0.0.0.0
        idx = argv.index("--host")
        assert argv[idx + 1] == "0.0.0.0", f"expected --host 0.0.0.0, got {argv[idx + 1]!r}"

    def test_n16_inference_server_binds_all_interfaces(self):
        """N1.6 protocol must include '--host 0.0.0.0'."""
        argv = self._build_argv(protocol="n1.6")
        idx = argv.index("--host")
        assert argv[idx + 1] == "0.0.0.0"

    def test_n17_inference_server_binds_all_interfaces(self):
        """N1.7 protocol must include '--host 0.0.0.0'."""
        argv = self._build_argv(protocol="n1.7")
        idx = argv.index("--host")
        assert argv[idx + 1] == "0.0.0.0"

    def test_build_inference_command_does_not_accept_host_kwarg(self):
        """Pin: host kwarg must be removed from _build_inference_command.

        AGENTS.md > Conventions: 'No dead code'. host is no longer used
        inside the cmd builder, so the parameter is removed; passing it
        must raise TypeError.
        """
        from strands_robots.tools.gr00t_inference import _build_inference_command

        with pytest.raises(TypeError, match="host"):
            _build_inference_command(
                container_name="x",
                checkpoint_path="/x",
                port=5555,
                host="127.0.0.1",  # type: ignore[call-arg]
                data_config="fourier_gr1_arms_only",
                embodiment_tag="gr1",
                denoising_steps=4,
                http_server=False,
                use_tensorrt=False,
                trt_engine_path="gr00t_engine",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                api_token=None,
                protocol="n1.5",
                use_sim_policy_wrapper=False,
            )
