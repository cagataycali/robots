"""Tests for gr00t_inference input validation.

Covers the validate_inputs() function which centralises all parameter
validation for the gr00t_inference tool.
"""

import pytest

from strands_robots.tools.gr00t_inference import validate_inputs


class TestValidateInputs:
    """Tests for the validate_inputs() public function."""

    def test_valid_defaults(self):
        """Default values must pass validation."""
        validate_inputs(
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            port=5555,
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
        )

    def test_valid_with_all_optional(self):
        validate_inputs(
            data_config="so100_dualcam",
            embodiment_tag="so100",
            port=8000,
            vit_dtype="fp16",
            llm_dtype="fp8",
            dit_dtype="fp16",
            checkpoint_path="/data/checkpoints/model",
            trt_engine_path="/engines/cache",
            container_name="gr00t-n17",
        )

    def test_invalid_data_config_uppercase(self):
        with pytest.raises(ValueError, match="data_config"):
            validate_inputs(
                data_config="FourierGR1",
                embodiment_tag="gr1",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    def test_invalid_data_config_shell_chars(self):
        with pytest.raises(ValueError, match="data_config"):
            validate_inputs(
                data_config="foo;rm -rf /",
                embodiment_tag="gr1",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    def test_invalid_embodiment_tag(self):
        with pytest.raises(ValueError, match="embodiment_tag"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="GR1-Sonic!",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    def test_port_zero(self):
        with pytest.raises(ValueError, match="port"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=0,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    def test_port_too_high(self):
        with pytest.raises(ValueError, match="port"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=70000,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    def test_invalid_vit_dtype(self):
        with pytest.raises(ValueError, match="vit_dtype"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="bf16",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    def test_invalid_llm_dtype(self):
        with pytest.raises(ValueError, match="llm_dtype"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="int4",
                dit_dtype="fp8",
            )

    def test_invalid_dit_dtype(self):
        with pytest.raises(ValueError, match="dit_dtype"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="bf16",
            )

    def test_checkpoint_path_traversal(self):
        with pytest.raises(ValueError, match="checkpoint_path"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path="/data/../../../etc/passwd",
            )

    def test_checkpoint_path_null_byte(self):
        with pytest.raises(ValueError, match="checkpoint_path"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                checkpoint_path="/data/model\x00.bin",
            )

    def test_trt_engine_path_shell_injection(self):
        with pytest.raises(ValueError, match="trt_engine_path"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                trt_engine_path="engine;rm -rf /",
            )

    def test_invalid_container_name(self):
        with pytest.raises(ValueError, match="container_name"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                container_name="-invalid-start",
            )

    def test_container_name_none_is_ok(self):
        """container_name=None should not raise."""
        validate_inputs(
            data_config="so100",
            embodiment_tag="so100",
            port=5555,
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            container_name=None,
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

        # Monkeypatch Path to point at our fake proc
        from pathlib import Path as RealPath

        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference.Path",
            lambda p: RealPath(str(p).replace("/proc", str(tmp_path / "proc"))),
        )

        assert _is_gr00t_host_process("123", port=80) is False

    def test_accepts_matching_port(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process should accept when port matches."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "456"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00inference_service.py\x00--port\x008000\x00")

        from pathlib import Path as RealPath

        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference.Path",
            lambda p: RealPath(str(p).replace("/proc", str(tmp_path / "proc"))),
        )

        assert _is_gr00t_host_process("456", port=8000) is True

    def test_rejects_non_gr00t_process(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process should reject non-GR00T processes."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "789"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00some_other_service.py\x00--port\x008000\x00")

        from pathlib import Path as RealPath

        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference.Path",
            lambda p: RealPath(str(p).replace("/proc", str(tmp_path / "proc"))),
        )

        assert _is_gr00t_host_process("789", port=8000) is False

    def test_no_port_check_when_none(self, tmp_path, monkeypatch):
        """_is_gr00t_host_process without port checks only process identity."""
        from strands_robots.tools.gr00t_inference import _is_gr00t_host_process

        proc_dir = tmp_path / "proc" / "321"
        proc_dir.mkdir(parents=True)
        cmdline_file = proc_dir / "cmdline"
        cmdline_file.write_text("python\x00inference_service.py\x00--port\x009999\x00")

        from pathlib import Path as RealPath

        monkeypatch.setattr(
            "strands_robots.tools.gr00t_inference.Path",
            lambda p: RealPath(str(p).replace("/proc", str(tmp_path / "proc"))),
        )

        # Without port kwarg, just checks identity
        assert _is_gr00t_host_process("321") is True
