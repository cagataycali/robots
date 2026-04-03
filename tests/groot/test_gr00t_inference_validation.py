"""Tests for gr00t_inference input validation."""

import pytest

from strands_robots.tools.gr00t_inference import validate_inputs


class TestValidateInputs:
    """Test the centralised validate_inputs() function."""

    def test_valid_defaults(self):
        """Default parameter values must pass validation."""
        validate_inputs(
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            port=5555,
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
        )

    def test_valid_with_all_optional(self):
        """All optional parameters provided with valid values."""
        validate_inputs(
            data_config="so100_dualcam",
            embodiment_tag="so100",
            port=8000,
            vit_dtype="fp16",
            llm_dtype="fp16",
            dit_dtype="fp16",
            checkpoint_path="/data/checkpoints/model",
            trt_engine_path="my_engine",
            container_name="isaac-gr00t-1",
        )

    # ── data_config ──────────────────────────────────────────────────

    def test_invalid_data_config_uppercase(self):
        with pytest.raises(ValueError, match="data_config"):
            validate_inputs(
                data_config="UPPER",
                embodiment_tag="gr1",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    def test_invalid_data_config_shell_chars(self):
        with pytest.raises(ValueError, match="data_config"):
            validate_inputs(
                data_config="config;rm -rf /",
                embodiment_tag="gr1",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    # ── embodiment_tag ───────────────────────────────────────────────

    def test_invalid_embodiment_tag(self):
        with pytest.raises(ValueError, match="embodiment_tag"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="BAD TAG!",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
            )

    # ── port ─────────────────────────────────────────────────────────

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

    # ── dtype allowlists ─────────────────────────────────────────────

    def test_invalid_vit_dtype(self):
        with pytest.raises(ValueError, match="vit_dtype"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="int8",
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
                llm_dtype="bf16",
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
                dit_dtype="nvfp4",
            )

    # ── path validation ──────────────────────────────────────────────

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
                trt_engine_path="engine; rm -rf /",
            )

    # ── container_name ───────────────────────────────────────────────

    def test_invalid_container_name(self):
        with pytest.raises(ValueError, match="container_name"):
            validate_inputs(
                data_config="so100",
                embodiment_tag="so100",
                port=5555,
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                container_name="-invalid",
            )

    def test_container_name_none_is_ok(self):
        """container_name=None should pass (auto-detect)."""
        validate_inputs(
            data_config="so100",
            embodiment_tag="so100",
            port=5555,
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            container_name=None,
        )
