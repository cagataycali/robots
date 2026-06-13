"""Unit tests for Qwen-VLA data config resolution + embodiment prompt."""

import pytest

from strands_robots.policies.qwen_vla import (
    DATA_CONFIG_MAP,
    QwenVlaDataConfig,
    build_embodiment_prompt,
    create_custom_data_config,
    load_data_config,
)


class TestEmbodimentPrompt:
    def test_single_arm_basic(self):
        prompt = build_embodiment_prompt(
            robot_tag="so100", arm_config="single", fps=30, chunk_size=16, instruction="pick up the cube"
        )
        assert prompt == (
            "The robot is so100 with single arm. The control frequency is 30 Hz. "
            "Please predict the next 16 control actions to execute the following task: pick up the cube."
        )

    def test_dual_arm(self):
        prompt = build_embodiment_prompt(
            robot_tag="aloha", arm_config="dual", fps=50, chunk_size=16, instruction="fold towel"
        )
        assert "dual arms" in prompt
        assert "50 Hz" in prompt

    def test_waist_and_mobile_base(self):
        prompt = build_embodiment_prompt(
            robot_tag="g1",
            arm_config="dual",
            fps=30,
            chunk_size=16,
            instruction="walk over",
            has_waist=True,
            has_mobile_base=True,
        )
        assert "dual arms, waist, and mobile base" in prompt

    def test_waist_only(self):
        prompt = build_embodiment_prompt(
            robot_tag="g1", arm_config="dual", fps=30, chunk_size=16, instruction="reach", has_waist=True
        )
        assert "dual arms, waist." in prompt
        assert "mobile base" not in prompt

    def test_trailing_period_not_doubled(self):
        prompt = build_embodiment_prompt(
            robot_tag="so100", arm_config="single", fps=30, chunk_size=16, instruction="do it."
        )
        assert prompt.endswith("do it.")
        assert not prompt.endswith("..")

    def test_invalid_arm_config_raises(self):
        with pytest.raises(ValueError, match="arm_config"):
            build_embodiment_prompt(robot_tag="x", arm_config="triple", fps=30, chunk_size=16, instruction="x")

    def test_empty_robot_tag_raises(self):
        with pytest.raises(ValueError, match="robot_tag"):
            build_embodiment_prompt(robot_tag="", arm_config="single", fps=30, chunk_size=16, instruction="x")

    def test_empty_instruction_raises(self):
        with pytest.raises(ValueError, match="instruction"):
            build_embodiment_prompt(robot_tag="x", arm_config="single", fps=30, chunk_size=16, instruction="   ")

    def test_nonpositive_fps_raises(self):
        with pytest.raises(ValueError, match="fps"):
            build_embodiment_prompt(robot_tag="x", arm_config="single", fps=0, chunk_size=16, instruction="x")

    def test_nonpositive_chunk_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            build_embodiment_prompt(robot_tag="x", arm_config="single", fps=30, chunk_size=-1, instruction="x")


class TestDataConfigResolution:
    def test_known_configs_present(self):
        for name in ("so100", "aloha_bimanual", "widowx", "unitree_g1", "franka_panda", "libero_panda"):
            assert name in DATA_CONFIG_MAP

    def test_aliases_resolve(self):
        assert load_data_config("aloha").name == "aloha_bimanual"
        assert load_data_config("g1").name == "unitree_g1"
        assert load_data_config("libero").name == "libero_panda"

    def test_extends_inherits_and_overrides(self):
        base = load_data_config("so100")
        child = load_data_config("so100_dualcam")
        # Inherited scalar
        assert child.fps == base.fps
        assert child.arm_config == base.arm_config
        # Overridden list
        assert child.video_keys == ["video.front", "video.wrist"]
        assert child.image_view_tags == {"video.front": "ego", "video.wrist": "cam_right_wrist"}

    def test_extends_scalar_override(self):
        g1 = load_data_config("unitree_g1")
        mobile = load_data_config("unitree_g1_mobile")
        assert g1.has_mobile_base is False
        assert mobile.has_mobile_base is True
        # waist inherited
        assert mobile.has_waist is True

    def test_config_morphology_fields(self):
        aloha = load_data_config("aloha_bimanual")
        assert aloha.arm_config == "dual"
        assert aloha.fps == 50
        assert aloha.chunk_size == 16

    def test_embodiment_prompt_method(self):
        cfg = load_data_config("unitree_g1")
        prompt = cfg.embodiment_prompt("open the door")
        assert "unitree_g1" in prompt
        assert "dual arms, waist." in prompt
        assert "open the door" in prompt

    def test_passthrough_instance(self):
        cfg = QwenVlaDataConfig(name="x", robot_tag="x", arm_config="single")
        assert load_data_config(cfg) is cfg

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown data_config"):
            load_data_config("does_not_exist")

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError, match="must be str or QwenVlaDataConfig"):
            load_data_config(123)  # type: ignore[arg-type]

    def test_create_custom_registers(self):
        cfg = create_custom_data_config(
            "my_robot",
            video_keys=["video.cam"],
            state_keys=["state.arm"],
            action_keys=["action.arm"],
            robot_tag="my_robot",
            arm_config="single",
            fps=25,
            chunk_size=8,
        )
        assert load_data_config("my_robot") is cfg
        assert cfg.action_indices == list(range(8))
        assert cfg.fps == 25
