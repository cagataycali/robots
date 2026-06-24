"""Tests for LerobotTrainer: factory wiring, validate, and command building.

These are pure/offline (no GPU, no actual lerobot_train launch). The real
end-to-end sim->train->load is exercised separately.
"""

import json

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.lerobot import LerobotTrainer


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


@pytest.fixture
def spec(dataset_root, tmp_path):
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="lerobot/act_aloha_sim",
        output_dir=str(tmp_path / "out"),
        steps=200,
        global_batch_size=8,
        save_freq=100,
        extra={"policy_type": "act"},
    )


class TestFactoryWiring:
    def test_resolves_from_registry(self):
        t = create_trainer("lerobot_local")
        assert isinstance(t, LerobotTrainer)
        assert t.provider_name == "lerobot_local"

    def test_alias_resolves(self):
        # 'lerobot' is a policies.json alias of lerobot_local
        t = create_trainer("lerobot")
        assert isinstance(t, LerobotTrainer)


class TestValidate:
    def test_clean(self, spec):
        assert LerobotTrainer().validate(spec) == []

    def test_non_native_policy_type(self, spec):
        spec.extra["policy_type"] = "openvla"
        problems = LerobotTrainer().validate(spec)
        assert any("not LeRobot-native" in p for p in problems)

    def test_lora_expert_clash(self, spec):
        spec.method = "lora"
        spec.tune = {"expert_only": True}
        problems = LerobotTrainer().validate(spec)
        assert any("mutually exclusive" in p for p in problems)

    def test_val_episodes_too_large(self, spec):
        spec.val_episodes = 99  # total is 10
        problems = LerobotTrainer().validate(spec)
        assert any("val_episodes" in p for p in problems)


class TestBuildCommand:
    def test_single_gpu_core_flags(self, spec):
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        # build_command is now a PURE argv-parity helper (no launcher prefix);
        # the module path is the first token.
        assert cmd[0] == "lerobot.scripts.lerobot_train"
        assert "--dataset.repo_id=local" in cmd
        assert f"--dataset.root={spec.dataset_root}" in cmd
        assert "--policy.type=act" in cmd
        assert "--policy.device=cpu" in cmd
        assert "--policy.push_to_hub=false" in cmd
        assert "--steps=200" in cmd
        assert "--batch_size=8" in cmd
        assert "--save_freq=100" in cmd
        assert "--wandb.enable=false" in cmd
        assert "--policy.pretrained_path=lerobot/act_aloha_sim" in cmd

    def test_build_command_is_launcher_free(self, spec):
        # build_command is parity-only: it never prepends accelerate/torchrun/
        # python. Multi-GPU is driven by elastic_launch in train(), not here.
        spec.num_gpus = 4
        cmd = LerobotTrainer(device="cuda").build_command(spec)
        assert cmd[0] == "lerobot.scripts.lerobot_train"
        assert "accelerate" not in cmd
        assert "torchrun" not in cmd
        assert "python" not in cmd

    def test_lora_flags(self, spec):
        spec.method = "lora"
        spec.lora_r = 16
        spec.lora_target_modules = "q_proj,v_proj"
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--peft.method_type=LORA" in cmd
        assert "--peft.r=16" in cmd
        assert "--peft.target_modules=q_proj,v_proj" in cmd

    def test_expert_only_flag(self, spec):
        spec.method = "expert_only"
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--policy.train_expert_only=true" in cmd

    def test_val_split_episodes_flag(self, spec):
        spec.val_episodes = 2  # total 10 -> train on [0..7]
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        ep_flags = [c for c in cmd if c.startswith("--dataset.episodes=")]
        assert ep_flags
        assert ep_flags[0] == "--dataset.episodes=[0, 1, 2, 3, 4, 5, 6, 7]"

    def test_seed_and_jobname_and_passthrough(self, spec):
        spec.seed = 42
        spec.extra["job_name"] = "my_run"
        spec.extra["num_workers"] = 4  # arbitrary passthrough
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--seed=42" in cmd
        assert "--job_name=my_run" in cmd
        assert "--num_workers=4" in cmd
        # consumed keys must NOT leak as flags
        assert not any(c.startswith("--policy_type=") for c in cmd)
        assert not any(c.startswith("--job_name=strands_ft") for c in cmd)


class TestBuildConfig:
    """build_config() yields lerobot's typed TrainPipelineConfig (the real lib path)."""

    def test_builds_typed_config(self, spec):
        pytest.importorskip("lerobot")
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.repo_id == "local"
        assert cfg.dataset.root == spec.dataset_root
        assert cfg.policy.type == "act"
        assert cfg.policy.device == "cpu"
        assert cfg.policy.push_to_hub is False
        assert str(cfg.policy.pretrained_path) == "lerobot/act_aloha_sim"
        assert cfg.steps == 200
        assert cfg.batch_size == 8
        assert cfg.save_freq == 100
        assert cfg.wandb.enable is False
        assert cfg.peft is None

    def test_lora_builds_peft(self, spec):
        pytest.importorskip("lerobot")
        spec.method = "lora"
        spec.lora_r = 16
        spec.lora_target_modules = "q_proj,v_proj"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.peft is not None
        assert cfg.peft.method_type == "LORA"
        assert cfg.peft.r == 16
        assert cfg.peft.target_modules == "q_proj,v_proj"
        assert cfg.policy.use_peft is True

    def test_val_split_episodes(self, spec):
        pytest.importorskip("lerobot")
        spec.val_episodes = 2  # total 10 -> [0..7]
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.episodes == [0, 1, 2, 3, 4, 5, 6, 7]


class TestParseLog:
    """_parse_log against lerobot's real MetricsTracker line format."""

    def test_expand_big_number(self):
        from strands_robots.training.lerobot import _expand_big_number

        assert _expand_big_number("1.2K") == 1200.0
        assert _expand_big_number("2") == 2.0
        assert _expand_big_number("3M") == 3_000_000.0
        assert _expand_big_number("1.5B") == 1.5e9
        assert _expand_big_number("nope") is None
        assert _expand_big_number("") is None

    def test_parses_real_metricstracker_line(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(
            "INFO 2026-06-23 ot_train.py:419 Start offline training\n"
            "step:1.2K smpl:4.9K ep:8 epch:2.00 loss:0.123\n"
            "step:1.3K smpl:5.0K ep:9 epch:2.10 loss:0.087\n"
        )
        m = LerobotTrainer(device="cpu")._parse_log(str(log))
        assert m["latest_step"] == 1300  # newest, K-expanded
        assert abs(m["latest_loss"] - 0.087) < 1e-9
        assert m["latest_epoch"] == 2.10
        assert m["learning"] is True
        assert m["liveness_ok"] is True

    def test_plain_integer_step(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("step:2 smpl:4 ep:1 epch:1.00 loss:0.5\n")
        m = LerobotTrainer(device="cpu")._parse_log(str(log))
        assert m["latest_step"] == 2
        assert m["latest_loss"] == 0.5

    def test_no_metrics_line_means_not_live(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("INFO booting...\nCreating dataset\n")
        m = LerobotTrainer(device="cpu")._parse_log(str(log))
        assert m["liveness_ok"] is False
        assert "latest_step" not in m

    def test_unreadable_log_returns_empty(self):
        assert LerobotTrainer(device="cpu")._parse_log("/no/such/log") == {}
