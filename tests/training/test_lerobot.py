"""Tests for LerobotTrainer: factory wiring, validate, and config building.

These are pure/offline (no GPU, no actual training launch). The trainer now
runs lerobot IN-PROCESS (no subprocess), so the build step produces a typed
``TrainPipelineConfig`` object rather than an argv list. The real end-to-end
sim->train->load is exercised separately (test_lerobot_e2e.py).
"""

import json

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.lerobot import LerobotTrainer

# build_config() touches lerobot dataclasses, so those tests need lerobot.
lerobot = pytest.importorskip("lerobot")


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

    def test_multi_node_rejected(self, spec):
        spec.num_nodes = 2
        problems = LerobotTrainer().validate(spec)
        assert any("multi-node" in p for p in problems)


class TestBuildConfig:
    """build_config() yields a typed TrainPipelineConfig (no argv strings)."""

    def test_core_fields(self, spec):
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        # dataset
        assert cfg.dataset.repo_id == "local"
        assert cfg.dataset.root == spec.dataset_root
        # policy
        assert cfg.policy.type == "act"
        assert cfg.policy.device == "cpu"
        assert cfg.policy.push_to_hub is False
        assert str(cfg.policy.pretrained_path) == "lerobot/act_aloha_sim"
        # training knobs
        assert cfg.steps == 200
        assert cfg.batch_size == 8
        assert cfg.save_freq == 100
        assert cfg.wandb.enable is False
        # no PEFT for full fine-tune
        assert cfg.peft is None

    def test_lora_builds_peft_config(self, spec):
        spec.method = "lora"
        spec.lora_r = 16
        spec.lora_target_modules = "q_proj,v_proj"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.peft is not None
        assert cfg.peft.method_type == "LORA"
        assert cfg.peft.r == 16
        assert cfg.peft.target_modules == "q_proj,v_proj"
        # policy must be flagged to actually wrap with PEFT
        assert cfg.policy.use_peft is True

    def test_expert_only_sets_policy_flag(self, spec):
        # use a policy type that exposes train_expert_only
        spec.extra["policy_type"] = "pi0"
        spec.method = "expert_only"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert getattr(cfg.policy, "train_expert_only", None) is True

    def test_val_split_episodes(self, spec):
        spec.val_episodes = 2  # total 10 -> train on [0..7]
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.episodes == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_seed_and_jobname(self, spec):
        spec.seed = 42
        spec.extra["job_name"] = "my_run"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.seed == 42
        assert cfg.job_name == "my_run"

    def test_typed_passthrough_known_field(self, spec):
        # a real top-level TrainPipelineConfig field is applied via setattr
        spec.extra["num_workers"] = 0
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.num_workers == 0

    def test_dotted_passthrough_subconfig(self, spec):
        # dotted extra resolves to a sub-config (dataset.video_backend)
        spec.extra["dataset.video_backend"] = "pyav"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.video_backend == "pyav"

    def test_unknown_extra_is_ignored_not_injected(self, spec, caplog):
        # An unknown key must NOT become a flag / attribute; it's dropped safely.
        spec.extra["totally_made_up_flag"] = "x; rm -rf /"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert not hasattr(cfg, "totally_made_up_flag")


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
