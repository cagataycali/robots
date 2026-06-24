"""Tests for Cosmos3Trainer: factory wiring, validate, prepare/train/export cmds.

Offline/pure - uses a fake cosmos_root + fake sft_toml; does not require the
cosmos-framework env. Verifies the DCP-convert (prepare) and DCP->safetensors
(export) command construction that distinguish Cosmos3 from the other backends.
"""

import json

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.cosmos3 import Cosmos3Trainer


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 8}))
    return str(tmp_path)


@pytest.fixture
def fake_cosmos_root(tmp_path):
    (tmp_path / "cosmos_framework").mkdir()
    return str(tmp_path)


@pytest.fixture
def sft_toml(tmp_path):
    f = tmp_path / "recipe.toml"
    f.write_text("[job]\nexperiment = 'action_policy_droid_nano'\n")
    return str(f)


@pytest.fixture
def spec(dataset_root, tmp_path, fake_cosmos_root, sft_toml):
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="nvidia/Cosmos3-Nano",
        output_dir=str(tmp_path / "out"),
        steps=1000,
        global_batch_size=8,
        learning_rate=2e-4,
        save_freq=500,
        num_gpus=8,
        extra={"cosmos_root": fake_cosmos_root, "sft_toml": sft_toml},
    )


class TestFactoryWiring:
    def test_resolves_from_registry(self):
        t = create_trainer("cosmos3")
        assert isinstance(t, Cosmos3Trainer)
        assert t.provider_name == "cosmos3"

    def test_hardware_floor_is_8xh100(self):
        floor = create_trainer("cosmos3").hardware_floor
        assert floor["min_gpus"] == 8
        assert floor["min_vram_gb"] == 80
        assert floor["multinode"] is True


class TestValidate:
    def test_clean(self, spec):
        assert Cosmos3Trainer().validate(spec) == []

    def test_sft_toml_required(self, spec):
        spec.extra.pop("sft_toml")
        problems = Cosmos3Trainer().validate(spec)
        assert any("needs a recipe TOML" in p for p in problems)

    def test_missing_cosmos_root(self, spec, monkeypatch):
        monkeypatch.delenv("COSMOS_ROOT", raising=False)
        spec.extra.pop("cosmos_root")
        problems = Cosmos3Trainer().validate(spec)
        assert any("cosmos-framework checkout not found" in p for p in problems)


class TestBuildOverrides:
    """build_overrides() yields the Hydra key=value LIST (no launcher/argv).

    prepare()/export() now call cosmos_framework's convert_model_to_dcp() /
    export_model() DIRECTLY with typed Args objects (verified against upstream),
    so there is no convert_command/export_command argv to assert anymore.
    """

    def test_core_overrides(self, spec):
        ov = Cosmos3Trainer().build_overrides(spec)
        assert all("=" in o and not o.startswith("-") for o in ov)
        assert "trainer.max_iter=1000" in ov
        assert "checkpoint.save_iter=500" in ov
        assert "optimizer.lr=0.0002" in ov
        assert any(o.startswith("checkpoint.load_path=") for o in ov)
        assert "dataloader_train.max_samples_per_batch=8" in ov

    def test_consumed_keys_not_leaked(self, spec):
        spec.extra["dcp_path"] = "/tmp/dcp"
        spec.extra["export_dir"] = "/tmp/exp"
        ov = Cosmos3Trainer().build_overrides(spec)
        assert not any(o.startswith("cosmos_root=") for o in ov)
        assert not any(o.startswith("sft_toml=") for o in ov)
        assert not any(o.startswith("dcp_path=") for o in ov)
        assert not any(o.startswith("export_dir=") for o in ov)


class TestBuildCommand:
    def test_torchrun_and_sft_toml(self, spec):
        cmd = Cosmos3Trainer().build_command(spec)
        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=8" in cmd
        assert "-m" in cmd and "cosmos_framework.scripts.train" in cmd
        assert any(c.startswith("--sft-toml=") for c in cmd)

    def test_hydra_tail_overrides_after_dashdash(self, spec):
        cmd = Cosmos3Trainer().build_command(spec)
        assert "--" in cmd
        tail = cmd[cmd.index("--") + 1 :]
        assert "trainer.max_iter=1000" in tail
        assert "checkpoint.save_iter=500" in tail
        assert "optimizer.lr=0.0002" in tail
        assert any(t.startswith("checkpoint.load_path=") for t in tail)
        assert "dataloader_train.max_samples_per_batch=8" in tail

    def test_multinode_hsdp_override(self, spec):
        spec.num_nodes = 4
        cmd = Cosmos3Trainer().build_command(spec)
        tail = cmd[cmd.index("--") + 1 :]
        assert "model.config.parallelism.data_parallel_replicate_degree=4" in tail

    def test_extra_hydra_passthrough(self, spec):
        spec.extra["dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict"] = "True"
        cmd = Cosmos3Trainer().build_command(spec)
        tail = cmd[cmd.index("--") + 1 :]
        assert any("use_filter_dict=True" in t for t in tail)
