"""Tests for Cosmos3Trainer: factory wiring, validate, Hydra override building.

The trainer now drives cosmos_framework AS A LIBRARY (build the typed Args /
Config objects and call convert_model_to_dcp / launch / export_model directly),
so there is no argv list to assert anymore. The cosmos package isn't importable
on laptops/CI, so the build_overrides() check stays pure (it produces the Hydra
``key=value`` LIST passed to load_experiment_from_toml(extra_overrides=...)).
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

    def test_missing_cosmos_root(self, spec):
        spec.extra.pop("cosmos_root")
        problems = Cosmos3Trainer().validate(spec)
        assert any("cosmos-framework checkout not found" in p for p in problems)


class TestBuildOverrides:
    """build_overrides() returns the Hydra key=value LIST (no argv flags)."""

    def test_core_overrides(self, spec):
        ov = Cosmos3Trainer().build_overrides(spec)
        # Pure dotted key=value Hydra overrides - no leading dashes, no launcher.
        assert all("=" in o and not o.startswith("-") for o in ov)
        assert "trainer.max_iter=1000" in ov
        assert "checkpoint.save_iter=500" in ov
        assert "optimizer.lr=0.0002" in ov
        assert any(o.startswith("checkpoint.load_path=") for o in ov)
        assert "dataloader_train.max_samples_per_batch=8" in ov

    def test_multinode_hsdp_override(self, spec):
        spec.num_nodes = 4
        ov = Cosmos3Trainer().build_overrides(spec)
        assert "model.config.parallelism.data_parallel_replicate_degree=4" in ov

    def test_seed_override(self, spec):
        spec.seed = 7
        ov = Cosmos3Trainer().build_overrides(spec)
        assert "trainer.seed=7" in ov

    def test_safe_extra_hydra_passthrough(self, spec):
        spec.extra["dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict"] = "True"
        ov = Cosmos3Trainer().build_overrides(spec)
        assert any("use_filter_dict=True" in o for o in ov)

    def test_unsafe_extra_key_is_dropped(self, spec):
        # Hydra keys are dotted, but spaces/metacharacters must be rejected.
        spec.extra["evil key=$(rm -rf /)"] = "x"
        ov = Cosmos3Trainer().build_overrides(spec)
        assert not any("evil key" in o for o in ov)
        assert not any("rm -rf" in o for o in ov)

    def test_consumed_keys_not_leaked(self, spec):
        # cosmos_root / sft_toml / dcp_path etc. must NOT become Hydra overrides.
        spec.extra["dcp_path"] = "/tmp/dcp"
        spec.extra["export_dir"] = "/tmp/exp"
        ov = Cosmos3Trainer().build_overrides(spec)
        assert not any(o.startswith("cosmos_root=") for o in ov)
        assert not any(o.startswith("sft_toml=") for o in ov)
        assert not any(o.startswith("dcp_path=") for o in ov)
        assert not any(o.startswith("export_dir=") for o in ov)
