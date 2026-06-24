"""Tests for Cosmos3Trainer: factory wiring, validate, prepare/train/export args.

Offline/pure - uses a fake cosmos_root + fake sft_toml; does not require the
cosmos-framework env. The trainer now runs the cosmos scripts IN-PROCESS (no
subprocess / torchrun), so the build steps return pure argument LISTs (no
launcher binary). Verifies the DCP-convert (prepare) and DCP->safetensors
(export) argument construction that distinguish Cosmos3 from the other backends.
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


class TestConvertArgs:
    def test_dcp_convert(self, spec):
        args = Cosmos3Trainer().convert_argv(spec)
        # No launcher / module token - just the convert script's own args.
        assert "nvidia/Cosmos3-Nano" in args
        assert "-o" in args
        # the -o target is the dcp path
        assert args[args.index("-o") + 1].endswith("_dcp_base")


class TestBuildArgs:
    """build_args() returns the cosmos train arg LIST (no torchrun/module)."""

    def test_sft_toml_and_no_launcher(self, spec):
        args = Cosmos3Trainer().build_argv(spec)
        assert not any(a in ("torchrun", "-m", "python") for a in args)
        assert not any("cosmos_framework.scripts.train" in a for a in args)
        assert any(a.startswith("--sft-toml=") for a in args)

    def test_hydra_tail_overrides_after_dashdash(self, spec):
        args = Cosmos3Trainer().build_argv(spec)
        assert "--" in args
        tail = args[args.index("--") + 1:]
        assert "trainer.max_iter=1000" in tail
        assert "checkpoint.save_iter=500" in tail
        assert "optimizer.lr=0.0002" in tail
        assert any(t.startswith("checkpoint.load_path=") for t in tail)
        assert "dataloader_train.max_samples_per_batch=8" in tail

    def test_multinode_hsdp_override(self, spec):
        spec.num_nodes = 4
        args = Cosmos3Trainer().build_argv(spec)
        tail = args[args.index("--") + 1:]
        assert "model.config.parallelism.data_parallel_replicate_degree=4" in tail

    def test_safe_extra_hydra_passthrough(self, spec):
        spec.extra["dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict"] = "True"
        args = Cosmos3Trainer().build_argv(spec)
        tail = args[args.index("--") + 1:]
        assert any("use_filter_dict=True" in t for t in tail)

    def test_unsafe_extra_key_is_dropped(self, spec):
        # Hydra keys are dotted, but spaces/metacharacters must be rejected.
        spec.extra["evil key=$(rm -rf /)"] = "x"
        args = Cosmos3Trainer().build_argv(spec)
        assert not any("evil key" in a for a in args)
        assert not any("rm -rf" in a for a in args)


class TestExportArgs:
    def test_dcp_to_safetensors(self, spec, tmp_path):
        out = str(tmp_path / "exported")
        args = Cosmos3Trainer().export_argv(spec, str(tmp_path / "ckpt"), out)
        assert any(a.startswith("--checkpoint-path=") for a in args)
        assert any(a.startswith("--output-dir=") for a in args)
        assert not any("cosmos_framework" in a for a in args)
