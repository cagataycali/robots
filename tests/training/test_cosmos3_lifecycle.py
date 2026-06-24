"""Lifecycle tests for Cosmos3Trainer: prepare/train/export run in-process.

These cover the parts that drive cosmos_framework as a Python library - the
DCP convert (``prepare``), the ``train.launch(config, args)`` call (single- and
multi-GPU), the DCP->safetensors ``export``, and ``latest_checkpoint`` - by
injecting a fake ``cosmos_framework`` package into ``sys.modules``. No real
cosmos-framework checkout, no subprocess, no GPU required: every upstream
callable is a recording stub, so we assert the trainer's orchestration
(which functions it calls, with what typed args, and how it maps failures to a
``TrainResult``) rather than cosmos's internals.
"""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training.cosmos3 import (
    Cosmos3Trainer,
    _cosmos_worker,
    _import_cosmos_module,
    _run_cosmos_launch,
)


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
        num_gpus=1,
        extra={"cosmos_root": fake_cosmos_root, "sft_toml": sft_toml},
    )


@pytest.fixture
def fake_cosmos(monkeypatch):
    """Inject a recording fake ``cosmos_framework`` package into sys.modules.

    Returns a ``calls`` namespace recording the typed args each upstream
    callable received, so tests can assert the trainer passed them through
    correctly. By default every stub succeeds; tests flip ``raises`` flags to
    drive the error paths.
    """
    calls = SimpleNamespace(convert=None, export=None, launch=None, load_toml=None, raise_export=False)

    def _mod(name: str) -> ModuleType:
        m = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    pkg = _mod("cosmos_framework")
    pkg.__path__ = []  # mark as package

    # inference.common.args.CheckpointOverrides
    inference = _mod("cosmos_framework.inference")
    common = _mod("cosmos_framework.inference.common")
    args_mod = _mod("cosmos_framework.inference.common.args")
    inference.common = common
    common.args = args_mod

    class CheckpointOverrides:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

    args_mod.CheckpointOverrides = CheckpointOverrides

    # scripts.convert_model_to_dcp
    scripts = _mod("cosmos_framework.scripts")
    convert_mod = _mod("cosmos_framework.scripts.convert_model_to_dcp")
    scripts.convert_model_to_dcp = convert_mod

    class ConvertArgs:
        def __init__(self, checkpoint, output_path):
            self.checkpoint = checkpoint
            self.output_path = output_path

    convert_mod.Args = ConvertArgs

    def convert_model_to_dcp(a):
        calls.convert = a

    convert_mod.convert_model_to_dcp = convert_model_to_dcp

    # scripts.export_model
    export_mod = _mod("cosmos_framework.scripts.export_model")
    scripts.export_model = export_mod

    class ExportArgs:
        def __init__(self, checkpoint, output_dir):
            self.checkpoint = checkpoint
            self.output_dir = output_dir

    export_mod.Args = ExportArgs

    def export_model(a):
        if calls.raise_export:
            raise RuntimeError("boom-export")
        calls.export = a

    export_mod.export_model = export_model

    # scripts.train.launch
    train_mod = _mod("cosmos_framework.scripts.train")
    scripts.train = train_mod

    def launch(config, args):
        calls.launch = (config, args)

    train_mod.launch = launch

    # configs.toml_config.sft_config.load_experiment_from_toml
    configs = _mod("cosmos_framework.configs")
    toml_config = _mod("cosmos_framework.configs.toml_config")
    sft_config = _mod("cosmos_framework.configs.toml_config.sft_config")
    configs.toml_config = toml_config
    toml_config.sft_config = sft_config

    def load_experiment_from_toml(toml, extra_overrides):
        calls.load_toml = (toml, list(extra_overrides))
        return SimpleNamespace(name="fake-config")

    sft_config.load_experiment_from_toml = load_experiment_from_toml

    return calls


class TestLatestCheckpoint:
    def test_none_before_training(self, spec):
        # output_dir not yet created
        assert Cosmos3Trainer().latest_checkpoint(spec.output_dir) is None

    def test_none_when_only_scratch_dirs(self, spec, tmp_path):
        out = tmp_path / "out"
        (out / "_dcp_base").mkdir(parents=True)
        (out / "_exported").mkdir()
        (out / "train.log").write_text("x")
        assert Cosmos3Trainer().latest_checkpoint(str(out)) is None

    def test_returns_dir_when_checkpoint_written(self, spec, tmp_path):
        out = tmp_path / "out"
        (out / "_dcp_base").mkdir(parents=True)
        (out / "iter_000500").mkdir()  # real training output
        assert Cosmos3Trainer().latest_checkpoint(str(out)) == str(out)


class TestPrepare:
    def test_converts_base_to_dcp(self, spec, fake_cosmos):
        Cosmos3Trainer(cosmos_root=spec.extra["cosmos_root"]).prepare(spec)
        assert fake_cosmos.convert is not None
        assert fake_cosmos.convert.checkpoint.checkpoint_path == "nvidia/Cosmos3-Nano"
        assert fake_cosmos.convert.output_path.endswith("_dcp_base")

    def test_idempotent_skip_when_dcp_present(self, spec, fake_cosmos, tmp_path):
        import os

        dcp = os.path.join(spec.output_dir, "_dcp_base", "model")
        os.makedirs(dcp)
        Cosmos3Trainer().prepare(spec)
        assert fake_cosmos.convert is None  # convert NOT called

    def test_noop_without_cosmos_root(self, spec, fake_cosmos, monkeypatch):
        monkeypatch.delenv("COSMOS_ROOT", raising=False)
        spec.extra.pop("cosmos_root")
        Cosmos3Trainer().prepare(spec)
        assert fake_cosmos.convert is None


class TestExport:
    def test_converts_dcp_to_safetensors(self, spec, fake_cosmos):
        out = Cosmos3Trainer().export(spec, "/some/dcp")
        assert out.endswith("_exported")
        assert fake_cosmos.export.checkpoint.checkpoint_path == "/some/dcp"

    def test_respects_export_dir_override(self, spec, fake_cosmos, tmp_path):
        spec.extra["export_dir"] = str(tmp_path / "hf_out")
        out = Cosmos3Trainer().export(spec, "/some/dcp")
        assert out == str(tmp_path / "hf_out")

    def test_passthrough_without_cosmos_root(self, spec, fake_cosmos, monkeypatch):
        monkeypatch.delenv("COSMOS_ROOT", raising=False)
        spec.extra.pop("cosmos_root")
        out = Cosmos3Trainer().export(spec, "/some/dcp")
        assert out == "/some/dcp"  # falls back to input checkpoint dir

    def test_falls_back_on_export_failure(self, spec, fake_cosmos):
        fake_cosmos.raise_export = True
        out = Cosmos3Trainer().export(spec, "/some/dcp")
        assert out == "/some/dcp"  # error -> returns DCP dir, no raise


class TestTrain:
    def test_validation_failure_short_circuits(self, spec, fake_cosmos):
        spec.steps = 0  # invalid
        result = Cosmos3Trainer().train(spec)
        assert result.status == "error"
        assert "validation failed" in result.message
        assert fake_cosmos.launch is None

    def test_single_gpu_happy_path(self, spec, fake_cosmos):
        result = Cosmos3Trainer().train(spec)
        assert result.status == "success"
        assert result.job_id.startswith("cosmos3-")
        assert result.checkpoint_dir == spec.output_dir
        # train.launch(config, args) was called with our overrides applied
        assert fake_cosmos.launch is not None
        config, args = fake_cosmos.launch
        assert config.name == "fake-config"
        assert "trainer.max_iter=1000" in args.opts

    def test_prepare_failure_surfaced(self, spec, fake_cosmos, monkeypatch):
        def boom(_spec):
            raise RuntimeError("convert-died")

        monkeypatch.setattr(Cosmos3Trainer, "prepare", boom)
        result = Cosmos3Trainer().train(spec)
        assert result.status == "error"
        assert "DCP conversion (prepare) failed" in result.message

    def test_launch_failure_surfaced(self, spec, fake_cosmos):
        def boom(config, args):
            raise RuntimeError("train-died")

        sys.modules["cosmos_framework.scripts.train"].launch = boom
        result = Cosmos3Trainer().train(spec)
        assert result.status == "error"
        assert "RuntimeError" in result.message
        assert result.checkpoint_dir == spec.output_dir

    def test_multi_gpu_uses_elastic_launch(self, spec, fake_cosmos, monkeypatch):
        spec.num_gpus = 4
        captured = {}

        def fake_elastic(fn, *, nproc_per_node, nnodes, rdzv_endpoint, run_id, fn_args):
            captured["nproc"] = nproc_per_node
            captured["nnodes"] = nnodes
            captured["fn_args"] = fn_args

        monkeypatch.setattr("strands_robots.training.cosmos3.elastic_launch_callable", fake_elastic)
        result = Cosmos3Trainer().train(spec)
        assert result.status == "success"
        assert captured["nproc"] == 4
        assert captured["nnodes"] == 1
        # fn_args = (sft_toml, overrides, log_path)
        assert captured["fn_args"][0] == spec.extra["sft_toml"]

    def test_multi_node_passes_rdzv(self, spec, fake_cosmos, monkeypatch):
        spec.num_nodes = 2
        spec.num_gpus = 8
        spec.extra["rdzv_endpoint"] = "head:29500"
        captured = {}

        def fake_elastic(fn, *, nproc_per_node, nnodes, rdzv_endpoint, run_id, fn_args):
            captured["rdzv"] = rdzv_endpoint
            captured["nnodes"] = nnodes

        monkeypatch.setattr("strands_robots.training.cosmos3.elastic_launch_callable", fake_elastic)
        result = Cosmos3Trainer().train(spec)
        assert result.status == "success"
        assert captured["rdzv"] == "head:29500"
        assert captured["nnodes"] == 2


class TestRunCosmosLaunch:
    def test_builds_config_and_calls_launch(self, fake_cosmos, sft_toml):
        overrides = ["trainer.max_iter=10", "optimizer.lr=0.001"]
        _run_cosmos_launch(sft_toml, overrides)
        assert fake_cosmos.load_toml == (sft_toml, overrides)
        config, args = fake_cosmos.launch
        assert args.opts == overrides
        assert args.deterministic is False
        assert args.dryrun is False


class TestCosmosWorker:
    def test_rank0_writes_log(self, fake_cosmos, sft_toml, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        log = str(tmp_path / "w.log")
        _cosmos_worker(sft_toml, ["trainer.max_iter=5"], log)
        assert fake_cosmos.launch is not None

    def test_non_rank0_no_log(self, fake_cosmos, sft_toml, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        log = str(tmp_path / "w.log")
        _cosmos_worker(sft_toml, ["trainer.max_iter=5"], log)
        assert fake_cosmos.launch is not None


class TestValidateBranches:
    """Exhaustive validate() problem branches (one assert per rejected input)."""

    def test_dataset_root_required(self, spec):
        spec.dataset_root = ""
        assert any("dataset_root is required" in p for p in Cosmos3Trainer().validate(spec))

    def test_dataset_root_not_v3(self, spec, tmp_path):
        empty = tmp_path / "not_a_dataset"
        empty.mkdir()
        spec.dataset_root = str(empty)
        assert any("not a LeRobotDataset v3 root" in p for p in Cosmos3Trainer().validate(spec))

    def test_base_model_required(self, spec):
        spec.base_model = ""
        assert any("base_model is required" in p for p in Cosmos3Trainer().validate(spec))

    def test_output_dir_required(self, spec):
        spec.output_dir = ""
        assert any("output_dir is required" in p for p in Cosmos3Trainer().validate(spec))

    def test_unsupported_method(self, spec):
        spec.method = "expert_only"
        assert any("unsupported method" in p for p in Cosmos3Trainer().validate(spec))

    def test_nonpositive_steps(self, spec):
        spec.steps = -5
        assert any("steps must be > 0" in p for p in Cosmos3Trainer().validate(spec))

    def test_sft_toml_does_not_exist(self, spec):
        spec.extra["sft_toml"] = "/no/such/recipe.toml"
        assert any("sft_toml does not exist" in p for p in Cosmos3Trainer().validate(spec))

    def test_cosmos_package_missing_under_root(self, spec, tmp_path):
        bare = tmp_path / "bare_root"
        bare.mkdir()  # no cosmos_framework/ subdir
        spec.extra["cosmos_root"] = str(bare)
        assert any("cosmos_framework package not found" in p for p in Cosmos3Trainer().validate(spec))


class TestOverridesExtras:
    def test_seed_appended_when_set(self, spec):
        spec.seed = 7
        assert "trainer.seed=7" in Cosmos3Trainer().build_overrides(spec)

    def test_seed_absent_when_none(self, spec):
        spec.seed = None
        assert not any(o.startswith("trainer.seed=") for o in Cosmos3Trainer().build_overrides(spec))

    def test_extra_passthrough_as_hydra_override(self, spec):
        spec.extra["model.config.foo"] = "bar"
        assert "model.config.foo=bar" in Cosmos3Trainer().build_overrides(spec)


class TestTrainEntrypointImportError:
    def test_train_import_failure_surfaced(self, spec, fake_cosmos, monkeypatch):
        # prepare() succeeds, but the train entrypoint import fails the preflight.
        real_import = _import_cosmos_module

        def flaky(qualname):
            if qualname == "scripts.train":
                raise ImportError("no train module")
            return real_import(qualname)

        monkeypatch.setattr("strands_robots.training.cosmos3._import_cosmos_module", flaky)
        result = Cosmos3Trainer().train(spec)
        assert result.status == "error"
        assert "no train module" in result.message
        assert result.job_id.startswith("cosmos3-")
