"""Tests for Gr00tTrainer: factory wiring + validate.

The trainer now drives GR00T AS A LIBRARY (build FinetuneConfig + call
gr00t.experiment.experiment.run), so the build step needs the real ``gr00t``
package importable. Those config-building assertions live in the parity test
(tests/training/test_native_parity.py), which runs only where GR00T_ROOT points
at a real Isaac-GR00T checkout. Here we keep the offline/pure checks: factory
wiring + validate (which only stat-checks the checkout layout, no import).
"""

import json

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.groot import Gr00tTrainer


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


@pytest.fixture
def fake_groot_root(tmp_path):
    """A fake Isaac-GR00T checkout (just the package dir validate() stat-checks)."""
    (tmp_path / "gr00t").mkdir()
    return str(tmp_path)


@pytest.fixture
def spec(dataset_root, tmp_path, fake_groot_root):
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="nvidia/GR00T-N1.5-3B",
        output_dir=str(tmp_path / "out"),
        embodiment="GR1",
        steps=500,
        global_batch_size=32,
        learning_rate=1e-4,
        save_freq=100,
        extra={"groot_root": fake_groot_root},
    )


class TestFactoryWiring:
    def test_resolves_from_registry(self):
        t = create_trainer("groot")
        assert isinstance(t, Gr00tTrainer)
        assert t.provider_name == "groot"

    def test_hardware_floor(self):
        assert create_trainer("groot").hardware_floor["min_gpus"] == 1


class TestValidate:
    def test_clean(self, spec):
        assert Gr00tTrainer().validate(spec) == []

    def test_embodiment_required(self, spec):
        spec.embodiment = None
        problems = Gr00tTrainer().validate(spec)
        assert any("embodiment is required" in p for p in problems)

    def test_missing_groot_root(self, spec):
        spec.extra.pop("groot_root")
        problems = Gr00tTrainer().validate(spec)
        assert any("Isaac-GR00T checkout not found" in p for p in problems)

    def test_missing_gr00t_package(self, spec, tmp_path):
        # groot_root exists but has no gr00t/ package dir.
        empty = tmp_path / "empty_root"
        empty.mkdir()
        spec.extra["groot_root"] = str(empty)
        problems = Gr00tTrainer().validate(spec)
        assert any("gr00t package not found" in p for p in problems)

    def test_bad_modality_config_path(self, spec):
        spec.extra["modality_config_path"] = "/does/not/exist.py"
        problems = Gr00tTrainer().validate(spec)
        assert any("modality_config_path does not exist" in p for p in problems)

    def test_multi_node_rejected(self, spec):
        spec.num_nodes = 2
        problems = Gr00tTrainer().validate(spec)
        assert any("multi-node" in p for p in problems)


class TestTuneResolution:
    """Pure tune-dict resolution (no gr00t import needed)."""

    def test_default_tune(self, spec):
        tune = Gr00tTrainer()._resolve_tune(spec)
        assert tune == {"llm": False, "visual": False, "projector": True, "diffusion": True}

    def test_custom_tune(self, spec):
        spec.tune = {"llm": True, "visual": True, "projector": False, "diffusion": False}
        tune = Gr00tTrainer()._resolve_tune(spec)
        assert tune == {"llm": True, "visual": True, "projector": False, "diffusion": False}

    def test_frozen_backbone_forces_backbone_off(self, spec):
        spec.method = "frozen_backbone"
        spec.tune = {"llm": True, "visual": True}
        tune = Gr00tTrainer()._resolve_tune(spec)
        assert tune["llm"] is False
        assert tune["visual"] is False
        # projector/diffusion keep their defaults
        assert tune["projector"] is True
        assert tune["diffusion"] is True


class TestBuildFinetuneConfig:
    """build_finetune_config builds GR00T's real FinetuneConfig object.

    Requires the real ``gr00t`` package; skipped where it isn't importable
    (laptops/CI without the Isaac-GR00T checkout). The argv/flag parity against
    the real FinetuneConfig fields is covered by test_native_parity.py.
    """

    def test_builds_real_finetune_config(self, spec):
        pytest.importorskip("gr00t")
        cfg = Gr00tTrainer().build_finetune_config(spec)
        assert cfg.base_model_path == "nvidia/GR00T-N1.5-3B"
        assert cfg.dataset_path == spec.dataset_root
        assert cfg.embodiment_tag == "GR1"
        assert cfg.max_steps == 500
        assert cfg.global_batch_size == 32
        assert cfg.save_steps == 100
        assert cfg.tune_projector is True
        assert cfg.tune_diffusion_model is True
