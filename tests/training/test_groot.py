"""Tests for Gr00tTrainer: factory wiring, validate, command building.

Offline/pure - does not require an Isaac-GR00T checkout to run (uses a fake
groot_root with a stub launch_finetune.py for the happy-path command tests).
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
    """A fake Isaac-GR00T checkout with a stub launch_finetune.py."""
    script = tmp_path / "gr00t" / "experiment" / "launch_finetune.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n")
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

    def test_multi_node_rejected(self, spec):
        spec.num_nodes = 2
        problems = Gr00tTrainer().validate(spec)
        assert any("multi-node" in p for p in problems)

    def test_embodiment_required(self, spec):
        spec.embodiment = None
        problems = Gr00tTrainer().validate(spec)
        assert any("embodiment is required" in p for p in problems)

    def test_missing_groot_root(self, spec, monkeypatch):
        monkeypatch.delenv("GR00T_ROOT", raising=False)
        spec.extra.pop("groot_root")
        problems = Gr00tTrainer().validate(spec)
        assert any("Isaac-GR00T checkout not found" in p for p in problems)

    def test_bad_modality_config_path(self, spec):
        spec.extra["modality_config_path"] = "/does/not/exist.py"
        problems = Gr00tTrainer().validate(spec)
        assert any("modality_config_path does not exist" in p for p in problems)


class TestBuildCommand:
    def test_single_gpu_core_flags(self, spec):
        cmd = Gr00tTrainer().build_command(spec)
        joined = " ".join(cmd)
        assert "launch_finetune.py" in joined
        assert "--base_model_path=nvidia/GR00T-N1.5-3B" in cmd
        assert f"--dataset_path={spec.dataset_root}" in cmd
        assert "--embodiment_tag=GR1" in cmd
        assert "--max_steps=500" in cmd
        assert "--global_batch_size=32" in cmd
        assert "--save_steps=100" in cmd
        assert "--num_gpus=1" in cmd

    def test_default_tune_flags(self, spec):
        cmd = Gr00tTrainer().build_command(spec)
        assert "--tune_llm=false" in cmd
        assert "--tune_visual=false" in cmd
        assert "--tune_projector=true" in cmd
        assert "--tune_diffusion_model=true" in cmd

    def test_custom_tune_dict(self, spec):
        spec.tune = {"llm": True, "visual": True, "projector": False, "diffusion": False}
        cmd = Gr00tTrainer().build_command(spec)
        assert "--tune_llm=true" in cmd
        assert "--tune_visual=true" in cmd
        assert "--tune_projector=false" in cmd
        assert "--tune_diffusion_model=false" in cmd

    def test_frozen_backbone_method(self, spec):
        spec.method = "frozen_backbone"
        spec.tune = {"llm": True, "visual": True}  # should be forced off
        cmd = Gr00tTrainer().build_command(spec)
        assert "--tune_llm=false" in cmd
        assert "--tune_visual=false" in cmd

    def test_multi_gpu_uses_torchrun(self, spec):
        spec.num_gpus = 4
        cmd = Gr00tTrainer().build_command(spec)
        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=4" in cmd
        assert "--num_gpus=4" in cmd

    def test_resume_flag(self, spec):
        spec.resume = True
        cmd = Gr00tTrainer().build_command(spec)
        assert "--resume_from_checkpoint" in cmd

    def test_modality_config_and_passthrough(self, spec, tmp_path):
        mcfg = tmp_path / "modality.py"
        mcfg.write_text("# modality\n")
        spec.extra["modality_config_path"] = str(mcfg)
        spec.extra["weight_decay"] = 1e-5
        cmd = Gr00tTrainer().build_command(spec)
        assert f"--modality_config_path={mcfg}" in cmd
        assert "--weight_decay=1e-05" in cmd
        # consumed keys must not leak
        assert not any(c.startswith("--groot_root=") for c in cmd)
