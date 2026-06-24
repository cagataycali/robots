"""Tests for Gr00tTrainer: factory wiring, validate, argument building.

Offline/pure - does not require an Isaac-GR00T checkout to run (uses a fake
groot_root with a stub launch_finetune.py for the happy-path arg tests). The
trainer now runs launch_finetune.py IN-PROCESS (no subprocess / torchrun), so
the build step returns a pure flag LIST (no launcher binary, no script path).
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

    def test_python_executable_kwarg_is_accepted_but_ignored(self):
        # Back-compat: old callers passed python_executable for the subprocess.
        # It must not raise; the script is now run in-process so it's ignored.
        t = Gr00tTrainer(python_executable="/usr/bin/python3")
        assert isinstance(t, Gr00tTrainer)


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

    def test_bad_modality_config_path(self, spec):
        spec.extra["modality_config_path"] = "/does/not/exist.py"
        problems = Gr00tTrainer().validate(spec)
        assert any("modality_config_path does not exist" in p for p in problems)

    def test_multi_node_rejected(self, spec):
        spec.num_nodes = 2
        problems = Gr00tTrainer().validate(spec)
        assert any("multi-node" in p for p in problems)


class TestBuildArgs:
    """build_args() returns the launch_finetune.py flag LIST (no launcher)."""

    def test_core_flags(self, spec):
        args = Gr00tTrainer().build_args(spec)
        # No launcher binary / script path - just the script's own flags.
        assert not any(a in ("torchrun", "python") for a in args)
        assert not any(a.endswith("launch_finetune.py") for a in args)
        assert "--base_model_path=nvidia/GR00T-N1.5-3B" in args
        assert f"--dataset_path={spec.dataset_root}" in args
        assert "--embodiment_tag=GR1" in args
        assert "--max_steps=500" in args
        assert "--global_batch_size=32" in args
        assert "--save_steps=100" in args
        assert "--num_gpus=1" in args

    def test_default_tune_flags(self, spec):
        args = Gr00tTrainer().build_args(spec)
        assert "--tune_llm=false" in args
        assert "--tune_visual=false" in args
        assert "--tune_projector=true" in args
        assert "--tune_diffusion_model=true" in args

    def test_custom_tune_dict(self, spec):
        spec.tune = {"llm": True, "visual": True, "projector": False, "diffusion": False}
        args = Gr00tTrainer().build_args(spec)
        assert "--tune_llm=true" in args
        assert "--tune_visual=true" in args
        assert "--tune_projector=false" in args
        assert "--tune_diffusion_model=false" in args

    def test_frozen_backbone_method(self, spec):
        spec.method = "frozen_backbone"
        spec.tune = {"llm": True, "visual": True}  # should be forced off
        args = Gr00tTrainer().build_args(spec)
        assert "--tune_llm=false" in args
        assert "--tune_visual=false" in args

    def test_num_gpus_flag_for_multi_gpu(self, spec):
        # build_args is launcher-agnostic; the >1 case is handled by
        # elastic_launch at train() time, but the script still gets --num_gpus.
        spec.num_gpus = 4
        args = Gr00tTrainer().build_args(spec)
        assert "--num_gpus=4" in args

    def test_resume_flag(self, spec):
        spec.resume = True
        args = Gr00tTrainer().build_args(spec)
        assert "--resume_from_checkpoint" in args

    def test_modality_config_and_safe_passthrough(self, spec, tmp_path):
        mcfg = tmp_path / "modality.py"
        mcfg.write_text("# modality\n")
        spec.extra["modality_config_path"] = str(mcfg)
        spec.extra["weight_decay"] = 1e-5
        args = Gr00tTrainer().build_args(spec)
        assert f"--modality_config_path={mcfg}" in args
        assert "--weight_decay=1e-05" in args
        # consumed keys must not leak
        assert not any(a.startswith("--groot_root=") for a in args)

    def test_unsafe_passthrough_key_is_dropped(self, spec):
        # A key with shell metacharacters / spaces must NOT become a token.
        spec.extra["evil key; rm -rf /"] = "boom"
        spec.extra["--inject"] = "x"
        args = Gr00tTrainer().build_args(spec)
        assert not any("evil key" in a for a in args)
        assert not any("rm -rf" in a for a in args)
        assert not any(a.startswith("----inject") for a in args)
