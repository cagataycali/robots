"""GR00T post-training is a lerobot policy type, and the spec's knobs reach its config.

lerobot ships GR00T N1.7 as a native policy: ``GrootConfig`` declares
``base_model_path``, ``embodiment_tag`` and the four ``tune_*`` component flags,
and ``lerobot_train`` fine-tunes it through the same entry point as every other
policy. :class:`~strands_robots.training.lerobot.LerobotTrainer` has listed
``groot`` as a policy type all along, but it read neither
:attr:`~strands_robots.training.base.TrainSpec.tune` nor
:attr:`~strands_robots.training.base.TrainSpec.embodiment` when building
``cfg.policy``, so a caller who asked to unfreeze the language backbone got a
frozen one and a caller who named an embodiment trained under the default tag -
both while ``validate()`` reported no problem.

These pins hold the whole route: the provider resolves to the lerobot trainer,
every component the caller names lands on the field lerobot declares for it, a
component the resolved policy does NOT declare is refused rather than dropped,
and the base model goes to the field lerobot says can read it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.training as training_pkg
from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.factory import trainer_defaults
from strands_robots.training.lerobot import LerobotTrainer

pytest.importorskip("lerobot", reason="the field mapping is graded against lerobot's own config classes")


@pytest.fixture
def dataset_root(tmp_path: Path) -> str:
    """A LeRobotDataset v3 root, enough for ``validate()`` to read its header."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 5, "codebase_version": "v3.0", "fps": 30}))
    return str(tmp_path)


def _spec(dataset_root: str, tmp_path: Path, **fields: Any) -> TrainSpec:
    return TrainSpec(dataset_root=dataset_root, output_dir=str(tmp_path / "run"), steps=10, **fields)


def _groot() -> LerobotTrainer:
    return LerobotTrainer(policy_type="groot", device="cpu")


class TestTheProviderIsTheLerobotTrainer:
    """``create_trainer("groot")`` post-trains through lerobot, with no second checkout."""

    def test_the_provider_resolves_to_the_lerobot_trainer_pinned_to_groot(self) -> None:
        trainer = create_trainer("groot")
        assert isinstance(trainer, LerobotTrainer)
        assert trainer.policy_type == "groot"
        assert trainer_defaults("groot") == {"policy_type": "groot"}

    def test_a_caller_kwarg_wins_over_the_registry_default(self) -> None:
        """A default fills a kwarg the caller left unset; it never overrides one."""
        trainer = create_trainer("groot", policy_type="act")
        assert isinstance(trainer, LerobotTrainer)
        assert trainer.policy_type == "act"

    def test_no_training_backend_reaches_for_an_out_of_tree_checkout(self) -> None:
        """The GR00T fine-tune needs one install: lerobot."""
        package_dir = Path(training_pkg.__file__).parent
        sources = sorted(package_dir.rglob("*.py"))
        assert len(sources) > 5, f"only {len(sources)} modules scanned"
        offenders = {
            path.relative_to(package_dir).as_posix(): marker
            for path in sources
            for marker in ("GR00T_ROOT", "groot_root", "Isaac-GR00T", "os.chdir")
            if marker in path.read_text(encoding="utf-8")
        }
        assert offenders == {}, f"training backends reaching outside lerobot: {offenders}"

    def test_the_isaac_trainer_module_is_gone(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("strands_robots.training.groot")


class TestEveryRequestedComponentReachesTheConfig:
    """``spec.tune`` writes the policy config field lerobot declares for it."""

    @pytest.mark.parametrize(
        "tune",
        [
            {"llm": True, "visual": True, "projector": False, "diffusion": False},
            {"llm": False, "visual": False, "projector": True, "diffusion": True},
            {"visual": True},
        ],
        ids=("unfreeze_the_backbone", "the_default_recipe", "one_component"),
    )
    def test_the_component_lands_on_its_field(self, tune: dict[str, bool], dataset_root: str, tmp_path: Path) -> None:
        cfg = _groot().build_config(_spec(dataset_root, tmp_path, tune=tune))
        fields = {
            "llm": "tune_llm",
            "visual": "tune_visual",
            "projector": "tune_projector",
            "diffusion": "tune_diffusion_model",
        }
        assert {c: getattr(cfg.policy, fields[c]) for c in tune} == tune

    def test_the_embodiment_tag_lands_on_the_config(self, dataset_root: str, tmp_path: Path) -> None:
        cfg = _groot().build_config(_spec(dataset_root, tmp_path, embodiment="my_robot"))
        assert cfg.policy.embodiment_tag == "my_robot"

    def test_a_policy_that_infers_its_embodiment_is_left_alone(self, dataset_root: str, tmp_path: Path) -> None:
        """TrainSpec.embodiment is documented as GR00T's; lerobot infers it from features."""
        cfg = LerobotTrainer(policy_type="act", device="cpu").build_config(
            _spec(dataset_root, tmp_path, embodiment="my_robot")
        )
        assert not hasattr(cfg.policy, "embodiment_tag")

    @pytest.mark.parametrize(
        ("policy_type", "requested"),
        [("pi0", True), ("smolvla", False)],
        ids=("freeze_the_vlm", "unfreeze_a_policy_that_freezes_by_default"),
    )
    def test_the_expert_only_component_reaches_the_field_method_sets(
        self, policy_type: str, requested: bool, dataset_root: str, tmp_path: Path
    ) -> None:
        """``tune={"expert_only": ...}`` writes the field ``method="expert_only"`` sets.

        The component is the more expressive door: ``method`` can only turn the
        freeze ON, so a policy whose config freezes the VLM by default (smolvla)
        is trainable in full only through ``tune``.
        """
        trainer = LerobotTrainer(policy_type=policy_type, device="cpu")
        cfg = trainer.build_config(_spec(dataset_root, tmp_path, tune={"expert_only": requested}))
        assert cfg.policy.train_expert_only is requested
        if requested:
            by_method = trainer.build_config(_spec(dataset_root, tmp_path, method="expert_only"))
            assert by_method.policy.train_expert_only is True


class TestAComponentThatReachesNoFieldIsRefused:
    """A request that cannot land is an error, never a silently dropped knob."""

    @pytest.mark.parametrize(
        ("policy_type", "tune", "expected"),
        [
            ("act", {"llm": True}, "tune['llm'] is not supported by policy_type 'act'"),
            ("groot", {"expert_only": True}, "tune['expert_only'] is not supported by policy_type 'groot'"),
            ("groot", {"typo": True}, "tune['typo'] is not a component this trainer maps"),
            ("groot", {"llm": "yes"}, "tune['llm']"),
        ],
        ids=("no_such_component", "component_of_another_family", "unmapped_name", "not_a_bool"),
    )
    def test_validate_reports_it_and_build_config_raises(
        self, policy_type: str, tune: dict[str, Any], expected: str, dataset_root: str, tmp_path: Path
    ) -> None:
        trainer = LerobotTrainer(policy_type=policy_type, device="cpu")
        spec = _spec(dataset_root, tmp_path, tune=tune)
        problems = trainer.validate(spec)
        assert any(expected in p for p in problems), problems
        with pytest.raises(ValueError, match="tune"):
            trainer.build_config(spec)

    def test_a_reward_model_run_has_no_components_to_tune(self, dataset_root: str, tmp_path: Path) -> None:
        """The policy-only knobs are refused on the reward-model path, like their siblings."""
        spec = _spec(
            dataset_root,
            tmp_path,
            tune={"llm": True},
            extra={"reward_model": {"type": "sarm"}},
        )
        problems = LerobotTrainer(device="cpu").validate(spec)
        assert any("names POLICY components" in p for p in problems), problems


class TestTheBaseModelGoesWhereLerobotCanReadIt:
    """A raw GR00T checkpoint is the policy's own ``base_model_path``, not a lerobot one."""

    def test_a_vendor_base_model_becomes_the_policys_source_field(self, dataset_root: str, tmp_path: Path) -> None:
        cfg = _groot().build_config(_spec(dataset_root, tmp_path, base_model="nvidia/GR00T-N1.7-3B"))
        assert cfg.policy.base_model_path == "nvidia/GR00T-N1.7-3B"
        assert cfg.policy.pretrained_path is None

    def test_a_saved_lerobot_checkpoint_still_loads_its_own_config(self, dataset_root: str, tmp_path: Path) -> None:
        """The warm-start path is unchanged for a directory whose config.json names a type."""
        from lerobot.policies.factory import make_policy_config

        checkpoint = tmp_path / "checkpoint"
        saved = make_policy_config("groot")
        saved.chunk_size = 24
        saved.n_action_steps = 24
        saved.save_pretrained(checkpoint)

        cfg = _groot().build_config(_spec(dataset_root, tmp_path, base_model=str(checkpoint)))
        assert cfg.policy.pretrained_path == checkpoint
        assert cfg.policy.chunk_size == 24, "the checkpoint's own architecture must survive the warm start"

    def test_the_documented_cli_sets_what_the_config_sets(self, dataset_root: str, tmp_path: Path) -> None:
        """``build_command`` is the argv the typed config corresponds to."""
        spec = _spec(
            dataset_root,
            tmp_path,
            base_model="nvidia/GR00T-N1.7-3B",
            embodiment="my_robot",
            tune={"llm": True, "projector": False},
        )
        cmd = _groot().build_command(spec)
        assert "--policy.base_model_path=nvidia/GR00T-N1.7-3B" in cmd
        assert "--policy.embodiment_tag=my_robot" in cmd
        assert "--policy.tune_llm=true" in cmd
        assert "--policy.tune_projector=false" in cmd
