"""pi0 / pi0.5 checkpoints keep their own processor pipeline through lerobot_local.

Measured on lerobot 0.6.1 with pi05_base / pi05_droid and SO-101 / panda:

* ``dim_policy="pad"`` padded the state but not the action, so every 32-D pi
  checkpoint (``max_action_dim`` padding; the arm's joints are the first N)
  was refused by every embodiment: "6 action_keys but model action dim is 32".
* A declared embodiment that failed validation DISCARDED the whole pipeline
  with only a warning: the run went on without normalization, and pi0.5's
  ``Task: ..., State: <256-bin tokens>;`` prompt became the bare instruction.
* That raw fallback then refused the partial camera set pi models are built
  for (pi05_droid declares three views; DROID has two).
* ``run_policy`` told the agent a pi0.5 "does not read the instruction", because
  its tokenizer lives in the pipeline, not in its config.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

_SO101 = ["1", "2", "3", "4", "5", "6"]


def _feature(*shape: int) -> SimpleNamespace:
    return SimpleNamespace(shape=shape)


def _policy(policy_type: str = "pi05", **kwargs) -> LerobotLocalPolicy:
    with patch.object(LerobotLocalPolicy, "_load_model"):
        pol = LerobotLocalPolicy(pretrained_name_or_path="lerobot/pi05_base", **kwargs)
    pol.policy_type = policy_type
    pol._device = torch.device("cpu")
    pol._input_features = {
        "observation.state": _feature(32),
        "observation.images.base_0_rgb": _feature(3, 224, 224),
        "observation.images.left_wrist_0_rgb": _feature(3, 224, 224),
        "observation.images.right_wrist_0_rgb": _feature(3, 224, 224),
    }
    pol._output_features = {"action": _feature(32)}
    return pol


class TestAPaddedActionMapsItsFirstJoints:
    def test_a_32_wide_model_validates_against_a_padded_six_joint_embodiment(self) -> None:
        emb = EmbodimentMap(name="so101", state_keys=_SO101, action_keys=_SO101, dim_policy="pad")
        emb.validate({"observation.state": _feature(32)}, {"action": _feature(32)})

    @pytest.mark.parametrize(("policy", "adim"), [("strict", 32), ("pad", 4), ("truncate", 4)])
    def test_strict_and_narrower_models_are_still_refused(self, policy: str, adim: int) -> None:
        emb = EmbodimentMap(name="so101", state_keys=[], action_keys=_SO101, dim_policy=policy)
        with pytest.raises(ValueError, match="action dim"):
            emb.validate({}, {"action": _feature(adim)})

    def test_the_robot_gets_the_first_six_values(self) -> None:
        pol = _policy()
        pol.robot_state_keys = list(_SO101)
        chunk = torch.arange(32, dtype=torch.float32).reshape(1, 32)
        [action] = pol._tensor_to_action_dicts(chunk)
        assert action == {k: float(i) for i, k in enumerate(_SO101)}


class TestAnEmbodimentThatCannotBeConfiguredIsRefused:
    def test_the_pipeline_is_not_discarded_for_the_raw_flow(self, monkeypatch) -> None:
        bridge = MagicMock(is_active=True, has_postprocessor=True, has_preprocessor=True)
        bridge.inert_normalization_features.return_value = []
        bridge.mismatched_normalization_widths.return_value = []
        monkeypatch.setattr(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            classmethod(lambda cls, *a, **k: bridge),
        )
        # so101's LIBERO-style camera renames target features pi05_base does not declare.
        pol = _policy(embodiment="so101")
        with pytest.raises(ValueError, match=r"cannot be configured.*refused rather than run without the pipeline"):
            pol._load_processor_bridge()


class TestAPartialCameraSetIsThePiModelsToHandle:
    def _obs(self) -> dict:
        return {
            "observation.state": torch.zeros(32),
            "observation.images.base_0_rgb": torch.zeros(3, 224, 224),
            "observation.images.left_wrist_0_rgb": torch.zeros(3, 224, 224),
        }

    def test_two_of_three_declared_views_run_on_pi05(self) -> None:
        batch = _policy("pi05")._build_observation_batch(self._obs(), "")
        assert "observation.images.right_wrist_0_rgb" not in batch

    def test_a_type_that_needs_every_view_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="right_wrist_0_rgb"):
            _policy("act")._build_observation_batch(self._obs(), "")

    def test_no_view_at_all_is_refused_even_for_pi05(self) -> None:
        with pytest.raises(ValueError, match="Missing required image feature"):
            _policy("pi05")._build_observation_batch({"observation.state": torch.zeros(32)}, "")


class TestAPipelineTokenizerMeansTheInstructionIsRead:
    def test_an_active_pipeline_with_a_tokenizer_step_reads_the_instruction(self, monkeypatch) -> None:
        step = SimpleNamespace(input_tokenizer=MagicMock(), tokenizer_name="google/paligemma-3b-pt-224")
        bridge = MagicMock(is_active=True, has_postprocessor=True, has_preprocessor=True, preprocessor_steps=[step])
        bridge.inert_normalization_features.return_value = []
        bridge.mismatched_normalization_widths.return_value = []
        monkeypatch.setattr(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            classmethod(lambda cls, *a, **k: bridge),
        )
        pol = _policy()
        pol._policy = SimpleNamespace(config=SimpleNamespace())  # pi05's config names no tokenizer
        assert pol._needs_language_tokens() is False
        pol._load_processor_bridge()
        assert pol._needs_language_tokens() is True

    def test_a_pipeline_without_one_changes_nothing(self) -> None:
        assert (
            LerobotLocalPolicy._pipeline_has_tokenizer(SimpleNamespace(preprocessor_steps=[SimpleNamespace()])) is False
        )
        assert LerobotLocalPolicy._pipeline_has_tokenizer(None) is False
