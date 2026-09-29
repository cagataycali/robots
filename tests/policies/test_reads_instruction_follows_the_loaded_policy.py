"""``reads_instruction`` follows the policy that actually runs, not the provider class.

``run_policy`` reported ``instruction_read=True`` for a lerobot_local ACT
checkpoint and for ``wbc`` (#4159), because the flag was a class attribute the
provider could not change once it knew what it had loaded. ACT has no language
input; the whole-body controller reads joint state, the IMU and a velocity
command. docs/learn/policies/index.md promises that a policy that never reads
the words says so in every report.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strands_robots.policies.base import instruction_not_read_notice
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.mock import MockPolicy
from strands_robots.policies.wbc.policy import WBCPolicy


def _loaded(policy: LerobotLocalPolicy, *, config: object, input_features: dict[str, object]) -> LerobotLocalPolicy:
    policy._policy = SimpleNamespace(config=config)  # type: ignore[assignment]
    policy._input_features = input_features
    policy._loaded = True
    return policy


class TestLerobotLocalAnswersFromTheLoadedCheckpoint:
    def test_before_the_load_the_class_default_holds(self) -> None:
        assert LerobotLocalPolicy.reads_instruction is not False  # the class cannot know yet
        assert LerobotLocalPolicy().reads_instruction is True
        assert instruction_not_read_notice(LerobotLocalPolicy()) is None

    def test_an_act_shaped_checkpoint_does_not_read_the_instruction(self) -> None:
        act = _loaded(
            LerobotLocalPolicy(policy_type="act"),
            config=SimpleNamespace(tokenizer_name=None, vlm_model_name=None),
            input_features={"observation.state": object(), "observation.images.front": object()},
        )
        assert act.reads_instruction is False
        notice = instruction_not_read_notice(act)
        assert notice is not None and notice.startswith("Note: LerobotLocalPolicy does not read the instruction.")
        assert "act checkpoint" in notice

    def test_a_language_conditioned_checkpoint_reads_it(self) -> None:
        smolvla = _loaded(
            LerobotLocalPolicy(policy_type="smolvla"),
            config=SimpleNamespace(tokenizer_name="HuggingFaceTB/SmolVLM2-500M", vlm_model_name=None),
            input_features={"observation.state": object()},
        )
        assert smolvla.reads_instruction is True
        assert instruction_not_read_notice(smolvla) is None

    def test_a_language_feature_alone_is_enough(self) -> None:
        pi0 = _loaded(
            LerobotLocalPolicy(policy_type="pi0"),
            config=SimpleNamespace(tokenizer_name=None, vlm_model_name=None),
            input_features={"observation.language.tokens": object()},
        )
        assert pi0.reads_instruction is True


class TestWbcDeclaresItself:
    def test_the_class_and_its_instances_say_no(self) -> None:
        assert WBCPolicy.reads_instruction is False
        notice = instruction_not_read_notice(WBCPolicy)
        assert notice is not None and "WBCPolicy does not read the instruction" in notice
        assert "velocity command" in notice


class TestTheContractStaysOverridable:
    def test_a_class_level_false_still_reads_as_before(self) -> None:
        assert MockPolicy.reads_instruction is False and MockPolicy().reads_instruction is False

    @pytest.mark.parametrize("attr", ["reads_instruction", "instruction_free_actions"])
    def test_the_two_attributes_are_plain_class_defaults_on_the_base(self, attr: str) -> None:
        from typing import get_type_hints

        from strands_robots.policies.base import Policy

        # Not ClassVar: a provider may override either with a property once it knows what it loaded.
        assert "ClassVar" not in str(get_type_hints(Policy, include_extras=True)[attr])
