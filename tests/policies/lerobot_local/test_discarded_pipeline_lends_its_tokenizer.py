"""A discarded pipeline lends its tokenizer to the raw obs/action flow.

``LerobotLocalPolicy._load_processor_bridge`` drops an ACTIVE pipeline when the
caller's embodiment cannot be configured against the model's declared features
and falls back to the raw obs/action flow. For the PaliGemma-based policies
(pi0, pi05) the pipeline's ``TokenizerProcessorStep`` is the ONLY owner of the
tokenizer: their config names neither ``tokenizer_name`` nor
``vlm_model_name`` and declares no language input feature, so the raw flow
found no tokenizer, ``_needs_language_tokens`` answered False, and
``predict_action_chunk`` raised ``KeyError: 'observation.language.tokens'`` at
the first inference (reproduced on lerobot/pi0_base with a 6-key embodiment).
The fallback now keeps the step's tokenizer, length and padding side.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _feature(dim: int) -> MagicMock:
    feat = MagicMock()
    feat.shape = (dim,)
    return feat


def _make_policy(**kwargs) -> LerobotLocalPolicy:
    with patch.object(LerobotLocalPolicy, "_load_model"):
        pol = LerobotLocalPolicy(pretrained_name_or_path="fake/pi0", **kwargs)
    pol._device = None
    pol._loaded = True
    # A pi0-shaped config: no tokenizer_name, no vlm_model_name, no processor.
    pol._policy = SimpleNamespace(config=SimpleNamespace(tokenizer_max_length=48))
    pol._input_features = {"observation.images.base_0_rgb": _feature(3), "observation.state": _feature(32)}
    pol._output_features = {"action": _feature(32)}
    return pol


def _tokenizer_step(tokenizer) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_name="google/paligemma-3b-pt-224",
        input_tokenizer=tokenizer,
        tokenizer=tokenizer,
        max_length=200,
        padding_side="left",
    )


class _FakeBridge:
    """The real bridge's shape: ``preprocessor_steps`` is a PROPERTY returning the step list.

    A ``MagicMock`` with ``preprocessor_steps.return_value`` passed the first
    version of this test while the real property never got called: the fix read
    it as a method. The fake mirrors the class, not the mock.
    """

    is_active = True
    has_postprocessor = True

    def __init__(self, steps: list) -> None:
        self._steps = steps

    @property
    def preprocessor_steps(self) -> list:
        return list(self._steps)

    def inert_normalization_features(self) -> list:
        return []

    def mismatched_normalization_widths(self) -> list:
        return []

    def apply_embodiment(self, *args, **kwargs) -> None:
        return None


def _bridge_with_steps(steps: list) -> _FakeBridge:
    return _FakeBridge(steps)


def _six_key_embodiment() -> EmbodimentMap:
    # 6 action_keys against a 32-wide action head -> EmbodimentMap.validate refuses.
    return EmbodimentMap(name="so101_six", obs_rename={}, state_keys=[], action_keys=list("abcdef"), dim_policy="pad")


def _patch_from_pretrained(monkeypatch, bridge) -> None:
    monkeypatch.setattr(
        "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
        classmethod(lambda cls, *a, **k: bridge),
    )


class TestTheFallbackKeepsTheTokenizer:
    def test_pre_fix_shape_the_raw_flow_had_no_tokenizer(self, monkeypatch):
        """Without a pipeline the pi0-shaped config yields no tokenizer and no language need."""
        pol = _make_policy(embodiment=None)
        pol._processor_bridge = None
        assert pol._resolve_tokenizer() is None
        assert pol._needs_language_tokens() is False

    def test_discarding_the_pipeline_adopts_its_tokenizer_step(self, monkeypatch, caplog):
        tokenizer = MagicMock(name="paligemma_tokenizer")
        bridge = _bridge_with_steps([SimpleNamespace(name="rename"), _tokenizer_step(tokenizer)])
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(embodiment=_six_key_embodiment())

        with caplog.at_level(logging.INFO):
            pol._load_processor_bridge()

        assert pol._processor_bridge is None and pol._embodiment_config_failed is True
        assert pol._resolve_tokenizer() is tokenizer
        assert pol._tokenizer_max_length == 200
        assert pol._tokenizer_padding_side == "left"
        assert pol._needs_language_tokens() is True
        assert any("kept the pipeline's tokenizer" in r.getMessage() for r in caplog.records)

    def test_a_pipeline_without_a_tokenizer_step_changes_nothing(self, monkeypatch):
        bridge = _bridge_with_steps([SimpleNamespace(name="rename"), SimpleNamespace(name="normalizer", stats={})])
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(embodiment=_six_key_embodiment())

        pol._load_processor_bridge()

        assert pol._processor_bridge is None
        assert pol._tokenizer is None
        assert pol._needs_language_tokens() is False
        assert pol._tokenizer_max_length == 48


class TestTheAdoptedTokenizerRunsOnTheDefaultInstruction:
    """``run_policy`` defaults to ``instruction=""``; pi0 / pi05 still need language tokens."""

    def _adopted(self, monkeypatch) -> tuple[LerobotLocalPolicy, MagicMock]:
        tokenizer = MagicMock(name="paligemma_tokenizer")
        tokenizer.return_value = {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4)),
        }
        bridge = _bridge_with_steps([_tokenizer_step(tokenizer)])
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(embodiment=_six_key_embodiment())
        pol._load_processor_bridge()
        assert pol._pipeline_tokenized is True
        return pol, tokenizer

    def test_an_empty_instruction_is_still_tokenized(self, monkeypatch):
        pol, tokenizer = self._adopted(monkeypatch)
        assert pol._tokenize_instruction("") is not None
        assert tokenizer.call_args.args == ("",)

    def test_the_batch_carries_language_tokens_for_the_default_instruction(self, monkeypatch):
        pol, _ = self._adopted(monkeypatch)
        image = torch.zeros((1, 3, 8, 8))
        passthrough = lambda obs, batch: {**batch, "observation.images.base_0_rgb": image}  # noqa: E731
        monkeypatch.setattr(pol, "_build_batch_from_strands_format", passthrough)
        monkeypatch.setattr(pol, "_build_batch_from_lerobot_format", passthrough)
        batch = pol._build_observation_batch({"observation.state": torch.zeros(6)}, "")
        assert batch["observation.language.tokens"].shape == (1, 4)
        assert batch["observation.language.attention_mask"].dtype == torch.bool

    def test_without_an_adopted_tokenizer_an_empty_instruction_adds_nothing(self):
        pol = _make_policy(embodiment=None)
        pol._processor_bridge = None
        assert pol._tokenize_instruction("") is None


class TestStateWidthAdaptationWarnsOnce:
    def test_the_raw_path_logs_the_zero_padding_once_per_policy(self, caplog):
        pol = _make_policy(embodiment=None)
        pol._input_features = {"observation.state": _feature(32)}  # a state-only model: no cameras to route
        pol.set_robot_state_keys(list("abcdef"))
        obs = {k: float(i) for i, k in enumerate("abcdef")}
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.policy"):
            for _ in range(5):
                out = pol._to_lerobot_observation(dict(obs))
        assert out["observation.state"].shape == (32,)
        lines = [r.getMessage() for r in caplog.records if "zero-padding" in r.getMessage()]
        assert len(lines) == 1, lines


class TestTheHeuristicPathHandsThePipelineATensor:
    def test_observation_state_is_a_float32_tensor(self):
        """pi05's ``Pi05PrepareStateTokenizerProcessorStep`` calls ``state.cpu()``.

        lerobot's inference helper and the declarative ``strands_pack_state``
        step both hand the pipeline a tensor; the heuristic remap built an
        ndarray, so ``lerobot/pi05_base`` routed by ``camera_key_map`` died with
        ``AttributeError: 'numpy.ndarray' object has no attribute 'cpu'`` inside
        the preprocessor at the first inference.
        """
        import torch

        pol = _make_policy(embodiment=None)
        pol._input_features = {"observation.state": _feature(6)}
        pol.set_robot_state_keys(list("abcdef"))
        out = pol._to_lerobot_observation({k: float(i) for i, k in enumerate("abcdef")})
        state = out["observation.state"]
        assert isinstance(state, torch.Tensor) and state.dtype == torch.float32
        assert state.cpu().numpy().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
