"""Stats a caller supplies, or a pi checkpoint ships, reach the numbers they are meant for.

* ``processor_overrides={"normalizer_processor": {"stats": ...}}`` - the
  documented remedy for missing or inert stats - normalized nothing on
  lerobot/pi05_base and pi05_droid: their ``policy_preprocessor.json`` declares
  ``features: {}``, and a normalizer only touches the features it declares. A
  -2.2 rad joint reached pi0.5's 256-bin state tokenizer as -2.2 (bin -1), and
  the inert-normalization check, which walks the declared features, found
  nothing to report.
* lerobot/pi0fast-libero declares ``observation.state`` at ``max_state_dim``
  (32) and ships 8-wide state stats; the width guard refused the official
  checkpoint with its own stats.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from strands_robots.policies.lerobot_local.processor import ProcessorBridge, complete_stats_overrides

_CONFIG = SimpleNamespace(
    input_features={"observation.state": "STATE-32", "observation.images.base_0_rgb": "VISUAL"},
    output_features={"action": "ACTION-32"},
    normalization_mapping={"STATE": "QUANTILES", "ACTION": "QUANTILES", "VISUAL": "IDENTITY"},
    max_state_dim=32,
)


class TestAStatsOverrideGetsTheFeaturesItNormalizes:
    def test_features_and_norm_map_come_from_the_policy_config(self) -> None:
        stats = {"observation.state": {"q01": [-1.0] * 8}}
        out = complete_stats_overrides(
            {"normalizer_processor": {"stats": stats}, "unnormalizer_processor": {"stats": stats}}, _CONFIG
        )
        assert out["normalizer_processor"]["features"] == {**_CONFIG.input_features, **_CONFIG.output_features}
        assert out["unnormalizer_processor"]["features"] == _CONFIG.output_features
        assert out["normalizer_processor"]["norm_map"] == _CONFIG.normalization_mapping
        assert out["normalizer_processor"]["stats"] is stats

    def test_a_callers_own_features_are_kept(self) -> None:
        mine = {"normalizer_processor": {"stats": {"a": {}}, "features": {"x": 1}, "norm_map": {"STATE": "MIN_MAX"}}}
        assert complete_stats_overrides(mine, _CONFIG) == mine

    @pytest.mark.parametrize(
        "overrides", [{}, {"device_processor": {"device": "cuda"}}, {"normalizer_processor": {"eps": 1e-6}}]
    )
    def test_anything_without_stats_is_untouched(self, overrides: dict[str, dict[str, object]]) -> None:
        assert complete_stats_overrides(overrides, _CONFIG) == overrides

    def test_without_a_policy_config_nothing_is_guessed(self) -> None:
        overrides: dict[str, dict[str, object]] = {"normalizer_processor": {"stats": {"a": {}}}}
        assert complete_stats_overrides(overrides, None) == overrides


def _step(name: str, *, features: dict | None, stats: dict | None = None) -> object:
    """A stand-in step whose class carries the lerobot step's name, as the bridge reads it."""
    cls = type(name, (), {})
    step = cls()
    step.features, step.norm_map, step.stats, step._tensor_stats = features or {}, {}, stats or {}, stats or {}
    return step


class TestANormalizerThatDeclaresNothingIsReportedInert:
    def test_both_empty_steps_are_named(self) -> None:
        bridge = ProcessorBridge(
            preprocessor=SimpleNamespace(steps=[_step("NormalizerProcessorStep", features=None)]),
            postprocessor=SimpleNamespace(steps=[_step("UnnormalizerProcessorStep", features=None)]),
        )
        assert bridge.inert_normalization_features() == [
            "observation (NormalizerProcessorStep declares no features)",
            "action (UnnormalizerProcessorStep declares no features)",
        ]


class TestPaddedStateStatsAreWidenedNeutrally:
    def _bridge(self, config: SimpleNamespace, *, declared_width: int | None = 32) -> ProcessorBridge:
        stats = {"observation.state": {"mean": torch.arange(8.0), "std": torch.ones(8), "q01": -torch.ones(8)}}
        features: dict = {"x": 1}
        if declared_width is not None:
            features["observation.state"] = SimpleNamespace(shape=(declared_width,))
        bridge = ProcessorBridge(
            preprocessor=SimpleNamespace(steps=[_step("NormalizerProcessorStep", features=features, stats=stats)])
        )
        bridge._policy_config = config
        bridge._pad_narrow_state_stats()
        return bridge

    def test_a_fine_tune_declaring_the_robots_width_keeps_its_stats(self) -> None:
        """``PI0Config.max_state_dim`` is always 32, but lerobot-train sets ``input_features`` from the
        dataset: a pi0 fine-tune on a 6-DOF arm declares ``observation.state`` at 6 with 6-wide stats and
        loads cleanly. Widening those to 32 would make the width guard refuse the checkpoint with its own
        stats, so the pad applies only when the declared feature IS the padded width."""
        pre = self._bridge(_CONFIG, declared_width=8)._preprocessor
        assert pre is not None
        assert pre.steps[0]._tensor_stats["observation.state"]["mean"].shape == (8,)

    def test_a_step_that_declares_no_state_feature_is_left_alone(self) -> None:
        pre = self._bridge(_CONFIG, declared_width=None)._preprocessor
        assert pre is not None
        assert pre.steps[0]._tensor_stats["observation.state"]["mean"].shape == (8,)

    def test_the_robots_columns_keep_their_stats_and_the_tail_maps_zero_to_zero(self) -> None:
        pre = self._bridge(_CONFIG)._preprocessor
        assert pre is not None
        stats = pre.steps[0]._tensor_stats["observation.state"]
        assert stats["mean"].shape == (32,) and torch.equal(stats["mean"][:8], torch.arange(8.0))
        assert torch.all(stats["mean"][8:] == 0) and torch.all(stats["std"][8:] == 1)
        assert torch.all(stats["q01"][8:] == -1)

    def test_a_model_without_state_padding_is_left_to_the_width_guard(self) -> None:
        pre = self._bridge(SimpleNamespace(max_state_dim=None))._preprocessor
        assert pre is not None
        stats = pre.steps[0]._tensor_stats
        assert stats["observation.state"]["mean"].shape == (8,)
