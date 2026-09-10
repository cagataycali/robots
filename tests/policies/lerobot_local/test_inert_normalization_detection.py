"""Detection of a present-but-inert normalization pipeline.

A LeRobot ``NormalizerProcessorStep`` silently returns a tensor unchanged when
the looked-up stats key is absent (``normalize_processor.py``:
``if norm_mode == IDENTITY or key not in self._tensor_stats: return tensor``).
Pretraining *base* checkpoints such as ``lerobot/smolvla_base`` ship stats keyed
by the training dataset (``so100.buffer.action``) with no ``observation.state``
stats and no bare ``action`` key, so a present, active pipeline normalizes
NOTHING while ``has_postprocessor`` stays ``True`` -- the load-time
missing-postprocessor guard never fires and the passthrough is silent.

These tests pin :meth:`ProcessorBridge.inert_normalization_features` against
REAL LeRobot processor steps (no network, no model download): the dataset-keyed
shape is flagged, the canonical ``action`` / ``observation.state`` shape is not,
and the detector stays robust to realistic pipeline variety -- non-normalizer
steps, an unnormalizer's non-action features, features whose type is
unresolved, and lerobot import drift.
"""

import sys
import types

import pytest

pytest.importorskip("lerobot.processor.pipeline")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.processor.normalize_processor import (  # noqa: E402
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: E402

from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402

_FEATS = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
}
_NORM_MAP = {
    FeatureType.VISUAL: NormalizationMode.IDENTITY,
    FeatureType.STATE: NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}
_MS = {"mean": [0.0] * 6, "std": [1.0] * 6}


def _bridge(pre_stats: dict, post_stats: dict) -> ProcessorBridge:
    """Build a real (network-free) bridge with the given normalizer stats."""
    norm = NormalizerProcessorStep(features=dict(_FEATS), norm_map=dict(_NORM_MAP), stats=pre_stats)
    unnorm = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        norm_map=dict(_NORM_MAP),
        stats=post_stats,
    )
    return ProcessorBridge(
        preprocessor=DataProcessorPipeline(steps=[norm]),
        postprocessor=DataProcessorPipeline(steps=[unnorm]),
        device="cpu",
    )


def test_dataset_prefixed_stats_are_flagged_inert():
    """smolvla_base shape: stats keyed 'so100.buffer.action' -> state+action inert."""
    dataset_keyed = {"so100.buffer.action": dict(_MS)}
    bridge = _bridge(dataset_keyed, dataset_keyed)
    inert = bridge.inert_normalization_features()
    # observation.state (declared MEAN_STD, no matching stats) is passed through.
    assert any(item.startswith("observation.state") for item in inert), inert
    # action unnormalization (declared MEAN_STD, no bare 'action' stats) too.
    assert any(item.startswith("action") for item in inert), inert
    # The IDENTITY visual feature is never flagged.
    assert not any(item.startswith("observation.image") for item in inert), inert


def test_canonical_stats_are_not_flagged():
    """A properly-keyed (fine-tuned) checkpoint normalizes everything -> no warning."""
    canonical_pre = {"observation.state": dict(_MS), "action": dict(_MS)}
    canonical_post = {"action": dict(_MS)}
    bridge = _bridge(canonical_pre, canonical_post)
    assert bridge.inert_normalization_features() == []


def test_no_pipelines_returns_empty():
    """A bridge with no pipelines has nothing to flag."""
    assert ProcessorBridge().inert_normalization_features() == []


def test_non_normalizer_steps_are_ignored():
    """A realistic pipeline mixes normalizer and non-normalizer steps.

    Only Normalizer/Unnormalizer steps carry normalization stats; every other
    step kind (rename, tokenizer, device transfer, ...) has no ``norm_map`` to
    honour and must be skipped without affecting the verdict. Here a rename step
    precedes a normalizer whose ``observation.state`` stats are absent: the
    detector must still flag exactly that one inert feature and ignore the
    rename step entirely.
    """
    from lerobot.processor.rename_processor import RenameObservationsProcessorStep

    norm = NormalizerProcessorStep(
        features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={},  # no matching stats -> observation.state is inert
    )
    bridge = ProcessorBridge(
        preprocessor=DataProcessorPipeline(steps=[RenameObservationsProcessorStep(rename_map={}), norm])
    )
    inert = bridge.inert_normalization_features()
    assert inert == ["observation.state (STATE/MEAN_STD)"], inert


def test_unnormalizer_ignores_non_action_features():
    """An UnnormalizerProcessorStep applies only to ACTION features.

    Upstream ``UnnormalizerProcessorStep`` unnormalizes the ACTION output and
    leaves observation features untouched. So even if it *declares* a STATE
    feature with a non-IDENTITY mode and no backing stats, that STATE feature is
    never applied and must NOT be reported as inert -- reporting it would raise
    a false alarm about a passthrough that cannot happen.
    """
    unnorm = UnnormalizerProcessorStep(
        features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": dict(_MS)},  # action is backed; state is not, but state is never applied
    )
    bridge = ProcessorBridge(postprocessor=DataProcessorPipeline(steps=[unnorm]))
    assert bridge.inert_normalization_features() == []


def test_feature_without_resolved_type_is_skipped():
    """A step exposing a feature whose ``type`` is unresolved must not crash.

    ``inert_normalization_features`` reads ``feature.type`` to look up the
    normalization mode. A step that surfaces a feature with ``type is None``
    (e.g. a partially-populated custom step) has no mode to honour and must be
    skipped defensively rather than raising or being falsely flagged.
    """
    fake_norm_cls = type("NormalizerProcessorStep", (), {})
    step = fake_norm_cls()
    step.features = {"weird": types.SimpleNamespace(type=None, shape=(6,))}
    step.norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    step.stats = {}
    bridge = ProcessorBridge(preprocessor=types.SimpleNamespace(steps=[step]))
    assert bridge.inert_normalization_features() == []


def test_import_drift_degrades_to_empty(monkeypatch):
    """If the lerobot type imports are unavailable, degrade to an empty verdict.

    Older / drifted lerobot layouts may not expose ``lerobot.configs.types`` or
    ``lerobot.utils.constants``. The detector guards those imports and returns
    an empty list instead of raising, so a present pipeline never turns an
    optional diagnostic into a hard failure.
    """
    norm = NormalizerProcessorStep(
        features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={},
    )
    bridge = ProcessorBridge(preprocessor=DataProcessorPipeline(steps=[norm]))
    # Sanity: with imports intact the feature IS flagged.
    assert bridge.inert_normalization_features() == ["observation.state (STATE/MEAN_STD)"]
    # Force the in-method import to raise ImportError.
    monkeypatch.setitem(sys.modules, "lerobot.configs.types", None)
    assert bridge.inert_normalization_features() == []


# --- stats present at the WRONG WIDTH: the opposite failure to an absent stat ---
#
# Absent stats make LeRobot return the tensor unchanged (silent). Stats present
# at the wrong width make it reach the arithmetic and raise from the broadcast,
# naming neither feature nor width, on the FIRST inference - after a rollout has
# started. Both widths are known at load, so a mismatch is reported there.

_MS7 = {"mean": [0.0] * 7, "std": [1.0] * 7}


def test_mismatched_stat_width_is_flagged_naming_both_widths():
    bridge = _bridge(
        {"observation.state": dict(_MS7), "action": dict(_MS7)},
        {"action": dict(_MS7)},
    )
    flagged = bridge.mismatched_normalization_widths()
    assert flagged, "7-wide stats on a 6-wide checkpoint were not reported"
    joined = " | ".join(flagged)
    assert "observation.state" in joined and "action" in joined
    assert "6" in joined and "7" in joined, f"neither width named: {joined}"


def test_matching_stat_width_is_not_flagged():
    bridge = _bridge({"observation.state": dict(_MS), "action": dict(_MS)}, {"action": dict(_MS)})
    assert bridge.mismatched_normalization_widths() == []


def test_absent_stats_are_the_inert_case_not_a_width_mismatch():
    """The two detectors report disjoint failures, so neither masks the other."""
    bridge = _bridge({"so100.buffer.action": dict(_MS)}, {"so100.buffer.action": dict(_MS)})
    assert bridge.inert_normalization_features(), "premise: absent stats are the inert case"
    assert bridge.mismatched_normalization_widths() == []


def test_visual_channel_stats_are_not_a_width_mismatch():
    """LeRobot reshapes a flat ``(C,)`` visual stat to ``(C, 1, 1)`` on purpose."""
    feats = {"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))}
    norm = NormalizerProcessorStep(
        features=feats,
        norm_map={FeatureType.VISUAL: NormalizationMode.MEAN_STD},
        stats={"observation.image": {"mean": [0.0] * 3, "std": [1.0] * 3}},
    )
    bridge = ProcessorBridge(preprocessor=DataProcessorPipeline(steps=[norm]), postprocessor=None, device="cpu")
    assert bridge.mismatched_normalization_widths() == []


def test_a_datasets_count_scalar_is_not_a_width_mismatch():
    """``LeRobotDataset.meta.stats`` carries ``count`` as shape ``(1,)`` per
    feature; the normalizer never reads it, so a 6-wide checkpoint fed its own
    dataset's stats must load (the recorded->trained->loaded round trip)."""
    with_count = {**dict(_MS), "count": [1]}
    bridge = _bridge({"observation.state": with_count}, {"action": with_count})
    assert bridge.mismatched_normalization_widths() == []


def test_both_detectors_read_one_scoping_rule():
    """A preprocessor's ``action`` entry is never exercised, so neither flags it."""
    bridge = _bridge({"action": dict(_MS7)}, {"action": dict(_MS)})
    assert bridge.mismatched_normalization_widths() == []
    assert bridge.inert_normalization_features() == ["observation.state (STATE/MEAN_STD)"]


def test_width_check_degrades_to_empty_on_import_drift(monkeypatch):
    bridge = _bridge({"observation.state": dict(_MS7)}, {"action": dict(_MS7)})
    monkeypatch.setitem(sys.modules, "lerobot.utils.constants", None)
    assert bridge.mismatched_normalization_widths() == []
