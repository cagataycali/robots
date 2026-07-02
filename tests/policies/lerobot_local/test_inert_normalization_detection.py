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
shape is flagged, the canonical ``action`` / ``observation.state`` shape is not.
"""

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
