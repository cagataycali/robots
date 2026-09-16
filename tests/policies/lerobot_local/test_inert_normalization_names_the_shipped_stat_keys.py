"""The inert-normalization diagnostic names the stats the checkpoint DOES ship.

:meth:`ProcessorBridge.inert_normalization_features` reports which declared
normalizations will silently pass through, and the load-time warning prescribes
``processor_overrides`` to supply the missing stats. It could not say WHERE
those stats come from -- and for a pretraining base checkpoint the answer is
usually the checkpoint already on disk, whose normalizer carries the
pretraining datasets' stats under prefixed keys (``so100.buffer.action``)
rather than the canonical ``action`` / ``observation.state`` LeRobot looks up.
A caller told only that ``action`` is missing has to open
``*_normalizer_processor.safetensors`` by hand to find that out.

These tests pin that the prefixed spellings are reported per missing key, that
a key which merely ends in the canonical word is not mistaken for one, that a
feature with no candidate says so explicitly rather than silently, and that the
names reach the warning a caller actually reads.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("lerobot.processor.pipeline")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.processor.normalize_processor import (  # noqa: E402
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: E402

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy  # noqa: E402
from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402

_FEATS = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
}
_NORM_MAP = {
    FeatureType.STATE: NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}
_MS = {"mean": [0.0] * 6, "std": [1.0] * 6}

# The three prefixes lerobot/smolvla_base really ships, read from
# policy_preprocessor_step_5_normalizer_processor.safetensors.
_SMOLVLA_KEYS = [
    "so100-blue.buffer.action",
    "so100-red.buffer.action",
    "so100.buffer.action",
]


def _bridge(stats: dict) -> ProcessorBridge:
    """A real (network-free) bridge whose normalizer steps carry ``stats``."""
    return ProcessorBridge(
        preprocessor=DataProcessorPipeline(
            steps=[NormalizerProcessorStep(features=dict(_FEATS), norm_map=dict(_NORM_MAP), stats=stats)]
        ),
        postprocessor=DataProcessorPipeline(
            steps=[
                UnnormalizerProcessorStep(
                    features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
                    norm_map=dict(_NORM_MAP),
                    stats=stats,
                )
            ]
        ),
        device="cpu",
    )


def test_the_prefixed_action_keys_are_named_and_a_missing_state_candidate_is_explicit():
    """smolvla_base shape: the three action prefixes are listed; state has none."""
    candidates = _bridge({key: dict(_MS) for key in _SMOLVLA_KEYS}).prefixed_stat_key_candidates()

    # Both inert features appear, so neither is silently omitted ...
    assert set(candidates) == {"observation.state", "action"}, candidates
    # ... the action half is recoverable from the checkpoint itself, sorted so
    # the warning text is deterministic ...
    assert candidates["action"] == _SMOLVLA_KEYS
    # ... and the state half genuinely is not, which is why the stats remedy
    # alone cannot restore proprioception for this checkpoint.
    assert candidates["observation.state"] == []


@pytest.mark.parametrize(
    ("stat_key", "is_candidate"),
    [
        ("so100.buffer.action", True),  # the shipped spelling
        ("dataset/action", True),  # slash-separated
        ("robot_action", True),  # underscore-separated
        ("so100-action", True),  # hyphen-separated
        ("reaction", False),  # ends in the word, no separator
        ("transaction", False),  # likewise
        ("action_left", False),  # canonical key is not the suffix
    ],
)
def test_only_a_separated_suffix_counts_as_a_prefixed_spelling(stat_key, is_candidate):
    """A key must end with the canonical key AFTER a separator to be offered."""
    candidates = _bridge({stat_key: dict(_MS)}).prefixed_stat_key_candidates()
    assert (stat_key in candidates["action"]) is is_candidate, candidates


def test_a_canonically_keyed_checkpoint_offers_nothing_because_nothing_is_inert():
    """A fine-tuned checkpoint normalizes everything, so there is no gap to fill."""
    canonical = {"observation.state": dict(_MS), "action": dict(_MS)}
    assert _bridge(canonical).prefixed_stat_key_candidates() == {}


def test_the_warning_names_the_keys_the_checkpoint_actually_ships():
    """The names reach the message, so no one has to open the safetensors by hand."""
    bridge = MagicMock(name="ProcessorBridge")
    bridge.is_active = True
    bridge.has_postprocessor = True
    bridge.mismatched_normalization_widths.return_value = []
    bridge.inert_normalization_features.return_value = [
        "observation.state (STATE/MEAN_STD)",
        "action (ACTION/MEAN_STD)",
    ]
    bridge.prefixed_stat_key_candidates.return_value = {
        "observation.state": [],
        "action": list(_SMOLVLA_KEYS),
    }

    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="lerobot/smolvla_base")
    policy._device = None

    with (
        patch.object(LerobotLocalPolicy, "_configure_embodiment"),
        patch(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            classmethod(lambda cls, *a, **k: bridge),
        ),
        caplog_at_warning() as records,
    ):
        policy._load_processor_bridge()

    message = next(m for m in records() if "ACTIVE normalization pipeline" in m)
    # Every shipped spelling is named, not just an illustrative one ...
    for key in _SMOLVLA_KEYS:
        assert key in message, message
    # ... and the message says why they are offered rather than adopted, so the
    # caller knows the choice between distributions is theirs to make.
    assert "silent guess" in message, message


class caplog_at_warning:
    """Minimal warning-record capture, so the test needs no fixture plumbing."""

    def __enter__(self):
        self._records: list[str] = []
        self._handler = logging.Handler()
        self._handler.emit = lambda record: self._records.append(record.getMessage())
        logger = logging.getLogger("strands_robots.policies.lerobot_local.policy")
        self._logger = logger
        self._prior = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(self._handler)
        return lambda: list(self._records)

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prior)
        return False
