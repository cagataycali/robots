"""Load telemetry for LerobotLocalPolicy: cache-hit + load-time observability.

The process-level model cache (see ``_MODEL_CACHE``) already skips the heavy
``from_pretrained`` weight read on repeat instantiation of the same checkpoint.
These tests pin that the saving is *observable*: every policy exposes
``load_time_s`` and ``load_cache_hit`` so a caller (or an LLM harness) can tell
a cold load apart from a cache hit and self-correct a per-episode reload, and
``list_cached_models()`` reports what is resident without poking the private
cache dict.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from strands_robots.policies.lerobot_local import (
    clear_model_cache,
    list_cached_models,
)
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _generic_inner():
    inner = MagicMock()
    inner.config = MagicMock(
        input_features={"observation.state": MagicMock(shape=(6,))},
        output_features={"action": MagicMock(shape=(6,))},
        device="cpu",
    )
    inner.eval.return_value = None
    return inner


def _load_generic(**kwargs):
    """Instantiate a generic lerobot_local policy with the weight load stubbed."""
    mock_cls = MagicMock()
    mock_cls.from_pretrained.side_effect = lambda _path: _generic_inner()
    with (
        patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=mock_cls,
        ),
        patch(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            return_value=MagicMock(is_active=False),
        ),
    ):
        return LerobotLocalPolicy(
            pretrained_name_or_path="test/model",
            policy_type="act",
            device="cpu",
            **kwargs,
        )


class TestLoadTelemetry:
    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    def test_cold_load_reports_miss(self):
        p = _load_generic()
        # First load actually read weights -> not a cache hit, time recorded.
        assert p.load_cache_hit is False
        assert isinstance(p.load_time_s, float)
        assert p.load_time_s >= 0.0

    def test_second_instance_reports_cache_hit(self):
        _load_generic()
        p2 = _load_generic()
        # Second instance of the same checkpoint skipped from_pretrained.
        assert p2.load_cache_hit is True

    def test_cache_model_false_never_reports_hit(self):
        _load_generic()
        p2 = _load_generic(cache_model=False)
        # Opt-out instance always loads privately, so it is never a hit.
        assert p2.load_cache_hit is False

    def test_telemetry_defaults_before_load(self):
        # A parameterless policy never loads on construction; telemetry stays
        # at its honest defaults rather than raising AttributeError.
        p = LerobotLocalPolicy()
        assert p.load_cache_hit is False
        assert p.load_time_s == 0.0


class TestListCachedModels:
    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    def test_empty_cache_lists_nothing(self):
        assert list_cached_models() == []

    def test_reports_resident_entry_fields(self):
        _load_generic()
        cached = list_cached_models()
        assert len(cached) == 1
        entry = cached[0]
        assert entry["namespace"] == "generic"
        assert entry["pretrained_name_or_path"] == "test/model"
        assert entry["device"] == "cpu"
        assert "policy_class" in entry

    def test_clear_empties_the_listing(self):
        _load_generic()
        assert list_cached_models()
        clear_model_cache()
        assert list_cached_models() == []
