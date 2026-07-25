"""Revision + norm_tag threading through the lerobot_local processor bridge.

``LerobotLocalPolicy`` accepts ``revision=`` to pin a checkpoint to a branch,
tag, or commit SHA and ``norm_tag=`` to select which embodiment's stats a
multi-tag ``norm_stats.json`` applies. Both must reach the processor pipeline
loader, not just the policy weights: otherwise a revision-pinned load silently
runs the DEFAULT-branch preprocessor/postprocessor JSONs and normalization
buffers against pinned weights (worst case: wrong normalization stats), and a
generic checkpoint ignores the user's ``norm_tag`` on the norm-stats fallback.

These tests pin:
  * the ``_load_processor_bridge`` call site forwards ``revision`` + ``norm_tag``
    to ``ProcessorBridge.from_pretrained``;
  * ``ProcessorBridge.from_pretrained`` threads ``revision`` into every
    ``DataProcessorPipeline.from_pretrained`` call (preprocessor + postprocessor);
  * an older lerobot pipeline loader whose ``from_pretrained`` predates the
    ``revision`` kwarg degrades to an unpinned load instead of crashing;
  * ``revision`` + ``norm_tag`` both reach the ``norm_stats.json`` fallback; and
  * a user ``norm_tag`` selects that tag from a multi-tag payload.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from strands_robots.policies.lerobot_local import norm_stats, processor
from strands_robots.policies.lerobot_local.norm_stats import (
    MOLMOACT2_NORM_STATS_FORMAT,
    select_norm_tag,
)
from strands_robots.policies.lerobot_local.policy import (
    LerobotLocalPolicy,
    clear_model_cache,
)
from strands_robots.policies.lerobot_local.processor import (
    POSTPROCESSOR_CONFIG,
    PREPROCESSOR_CONFIG,
    ProcessorBridge,
)


def _generic_inner():
    inner = MagicMock()
    inner.config = MagicMock(
        input_features={"observation.state": MagicMock(shape=(6,))},
        output_features={"action": MagicMock(shape=(6,))},
        device="cpu",
    )
    inner.eval.return_value = None
    return inner


class TestBridgeCallSiteForwarding:
    """The policy call site must forward revision + norm_tag to the bridge."""

    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    def test_load_processor_bridge_forwards_revision_and_norm_tag(self):
        captured: dict = {}

        def _fake_bridge(_path, **kwargs):
            captured.update(kwargs)
            return MagicMock(is_active=False)

        mock_cls = MagicMock()
        mock_cls.from_pretrained.side_effect = lambda _p, **_kw: _generic_inner()
        with (
            patch(
                "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
                return_value=mock_cls,
            ),
            patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                side_effect=_fake_bridge,
            ),
        ):
            LerobotLocalPolicy(
                pretrained_name_or_path="test/model",
                policy_type="act",
                device="cpu",
                cache_model=False,
                revision="v1.2.3",
                norm_tag="so101",
            )
        assert captured.get("revision") == "v1.2.3"
        assert captured.get("norm_tag") == "so101"


class _RecordingPipeline:
    """Stub pipeline loader that records each from_pretrained call."""

    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, path, *, config_filename, overrides, revision=None, **_kw):
        cls.calls.append({"config_filename": config_filename, "revision": revision})
        return [object()]  # len()==1 so the load-diagnostic logger works


class _OldPipeline:
    """Stub loader whose from_pretrained predates the ``revision`` kwarg."""

    seen: list[str] = []

    @classmethod
    def from_pretrained(cls, path, *, config_filename, overrides, **kwargs):
        if "revision" in kwargs:
            raise TypeError("from_pretrained() got an unexpected keyword argument 'revision'")
        cls.seen.append(config_filename)
        return [object()]


class TestBridgeThreadsRevisionToPipeline:
    def test_revision_reaches_every_pipeline_loader_call(self, monkeypatch):
        _RecordingPipeline.calls = []
        monkeypatch.setattr(processor, "_try_import_processor", lambda: _RecordingPipeline)
        monkeypatch.setattr(processor, "_register_policy_processor_steps", lambda *_a, **_k: None)

        ProcessorBridge.from_pretrained("owner/model", revision="deadbeef", policy_type=None)

        by_config = {c["config_filename"]: c["revision"] for c in _RecordingPipeline.calls}
        assert by_config[PREPROCESSOR_CONFIG] == "deadbeef"
        assert by_config[POSTPROCESSOR_CONFIG] == "deadbeef"

    def test_no_revision_keeps_unpinned_pipeline_load(self, monkeypatch):
        _RecordingPipeline.calls = []
        monkeypatch.setattr(processor, "_try_import_processor", lambda: _RecordingPipeline)
        monkeypatch.setattr(processor, "_register_policy_processor_steps", lambda *_a, **_k: None)

        ProcessorBridge.from_pretrained("owner/model", policy_type=None)

        assert _RecordingPipeline.calls  # loader was invoked
        assert all(c["revision"] is None for c in _RecordingPipeline.calls)

    def test_old_pipeline_loader_degrades_to_unpinned(self, monkeypatch, caplog):
        _OldPipeline.seen = []
        monkeypatch.setattr(processor, "_try_import_processor", lambda: _OldPipeline)
        monkeypatch.setattr(processor, "_register_policy_processor_steps", lambda *_a, **_k: None)

        with caplog.at_level("WARNING"):
            bridge = ProcessorBridge.from_pretrained("owner/model", revision="v1", policy_type=None)

        # Both pipelines loaded (unpinned retry) instead of crashing on TypeError.
        assert PREPROCESSOR_CONFIG in _OldPipeline.seen
        assert POSTPROCESSOR_CONFIG in _OldPipeline.seen
        assert bridge.is_active
        assert any("does not accept revision" in r.message for r in caplog.records)


def _multi_tag_payload():
    stats = {
        "min": [0.0] * 6,
        "max": [1.0] * 6,
    }
    return {
        "format": MOLMOACT2_NORM_STATS_FORMAT,
        "norm_mode": "min_max",
        "metadata_by_tag": {
            "so100": {"state_stats": stats, "action_stats": stats},
            "so101": {"state_stats": stats, "action_stats": stats},
        },
    }


class TestNormStatsFallbackReceivesRevisionAndTag:
    def test_revision_and_norm_tag_reach_norm_stats_fallback(self, monkeypatch):
        # No processor JSONs -> both pipelines None -> norm_stats fallback runs.
        class _NoConfigPipeline:
            @classmethod
            def from_pretrained(cls, *_a, **_k):
                raise FileNotFoundError("no processor config shipped")

        monkeypatch.setattr(processor, "_try_import_processor", lambda: _NoConfigPipeline)
        monkeypatch.setattr(processor, "_register_policy_processor_steps", lambda *_a, **_k: None)

        captured: dict = {}

        def _fake_load(path, *, revision=None, **_kw):
            captured["load_revision"] = revision
            return _multi_tag_payload()

        def _fake_build(payload, norm_tag=None, **_kw):
            captured["build_norm_tag"] = norm_tag
            return ("PRE", "POST")

        monkeypatch.setattr(norm_stats, "load_norm_stats", _fake_load)
        monkeypatch.setattr(norm_stats, "build_norm_stats_processors", _fake_build)

        bridge = ProcessorBridge.from_pretrained("owner/model", revision="rev9", norm_tag="so101", policy_type=None)
        assert captured["load_revision"] == "rev9"
        assert captured["build_norm_tag"] == "so101"
        assert bridge.has_preprocessor

    def test_requested_norm_tag_selected_from_multi_tag_payload(self):
        # Acceptance criterion: an explicit norm_tag wins over auto-resolution.
        assert select_norm_tag(_multi_tag_payload(), "so101") == "so101"
        # Unresolvable without a hint (multiple tags, no matching default).
        assert select_norm_tag(_multi_tag_payload()) is None
