"""Revision threading through the lerobot_local processor bridge.

``LerobotLocalPolicy`` accepts ``revision=`` to pin a checkpoint to a branch,
tag, or commit SHA. It must reach the processor pipeline loader, not just the
policy weights: otherwise a revision-pinned load silently runs the
DEFAULT-branch preprocessor/postprocessor JSONs against pinned weights, whose
worst case is wrong normalization stats.

These tests pin:
  * the ``_load_processor_bridge`` call site forwards ``revision`` to
    ``ProcessorBridge.from_pretrained``;
  * ``ProcessorBridge.from_pretrained`` threads ``revision`` into every
    ``DataProcessorPipeline.from_pretrained`` call (preprocessor + postprocessor);
  * an older lerobot pipeline loader whose ``from_pretrained`` predates the
    ``revision`` kwarg degrades to an unpinned load instead of crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from strands_robots.policies.lerobot_local import processor
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
    """The policy call site must forward revision to the bridge."""

    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    def test_load_processor_bridge_forwards_revision(self):
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
            )
        assert captured.get("revision") == "v1.2.3"


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
