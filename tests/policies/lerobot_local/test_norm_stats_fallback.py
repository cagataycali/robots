"""Regression tests for the ``norm_stats.json`` processor fallback.

Covers :mod:`strands_robots.policies.lerobot_local.norm_stats` and its wiring
into :class:`~strands_robots.policies.lerobot_local.processor.ProcessorBridge`.

The bug these guard against: a checkpoint that ships only ``norm_stats.json``
(no ``policy_preprocessor.json`` / ``policy_postprocessor.json``, e.g. the
MolmoAct2 SO-100/101 family) used to produce a passthrough bridge -- state
reached the policy un-normalized and predicted actions reached the motors
un-unnormalized, the single biggest cause of off-policy arm motion.

The numeric transform is validated bit-for-bit against the q01_q99 reference
formula used by lerobot's MolmoAct2 normalizer
(``modeling_molmoact2._FeatureNormalizer``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from strands_robots.policies.lerobot_local import norm_stats as ns

FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_norm_stats.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _ref_q01_q99_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Reference q01_q99 normalize (matches modeling_molmoact2._FeatureNormalizer)."""
    normed = 2.0 * (x - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
    return np.clip(normed, -1.0, 1.0)


def _ref_q01_q99_unnormalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    arr = np.clip(x, -1.0, 1.0)
    return (arr + 1.0) * (q99 - q01) / 2.0 + q01


class TestFeatureNormalizer:
    """The ported per-feature normalizer."""

    def test_q01_q99_normalize_matches_reference(self):
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        x = np.array([0.0, 100.0, 90.0, 40.0, -10.0, 20.0], dtype=np.float32)
        got = fn.normalize(x)
        want = _ref_q01_q99_normalize(x, q01, q99)
        assert np.allclose(got, want, atol=1e-4)

    def test_q01_q99_unnormalize_matches_reference(self):
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["action_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        a = np.array([0.1, -0.5, 0.3, 0.9, -1.0, 0.0], dtype=np.float32)
        got = fn.unnormalize(a)
        want = _ref_q01_q99_unnormalize(a, q01, q99)
        assert np.allclose(got, want, atol=1e-4)

    def test_round_trip_within_quantile_range_is_identity(self):
        # Values strictly inside [q01, q99] survive normalize->unnormalize.
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        x = (q01 + q99) / 2.0  # midpoint, well inside range
        round_tripped = fn.unnormalize(fn.normalize(x))
        assert np.allclose(round_tripped, x, atol=1e-3)

    def test_clip_saturates_out_of_range(self):
        stats = {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")
        # 100 >> q99 -> normalized clips to +1.0
        got = fn.normalize(np.array([100.0, -100.0], dtype=np.float32))
        assert np.allclose(got, [1.0, -1.0])

    def test_mean_std_mode(self):
        stats = {"mean": [1.0, 2.0], "std": [2.0, 4.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "mean_std")
        got = fn.normalize(np.array([3.0, 6.0], dtype=np.float32))
        assert np.allclose(got, [1.0, 1.0])
        assert np.allclose(fn.unnormalize(got), [3.0, 6.0])

    def test_min_max_mode(self):
        stats = {"min": [0.0, 0.0], "max": [10.0, 10.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "min_max")
        got = fn.normalize(np.array([5.0, 0.0], dtype=np.float32))
        assert np.allclose(got, [0.0, -1.0])

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported robot normalization mode"):
            ns.FeatureNormalizer.from_stats({"q01": [0.0]}, "bogus_mode")

    def test_missing_required_stats_raises(self):
        with pytest.raises(ValueError, match="requires q01 and q99"):
            ns.FeatureNormalizer.from_stats({"min": [0.0]}, "q01_q99")

    def test_none_input_returns_none(self):
        fn = ns.FeatureNormalizer.from_stats({"q01": [0.0], "q99": [1.0]}, "q01_q99")
        assert fn.normalize(None) is None
        assert fn.unnormalize(None) is None


class TestSchemaDetection:
    """Payload schema detection and tag selection."""

    def test_recognizes_molmoact2_schema(self):
        assert ns.is_norm_stats_payload(_load_fixture()) is True

    def test_rejects_unrelated_json(self):
        assert ns.is_norm_stats_payload({"foo": "bar"}) is False
        assert ns.is_norm_stats_payload({"format": "molmoact2_norm_stats.v1"}) is False
        assert ns.is_norm_stats_payload(None) is False

    def test_select_sole_tag(self):
        assert ns.select_norm_tag(_load_fixture()) == "so100_so101_molmoact2"

    def test_explicit_tag_wins(self):
        payload = {"metadata_by_tag": {"a": {}, "b": {}}}
        assert ns.select_norm_tag(payload, "b") == "b"

    def test_unknown_explicit_tag_returns_none(self):
        payload = {"metadata_by_tag": {"a": {}}}
        assert ns.select_norm_tag(payload, "missing") is None

    def test_default_tag_used_on_ambiguity(self):
        payload = {"metadata_by_tag": {ns.DEFAULT_SO_NORM_TAG: {}, "other": {}}}
        assert ns.select_norm_tag(payload) == ns.DEFAULT_SO_NORM_TAG

    def test_ambiguous_no_default_returns_none(self):
        payload = {"metadata_by_tag": {"x": {}, "y": {}}}
        assert ns.select_norm_tag(payload) is None


class TestLoadNormStats:
    """Loading norm_stats.json from a local checkpoint directory."""

    def test_loads_local_norm_stats(self, tmp_path):
        (tmp_path / "norm_stats.json").write_text(FIXTURE.read_text())
        payload = ns.load_norm_stats(str(tmp_path))
        assert ns.is_norm_stats_payload(payload)

    def test_honors_config_filename_override(self, tmp_path):
        (tmp_path / "custom_stats.json").write_text(FIXTURE.read_text())
        (tmp_path / "config.json").write_text(json.dumps({"norm_stats_filename": "custom_stats.json"}))
        payload = ns.load_norm_stats(str(tmp_path))
        assert ns.is_norm_stats_payload(payload)

    def test_missing_file_returns_none(self, tmp_path):
        assert ns.load_norm_stats(str(tmp_path)) is None

    def test_empty_path_returns_none(self):
        assert ns.load_norm_stats("") is None


# Tests below need LeRobot's processor framework (real pipeline steps).
pytest.importorskip("lerobot.processor.pipeline")


class TestBuildProcessors:
    """build_norm_stats_processors against the real LeRobot pipeline."""

    def test_builds_active_pre_and_post(self):
        pre, post = ns.build_norm_stats_processors(_load_fixture())
        assert pre is not None and post is not None
        assert len(pre) == 1 and len(post) == 1

    def test_preprocessor_normalizes_state_through_pipeline(self):
        from lerobot.processor import TransitionKey
        from lerobot.processor.converters import create_transition

        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)

        pre, _ = ns.build_norm_stats_processors(fix)
        x = np.array([0.0, 100.0, 90.0, 40.0, -10.0, 20.0], dtype=np.float32)
        out = pre._forward(create_transition(observation={"observation.state": x.copy()}))
        normed = out[TransitionKey.OBSERVATION]["observation.state"]
        assert np.allclose(normed, _ref_q01_q99_normalize(x, q01, q99), atol=1e-4)

    def test_postprocessor_unnormalizes_action_through_pipeline(self):
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["action_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)

        _, post = ns.build_norm_stats_processors(fix)
        a = np.array([0.1, -0.5, 0.3, 0.9, -1.0, 0.0], dtype=np.float32)
        out = post.process_action(a)
        assert np.allclose(out, _ref_q01_q99_unnormalize(a, q01, q99), atol=1e-4)

    def test_unresolved_tag_returns_none_pair(self):
        payload = {"format": ns.MOLMOACT2_NORM_STATS_FORMAT, "metadata_by_tag": {"x": {}, "y": {}}}
        assert ns.build_norm_stats_processors(payload) == (None, None)


class TestProcessorBridgeFallback:
    """ProcessorBridge.from_pretrained wires the norm_stats fallback.

    Regression: pre-fix, a checkpoint dir with ONLY norm_stats.json yielded an
    inactive (passthrough) bridge. Post-fix it builds working normalizers.
    """

    def test_fallback_activates_bridge(self, tmp_path):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        (tmp_path / "norm_stats.json").write_text(FIXTURE.read_text())
        bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")
        assert bridge.is_active
        assert bridge.has_preprocessor and bridge.has_postprocessor

    def test_fallback_normalizes_and_unnormalizes(self, tmp_path):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        fix = _load_fixture()
        (tmp_path / "norm_stats.json").write_text(json.dumps(fix))
        bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")

        sstats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(sstats["q01"], dtype=np.float32)
        q99 = np.asarray(sstats["q99"], dtype=np.float32)
        x = np.array([0.0, 100.0, 90.0, 40.0, -10.0, 20.0], dtype=np.float32)
        out = bridge.preprocess({"observation.state": x.copy()})
        assert np.allclose(out["observation.state"], _ref_q01_q99_normalize(x, q01, q99), atol=1e-4)

    def test_empty_checkpoint_stays_passthrough(self, tmp_path):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")
        assert not bridge.is_active

    def test_migration_error_is_treated_as_missing_config(self):
        # LeRobot 0.5.2 raises ProcessorMigrationError for missing configs; the
        # bridge must classify it as "no standard config" so the fallback runs.
        from strands_robots.policies.lerobot_local.processor import _missing_config_errors

        try:
            from lerobot.processor.pipeline import ProcessorMigrationError
        except ImportError:
            pytest.skip("installed lerobot has no ProcessorMigrationError")
        assert ProcessorMigrationError in _missing_config_errors()
