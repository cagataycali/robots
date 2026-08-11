"""LeRobot policy-type validation is discovered from lerobot's live registry.

``LerobotTrainer.validate`` guards ``extra['policy_type']`` two ways: it rejects
types that are not LeRobot-native, and it rejects ``relative_actions`` for types
whose config lacks ``use_relative_actions``. Both gates used to consult
hardcoded sets that drifted behind lerobot: the native-type set listed only a
subset of the policies lerobot actually ships (so newer types such as ``eo1`` /
``molmoact2`` / ``vla_jepa`` / ``wall_x`` were wrongly reported "not
LeRobot-native"), and the relative-action set omitted ``groot`` (which exposes
``use_relative_actions``), so a valid ``groot`` + relative-actions run was
wrongly rejected. Both are now discovered live from lerobot's
``PreTrainedConfig`` ChoiceRegistry - the same zero-maintenance discovery the
reward-model, robot, teleop, and camera surfaces already use. The
``method='expert_only'`` gate is discovered the same way, off each config's
``train_expert_only`` field.

One gate is discovered slightly differently. The QUANTILES-normalization gate
reads each config's ``normalization_mapping`` *default* rather than a field
name, so it is the only probe here that can hold the type and still fail to
read its answer - and it resolves "unknown" and "declares no such field" two
deliberately different ways. ``TestQuantileNormRegistryProbe`` pins both.

These tests pin the invariant against whatever lerobot is installed (they read
its live registry rather than hardcoding type names), so they hold across
lerobot versions and fail on the pre-fix hardcoded gates whenever the registry
contains a type outside the stale sets.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training.lerobot import (
    _EXPERT_ONLY_POLICY_TYPES_FALLBACK,
    _LEROBOT_POLICY_TYPES_FALLBACK,
    _QUANTILE_NORM_POLICY_TYPES_FALLBACK,
    _RELATIVE_ACTION_POLICY_TYPES_FALLBACK,
    LerobotTrainer,
    _lerobot_policy_types,
    _policy_registry,
    _policy_supports_expert_only,
    _policy_supports_relative_actions,
    _policy_uses_quantile_norm,
)


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


def _broken_default_factory() -> dict[str, str]:
    """A ``default_factory`` that cannot be evaluated (a broken config default)."""
    raise RuntimeError("config default_factory is broken")


@dataclasses.dataclass
class _UnreadableNormalizationConfig:
    """Declares ``normalization_mapping``, but its default cannot be read."""

    normalization_mapping: dict[str, str] = dataclasses.field(default_factory=_broken_default_factory)


@dataclasses.dataclass
class _NoNormalizationMappingConfig:
    """Declares no ``normalization_mapping`` field at all."""

    chunk_size: int = 8


def _patch_registry(monkeypatch, cfg_cls) -> None:
    """Point the live-registry probe at ``cfg_cls`` for the types used below."""
    assert _QUANTILE_NORM_POLICY_TYPES_FALLBACK, "the static set must be non-empty for the loops below"
    registry = dict.fromkeys([*_QUANTILE_NORM_POLICY_TYPES_FALLBACK, "act"], cfg_cls)
    monkeypatch.setattr("strands_robots.training.lerobot._policy_registry", lambda: registry)


def _write_stats_without_quantiles(tmp_path) -> None:
    """Write a v3 ``meta/stats.json`` carrying mean/std/min/max but no ``q01``/``q99``."""
    feature = {"mean": [0.0], "std": [1.0], "min": [-1.0], "max": [1.0]}
    stats = {"observation.state": dict(feature), "action": dict(feature)}
    (tmp_path / "meta" / "stats.json").write_text(json.dumps(stats))


def _spec(dataset_root, tmp_path, **extra) -> TrainSpec:
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="",
        output_dir=str(tmp_path / "out"),
        steps=10,
        extra=extra,
    )


class TestNativePolicyTypeDiscovery:
    def test_every_registry_type_is_accepted_as_native(self, dataset_root, tmp_path):
        """No policy type lerobot registers may be flagged "not LeRobot-native".

        Pre-fix the check consulted a hardcoded 10-name set; any registered type
        beyond it (eo1, molmoact2, vla_jepa, wall_x, ... - present even before
        the latest additions) was wrongly rejected. This iterates lerobot's live
        registry, so it is version-agnostic and fails on the stale set.
        """
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed; registry-driven check not applicable")
        assert reg, "lerobot registry unexpectedly empty"
        trainer = LerobotTrainer(device="cpu")
        for ptype in reg:
            problems = trainer.validate(_spec(dataset_root, tmp_path, policy_type=ptype))
            not_native = [p for p in problems if "not LeRobot-native" in p]
            assert not not_native, f"registry type {ptype!r} wrongly rejected: {not_native}"

    def test_lerobot_policy_types_matches_registry(self):
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        assert _lerobot_policy_types() == set(reg)

    def test_unknown_type_still_rejected(self, dataset_root, tmp_path):
        # A genuinely non-native name must still be caught (no over-permissiveness).
        problems = LerobotTrainer(device="cpu").validate(
            _spec(dataset_root, tmp_path, policy_type="definitely_not_a_policy")
        )
        assert any("not LeRobot-native" in p for p in problems)


class TestRelativeActionDiscovery:
    def test_gate_tracks_config_field_for_every_registry_type(self, dataset_root, tmp_path):
        """`relative_actions` is rejected iff the config class lacks the field.

        Pre-fix the gate used a hardcoded {pi0, pi05, pi0_fast} set; groot also
        exposes ``use_relative_actions`` on current lerobot, so a groot +
        relative-actions run was wrongly rejected. This derives the expectation
        from the actual dataclass field, so it fails whenever the hardcoded set
        diverges from lerobot's configs.
        """
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        trainer = LerobotTrainer(device="cpu")
        for ptype, cfg_cls in reg.items():
            has_field = any(f.name == "use_relative_actions" for f in dataclasses.fields(cfg_cls))
            problems = trainer.validate(_spec(dataset_root, tmp_path, policy_type=ptype, relative_actions=True))
            rejected = any("relative_actions is not supported" in p for p in problems)
            assert rejected == (not has_field), (
                f"{ptype!r}: config has use_relative_actions={has_field} but "
                f"validate {'rejected' if rejected else 'accepted'} relative_actions"
            )

    def test_helper_agrees_with_config_field(self):
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        for ptype, cfg_cls in reg.items():
            has_field = any(f.name == "use_relative_actions" for f in dataclasses.fields(cfg_cls))
            assert _policy_supports_relative_actions(ptype) == has_field


class TestExpertOnlyDiscovery:
    def test_gate_tracks_config_field_for_every_registry_type(self, dataset_root, tmp_path):
        """`method='expert_only'` is rejected iff the config lacks train_expert_only.

        expert_only freezes the VLM and trains only the action expert; lerobot
        implements it as a per-policy ``config.train_expert_only`` field. The
        gate is derived from the actual dataclass field (not a hardcoded set),
        so it fails whenever the static fallback diverges from lerobot's configs.
        """
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        trainer = LerobotTrainer(device="cpu")
        for ptype, cfg_cls in reg.items():
            has_field = any(f.name == "train_expert_only" for f in dataclasses.fields(cfg_cls))
            spec = TrainSpec(
                dataset_root=dataset_root,
                base_model="",
                output_dir=str(tmp_path / "out"),
                steps=10,
                method="expert_only",
                extra={"policy_type": ptype},
            )
            problems = trainer.validate(spec)
            rejected = any("method 'expert_only' is not supported" in p for p in problems)
            assert rejected == (not has_field), (
                f"{ptype!r}: config has train_expert_only={has_field} but "
                f"validate {'rejected' if rejected else 'accepted'} expert_only"
            )

    def test_helper_agrees_with_config_field(self):
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        for ptype, cfg_cls in reg.items():
            has_field = any(f.name == "train_expert_only" for f in dataclasses.fields(cfg_cls))
            assert _policy_supports_expert_only(ptype) == has_field


class TestOfflineFallback:
    """When lerobot's registry is unavailable, the static fallbacks drive the gate."""

    def test_native_types_fall_back_to_static_set(self, monkeypatch):
        monkeypatch.setattr("strands_robots.training.lerobot._policy_registry", lambda: None)
        assert _lerobot_policy_types() == set(_LEROBOT_POLICY_TYPES_FALLBACK)

    def test_relative_actions_fall_back_to_static_set(self, monkeypatch):
        monkeypatch.setattr("strands_robots.training.lerobot._policy_registry", lambda: None)
        for ptype in _RELATIVE_ACTION_POLICY_TYPES_FALLBACK:
            assert _policy_supports_relative_actions(ptype) is True
        assert _policy_supports_relative_actions("act") is False
        assert _policy_supports_relative_actions("definitely_not_a_policy") is False

    def test_expert_only_falls_back_to_static_set(self, monkeypatch):
        monkeypatch.setattr("strands_robots.training.lerobot._policy_registry", lambda: None)
        for ptype in _EXPERT_ONLY_POLICY_TYPES_FALLBACK:
            assert _policy_supports_expert_only(ptype) is True
        assert _policy_supports_expert_only("act") is False
        assert _policy_supports_expert_only("definitely_not_a_policy") is False

    def test_quantile_norm_falls_back_to_static_set(self, monkeypatch):
        monkeypatch.setattr("strands_robots.training.lerobot._policy_registry", lambda: None)
        for ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK:
            assert _policy_uses_quantile_norm(ptype) is True
        assert _policy_uses_quantile_norm("act") is False
        assert _policy_uses_quantile_norm("definitely_not_a_policy") is False


class TestQuantileNormRegistryProbe:
    """The quantile gate reads a field's *default*, not just its name.

    :func:`~strands_robots.training.lerobot._policy_uses_quantile_norm` is the
    only registry probe here that calls a ``default_factory`` instead of looking
    a field name up, so it is the only one that can hold the type and still fail
    to read its answer. It resolves that case and "the config declares no such
    field" two deliberately different ways:

    * an **unreadable default** means the answer is unknown, so the documented
      static set decides - a ``molmoact2`` run keeps its quantile-stats warning;
    * a **missing field** is itself the answer - the config does not declare
      quantile normalization - so the result is ``False`` and the static set is
      not consulted.

    Collapsing the two would silence ``validate``'s quantile-stats preflight for
    exactly the policies that need it, which is the consequence the last test
    here measures at the surface a caller reads.
    """

    def test_unreadable_default_factory_falls_back_to_static_set(self, monkeypatch):
        """An unreadable default is unknown, so the static set decides."""
        _patch_registry(monkeypatch, _UnreadableNormalizationConfig)
        for ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK:
            assert _policy_uses_quantile_norm(ptype) is True, ptype
        assert _policy_uses_quantile_norm("act") is False

    def test_config_without_normalization_mapping_is_definitively_false(self, monkeypatch):
        """A missing field is an answer, not an unknown: the static set is not consulted."""
        _patch_registry(monkeypatch, _NoNormalizationMappingConfig)
        for ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK:
            assert _policy_uses_quantile_norm(ptype) is False, ptype
        assert _policy_uses_quantile_norm("act") is False

    def test_every_registry_config_declares_normalization_mapping(self):
        """Why the test above must inject a registry rather than name a real type.

        Every config lerobot ships declares ``normalization_mapping``, so the
        "no such field" branch is unreachable through the live registry today.
        This fails the day lerobot registers a config without it - at which point
        that branch becomes reachable for a real policy type.
        """
        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot not installed")
        assert reg, "lerobot registry unexpectedly empty"
        lacking = sorted(
            ptype
            for ptype, cfg_cls in reg.items()
            if not any(f.name == "normalization_mapping" for f in dataclasses.fields(cfg_cls))
        )
        assert lacking == [], f"now reachable through the live registry: {lacking}"

    def test_unreadable_default_keeps_the_quantile_stats_preflight_firing(self, dataset_root, tmp_path, monkeypatch):
        """The fallback's purpose, at the surface a caller reads.

        ``validate`` warns when a QUANTILES-normalizing policy is pointed at a
        dataset whose stats carry no ``q01``/``q99``, because lerobot would
        mis-normalize or fail at train time. With the registry answer unreadable,
        the static set is what keeps that warning firing - answering ``False``
        there instead would let the run start with nothing reported.
        """
        _write_stats_without_quantiles(tmp_path)
        _patch_registry(monkeypatch, _UnreadableNormalizationConfig)
        problems = LerobotTrainer(device="cpu").validate(_spec(dataset_root, tmp_path, policy_type="molmoact2"))
        assert any("augment_dataset_quantile_stats" in p for p in problems), problems
