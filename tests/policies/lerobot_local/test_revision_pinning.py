"""Hub revision pinning for the lerobot_local provider.

lerobot's ``PreTrainedPolicy.from_pretrained`` accepts ``revision=`` to pin a
checkpoint to a branch, tag, or commit SHA for reproducible loads. These tests
pin that ``LerobotLocalPolicy`` threads a caller-supplied ``revision`` all the
way to ``from_pretrained`` and the hub-side class resolution, that omitting it
preserves the default (unpinned) call shape, and that asking for a revision on a
transformers-native MolmoAct2 checkpoint fails loudly instead of silently
ignoring the request.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies.lerobot_local.policy import (
    LerobotLocalPolicy,
    clear_model_cache,
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


def _build(policy_type="act", **kwargs):
    """Construct a generic lerobot_local policy with the weight load stubbed.

    Returns ``(policy, from_pretrained_mock)`` so a test can inspect exactly how
    the underlying ``PreTrainedPolicy.from_pretrained`` was invoked.
    """
    mock_cls = MagicMock()
    captured: dict = {}

    def _fp(path, **kw):
        captured["path"] = path
        captured["kwargs"] = kw
        return _generic_inner()

    mock_cls.from_pretrained.side_effect = _fp
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
        policy = LerobotLocalPolicy(
            pretrained_name_or_path="test/model",
            policy_type=policy_type,
            device="cpu",
            cache_model=False,
            **kwargs,
        )
    return policy, captured


class TestRevisionPinning:
    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    def test_revision_threaded_to_from_pretrained(self):
        _policy, captured = _build(revision="v1.2.3")
        assert captured["kwargs"].get("revision") == "v1.2.3"

    def test_no_revision_keeps_default_call_shape(self):
        # Without a revision the wrapper must not inject revision=None, so it
        # stays compatible with policy classes whose from_pretrained predates
        # the kwarg.
        _policy, captured = _build()
        assert "revision" not in captured["kwargs"]

    def test_revision_threaded_to_hub_class_resolution(self):
        # When policy_type is not given, the wrapper resolves the class from the
        # hub; the revision must pin that config read too.
        seen: dict = {}

        def _resolve(path, revision=None):
            seen["revision"] = revision
            mock_cls = MagicMock()
            mock_cls.from_pretrained.side_effect = lambda _p, **_kw: _generic_inner()
            return mock_cls, "act"

        with (
            patch(
                "strands_robots.policies.lerobot_local.policy.resolve_policy_class_from_hub",
                side_effect=_resolve,
            ),
            patch(
                "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
                return_value=MagicMock(is_active=False),
            ),
        ):
            LerobotLocalPolicy(
                pretrained_name_or_path="test/model",
                policy_type=None,
                device="cpu",
                cache_model=False,
                revision="abc1234",
            )
        assert seen["revision"] == "abc1234"

    def test_revision_on_molmoact2_raises(self):
        # MolmoAct2 SO-100/101 checkpoints load weights via checkpoint_path, not
        # PreTrainedPolicy.from_pretrained, so revision cannot be honored. Asking
        # for one must fail loudly rather than silently load the default branch.
        with pytest.raises(ValueError, match="revision pinning is not supported"):
            LerobotLocalPolicy(
                pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
                policy_type="molmoact2",
                device="cpu",
                cache_model=False,
                revision="v1.0",
            )
