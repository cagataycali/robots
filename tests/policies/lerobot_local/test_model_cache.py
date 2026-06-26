"""Process-level model cache for LerobotLocalPolicy.

Loading a LeRobot/VLA checkpoint (e.g. MolmoAct2 SO-100/101 = 1295 weight
files) reads gigabytes from disk and uploads them to the GPU. Re-instantiating
the policy for the same checkpoint - the common pattern when an eval driver
calls ``create_policy`` per rollout - used to pay that full load cost every
time. These tests pin that the heavy load happens ONCE per (checkpoint, type,
device) and is shared by later instances, that opting out forces a private
load, and that clearing the cache restores a cold load.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from strands_robots.policies.lerobot_local import molmoact2
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
        policy = LerobotLocalPolicy(
            pretrained_name_or_path="test/model",
            policy_type="act",
            device="cpu",
            **kwargs,
        )
    return policy, mock_cls


class TestGenericModelCache:
    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    def test_second_instance_reuses_cached_model(self):
        p1, cls1 = _load_generic()
        assert cls1.from_pretrained.call_count == 1

        p2, cls2 = _load_generic()
        # The second instance hits the cache: no new from_pretrained call on
        # its resolver, and it shares the SAME underlying module as the first.
        assert cls2.from_pretrained.call_count == 0
        assert p2._policy is p1._policy
        assert p1._loaded and p2._loaded

    def test_cache_model_false_forces_private_load(self):
        p1, _ = _load_generic()
        p2, cls2 = _load_generic(cache_model=False)
        # Opt-out instance always loads its own module, never sharing.
        assert cls2.from_pretrained.call_count == 1
        assert p2._policy is not p1._policy

    def test_clear_model_cache_restores_cold_load(self):
        _load_generic()
        evicted = clear_model_cache()
        assert evicted >= 1
        _, cls = _load_generic()
        # After eviction the next instance must rebuild from scratch.
        assert cls.from_pretrained.call_count == 1

    def test_distinct_device_keys_do_not_collide(self):
        p_cpu, _ = _load_generic()  # device="cpu"
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
            p_other = LerobotLocalPolicy(
                pretrained_name_or_path="test/model",
                policy_type="act",
                device="meta",
            )
        # A different requested device is a distinct key -> fresh load.
        assert mock_cls.from_pretrained.call_count == 1
        assert p_other._policy is not p_cpu._policy


class _FakeConfig:
    def __init__(self):
        self.n_action_steps = 30
        self.input_features = {"observation.state": MagicMock(shape=(6,))}
        self.output_features = {"action": MagicMock(shape=(6,))}


class _FakeParam:
    def __init__(self, device):
        self.device = device


class _FakePolicy:
    def __init__(self):
        self.config = _FakeConfig()

    def parameters(self):
        return iter([_FakeParam(torch.device("cpu"))])


class TestMolmoAct2ModelCache:
    """The MolmoAct2 transformers-native load path shares the same cache.

    Only the heavy ``policy`` module is cached; ``build_policy`` still rebuilds
    the cheap config + pre/post processors each time and reuses the model via
    its ``prebuilt_policy`` parameter, so per-instance processor state is never
    shared.
    """

    def setup_method(self):
        clear_model_cache()

    def teardown_method(self):
        clear_model_cache()

    def _instantiate(self, build_calls):
        def fake_build_policy(path, **kwargs):
            build_calls.append(kwargs.get("prebuilt_policy"))
            policy = kwargs.get("prebuilt_policy") or _FakePolicy()
            return policy, None, None, _FakeConfig()

        with (
            patch.object(molmoact2, "is_molmoact2", return_value=True),
            patch.object(molmoact2, "build_policy", side_effect=fake_build_policy),
        ):
            return LerobotLocalPolicy(
                pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
                device="cpu",
                use_processor=False,
            )

    def test_weights_built_once_and_shared(self):
        calls: list = []
        p1 = self._instantiate(calls)
        p2 = self._instantiate(calls)

        # First build constructs the model (prebuilt=None); the second receives
        # the cached model as prebuilt_policy, skipping the weight load.
        assert calls[0] is None
        assert calls[1] is p1._policy
        assert p2._policy is p1._policy

    def test_clear_cache_forces_fresh_molmoact2_build(self):
        calls: list = []
        self._instantiate(calls)
        clear_model_cache()
        self._instantiate(calls)
        # Both builds were cold (prebuilt_policy=None) because the cache was
        # cleared between them.
        assert calls[0] is None
        assert calls[1] is None
