"""One policy's RTC request does not reach another built from the same checkpoint.

RTC is not configured beside the model, it is configured ON it: ``_init_rtc``
writes ``config.rtc_config`` and has lerobot build the model's ``rtc_processor``
from that field, and lerobot branches on the same field afterwards (measured on
lerobot 0.6.2, ``SmolVLAPolicy.select_action``: ``assert not
self._rtc_enabled()``, i.e. ``config.rtc_config.enabled``). The process-level
``_MODEL_CACHE`` hands one live module to every wrapper with the same key, so
until the key carried the caller's RTC request a single ``rtc_enabled=True``
policy rewrote RTC for every wrapper sharing that module:

* a wrapper built LATER with defaults auto-detected the RTC another caller
  asked for - ``supports_rtc`` True and the re-query interval collapsed from the
  trained chunk to the RTC horizon, silently;
* a wrapper built EARLIER kept ``supports_rtc`` False and so kept driving the
  module through ``select_action``, which lerobot asserts against.

Both directions make an RTC-on/RTC-off comparison in one process meaningless,
and the contaminated arm is the one that asked for nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from lerobot.policies.rtc.configuration_rtc import RTCConfig

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

CHECKPOINT = "test/flow-matching-ckpt"
TRAINED_CHUNK = 50


class _Feature:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FlowMatchingConfig:
    """Shaped like ``SmolVLAConfig``: declares ``rtc_config``, chunks 50 actions."""

    def __init__(self, shipped: RTCConfig | None) -> None:
        self.rtc_config = shipped
        self.n_action_steps = TRAINED_CHUNK
        self.device = "cpu"
        self.input_features = {"observation.state": _Feature((6,))}
        self.output_features = {"action": _Feature((6,))}


class _FlowMatchingModel:
    """Shaped like a loaded SmolVLA/Pi0 module: ``rtc_config`` on its config.

    ``rtc_config`` starts as ``shipped``, which is ``None`` for every public
    flow-matching checkpoint (RTC is an inference-time choice) and an enabled
    ``RTCConfig`` for one saved with RTC on.
    """

    def __init__(self, shipped: RTCConfig | None) -> None:
        self.config = _FlowMatchingConfig(shipped)

    def eval(self) -> None:
        return None

    def parameters(self) -> Any:
        return iter(())

    def predict_action_chunk(self, *_a: Any, **_k: Any) -> None:
        raise AssertionError("_init_rtc must not run inference")

    def init_rtc_processor(self) -> None:
        """lerobot builds the module's RTC processor from ``config.rtc_config``."""


@contextmanager
def _weight_load_stubbed(shipped: RTCConfig | None) -> Iterator[MagicMock]:
    """Serve ``CHECKPOINT`` from a fake module; yield the resolved policy class.

    ``resolved.from_pretrained.call_count`` is the number of weight loads paid
    inside the block - ``0`` means the process model cache served it.
    """
    resolved = MagicMock()
    resolved.from_pretrained.side_effect = lambda _path: _FlowMatchingModel(shipped)
    with (
        patch(
            "strands_robots.policies.lerobot_local.policy.resolve_policy_class_by_name",
            return_value=resolved,
        ),
        patch(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            return_value=MagicMock(is_active=False),
        ),
    ):
        yield resolved


def _module_rtc_config(policy: LerobotLocalPolicy) -> Any:
    """The ``rtc_config`` on the loaded module - the field lerobot itself reads."""
    module = policy._policy
    assert module is not None, "the policy never loaded a module"
    return module.config.rtc_config


def _build(shipped: RTCConfig | None = None, **request: Any) -> tuple[LerobotLocalPolicy, int]:
    """Construct a policy on ``CHECKPOINT``; report how many weight loads it cost."""
    with _weight_load_stubbed(shipped) as resolved:
        policy = LerobotLocalPolicy(
            pretrained_name_or_path=CHECKPOINT,
            policy_type="smolvla",
            device="cpu",
            **request,
        )
    return policy, resolved.from_pretrained.call_count


@pytest.mark.parametrize(
    "rtc_request",
    [{"rtc_enabled": True, "rtc_execution_horizon": 10}, {"rtc_enabled": True}],
    ids=["horizon-named", "horizon-from-lerobot"],
)
def test_a_default_policy_built_after_an_rtc_one_keeps_the_trained_chunk(
    rtc_request: dict[str, Any],
) -> None:
    """default -> rtc_enabled=True -> default: the third policy matches the first."""
    first, first_loads = _build()
    rtc, rtc_loads = _build(**rtc_request)
    after, after_loads = _build()

    # The RTC wrapper is the only one whose module carries RTC.
    horizon = rtc_request.get("rtc_execution_horizon", RTCConfig().execution_horizon)
    assert (rtc.supports_rtc, rtc.execution_horizon) == (True, horizon)
    assert _module_rtc_config(rtc).enabled is True
    assert (first_loads, rtc_loads) == (1, 1), "a different RTC request cannot reuse the module"

    assert (after.supports_rtc, after.execution_horizon) == (first.supports_rtc, first.execution_horizon)
    assert (after.supports_rtc, after.execution_horizon) == (False, TRAINED_CHUNK)
    # ... and it costs nothing: same RTC request as the first, so same module.
    assert after_loads == 0 and after._policy is first._policy
    assert _module_rtc_config(first) is None, "the checkpoint shipped no RTC and nobody asked it to"


def test_a_caller_guidance_ceiling_does_not_reach_a_policy_that_asked_for_the_models_own() -> None:
    """The ceiling is written onto the config lerobot reads, so it travels too."""
    shipped_default = RTCConfig(enabled=True).max_guidance_weight
    tuned, _ = _build(shipped=RTCConfig(enabled=True), rtc_max_guidance_weight=2.0)
    default, _ = _build(shipped=RTCConfig(enabled=True))

    assert _module_rtc_config(tuned).max_guidance_weight == 2.0
    assert _module_rtc_config(default).max_guidance_weight == shipped_default
    assert default._rtc_max_guidance_weight == shipped_default


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {},
        {"rtc_enabled": True, "rtc_execution_horizon": 10},
        {"rtc_max_guidance_weight": 2.0},
    ],
    ids=["defaults", "rtc-on", "ceiling-override"],
)
def test_the_same_rtc_request_still_shares_one_resident_module(request_kwargs: dict[str, Any]) -> None:
    """Separating RTC requests must not cost the cache its reason to exist."""
    first, first_loads = _build(shipped=RTCConfig(enabled=True), **request_kwargs)
    second, second_loads = _build(shipped=RTCConfig(enabled=True), **request_kwargs)

    assert (first_loads, second_loads) == (1, 0)
    assert second._policy is first._policy


def test_a_reload_after_rtc_resolved_the_horizon_finds_its_own_module_again() -> None:
    """The key records the request, not the resolved values, which move.

    ``_init_rtc`` writes the checkpoint's horizon onto the policy, so a policy
    that asked for RTC and named no horizon no longer looks like its own cache
    entry once loaded. It reloads whenever it is driven after a release, and
    reading the live values there would buy a second copy of the weights.
    """
    policy, loads = _build(rtc_enabled=True)
    module = policy._policy
    assert (loads, policy._rtc_execution_horizon) == (1, RTCConfig().execution_horizon)

    policy._loaded = False
    policy._policy = None
    with _weight_load_stubbed(None) as resolved:
        policy._load_model()

    assert resolved.from_pretrained.call_count == 0
    assert policy._policy is module
