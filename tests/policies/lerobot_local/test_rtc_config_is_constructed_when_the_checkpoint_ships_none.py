"""``rtc_enabled=True`` constructs the RTC config a public checkpoint never ships.

lerobot declares ``rtc_config`` on the flow-matching config classes only
(measured on lerobot 0.6.2: ``SmolVLAConfig``, ``PI0Config`` and ``PI05Config``
declare it and default it to ``None``, and their policies carry
``init_rtc_processor``; ``ACTConfig``, ``DiffusionConfig`` and ``VQBeTConfig``
declare neither). A ``None`` therefore says "this checkpoint was saved with RTC
off", not "this policy cannot do RTC" - and reading the two as one, which
``_init_rtc`` used to, disabled RTC on every checkpoint a caller can download.

The GPU sibling ``test_rtc_engages_on_a_public_smolvla_checkpoint`` proves the
same seam against the real ``lerobot/smolvla_base``. This module pins the
decision on both policy shapes and needs no GPU, so the refusal half stays
graded on a CPU runner too.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from lerobot.policies.rtc.configuration_rtc import RTCConfig

from tests.policies.lerobot_local.test_policy import _make_loaded_policy


class _FlowMatchingPolicy:
    """Shaped like SmolVLA/Pi0: declares ``rtc_config``, ships it as ``None``."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(rtc_config=None)
        self.rtc_processors_built = 0

    def predict_action_chunk(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("_init_rtc must not run inference")

    def init_rtc_processor(self) -> None:
        self.rtc_processors_built += 1


class _NotFlowMatchingPolicy:
    """Shaped like ACT/Diffusion: inherits ``predict_action_chunk`` and nothing else."""

    def __init__(self) -> None:
        self.config = SimpleNamespace()

    def predict_action_chunk(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("_init_rtc must not run inference")


def _init_rtc_on(inner: object, **overrides: Any) -> Any:
    policy = _make_loaded_policy(include_images=False)
    policy._policy = inner
    policy._rtc_requested = True
    for name, value in overrides.items():
        setattr(policy, name, value)
    policy._init_rtc()
    return policy


@pytest.mark.parametrize(
    ("overrides", "horizon", "ceiling"),
    [
        ({}, RTCConfig().execution_horizon, RTCConfig().max_guidance_weight),
        ({"_rtc_execution_horizon": 8, "_rtc_max_guidance_weight": 2.0}, 8, 2.0),
    ],
    ids=["lerobot-defaults", "caller-overrides"],
)
def test_a_flow_matching_policy_with_no_rtc_config_is_given_one(
    overrides: dict[str, Any], horizon: int, ceiling: float
) -> None:
    inner = _FlowMatchingPolicy()
    policy = _init_rtc_on(inner, **overrides)

    assert policy._rtc_enabled is True
    assert inner.rtc_processors_built == 1, "lerobot must build the processor for the model"
    built = inner.config.rtc_config
    assert (built.enabled, built.execution_horizon, built.max_guidance_weight) == (True, horizon, ceiling)
    assert policy.execution_horizon == horizon, "the re-query interval follows the constructed horizon"


def test_a_policy_that_declares_no_rtc_config_field_is_still_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ACT and Diffusion cannot blend a chunk seam, so asking is answered, not obeyed."""
    inner = _NotFlowMatchingPolicy()
    with caplog.at_level(logging.WARNING):
        policy = _init_rtc_on(inner)

    assert policy._rtc_enabled is False
    assert "has no rtc_config" in caplog.text
    assert not hasattr(inner.config, "rtc_config"), "a refused request writes nothing onto the config"
