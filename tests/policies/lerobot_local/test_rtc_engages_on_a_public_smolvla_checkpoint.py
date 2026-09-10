"""``rtc_enabled=True`` engages RTC on a public SmolVLA checkpoint.

Every public flow-matching checkpoint (``lerobot/smolvla_base`` included) ships
``config.rtc_config = None``: RTC is an inference-time choice, not a training
artifact. ``_init_rtc`` used to treat that ``None`` as "not a flow-matching
policy", warn ``has no rtc_config`` and fall back to ``select_action()`` - so
the documented ``create_policy("lerobot_local", ..., rtc_enabled=True)`` never
ran RTC on any checkpoint a customer can download.

Real model on a real GPU by design (measured on an L40S): the defect is in the
seam between our adapter and lerobot's ``init_rtc_processor``, which a fake
policy cannot exercise.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="needs lerobot"),
]


def test_rtc_enabled_true_constructs_rtc_config_and_engages():
    pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla", reason="needs the [smolvla] extra")
    from strands_robots import create_policy

    policy = create_policy(
        "lerobot_local", pretrained_name_or_path="lerobot/smolvla_base", device="cuda", rtc_enabled=True
    )
    inner = policy._policy
    assert inner.config.rtc_config is not None and inner.config.rtc_config.enabled is True
    assert inner.rtc_processor is not None
    assert inner.model.rtc_processor is inner.rtc_processor
    assert policy.supports_rtc is True
    assert policy.execution_horizon == inner.config.rtc_config.execution_horizon


def test_rtc_overrides_land_in_the_constructed_config():
    pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla", reason="needs the [smolvla] extra")
    from strands_robots import create_policy

    policy = create_policy(
        "lerobot_local",
        pretrained_name_or_path="lerobot/smolvla_base",
        device="cuda",
        rtc_enabled=True,
        rtc_execution_horizon=8,
        rtc_max_guidance_weight=2.0,
    )
    rtc_config = policy._policy.config.rtc_config
    assert (rtc_config.execution_horizon, rtc_config.max_guidance_weight) == (8, 2.0)
