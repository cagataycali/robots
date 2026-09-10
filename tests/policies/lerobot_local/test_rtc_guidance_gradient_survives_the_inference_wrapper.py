"""CPU shadow of the SmolVLA GPU test: the inference wrapper must let RTC differentiate.

lerobot's ``RTCProcessor.denoise_step`` guides the denoiser with
``torch.autograd.grad`` taken inside ``with torch.enable_grad():``. That works
under ``torch.no_grad`` and is impossible under ``torch.inference_mode`` - the
mode cannot be re-enabled from within, so the grad call fails with "element 0
of tensors does not require grad and does not have a grad_fn". The adapter used
to wrap inference in ``inference_mode``, so the second RTC chunk of every
flow-matching run died there (measured on an L40S with lerobot/smolvla_base).

The fake ``predict_action_chunk`` below does exactly lerobot's autograd step and
nothing else, so this fails on the old wrapper and passes on ``no_grad`` -
unlike the ``test_rtc_*.py`` suites, which call ``_predict_with_rtc`` directly
under their own ``inference_mode`` and never reach the wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from tests.policies.lerobot_local.test_policy import _make_loaded_policy

_HORIZON = 10
_CHUNK = 50
_DIM = 6


def _guided_chunk(_batch, **kwargs) -> torch.Tensor:
    x_t = torch.zeros(1, _CHUNK, _DIM)
    with torch.enable_grad():  # verbatim lerobot/policies/rtc/modeling_rtc.py denoise_step
        x_t.requires_grad_(True)
        x1_t = x_t - 0.5 * torch.ones_like(x_t)
        correction = torch.autograd.grad(x1_t, x_t, torch.ones_like(x1_t), retain_graph=False)[0]
    return (x1_t + correction).detach()


def test_rtc_get_actions_lets_lerobot_take_its_guidance_gradient():
    policy = _make_loaded_policy(action_dim=_DIM, state_dim=_DIM, include_images=False)
    policy.set_robot_state_keys([f"j{i}" for i in range(_DIM)])
    policy._policy.predict_action_chunk = MagicMock(side_effect=_guided_chunk)
    policy._rtc_enabled = True
    policy._rtc_execution_horizon = _HORIZON
    policy.rtc_observed_delay_steps = 0

    obs = {f"j{i}": 0.0 for i in range(_DIM)}
    first = policy.get_actions_sync(obs, "test")
    second = policy.get_actions_sync(obs, "test")  # carries prev_chunk_left_over -> guidance runs

    assert len(first) == len(second) == _HORIZON
    assert policy._policy.predict_action_chunk.call_args.kwargs["prev_chunk_left_over"] is not None
