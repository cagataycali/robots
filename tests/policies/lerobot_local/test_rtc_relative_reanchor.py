"""Relative-action RTC prefix re-anchoring (LeRobot parity).

A relative-action flow policy (pi0 / pi0.5 / pi0-FAST with an enabled
``RelativeActionsProcessorStep``) trains on actions expressed as offsets from
the current robot state. The unexecuted tail of the previous chunk
(``prev_chunk_left_over``) is therefore only valid in the coordinate frame of
the observation that produced it. When the robot state moves between chunks,
the leftover must be re-expressed against the NEW state before it is fed back
to the policy, otherwise the model blends a stale-frame prefix into the next
chunk and the seam is corrupted.

These tests pin that the LeRobot RTC consumer in ``LerobotLocalPolicy``
re-anchors the leftover via LeRobot's ``reanchor_relative_rtc_prefix`` for
relative-action policies, and carries it verbatim for absolute-action policies
(whose frame does not move). Skips cleanly when LeRobot is unavailable.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("lerobot.processor")

from lerobot.processor import (  # noqa: E402
    AbsoluteActionsProcessorStep,
    IdentityProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy  # noqa: E402
from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402

_ACTION_DIM = 4
_CHUNK_LEN = 6
_EXEC_HORIZON = 2


def _model_chunk() -> torch.Tensor:
    """A deterministic (1, T, A) action chunk in model (normalized-relative) space."""
    return torch.arange(_CHUNK_LEN * _ACTION_DIM, dtype=torch.float32).reshape(1, _CHUNK_LEN, _ACTION_DIM)


def _model_leftover() -> torch.Tensor:
    """The tail of the chunk consumers do NOT execute this step: chunk[exec_horizon:]."""
    return _model_chunk().squeeze(0)[_EXEC_HORIZON:]


def _make_rtc_policy(preprocessor: DataProcessorPipeline, postprocessor: DataProcessorPipeline):
    """Build an RTC-enabled LerobotLocalPolicy wired to a real processor pipeline.

    The inner LeRobot policy is a mock whose ``predict_action_chunk`` records the
    ``prev_chunk_left_over`` it receives so the test can assert on the prefix that
    was actually fed to the model.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")

    policy._loaded = True
    policy._device = torch.device("cpu")

    inner = MagicMock()
    inner.config = MagicMock()
    inner.config.action_feature_names = [f"j{i}.pos" for i in range(_ACTION_DIM)]

    captured: list[torch.Tensor | None] = []

    def _predict(_batch, **kwargs):
        captured.append(kwargs.get("prev_chunk_left_over"))
        return _model_chunk()

    inner.predict_action_chunk.side_effect = _predict
    policy._policy = inner

    policy._rtc_enabled = True
    policy._rtc_execution_horizon = _EXEC_HORIZON
    # Deterministic zero inference delay: world is paused, exactly 0 steps elapse.
    policy.rtc_observed_delay_steps = 0

    policy._processor_bridge = ProcessorBridge(preprocessor=preprocessor, postprocessor=postprocessor)
    return policy, captured


def _prime_state(relative_step: RelativeActionsProcessorStep, state: torch.Tensor) -> None:
    """Cache ``state`` as the relative step's reference (mimics a preprocess pass)."""
    relative_step(create_transition(observation={OBS_STATE: state}))


def test_relative_action_rtc_prefix_is_reanchored_to_current_state():
    names = [f"j{i}.pos" for i in range(_ACTION_DIM)]
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=names)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
    policy, captured = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(steps=[absolute_step]),
    )

    state1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _prime_state(relative_step, state1)
    with torch.inference_mode():
        policy._predict_with_rtc({})

    # First call has no prior chunk to blend.
    assert captured[0] is None
    leftover_model = _model_leftover()
    assert torch.allclose(policy._rtc_prev_chunk, leftover_model)
    # Absolute leftover = unnormalize (identity here) + add the current state.
    assert torch.allclose(policy._rtc_prev_chunk_abs, leftover_model + state1)

    # State moves: the leftover must be re-expressed against the new frame.
    state2 = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    _prime_state(relative_step, state2)
    with torch.inference_mode():
        policy._predict_with_rtc({})

    prev = captured[1]
    assert prev is not None
    # Re-anchored relative prefix = absolute - new_state = leftover + state1 - state2.
    expected = leftover_model + state1 - state2
    assert torch.allclose(prev, expected, atol=1e-5)
    # Crucially NOT the stale model-space leftover that pre-fix code carried.
    assert not torch.allclose(prev, leftover_model)


def test_absolute_action_policy_carries_leftover_verbatim():
    # No RelativeActionsProcessorStep -> the prefix frame never moves, so the
    # leftover is fed back unchanged and no absolute copy is kept.
    policy, captured = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )

    with torch.inference_mode():
        policy._predict_with_rtc({})
    assert captured[0] is None
    leftover_model = _model_leftover()
    assert torch.allclose(policy._rtc_prev_chunk, leftover_model)
    assert policy._rtc_prev_chunk_abs is None

    with torch.inference_mode():
        policy._predict_with_rtc({})
    prev = captured[1]
    assert prev is not None
    assert torch.allclose(prev, leftover_model)


def test_resolve_rtc_rebase_steps_is_idempotent_and_detects_relative():
    names = [f"j{i}.pos" for i in range(_ACTION_DIM)]
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=names)
    policy, _ = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )

    policy._resolve_rtc_rebase_steps()
    assert policy._rtc_rebase_resolved is True
    assert policy._rtc_relative_step is relative_step
    assert policy._rtc_reanchor_fn is not None

    # A disabled relative step must NOT trigger re-anchoring.
    disabled = RelativeActionsProcessorStep(enabled=False, action_names=names)
    policy2, _ = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[disabled]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
    policy2._resolve_rtc_rebase_steps()
    assert policy2._rtc_relative_step is None
