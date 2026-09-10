"""The RTC prefix describes the actions the robot has not executed yet.

LeRobot's Real-Time Chunking contract for ``prev_chunk_left_over`` is fixed by
``ActionQueue.get_left_over()``, which the inference loop snapshots immediately
before it starts denoising::

    idx_before = queue.get_action_index()
    prev_actions = queue.get_left_over()      # original_queue[last_index:]

``original_queue`` is the previous chunk with its own inference delay already
dropped - exactly what a strands RTC policy hands its consumer - and
``last_index`` is how much of it the robot has executed. So row 0 of the prefix
is the action the robot executes on the tick right after the observation, and
row *i* is the action it executes on the tick the new chunk's row *i* would land
on. That index alignment is the whole mechanism: ``RTCProcessor.denoise_step``
builds ``get_prefix_weights(inference_delay, execution_horizon, T)``, which
pins weight 1.0 on ``[0, inference_delay)`` - the steps that elapse *during*
inference - and blends ``[inference_delay, execution_horizon)`` toward the
prefix. A prefix that starts anywhere else freezes the new chunk onto actions
the robot is not executing.

These cells drive the real async pipeline (``_ChunkPipeline``) so the runner's
own prefetch arithmetic - not a restatement of it - decides the observation tick
and the delay, and then assert which rows of the previous chunk arrive.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from strands_robots.policies.base import resolve_chunk_length
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.simulation.policy_runner import _ChunkPipeline

_ACTION_DIM = 1
_CHUNK_LEN = 50
_HORIZON = 10


def _arange_chunk() -> torch.Tensor:
    """A ``(1, 50, 1)`` chunk whose row *j* carries the value ``j``.

    Every row is its own index, so an assertion on a prefix value names the row
    of the previous chunk that arrived without any bookkeeping in the test.
    """
    return torch.arange(_CHUNK_LEN, dtype=torch.float32).reshape(1, _CHUNK_LEN, _ACTION_DIM)


def _rtc_policy() -> LerobotLocalPolicy:
    """An RTC-active policy whose model returns the arange chunk on every call."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")
    policy._rtc_enabled = True
    policy._rtc_execution_horizon = _HORIZON
    policy.actions_per_step = _CHUNK_LEN
    stub = MagicMock()
    stub.predict_action_chunk.side_effect = lambda batch, **kwargs: _arange_chunk()
    policy._policy = stub
    return policy


def _run_async_rollout(policy: LerobotLocalPolicy, ticks: int) -> tuple[list[float], list[dict], list[int]]:
    """Drive ``ticks`` control steps of the async-RTC pipeline over ``policy``.

    Returns the action values the consumer actually executed in order, the kwargs
    of every ``predict_action_chunk`` call, and the tick each of those calls'
    observation was captured on - so a prefix can be compared against the
    execution it claims to describe. The pipeline captures a prefetch
    observation on the consuming thread immediately before it submits the
    inference, so the length of the executed list at that moment IS the
    observation tick.
    """
    calls: list[dict] = []
    observation_ticks: list[int] = []
    stub = policy._policy
    assert stub is not None, "_rtc_policy installs the stub model"

    def query_chunk(observation: dict, observed_delay: int) -> list[dict]:
        policy.set_rtc_observed_delay(observed_delay)
        chunk = policy._predict_with_rtc({})
        calls.append(dict(stub.predict_action_chunk.call_args.kwargs))
        actions = [{"joint": float(row[0])} for row in chunk]
        return actions[: resolve_chunk_length(policy, 1)]

    executed: list[float] = []

    def observe() -> dict:
        observation_ticks.append(len(executed))
        return {}

    with _ChunkPipeline(query_chunk, observe, async_rtc=True, rtc_inference_timeout_s=10.0) as chunks:
        for _observation, action in chunks:
            executed.append(action["joint"])
            if len(executed) >= ticks:
                break
    return executed, calls, observation_ticks


def test_prefix_row_zero_is_the_action_executed_right_after_the_observation():
    """The async prefix starts at the consumer's next action, not past the horizon.

    The runner fires the prefetch when the current chunk is half drained, so at
    the observation tick five of the ten handed-out actions are still pending.
    Those five execute *during* inference and are what ``inference_delay=5``
    tells the denoiser to freeze. Handing back the tail past the execution
    horizon instead shifted the whole prefix five steps into the future: the
    denoiser froze the new chunk onto rows the robot would not reach for another
    five ticks, and the seam RTC exists to smooth was blended against actions
    that were never executed.
    """
    policy = _rtc_policy()

    executed, calls, _ = _run_async_rollout(policy, ticks=2 * _HORIZON)

    assert len(calls) >= 2, "the pipeline should have re-queried at the chunk seam"
    prefetch = calls[1]
    assert prefetch["inference_delay"] == _HORIZON // 2
    prefix = prefetch["prev_chunk_left_over"]
    # The observation is captured just before the action at index 5 is applied,
    # so that action is what the prefix must open on.
    assert prefix[0, 0] == pytest.approx(executed[_HORIZON // 2])
    # ... and the prefix runs to the end of the previous chunk, which is what
    # anchors the blended region [inference_delay, execution_horizon).
    assert prefix.shape == (_CHUNK_LEN - _HORIZON // 2, _ACTION_DIM)
    assert prefix[:, 0].tolist() == [float(v) for v in range(_HORIZON // 2, _CHUNK_LEN)]


def test_prefix_rows_stay_aligned_with_the_ticks_the_new_chunk_lands_on():
    """Prefix row *i* is the action executed on the new chunk's row *i* tick.

    The frozen region is an index-wise comparison inside the denoiser, so the
    guarantee has to hold per row, not just at row 0: for every step that
    elapses during inference, the prefix row and the action the consumer really
    applies on that tick must be the same.
    """
    policy = _rtc_policy()

    executed, calls, _ = _run_async_rollout(policy, ticks=2 * _HORIZON)

    delay = calls[1]["inference_delay"]
    prefix = calls[1]["prev_chunk_left_over"]
    during_inference = executed[_HORIZON - delay : _HORIZON]
    assert prefix[:delay, 0].tolist() == pytest.approx(during_inference)


def test_a_fully_drained_chunk_carries_the_tail_past_the_horizon():
    """With no overlap the prefix is the undrained tail - unchanged behaviour.

    The synchronous path pauses the world during inference and re-queries only
    once the handed-out chunk has drained, so nothing is pending at the
    observation tick and ``observed_delay`` is 0. The prefix is then the
    continuation the previous chunk would have run past the execution horizon,
    which is what the consumer's next actions are blended against.
    """
    policy = _rtc_policy()

    policy.set_rtc_observed_delay(0)
    policy._predict_with_rtc({})
    policy.set_rtc_observed_delay(0)
    policy._predict_with_rtc({})

    prefix = policy._policy.predict_action_chunk.call_args.kwargs["prev_chunk_left_over"]
    assert prefix[0, 0] == pytest.approx(float(_HORIZON))
    assert prefix.shape == (_CHUNK_LEN - _HORIZON, _ACTION_DIM)


def test_an_overlap_longer_than_the_horizon_anchors_at_the_chunk_start():
    """A delay past the horizon opens the prefix at the chunk the consumer got.

    Only the wall-clock fallback can report an overlap longer than the
    handed-out chunk (the counted path never exceeds it): the estimate says more
    steps elapse during inference than the consumer was given actions for, so it
    has stalled on the chunk's last action. The prefix cannot describe ticks
    before the chunk began, so it is clamped to the chunk's first row rather
    than indexed off its front.
    """
    policy = _rtc_policy()

    policy.set_rtc_observed_delay(0)
    policy._predict_with_rtc({})
    policy.set_rtc_observed_delay(_HORIZON + 25)
    policy._predict_with_rtc({})

    prefix = policy._policy.predict_action_chunk.call_args.kwargs["prev_chunk_left_over"]
    assert prefix[0, 0] == pytest.approx(0.0)
    assert prefix.shape == (_CHUNK_LEN, _ACTION_DIM)


def test_the_absolute_prefix_is_sliced_where_the_model_prefix_is():
    """A relative-action policy re-anchors the same rows it blends.

    The absolute-coordinate copy of the chunk exists so a relative-action policy
    can re-express the prefix against the current state. It has to be cut at the
    same consumption index as the model-space prefix - a copy sliced elsewhere
    would re-anchor a different span of the chunk than the one being blended,
    reintroducing the shift this module pins in the coordinate frame instead of
    the row index.
    """
    policy = _rtc_policy()
    # Element-wise offset, mirroring the real conversion's per-action contract.
    policy._absolute_rtc_leftover = lambda tail: tail + 100.0  # type: ignore[method-assign]
    policy._rtc_rebase_resolved = True
    relative_step = MagicMock()
    relative_step.get_cached_state.return_value = torch.zeros(_ACTION_DIM)
    policy._rtc_relative_step = relative_step
    seen: dict[str, torch.Tensor] = {}

    def reanchor(*, prev_actions_absolute, current_state, relative_step, normalizer_step, policy_device):
        seen["absolute"] = prev_actions_absolute
        return prev_actions_absolute

    policy._rtc_reanchor_fn = reanchor

    policy.set_rtc_observed_delay(0)
    policy._predict_with_rtc({})
    policy.set_rtc_observed_delay(_HORIZON // 2)
    policy._predict_with_rtc({})

    model_prefix = policy._policy.predict_action_chunk.call_args.kwargs["prev_chunk_left_over"]
    assert torch.allclose(seen["absolute"], model_prefix - 100.0 + 100.0)
    assert seen["absolute"].shape == model_prefix.shape
    assert seen["absolute"][0, 0] == pytest.approx(float(_HORIZON // 2) + 100.0)


def test_every_seam_opens_its_prefix_on_that_seam_own_next_action():
    """The alignment holds at every re-query, not only the first.

    Each chunk is handed out with its own inference delay already dropped, so the
    chunk a later seam slices is not the raw model output - it is what the
    consumer received. Keeping the raw chunk instead is indistinguishable on the
    first seam (its delay is 0) and wrong on every one after it, so the guarantee
    has to be checked across several chunks against the tick each observation was
    actually captured on.
    """
    policy = _rtc_policy()

    executed, calls, observation_ticks = _run_async_rollout(policy, ticks=3 * _HORIZON)

    seams = [i for i, call in enumerate(calls) if call.get("prev_chunk_left_over") is not None]
    assert len(seams) >= 2, f"expected at least two seams, saw calls={len(calls)}"
    for i in seams:
        prefix = calls[i]["prev_chunk_left_over"]
        tick = observation_ticks[i]
        assert prefix[0, 0] == pytest.approx(executed[tick]), (
            f"seam {i}: prefix opens on {prefix[0, 0]} but tick {tick} executes {executed[tick]}"
        )
