"""Relative-action RTC re-anchoring reports the conversions it cannot make.

:meth:`~strands_robots.policies.lerobot_local.policy.LerobotLocalPolicy._absolute_rtc_leftover`
converts the unexecuted tail of a chunk into absolute robot coordinates so the
NEXT chunk can be re-anchored against the moved robot state. It returns ``None``
in three cases, and only the first is benign:

* the policy is not relative-action - its frame does not move, so there is
  nothing to re-anchor and nothing to report;
* the processor bridge has no postprocessor;
* the postprocessor does not yield a plain action tensor.

The last two are degradations rather than no-ops. The policy *is*
relative-action, so every following chunk blends a stale-frame prefix - the
outcome ``_resolve_rtc_rebase_steps`` warns about when LeRobot's re-anchor
helper is unavailable, and which the sibling module's
``test_relative_action_falls_back_when_reanchor_helper_unavailable`` pins as
"warn once ... never crash or silently drop the prefix". Both of these took that
same degradation silently, behind the INFO line announcing that re-anchoring was
enabled.

These tests drive all three fallbacks: the benign one stays silent, both
degradations report once naming their cause and the stale-frame consequence, the
latch survives ``reset()`` (a pipeline shape does not change between episodes),
and the degraded prefix that actually reaches the denoiser is measured. The scope
boundary is pinned too - a postprocessor that *raises* stays fatal rather than
being downgraded to a silent ``None``.
"""

import importlib
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("lerobot.processor")

import torch  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy  # noqa: E402
from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402

_LOGGER_NAME = "strands_robots.policies.lerobot_local.policy"
_ACTION_DIM = 4
_CHUNK_LEN = 6
_EXEC_HORIZON = 2


def _lerobot_has_reanchor_helper() -> bool:
    """True when the installed lerobot ships ``reanchor_relative_rtc_prefix``.

    Probed by string import + ``hasattr`` (never ``from ... import``) because on
    lerobot 0.5.1 importing ``lerobot.policies.rtc`` executes a module whose
    dataclass fails to build, raising TypeError at load time rather than cleanly
    missing the symbol.
    """
    try:
        module = importlib.import_module("lerobot.policies.rtc")
    except (ImportError, TypeError):
        return False
    return hasattr(module, "reanchor_relative_rtc_prefix")


# _absolute_rtc_leftover is reached only once _rtc_relative_step is set, and
# _resolve_rtc_rebase_steps sets it only when the re-anchor helper resolves. The
# end-to-end cases therefore need the helper; the direct-call cases set the
# precondition themselves and do not.
_requires_reanchor = pytest.mark.skipif(
    not _lerobot_has_reanchor_helper(),
    reason="lerobot.policies.rtc.reanchor_relative_rtc_prefix unavailable (added after lerobot 0.5.1)",
)


def _model_chunk() -> torch.Tensor:
    """A deterministic ``(1, T, A)`` action chunk in model (relative) space."""
    return torch.arange(_CHUNK_LEN * _ACTION_DIM, dtype=torch.float32).reshape(1, _CHUNK_LEN, _ACTION_DIM)


def _model_leftover() -> torch.Tensor:
    """The tail a consumer does not execute this step: ``chunk[exec_horizon:]``."""
    return _model_chunk().squeeze(0)[_EXEC_HORIZON:]


def _make_rtc_policy(
    preprocessor: DataProcessorPipeline | None,
    postprocessor: DataProcessorPipeline | None,
) -> tuple[Any, list[Any]]:
    """An RTC-enabled policy wired to a real bridge, capturing the fed prefix.

    The inner LeRobot policy is a mock whose ``predict_action_chunk`` records the
    ``prev_chunk_left_over`` it is handed, so a test can assert on the prefix the
    denoiser actually receives rather than on internal state.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")

    policy._loaded = True
    policy._device = torch.device("cpu")

    inner = MagicMock()
    inner.config = MagicMock()
    inner.config.action_feature_names = [f"j{i}.pos" for i in range(_ACTION_DIM)]

    captured: list[Any] = []

    def _predict(_batch: Any, **kwargs: Any) -> torch.Tensor:
        captured.append(kwargs.get("prev_chunk_left_over"))
        return _model_chunk()

    inner.predict_action_chunk.side_effect = _predict
    policy._policy = inner

    policy._rtc_enabled = True
    policy._rtc_execution_horizon = _EXEC_HORIZON
    # Deterministic zero inference delay: the world is paused, 0 steps elapse.
    policy.rtc_observed_delay_steps = 0
    policy._processor_bridge = ProcessorBridge(preprocessor=preprocessor, postprocessor=postprocessor)
    return policy, captured


def _relative_policy_with_relative_step() -> tuple[Any, Any, list[Any]]:
    """A policy whose preprocessor carries an enabled relative-action step."""
    names = [f"j{i}.pos" for i in range(_ACTION_DIM)]
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=names)
    policy, captured = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(
            steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)]
        ),
    )
    return policy, relative_step, captured


def _prime_state(relative_step: Any, state: torch.Tensor) -> None:
    """Cache ``state`` as the relative step's reference (mimics a preprocess pass)."""
    relative_step(create_transition(observation={OBS_STATE: state}))


def _drop_postprocessor(policy: Any) -> None:
    """Rebuild the bridge with its preprocessor intact and no postprocessor.

    The preprocessor's step objects are carried over, so the enabled
    relative-action step survives: replacing them would make the policy read as
    absolute-action and take the benign fallback instead of this one.
    """
    policy._processor_bridge = ProcessorBridge(
        preprocessor=DataProcessorPipeline(steps=policy._processor_bridge.preprocessor_steps),
        postprocessor=None,
    )


def _postprocessor_yields_a_dict(policy: Any) -> None:
    """Make the postprocessor return a mapping instead of a plain action tensor."""
    policy._processor_bridge.postprocess = lambda action: {"action": action}


def _postprocessor_raises(policy: Any) -> None:
    """Make the postprocessor fail the way ``ProcessorBridge.postprocess`` documents."""

    def _boom(_action: Any) -> Any:
        raise RuntimeError("Postprocessor pipeline failed: boom")

    policy._processor_bridge.postprocess = _boom


# Every fallback that is a degradation, with the cause its report must name.
_DEGRADATIONS: list[tuple[str, Any, str]] = [
    ("no_postprocessor", _drop_postprocessor, "no postprocessor"),
    ("non_tensor_output", _postprocessor_yields_a_dict, "not a tensor"),
]
_DEGRADATION_IDS = [name for name, _apply, _reason in _DEGRADATIONS]


def _degraded_policy(apply_degradation: Any) -> Any:
    """A relative-action policy whose absolute conversion cannot be made."""
    policy, _relative_step, _captured = _relative_policy_with_relative_step()
    # The documented precondition for reaching the bridge guards: the policy was
    # detected as relative-action, so its leftover genuinely needs re-anchoring.
    policy._rtc_relative_step = object()
    apply_degradation(policy)
    return policy


class TestTheBenignFallbackStaysSilent:
    """An absolute-action policy has nothing to re-anchor, so it reports nothing."""

    def test_an_absolute_action_policy_returns_none_without_reporting(self, caplog):
        policy, _relative_step, _captured = _relative_policy_with_relative_step()
        policy._rtc_relative_step = None  # absolute-action: frame does not move

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            assert policy._absolute_rtc_leftover(_model_leftover()) is None

        assert caplog.records == []
        assert policy._rtc_reanchor_degraded_warned is False

    @_requires_reanchor
    def test_a_healthy_relative_action_policy_reports_nothing(self, caplog):
        policy, relative_step, _captured = _relative_policy_with_relative_step()
        state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        _prime_state(relative_step, state)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            with torch.inference_mode():
                policy._predict_with_rtc({})

        # The conversion succeeded, so there is a leftover to re-anchor with.
        assert policy._rtc_prev_chunk_abs is not None
        assert torch.allclose(policy._rtc_prev_chunk_abs, _model_leftover() + state)
        assert caplog.records == []


class TestEveryDegradedFallbackIsReported:
    """A relative-action leftover that cannot be converted says so, once."""

    @pytest.mark.parametrize(
        ("apply_degradation", "expected_cause"),
        [(apply, cause) for _name, apply, cause in _DEGRADATIONS],
        ids=_DEGRADATION_IDS,
    )
    def test_it_returns_none_and_warns(self, apply_degradation, expected_cause, caplog):
        policy = _degraded_policy(apply_degradation)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            assert policy._absolute_rtc_leftover(_model_leftover()) is None

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
        assert expected_cause in warnings[0].getMessage()

    @pytest.mark.parametrize(
        "apply_degradation",
        [apply for _name, apply, _cause in _DEGRADATIONS],
        ids=_DEGRADATION_IDS,
    )
    def test_the_report_names_the_stale_frame_consequence(self, apply_degradation, caplog):
        # Naming the cause is not enough: the reader has to learn what it costs,
        # which is the wording _resolve_rtc_rebase_steps already uses.
        policy = _degraded_policy(apply_degradation)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            policy._absolute_rtc_leftover(_model_leftover())

        assert caplog.records, "the degradation was not reported at all"
        message = caplog.records[0].getMessage()
        assert "STALE coordinate frame" in message
        assert "chunk-seam prefix" in message


class TestTheReportIsLatchedPerPolicy:
    """One warning per policy, not one per chunk - and not one per episode."""

    @pytest.mark.parametrize(
        "apply_degradation",
        [apply for _name, apply, _cause in _DEGRADATIONS],
        ids=_DEGRADATION_IDS,
    )
    def test_many_chunks_report_once(self, apply_degradation, caplog):
        policy = _degraded_policy(apply_degradation)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            for _ in range(12):
                assert policy._absolute_rtc_leftover(_model_leftover()) is None

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_reset_leaves_the_latch_set(self, caplog):
        # The bridge's postprocessor shape does not change between episodes, so
        # re-arming would re-report a condition the operator has already been
        # told about. reset() re-arms the per-episode action diagnostics only.
        policy = _degraded_policy(_drop_postprocessor)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            policy._absolute_rtc_leftover(_model_leftover())
            assert policy._rtc_reanchor_degraded_warned is True

            policy.reset()
            assert policy._rtc_reanchor_degraded_warned is True
            policy._absolute_rtc_leftover(_model_leftover())

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


class TestTheDegradationReachesTheDenoiser:
    """What the report is about: the prefix actually fed to the model."""

    @_requires_reanchor
    def test_a_degraded_policy_feeds_the_stale_model_space_prefix(self, caplog):
        policy, relative_step, captured = _relative_policy_with_relative_step()
        _drop_postprocessor(policy)

        state1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        _prime_state(relative_step, state1)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            with torch.inference_mode():
                policy._predict_with_rtc({})

            # No absolute copy, so the next chunk has nothing to re-anchor with.
            assert policy._rtc_prev_chunk_abs is None

            # The state moves between chunks - exactly when a stale frame hurts.
            state2 = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
            _prime_state(relative_step, state2)
            with torch.inference_mode():
                policy._predict_with_rtc({})

        leftover_model = _model_leftover()
        prefix = captured[1]
        assert prefix is not None
        # Carried in the frame of the PREVIOUS observation, not the current one.
        assert torch.allclose(prefix, leftover_model)
        assert not torch.allclose(prefix, leftover_model + state1 - state2)
        # And the operator was told, rather than left to infer it from the seam.
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


class TestARaisingPostprocessorStaysFatal:
    """Scope boundary: a broken pipeline is not downgraded to a silent fallback."""

    def test_the_runtime_error_propagates(self, caplog):
        policy = _degraded_policy(_postprocessor_raises)

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            with pytest.raises(RuntimeError, match="Postprocessor pipeline failed"):
                policy._absolute_rtc_leftover(_model_leftover())

        # Loud already: adding a warning here would report a failure that is
        # about to be raised anyway.
        assert caplog.records == []
        assert policy._rtc_reanchor_degraded_warned is False
