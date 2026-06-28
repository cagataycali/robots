"""Guard tests pinning the intentional divergence from LeRobot's rollout subsystem.

LeRobot ships a ``rollout/`` deployment engine: ``RolloutStrategyConfig`` (base,
sentry, highlight, dagger, episodic) session managers and ``InferenceEngineConfig``
(sync, rtc) action backends. strands-robots deliberately does NOT consume that
engine in its inference path. Instead it owns the re-query loop and drives Real-Time
Chunking by an integer *control-step count*, so seeded episodes are reproducible.

The divergence is by design (see ``docs/policies/lerobot-local.md`` -> "Relationship
to LeRobot's rollout subsystem" and ``AGENTS.md``):

* strands adopts LeRobot's RTC *algorithm* at the policy layer (``LerobotLocalPolicy``
  consumes LeRobot's ``RTCConfig`` + ``predict_action_chunk``).
* strands diverges on the *loop driver*: LeRobot's ``RTCInferenceEngine`` swaps chunks
  off a wall-clock ``queue_threshold`` (non-deterministic by step count), whereas
  strands re-queries at exactly ``execution_horizon`` and feeds the seam offset as a
  counted integer via ``set_rtc_observed_delay``.

These tests fail if a future change bridges LeRobot's wall-clock rollout inference
engine into the strands inference path without re-reading that rationale.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.base import Policy, resolve_chunk_length

# Modules that make up the strands inference path. None of them may *import* the
# LeRobot rollout deployment engine (the teleop tool may still invoke
# ``lerobot-rollout`` as a CLI subprocess - that is a string, not an import).
_INFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "strands_robots" / "policies",
    Path(__file__).resolve().parents[2] / "strands_robots" / "simulation",
)

# Import targets that would signal a bridge of LeRobot's rollout engine.
_FORBIDDEN_IMPORT_PREFIXES = ("lerobot.rollout",)
_FORBIDDEN_IMPORT_NAMES = (
    "RolloutStrategyConfig",
    "InferenceEngineConfig",
    "RTCInferenceEngine",
    "SyncInferenceEngine",
)


def _imported_modules(tree: ast.AST) -> list[str]:
    """Return every module path referenced by ``import`` / ``from ... import``."""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
            modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in _INFERENCE_PATH:
        files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def test_inference_path_does_not_import_lerobot_rollout_engine() -> None:
    """No strands inference module imports LeRobot's rollout deployment engine.

    Pins decision (c)+(b): strands implements RTC at the policy layer and drives
    the loop itself; it never consumes LeRobot's ``InferenceEngineConfig`` /
    ``RolloutStrategyConfig``. A subprocess CLI invocation of ``lerobot-rollout``
    (a string, not an import) is allowed and is how ``dagger`` is adopted.
    """
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_modules(tree):
            if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                offenders.append(f"{path.name}: imports {module}")
            elif module.rsplit(".", 1)[-1] in _FORBIDDEN_IMPORT_NAMES:
                offenders.append(f"{path.name}: imports {module}")
    assert not offenders, (
        "strands inference path must not bridge LeRobot's rollout engine; see "
        "docs/policies/lerobot-local.md 'Relationship to LeRobot's rollout subsystem'. "
        f"Offending imports: {offenders}"
    )


class _FakeRtcPolicy(Policy):
    """Weight-free RTC policy: declares an execution horizon shorter than its chunk."""

    requires_images = False
    supports_rtc = True

    def __init__(self, execution_horizon: int = 10, actions_per_step: int = 50) -> None:
        self._exec = execution_horizon
        self.actions_per_step = actions_per_step

    @property
    def provider_name(self) -> str:
        return "fake_rtc"

    @property
    def execution_horizon(self) -> int:
        return self._exec

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"j0": 0.0} for _ in range(self.actions_per_step)]


def test_rtc_observed_delay_is_deterministic_step_count() -> None:
    """The seam offset is a counted non-negative integer, not a wall-clock measure.

    This is the determinism guarantee that LeRobot's wall-clock ``RTCInferenceEngine``
    cannot provide: the paused-world sync loop reports exactly 0 steps elapsed, and
    any async overlap reports an integer step count.
    """
    policy = _FakeRtcPolicy()

    policy.set_rtc_observed_delay(0)  # paused-world sync eval loop
    assert policy.rtc_observed_delay_steps == 0

    policy.set_rtc_observed_delay(7)  # async overlap: still-pending steps
    assert policy.rtc_observed_delay_steps == 7

    policy.set_rtc_observed_delay(None)  # clear -> provider falls back to estimate
    assert policy.rtc_observed_delay_steps is None

    with pytest.raises(ValueError):
        policy.set_rtc_observed_delay(-1)


def test_rtc_requery_interval_is_owned_by_strands_not_caller() -> None:
    """An RTC policy is re-queried at ``execution_horizon`` regardless of caller hint.

    strands owns the re-query interval (the divergence from delegating it to an
    inference engine). A caller-supplied ``action_horizon`` cannot stretch or shrink
    an RTC policy's interval - that would leave the prev-chunk tail empty and degrade
    RTC to open-loop replay.
    """
    rtc = _FakeRtcPolicy(execution_horizon=10, actions_per_step=50)
    assert resolve_chunk_length(rtc, action_horizon=1) == 10
    assert resolve_chunk_length(rtc, action_horizon=999) == 10
