# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A constructor verdict that is not a TypeError or ValueError is still a result, not a raise.

``SimEngine._build_policy`` turned a constructor's ``TypeError`` / ``ValueError``
into the ``status=error`` envelope every rollout surface documents, and let
every other exception escape. Measured on main 40eec5bee, two interpreters each:

* ``lerobot_local`` with ``pretrained_name_or_path="nobody/does_not_exist_xyz"``
  raised ``FileNotFoundError: config.json not found on the HuggingFace Hub`` out
  of ``run_policy``;
* ``remote`` with nothing listening raised ``ConnectionError: RemotePolicy could
  not reach a PolicyServer at ws://127.0.0.1:59999``;
* ``rl`` with a missing ``checkpoint_dir`` raised ``FileNotFoundError: no
  policy_meta.json in checkpoint dir``;
* ``wbc`` with a checkpoint directory without its ONNX raised ``RuntimeError:
  WBCPolicy main ONNX checkpoint not found``.

The refusal texts were right; only the shape was wrong. The boundary is now one
shared rule, :func:`~strands_robots.policies.construction_failure_keeps_its_raise`:
the remote-code gate and a missing optional dependency (an ``ImportError``, or a
provider's error wrapping one as its cause) keep their raise, everything else a
constructor raises is this configuration's verdict and travels in the envelope,
followed by the configuration that was judged so the checkpoint id or address
is named even when the constructor's own text does not repeat it.

The providers here are registered stand-ins whose constructors raise exactly
what the real ones raised, so no Hub, socket or ONNX file is involved.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import construction_failure_keeps_its_raise, register_policy
from strands_robots.policies import factory as policy_factory
from strands_robots.policies.factory import UntrustedRemoteCodeError
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.benchmark import register_benchmark, unregister_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

_MISSING_ID = "nobody/does_not_exist_xyz"
_HUB_MISS = FileNotFoundError("config.json not found on the HuggingFace Hub")
_NO_SERVER = ConnectionError("RemotePolicy could not reach a PolicyServer at ws://127.0.0.1:59999")
_NO_ONNX = RuntimeError("WBCPolicy main ONNX checkpoint not found")
_NO_META = FileNotFoundError("no policy_meta.json in checkpoint dir: /tmp/does-not-exist")

#: (provider name, what its constructor raises), the four measured shapes.
VERDICTS: list[tuple[str, Exception]] = [
    ("probe_hub_miss", _HUB_MISS),
    ("probe_no_server", _NO_SERVER),
    ("probe_no_onnx", _NO_ONNX),
    ("probe_no_meta", _NO_META),
]

_BENCHMARK = "constructor_verdict_probe"


def _raising_provider(raised: Exception) -> type[MockPolicy]:
    class _RaisingPolicy(MockPolicy):
        def __init__(self, **kwargs: Any) -> None:
            raise raised

    return _RaisingPolicy


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine()
    engine.create_world()
    engine.add_robot("so101")
    yield engine
    engine.cleanup()


@pytest.fixture
def providers():
    """The four stand-ins, registered for the test and removed after it."""
    for name, raised in VERDICTS:
        klass = _raising_provider(raised)
        register_policy(name, lambda klass=klass: klass)
    try:
        yield dict(VERDICTS)
    finally:
        for name, _raised in VERDICTS:
            policy_factory._runtime_registry.pop(name, None)


@pytest.fixture
def benchmark():
    bench = DeclarativeBenchmark.from_dict(
        {"name": _BENCHMARK, "max_steps": 4, "supported_robots": ["so101"], "default_robot": "so101"}
    )
    register_benchmark(bench.name, bench)
    yield bench.name
    unregister_benchmark(bench.name)


class TestTheRuleItself:
    """What keeps its raise is decided by identity, not by message text."""

    @pytest.mark.parametrize(("_name", "raised"), VERDICTS, ids=[name for name, _ in VERDICTS])
    def test_a_configuration_verdict_is_reported(self, _name: str, raised: Exception) -> None:
        assert construction_failure_keeps_its_raise(raised) is False

    def test_the_remote_code_gate_keeps_its_raise(self) -> None:
        assert construction_failure_keeps_its_raise(UntrustedRemoteCodeError("opt in with STRANDS_TRUST_REMOTE_CODE=1"))

    def test_a_missing_dependency_keeps_its_raise(self) -> None:
        assert construction_failure_keeps_its_raise(ImportError("No module named 'onnxruntime'"))

    def test_a_dependency_error_wrapped_by_the_provider_keeps_its_raise(self) -> None:
        """``raise RuntimeError(...) from e``: the wbc provider's shape when onnxruntime is absent."""
        try:
            try:
                raise ImportError("No module named 'onnxruntime'")
            except ImportError as e:
                raise RuntimeError("WBCPolicy requires onnxruntime (the [wbc] extra) but it is not installed.") from e
        except RuntimeError as wrapped:
            assert construction_failure_keeps_its_raise(wrapped) is True

    def test_a_runtime_error_with_another_cause_is_a_verdict(self) -> None:
        try:
            try:
                raise OSError("disk full")
            except OSError as e:
                raise RuntimeError("could not write the checkpoint") from e
        except RuntimeError as wrapped:
            assert construction_failure_keeps_its_raise(wrapped) is False


class TestTheBlockingSurfaces:
    """Each entry point returns the verdict it used to raise, naming itself and the configuration."""

    @pytest.mark.parametrize(("name", "raised"), VERDICTS, ids=[name for name, _ in VERDICTS])
    def test_run_policy_reports_it(self, sim, providers, name: str, raised: Exception) -> None:
        config = {"pretrained_name_or_path": _MISSING_ID}
        result = sim.run_policy(robot_name="so101", policy_provider=name, policy_config=config, duration=0.2)
        assert result["status"] == "error"
        assert _text(result).startswith(
            f"run_policy: policy provider {name!r} refused its configuration, so no rollout was started. {raised}"
        )
        assert _MISSING_ID in _text(result)

    def test_eval_policy_reports_it(self, sim, providers) -> None:
        config = {"pretrained_name_or_path": _MISSING_ID}
        result = sim.eval_policy(
            robot_name="so101", policy_provider="probe_hub_miss", policy_config=config, n_episodes=1, max_steps=2
        )
        assert result["status"] == "error"
        assert _text(result).startswith(
            "eval_policy: policy provider 'probe_hub_miss' refused its configuration, "
            f"so no rollout was started. {_HUB_MISS}"
        )
        assert _MISSING_ID in _text(result)

    def test_evaluate_benchmark_reports_it(self, sim, providers, benchmark) -> None:
        config = {"pretrained_name_or_path": _MISSING_ID}
        result = sim.evaluate_benchmark(
            benchmark, robot_name="so101", policy_provider="probe_hub_miss", policy_config=config, n_episodes=1
        )
        assert result["status"] == "error"
        assert _text(result).startswith(
            "evaluate_benchmark: policy provider 'probe_hub_miss' refused its configuration, "
            f"so no rollout was started. {_HUB_MISS}"
        )
        assert _MISSING_ID in _text(result)

    def test_the_configuration_is_named_with_the_shared_renderer(self, sim, providers) -> None:
        """A value whose repr raises cannot take the envelope down with it."""

        class _Unprintable:
            def __repr__(self) -> str:
                raise RuntimeError("no repr")

        config = {"pretrained_name_or_path": _MISSING_ID, "extra": _Unprintable()}
        result = sim.run_policy(
            robot_name="so101", policy_provider="probe_hub_miss", policy_config=config, duration=0.2
        )
        assert result["status"] == "error"
        assert f"pretrained_name_or_path={_MISSING_ID!r}" in _text(result)
        assert "extra=" in _text(result)

    def test_an_empty_configuration_adds_no_clause(self, sim, providers) -> None:
        result = sim.run_policy(robot_name="so101", policy_provider="probe_no_server", duration=0.2)
        assert result["status"] == "error"
        assert _text(result).endswith(str(_NO_SERVER))

    def test_the_trust_gate_still_raises(self, sim) -> None:
        """The boundary moved from a type tuple to a rule; the gate is on the same side of it."""
        with pytest.raises(UntrustedRemoteCodeError):
            sim.run_policy(robot_name="so101", policy_provider="kimodo", duration=0.2)

    def test_the_robot_is_left_free(self, sim, providers) -> None:
        sim.run_policy(robot_name="so101", policy_provider="probe_hub_miss", duration=0.2)
        assert _text(sim.list_policies_running()) == "No policies running."
        assert sim.run_policy(robot_name="so101", policy_provider="mock", duration=0.1)["status"] == "success"
