"""Every rollout loop acquires chunks through one pipeline and one query helper.

``PolicyRunner`` used to carry three hand-rolled copies of the chunk-query body
and two copies of the async-RTC prefetch pipeline - ``run()`` re-implemented
inline what :class:`_ChunkPipeline` already did for ``evaluate()``. The copies
had drifted, so the same degenerate policy produced four different outcomes
depending on which entry point drove it:

* ``run(async_rtc=False)`` spun forever (no empty-chunk guard at all),
* ``run(async_rtc=True)`` refused,
* ``evaluate(async_rtc=False)`` advanced one physics step per query,
* ``evaluate(async_rtc=True)`` refused via the pipeline.

and ``run()``'s synchronous path never counted the chunks it acquired, so the
documented ``rtc_chunks_acquired`` field read 0 beside a non-zero
``rtc_avg_inference_ms``. These pin the collapsed seam: the structural facts that
keep it single, and the behaviour each duplicate had got wrong.
"""

from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation import policy_runner as policy_runner_module
from strands_robots.simulation.benchmark_spec import BenchmarkProtocol, StepInfo
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner, query_policy_chunk

_SOURCE = inspect.getsource(policy_runner_module)


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="one_pipeline", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


def _payload(result: dict) -> dict:
    for block in result["content"]:
        if "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in {result['content']!r}")


class _EmptyChunkPolicy(MockPolicy):
    """Degenerate policy: always returns an empty chunk.

    ``call_budget`` makes the pre-fix defect TERMINATE instead of hanging the
    suite: the synchronous ``run()`` loop re-queried without ever advancing
    ``step_count``, so it called ``get_actions`` unboundedly (measured at
    112,200+ calls in 25s for a 4-step budget). Exceeding a budget generous
    enough for any legitimate loop is the diagnosis, not a timeout.
    """

    def __init__(self, call_budget: int = 40) -> None:
        super().__init__()
        self.calls = 0
        self.call_budget = call_budget

    @property
    def provider_name(self) -> str:
        return "empty"

    async def get_actions(self, observation_dict, instruction, **kwargs):
        self.calls += 1
        if self.calls > self.call_budget:
            raise AssertionError(
                f"policy was queried {self.calls} times: the empty chunk is not being "
                "refused, so the loop is re-querying without ever advancing a step"
            )
        return []


class _FixedChunkPolicy(MockPolicy):
    """Emits a fixed-length chunk and records the delay it was told to expect."""

    def __init__(self, chunk_len: int = 4) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.declared_delays: list[int | None] = []

    @property
    def provider_name(self) -> str:
        return "fixed"

    async def get_actions(self, observation_dict, instruction, **kwargs):
        self.declared_delays.append(self.rtc_observed_delay_steps)
        return [{k: 0.0 for k in self.robot_state_keys} for _ in range(self.chunk_len)]


class _SixStepSpec(BenchmarkProtocol):
    max_steps = 6

    @property
    def supported_robots(self) -> list[str]:
        # Compatibility is checked against the robot's data_config, not the
        # name it was added to the sim under.
        return ["so100"]

    @property
    def default_robot(self) -> str:
        return "so100"

    def on_step(self, sim, obs, action) -> StepInfo:
        return StepInfo(reward=1.0)

    def is_success(self, sim) -> bool:
        return False


# --------------------------------------------------------------- structure


def test_module_has_exactly_one_chunk_query_and_one_executor():
    """One acquisition seam, one prefetch worker, one policy call site.

    The three greps that made the duplication visible. ``policy.get_actions``
    appearing once is the strongest of the three: a fourth hand-rolled copy
    cannot be added without failing here.
    """
    tree = ast.parse(_SOURCE)
    query_defs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.endswith("_query_chunk")]
    assert query_defs == [], f"nested chunk-query copies are back: {query_defs}"
    assert _SOURCE.count("ThreadPoolExecutor(") == 1, "the prefetch executor is built in more than one place"
    # AST, not grep: two docstrings legitimately QUOTE ``policy.get_actions(...)``
    # while documenting the forwarding contract, and a text count reads those as
    # call sites.
    call_sites = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_actions"
    ]
    assert len(call_sites) == 1, f"the policy is invoked outside query_policy_chunk, at lines {call_sites}"


def test_run_and_evaluate_drive_the_same_pipeline_class():
    """Both rollout entry points construct ``_ChunkPipeline``.

    ``run()`` re-implemented the prefetch loop inline, so an async-RTC fix
    applied to the pipeline silently missed it.
    """
    for method in (PolicyRunner.run, PolicyRunner.evaluate):
        source = inspect.getsource(method)
        assert "_ChunkPipeline(" in source, f"{method.__name__} does not use the shared pipeline"


# ------------------------------------------------- the empty-chunk divergence


@pytest.mark.parametrize("async_rtc", [False, True])
def test_run_refuses_an_empty_chunk_on_both_paths(sim_with_robot, async_rtc):
    """A rollout with no actions is refused after ONE query, not spun on.

    Pre-fix the ``async_rtc=False`` case never terminated: the synchronous loop
    had no empty-chunk guard, so ``step_count`` stayed 0 while the policy was
    re-queried forever. The async path already refused, from the copy that had
    the guard.
    """
    policy = _EmptyChunkPolicy()
    policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

    result = PolicyRunner(sim_with_robot).run(
        "alice", policy, n_steps=4, control_frequency=50, fast_mode=True, async_rtc=async_rtc
    )

    assert result["status"] == "error"
    assert "empty action chunk" in result["content"][0]["text"]
    assert policy.calls == 1, "the rollout must give up on the first empty chunk, not retry"


# --------------------------------------------------- the telemetry divergence


@pytest.mark.parametrize("async_rtc", [False, True])
def test_run_counts_acquired_chunks_on_both_paths(sim_with_robot, async_rtc):
    """``rtc_chunks_acquired`` reports the chunks a rollout actually acquired.

    A chunk is acquired on BOTH paths, but only the async copy counted, so a
    synchronous rollout reported 0 chunks beside a non-zero average inference
    time - self-contradictory telemetry in a documented result field. 12 steps
    of a 4-action chunk is exactly 3 acquisitions, and the pipeline must not
    pull a 4th chunk it can never apply.
    """
    policy = _FixedChunkPolicy(chunk_len=4)
    policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

    result = PolicyRunner(sim_with_robot).run(
        "alice",
        policy,
        n_steps=12,
        control_frequency=50,
        action_horizon=4,
        fast_mode=True,
        async_rtc=async_rtc,
    )

    payload = _payload(result)
    assert result["status"] == "success"
    assert payload["n_steps"] == 12
    assert payload["rtc_chunks_acquired"] == 3
    assert payload["rtc_avg_inference_ms"] > 0.0


def test_run_reports_the_chunks_a_failed_rollout_had_acquired(sim_with_robot):
    """The error payload carries the telemetry too, read off the same pipeline.

    The counters live on the pipeline instance rather than in local tallies, so
    there is one source of truth for the success and the error payload.
    """

    class _Boom(_FixedChunkPolicy):
        async def get_actions(self, observation_dict, instruction, **kwargs):
            if self.declared_delays:
                raise RuntimeError("policy exploded on the second chunk")
            return await super().get_actions(observation_dict, instruction, **kwargs)

    policy = _Boom(chunk_len=2)
    policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

    result = PolicyRunner(sim_with_robot).run(
        "alice", policy, n_steps=8, control_frequency=50, action_horizon=2, fast_mode=True
    )

    payload = _payload(result)
    assert result["status"] == "error"
    assert payload["stopped_reason"] == "error"
    assert payload["rtc_chunks_acquired"] == 1, "a rollout that died mid-flight still acquired one chunk"


# ------------------------------------------------- the RTC delay declaration


def test_spec_eval_declares_a_zero_delay_instead_of_inheriting_a_stale_one(sim_with_robot):
    """A synchronous spec eval tells the policy the delay is exactly 0.

    ``rtc_observed_delay_steps`` is persistent instance state. The benchmark
    loop was the one rollout path that never declared it, so a policy handed
    over from an async rollout kept slicing its chunk seam against that old
    count on a loop where the world advances 0 steps during inference.
    """
    policy = _FixedChunkPolicy(chunk_len=1)
    policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
    policy.set_rtc_observed_delay(7)  # left behind by a previous async rollout

    result = PolicyRunner(sim_with_robot).evaluate(
        "alice", policy, spec=_SixStepSpec(), n_episodes=1, max_steps=6, action_horizon=1
    )

    assert result["status"] == "success"
    assert policy.declared_delays, "the spec loop must have queried the policy"
    assert set(policy.declared_delays) == {0}, f"stale seam offset reached the policy: {policy.declared_delays}"


def test_query_helper_never_truncates_below_the_policys_own_chunk(sim_with_robot):
    """``action_horizon`` raises to the policy's chunk size, never cuts below it.

    The one place this is decided now, so a model trained for N-step open-loop
    replay cannot have its chunk clipped by one entry point and not another.
    """
    policy = _FixedChunkPolicy(chunk_len=6)
    policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
    policy.actions_per_step = 6

    chunk = query_policy_chunk(
        policy,
        {"1": 0.0},
        0,
        instruction="",
        policy_kwargs={},
        action_horizon=2,
    )

    assert len(chunk) == 6, "a 6-action chunk must survive a smaller action_horizon"
    assert policy.declared_delays == [0]
