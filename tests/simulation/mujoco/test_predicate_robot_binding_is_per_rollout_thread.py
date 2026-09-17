"""The predicate-robot binding is scoped to the thread driving a rollout, not to the scene.

``bind_predicate_robot`` makes an unnamed ``base_*`` clause read the robot under
evaluation. Rollouts are per-robot and explicitly concurrent - ``start_policy``
submits each to the engine's executor and "policies on different robots can
execute concurrently" is a documented surface - so a binding kept as ONE
scene-wide attribute made the last bind win: the moment a second rollout (or a
second call that was then refused) bound its robot, the first rollout's unnamed
clauses, evaluated every step through ``_bound_robot(sim, None)``, read the
second robot for the rest of the episode, silently, under ``status=success``.
That is the defect class the binding exists to remove, reintroduced through a
supported concurrency surface, and with a new mode the pre-binding code did not
have: a mid-episode flip that blends two robots' signals into one verdict.

Pinned: the binding is per thread - two threads each bind their own robot and
each reads its own; a bind on one thread is invisible on another; and, through
the real surfaces, a ``run_policy`` rollout on the robot that is NOT past the
line runs to its budget even though a second rollout on the robot that IS past
the line binds, fires and returns while the first is mid-episode.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.observers import RunPolicyStep


@pytest.fixture
def two_quadrupeds():
    """Two robots of the SAME floating-base model - the silent case.

    The fixed-base-arm refusal and the data_config compatibility check cannot
    tell these two apart, so a wrong-robot read here is caught by nothing but
    the binding's scope.
    """
    s = Simulation(tool_name="bind_per_thread", mesh=False)
    s.create_world()
    assert s.add_robot(name="go2_a", data_config="unitree_go2", position=[0.0, 0.0, 0.3])["status"] == "success"
    assert s.add_robot(name="go2_b", data_config="unitree_go2", position=[2.0, 0.0, 0.3])["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def _payload(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def test_a_bind_on_one_thread_is_not_seen_by_another(two_quadrupeds):
    sim = two_quadrupeds
    sim.bind_predicate_robot("go2_a")
    seen_on_other_thread: list[str | None] = []

    def other() -> None:
        seen_on_other_thread.append(sim.predicate_robot)
        sim.bind_predicate_robot("go2_b")
        seen_on_other_thread.append(sim.predicate_robot)

    t = threading.Thread(target=other)
    t.start()
    t.join(5.0)
    assert not t.is_alive()
    # The other thread started unbound, then saw only what it bound itself.
    assert seen_on_other_thread == [None, "go2_b"]
    # Its bind did not disturb this thread's binding.
    assert sim.predicate_robot == "go2_a"


def test_two_threads_each_read_the_robot_they_bound(two_quadrupeds):
    """Both threads hold a binding at once; each reads its own, not the last one written."""
    sim = two_quadrupeds
    both_bound = threading.Barrier(2, timeout=5.0)
    read_back: dict[str, str | None] = {}

    def bind_then_read(robot: str) -> None:
        sim.bind_predicate_robot(robot)
        both_bound.wait()  # the other thread has bound too - a scene-wide slot would now hold one name
        read_back[robot] = sim.predicate_robot

    threads = [threading.Thread(target=bind_then_read, args=(r,)) for r in ("go2_a", "go2_b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5.0)
        assert not t.is_alive()
    assert read_back == {"go2_a": "go2_a", "go2_b": "go2_b"}


def test_a_rollout_keeps_reading_its_own_robot_while_another_rollout_binds_and_fires(two_quadrupeds):
    """The behavioural pin, through the real surfaces and the real DSL clause.

    Rollout A drives ``go2_a`` (x=0) with ``stop_when = base_beyond_x(1.0)``; a
    mock policy does not walk a metre in 0.4 s, so A must run to its budget.
    Rollout B drives ``go2_b`` (x=2) with the same clause, which already holds,
    so B fires at once. The interleaving is forced, not hoped for: A's observer
    blocks at its first step until B has bound, run and returned - so B's bind
    lands while A is mid-episode, exactly the window a scene-wide slot loses.
    Under that slot A's later steps read ``go2_b`` (x=2 > 1) and A reports
    ``stopped_reason="predicate"`` for a robot that never moved.
    """
    sim = two_quadrupeds
    clause = {"predicate": "base_beyond_x", "x": 1.0}
    a_is_stepping = threading.Event()
    b_has_returned = threading.Event()

    def hold_a_until_b_returns(event: object) -> None:
        if isinstance(event, RunPolicyStep) and not a_is_stepping.is_set():
            a_is_stepping.set()
            assert b_has_returned.wait(10.0), "rollout B did not return while A was mid-episode"

    def rollout_a() -> dict:
        return sim.run_policy(
            robot_name="go2_a",
            policy_provider="mock",
            duration=0.4,
            control_frequency=50.0,
            stop_when=clause,
            observer=hold_a_until_b_returns,
        )

    def rollout_b() -> dict:
        assert a_is_stepping.wait(10.0), "rollout A never reached its first step"
        try:
            return sim.run_policy(
                robot_name="go2_b", policy_provider="mock", duration=0.4, control_frequency=50.0, stop_when=clause
            )
        finally:
            b_has_returned.set()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bind-per-thread") as pool:
        fut_a = pool.submit(rollout_a)
        fut_b = pool.submit(rollout_b)
        result_b = fut_b.result(timeout=30.0)
        result_a = fut_a.result(timeout=30.0)

    assert result_b["status"] == "success", _text(result_b)
    assert _payload(result_b)["stopped_reason"] == "predicate", _text(result_b)  # go2_b IS past x=1

    assert result_a["status"] == "success", _text(result_a)
    payload_a = _payload(result_a)
    # go2_a never crossed x=1: its clause must not have fired on go2_b's position.
    assert payload_a["stopped_reason"] == "budget", _text(result_a)
    assert payload_a["stopped_early"] is False, payload_a
