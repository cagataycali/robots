"""Tearing the world down while a motion primitive runs aborts it, never raises.

``move_to`` / ``set_gripper`` / ``rotate_wrist`` run on their CALLER's thread:
no Future awaits them, and the ``policy_running`` flag ``cleanup`` signals does
not apply to them (a primitive refuses to start while a policy runs, so it never
sets that flag). ``self._lock`` - taken per control tick - is their only
synchronisation against another thread reshaping the world.

Each primitive documents "Never raises.", and
``_primitive_abort_reason`` documents the intended outcome: "the world can
legitimately be destroyed, the model recompiled, or a policy started while a
primitive runs. Each of those aborts the primitive with a structured error
rather than stepping a stale/contended model."

These tests pin that contract for the teardown path, which handed ``self._world``
off without holding the lock: the handoff landed inside a tick's physics substep
loop and the tick's write-back of ``sim_time`` raised ``AttributeError`` out of
all three primitives.

The interleave is forced rather than raced: a ``mj_step`` hook starts the
teardown from another thread at the exact production window - inside the substep
loop, after the tick's own ``assert self._world is not None``. Without a locked
handoff that window is fatal every time; with one, the teardown either lands
between ticks (structured abort) or after the primitive finishes (normal result).
Both are correct, so these tests assert the contract - the call returns a result
instead of raising - rather than which of the two the scheduler picked. The lock
itself is pinned directly in :class:`TestWorldHandoffLock`, and the structured
abort text in :class:`TestPolicyStartedMidRun`.
"""

import threading
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

# Long enough that the servo cannot converge before the hook fires, so every
# primitive is genuinely mid-flight when the world is taken away.
_LONG_RUN = {"max_steps": 4000}
# Looser than the IK residual this 4-DOF arm leaves on REACHABLE, so ``move_to``
# enters its servo loop instead of refusing the target up front.
_REACHABLE_TOL = 0.002


@pytest.fixture
def sim(tmp_path):
    path = tmp_path / "prim_arm.xml"
    path.write_text(ARM_XML)
    s = Simulation(tool_name="test_primitive_teardown", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert s.add_robot("arm", urdf_path=str(path))["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=2.0)


def _text(result: dict[str, Any]) -> str:
    return " ".join(b["text"] for b in result["content"] if "text" in b)


class _MjHook:
    """Delegate to the real ``mujoco`` module, firing a hook inside ``mj_step``.

    ``mj_step`` is where a primitive control tick spends its time, so a hook
    installed here runs at the same point a concurrent teardown would land in
    production: after the tick's own world check, before it writes ``sim_time``
    back to the world.
    """

    def __init__(self, mj: Any, hook: Any, at_call: int) -> None:
        self._mj = mj
        self._hook = hook
        self._at_call = at_call
        self._calls = 0
        self.fired = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mj, name)

    def mj_step(self, model: Any, data: Any) -> Any:
        self._calls += 1
        if self._calls == self._at_call:
            self.fired = True
            self._hook()
        return self._mj.mj_step(model, data)


def _teardown_from_another_thread(s: Simulation, join_timeout: float = 0.75) -> list[threading.Thread]:
    """Install a hook that calls ``cleanup()`` from a second thread mid-tick.

    The hook joins that thread with a bounded timeout: without a locked handoff
    the teardown needs nothing this tick holds, so it completes and the tick then
    writes into a nulled world; with a locked handoff the join times out because
    the teardown is blocked on the lock this tick holds, and it lands only once
    the tick has finished.
    """
    threads: list[threading.Thread] = []

    def hook() -> None:
        t = threading.Thread(target=lambda: s.cleanup(policy_stop_timeout=0.1), daemon=True)
        threads.append(t)
        t.start()
        t.join(timeout=join_timeout)

    s._mj = _MjHook(s._mj, hook, at_call=3)
    return threads


class TestTeardownDuringAPrimitive:
    """A world torn down mid-primitive returns a result; it never raises.

    Pre-fix every one of these raised
    ``AttributeError: 'NoneType' object has no attribute 'sim_time'`` from the
    control tick's write-back, out of three methods documented "Never raises."
    """

    def _assert_returned_a_result(self, sim, result, threads):
        for t in threads:
            t.join(timeout=30.0)
        assert result["status"] in ("success", "error")
        if result["status"] == "error":
            # The teardown won the lock between two ticks: the documented abort.
            assert "aborting" in _text(result) or "did not reach" in _text(result)
        assert sim._world is None

    def test_move_to_returns_a_result_instead_of_raising(self, sim):
        pytest.importorskip("mink")
        threads = _teardown_from_another_thread(sim)

        result = sim.move_to(robot_name="arm", position=REACHABLE, tol=_REACHABLE_TOL, **_LONG_RUN)

        self._assert_returned_a_result(sim, result, threads)

    def test_set_gripper_returns_a_result_instead_of_raising(self, sim):
        threads = _teardown_from_another_thread(sim)

        result = sim.set_gripper(robot_name="arm", state="close", steps=_LONG_RUN["max_steps"])

        self._assert_returned_a_result(sim, result, threads)

    def test_rotate_wrist_returns_a_result_instead_of_raising(self, sim):
        threads = _teardown_from_another_thread(sim)

        result = sim.rotate_wrist(robot_name="arm", target_yaw=1.2, tol=_REACHABLE_TOL, **_LONG_RUN)

        self._assert_returned_a_result(sim, result, threads)


class TestWorldHandoffLock:
    """``cleanup`` hands the world off under the per-tick lock, but bounded."""

    def test_handoff_waits_for_a_control_tick_to_finish(self, sim):
        """The world survives until the lock a tick holds is released."""
        done = threading.Event()

        with sim._lock:
            worker = threading.Thread(target=lambda: (sim.cleanup(policy_stop_timeout=0.1), done.set()), daemon=True)
            worker.start()
            # Cannot observe "blocked" directly; what matters is that the world
            # is still intact for as long as a tick could be using it.
            assert not done.wait(timeout=0.5)
            assert sim._world is not None

        assert done.wait(timeout=30.0)
        worker.join(timeout=30.0)
        assert sim._world is None

    def test_teardown_does_not_hang_when_the_lock_is_never_released(self, sim, monkeypatch, caplog):
        """A wedged lock holder must not hang the host process on exit.

        The same tradeoff step 2 makes for a policy worker that ignores the
        cooperative-stop flag: warn, then tear down regardless.
        """
        monkeypatch.setattr(type(sim), "_WORLD_HANDOFF_LOCK_TIMEOUT", 0.2)
        holder_has_lock = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with sim._lock:
                holder_has_lock.set()
                release.wait(timeout=30.0)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        assert holder_has_lock.wait(timeout=10.0)
        try:
            finished = threading.Event()
            worker = threading.Thread(
                target=lambda: (sim.cleanup(policy_stop_timeout=0.1), finished.set()), daemon=True
            )
            worker.start()
            assert finished.wait(timeout=30.0), "cleanup hung on the world-handoff lock"
            worker.join(timeout=30.0)
            assert sim._world is None
            assert "world-handoff lock still held" in caplog.text
        finally:
            release.set()
            holder.join(timeout=30.0)


class TestPolicyStartedMidRun:
    """The other documented mid-run abort: a policy claims the robot."""

    def test_primitive_aborts_when_a_policy_claims_the_robot(self, sim):
        """``policy_running`` is what ``start_policy`` sets and the primitive reads."""

        def claim() -> None:
            sim._world.robots["arm"].policy_running = True

        sim._mj = _MjHook(sim._mj, claim, at_call=3)

        result = sim.set_gripper(robot_name="arm", state="close", steps=4000)

        assert result["status"] == "error"
        text = _text(result)
        assert "a policy started on 'arm' mid-run" in text
        assert "stop_policy" in text
