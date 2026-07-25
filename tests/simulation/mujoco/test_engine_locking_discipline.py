"""Locking-discipline regression tests for the MuJoCo simulation engine.

Each test pins a concurrency defect that races the agent-dispatch thread
against a PolicyRunner worker or the recorder/render daemon:

* ``remove_robot`` dispatched during an active policy must return promptly
  (no swallowed cooperative-stop timeout) and let the worker exit cleanly,
  instead of stalling on a join held under the dispatch lock and then
  rebuilding the scene under a still-live worker.
* A long-running dispatched ``step`` must not block ``stop_policy`` (or any
  other lightweight action): ``step`` releases the engine lock between
  batches and ``stop_policy`` runs without the blanket dispatch lock.
* A failed ``load_scene`` must leave the previously-live world intact.
* ``move_object`` must mutate ``qpos``/``qvel`` (and recompile static bodies)
  under the engine lock, like every sibling mutator.
* ``render`` must read ``mjData`` under the engine lock so a concurrent
  ``mj_step`` cannot tear the frame or crash on ``data.contact``.
"""

import threading
import time

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (headless without EGL/OSMesa)",
)

_ARM_XML = """
<mujoco model="lock_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def arm_xml_path(tmp_path):
    path = tmp_path / "lock_arm.xml"
    path.write_text(_ARM_XML)
    return str(path)


@pytest.fixture
def sim_with_arm(arm_xml_path):
    sim = Simulation(tool_name="test_locking", mesh=False)
    assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert sim.add_robot("arm1", urdf_path=arm_xml_path)["status"] == "success"
    yield sim
    sim.cleanup(policy_stop_timeout=1.0)


class _SlowPolicy:
    """A trivial policy whose per-step inference sleeps, keeping the worker
    demonstrably alive while ``remove_robot`` races it."""

    provider_name = "slow_test"
    requires_images = False

    def set_robot_state_keys(self, keys):  # noqa: D401 - test stub
        pass

    def reset(self, seed=None):
        pass

    def get_actions_sync(self, observation, instruction):
        time.sleep(0.05)
        return {"shoulder_pan": 0.1}

    async def get_actions(self, observation, instruction):
        return {"shoulder_pan": 0.1}


def test_remove_robot_during_active_policy_returns_and_worker_exits_clean(sim_with_arm, arm_xml_path):
    """remove_robot dispatched while a policy runs returns promptly.

    Pre-fix, the dispatch path held the reentrant engine lock across the
    cooperative-stop join; the worker (which needs that same lock to observe
    the stop flag) could never exit, so the join hit its 5s timeout, the
    TimeoutError was swallowed, and the scene was rebuilt under a live worker
    holding stale model/data ids. The dispatched call therefore stalled ~5s
    and left the worker crashing on freed state.

    Post-fix, remove_robot runs without the blanket lock, so the worker sees
    the stop flag and exits cleanly and the call returns near-instantly.
    """
    sim = sim_with_arm
    started = sim.start_policy(robot_name="arm1", policy_object=_SlowPolicy(), duration=30.0, control_frequency=50.0)
    assert started["status"] == "success"
    # Let the worker enter its control loop.
    time.sleep(0.3)
    assert "arm1" in sim._policy_threads

    # Dispatch through __call__ -> _dispatch_action (the path that used to
    # hold the blanket lock across the join).
    start = time.monotonic()
    result = sim(action="remove_robot", name="arm1")
    elapsed = time.monotonic() - start

    assert result["status"] == "success", result
    # The swallowed timeout was 5s; anything under 3s proves no deadlock/stall.
    assert elapsed < 3.0, f"remove_robot stalled {elapsed:.2f}s (cooperative-stop deadlock)"
    assert "arm1" not in sim.list_robots()

    # Clean scene integrity: the worker unwound without corrupting the rebuilt
    # scene, so the same name can be re-added and the engine still steps.
    re_added = sim.add_robot("arm1", urdf_path=arm_xml_path)
    assert re_added["status"] == "success", re_added
    assert sim.step(n_steps=1)["status"] == "success"


def test_long_running_step_does_not_block_stop_policy(sim_with_arm, monkeypatch):
    """A dispatched multi-batch step must not gate a concurrent stop_policy.

    Pre-fix, step ran entirely under the blanket dispatch lock, so any other
    dispatched action (here stop_policy) blocked until the whole step
    finished. Post-fix, step releases the lock between batches and stop_policy
    dispatches without the blanket lock, so it returns while step is still
    running.
    """
    sim = sim_with_arm
    real_mj = sim._mj

    class _SlowMj:
        """Delegates to the real mujoco module but sleeps in mj_step so a
        multi-batch step() takes a measurable, deterministic wall-time."""

        def mj_step(self, model, data):
            time.sleep(0.003)
            return real_mj.mj_step(model, data)

        def __getattr__(self, name):
            return getattr(real_mj, name)

    monkeypatch.setattr(sim, "_mj", _SlowMj())

    step_done = threading.Event()
    timings = {}

    def _long_step():
        t0 = time.monotonic()
        # > _STEPS_PER_BATCH (1000) so the batched lock-release path is taken.
        sim(action="step", n_steps=1500)
        timings["step"] = time.monotonic() - t0
        step_done.set()

    stepper = threading.Thread(target=_long_step)
    stepper.start()
    try:
        time.sleep(0.2)  # ensure step is underway
        t0 = time.monotonic()
        stop_result = sim(action="stop_policy", robot_name="arm1")
        stop_elapsed = time.monotonic() - t0
        step_running_when_stop_returned = not step_done.is_set()
    finally:
        step_done.wait(30.0)
        stepper.join(timeout=30.0)

    assert stop_result["status"] == "success"
    # The killer discriminator: pre-fix, stop_policy only returns AFTER step
    # finishes (blanket lock), so step is no longer running. Post-fix it
    # returns mid-step.
    assert step_running_when_stop_returned, "stop_policy blocked until step finished (blanket lock)"
    assert stop_elapsed < 1.0, f"stop_policy took {stop_elapsed:.2f}s while step ran {timings.get('step')}s"


def test_load_scene_failure_preserves_live_world(sim_with_arm, tmp_path):
    """A malformed MJCF must not destroy the currently-live world.

    Pre-fix, load_scene reassigned self._world to a fresh empty SimWorld
    before compiling, so a bad file discarded the live scene AND returned an
    error dict. Post-fix, the scene compiles into locals first and self._world
    is only swapped on success.
    """
    sim = sim_with_arm
    live_model = sim._world._model
    assert "arm1" in sim._world.robots

    bad = tmp_path / "bad.xml"
    bad.write_text("<mujoco><worldbody><geom type='box' size='oops'/></worldbody></mujoco>")

    result = sim(action="load_scene", scene_path=str(bad))
    assert result["status"] == "error"
    # World intact: same live model object, robot still present, still steppable.
    assert sim._world is not None
    assert sim._world._model is live_model
    assert "arm1" in sim._world.robots
    assert sim.step(n_steps=1)["status"] == "success"


def test_load_scene_success_still_swaps_world(sim_with_arm, arm_xml_path):
    """The happy path still replaces the world (guards against over-correction)."""
    sim = sim_with_arm
    result = sim(action="load_scene", scene_path=arm_xml_path)
    assert result["status"] == "success"
    assert sim._world is not None and sim._world._model is not None
    assert sim._world._backend_state.get("scene_loaded") is True


def test_move_object_mutates_under_engine_lock(sim_with_arm):
    """move_object must hold self._lock while writing qpos/qvel + mj_forward.

    Deterministic proof: hold the engine lock on the main thread, then invoke
    move_object from a worker thread. Post-fix the worker blocks until the
    lock is released; pre-fix (no lock) it completes immediately.
    """
    sim = sim_with_arm
    added = sim.add_object("ball", shape="sphere", position=[0.2, 0.0, 0.5], size=[0.05], is_static=False)
    assert added["status"] == "success"

    done = threading.Event()

    sim._lock.acquire()
    try:
        worker = threading.Thread(target=lambda: (sim.move_object("ball", position=[0.3, 0.1, 0.5]), done.set()))
        worker.start()
        # Blocked because a different thread holds the RLock.
        assert not done.wait(0.4), "move_object did not serialize under the engine lock"
    finally:
        sim._lock.release()

    assert done.wait(3.0), "move_object never completed after lock release"
    worker.join(timeout=3.0)


@requires_gl
def test_render_reads_mjdata_under_engine_lock(sim_with_arm):
    """render must read mjData under self._lock so a concurrent mj_step cannot
    tear the frame or crash on data.contact.

    Deterministic proof mirrors move_object: hold the lock, render from a
    worker thread, assert it blocks until release.
    """
    sim = sim_with_arm
    result_box = {}
    done = threading.Event()

    sim._lock.acquire()
    try:
        worker = threading.Thread(
            target=lambda: (
                result_box.__setitem__("r", sim.render(camera_name="default", width=64, height=64)),
                done.set(),
            )
        )
        worker.start()
        assert not done.wait(0.4), "render did not serialize its mjData read under the engine lock"
    finally:
        sim._lock.release()

    assert done.wait(10.0), "render never completed after lock release"
    worker.join(timeout=10.0)
    assert result_box["r"]["status"] == "success"


@requires_gl
def test_concurrent_render_and_step_do_not_crash(sim_with_arm):
    """Smoke: a render loop concurrent with a stepping loop must not SIGSEGV
    or raise (both now serialize on the engine lock)."""
    sim = sim_with_arm
    errors = []
    stop = threading.Event()

    def _render_loop():
        try:
            while not stop.is_set():
                r = sim.render(camera_name="default", width=64, height=64)
                if r["status"] != "success":
                    errors.append(f"render: {r}")
                    return
        except Exception as e:  # noqa: BLE001 - surface any crash as a failure
            errors.append(f"render raised: {e}")

    renderer = threading.Thread(target=_render_loop)
    renderer.start()
    try:
        for _ in range(50):
            sim.step(n_steps=5)
    finally:
        stop.set()
        renderer.join(timeout=10.0)
    assert not errors, errors
