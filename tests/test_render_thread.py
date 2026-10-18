"""Every MuJoCo render runs on one GL thread: a context is bound to its thread (macOS hangs otherwise)."""

import threading

from strands_robots import Robot


def test_renders_from_any_thread_run_on_the_one_gl_thread(monkeypatch):
    from strands_robots.simulation.backends.mujoco import backend

    mujoco = backend.mujoco  # the module the backend renders with, even after a test re-imported the package
    made: list[str] = []
    rendered: list[str] = []
    real = mujoco.Renderer

    class Spy(real):
        def __init__(self, *a, **kw):
            made.append(threading.current_thread().name)
            super().__init__(*a, **kw)

        def render(self, *a, **kw):
            rendered.append(threading.current_thread().name)
            return super().render(*a, **kw)

    monkeypatch.setattr(mujoco, "Renderer", Spy)
    arm = Robot("so101")
    arm.sim.camera("front", position=[0.6, 0, 0.4], target=[0, 0, 0.2], size=(64, 48), attach=arm)
    arm.run(lambda o: o.joints.positions, steps=2)  # the main thread renders first
    task = arm.start(lambda o: o.joints.positions, steps=2)  # then a background Task: it used to hang on macOS
    task._thread.join(10)
    assert task.result is not None and task.result.steps == 2
    other = threading.Thread(target=lambda: arm.sim.render("front"))
    other.start()
    other.join(10)
    assert len(rendered) >= 3 and len(set(made)) == 1 and set(rendered) == set(made)
    assert made[0].startswith("strands-gl")
    arm.close()
