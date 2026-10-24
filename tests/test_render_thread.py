"""Every MuJoCo render runs on one GL thread: a context is bound to its thread (macOS hangs otherwise)."""

import threading

import pytest

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


def test_a_base_exception_on_the_gl_thread_reaches_the_caller_and_ends_the_worker():
    from strands_robots import _gl

    class Stop(BaseException):  # not an Exception: the worker hands it over, then stops instead of swallowing it
        pass

    def boom() -> None:
        raise Stop()

    worker = _gl._GLThread()
    try:
        with pytest.raises(Stop):
            worker.submit(boom).result(5)
        worker.t.join(5)
        assert not worker.t.is_alive()
        with pytest.raises(RuntimeError, match="GL thread has stopped"):
            worker.submit(lambda: None)
    finally:
        worker.q.put(None)


def test_a_script_that_rendered_exits_without_egl_errors():
    """mujoco.egl registers ``eglTerminate`` at exit when it makes its first context; the renderers must be freed
    before it runs, or every rendering script ends with ``EGL_NOT_INITIALIZED`` tracebacks on stderr."""
    import ctypes
    import subprocess
    import sys

    try:
        ctypes.CDLL("libEGL.so.1")
    except OSError:
        pytest.skip("no libEGL on this host")
    script = (
        "from strands_robots import Robot\n"
        "arm = Robot('so101')\n"
        "arm.sim.camera('front', position=[0.6, 0, 0.4], target=[0, 0, 0.2], size=(64, 48))\n"
        "assert arm.sim.render('front').shape == (48, 64, 3)\n"
    )
    env = {**__import__("os").environ, "MUJOCO_GL": "egl"}
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "EGL" not in out.stderr and "Exception ignored" not in out.stderr, out.stderr
