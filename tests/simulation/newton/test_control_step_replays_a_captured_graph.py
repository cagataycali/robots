"""A Newton control step is one captured CUDA graph replay, re-captured whenever a buffer it holds changes.

Launching a solver step's kernels one at a time from Python dominated the
step: 81.7 ms per 10-substep control step on an L40S against 4.1 ms as a graph
replay, so ``run_policy`` ran at about 2 percent of real time. The graph holds
device pointers, so the cells below pin the rules that keep a replay equal to
the loop it replaced: a new key (and a new capture) for any new buffer, two
alternating graphs for an odd substep count, the Python loop on CPU, on
opt-out, or when capture fails, and joint targets written in place.

The first class runs anywhere against a stand-in Warp; the last cell steps the
real solver both ways and is skipped without Newton on a CUDA device.
"""

from __future__ import annotations

import importlib.util
import types

import pytest

from strands_robots.simulation.newton.simulation import NewtonSimEngine


class _Capture:
    def __init__(self, wp, device=None):
        self.wp = wp

    def __enter__(self):
        self.wp.capturing = True
        return self

    def __exit__(self, *exc):
        self.wp.capturing = False
        if self.wp.fail_capture:
            raise RuntimeError("host sync during capture")
        self.graph = object()
        self.wp.captures += 1
        return False


class _Wp:
    def __init__(self, cuda=True, fail_capture=False):
        self.cuda, self.fail_capture = cuda, fail_capture
        self.captures = self.launches = 0
        self.capturing = False

    def get_device(self, _device):
        return types.SimpleNamespace(is_cuda=self.cuda)

    def ScopedCapture(self, device=None):  # noqa: N802 - Warp's spelling
        return _Capture(self, device)

    def capture_launch(self, _graph):
        self.launches += 1


class SolverMuJoCo:  # noqa: D101 - named like the solver the allowlist admits
    def __init__(self, step):
        self.step = step


class SolverKamino(SolverMuJoCo):  # noqa: D101
    pass


class _Engine:
    """Only what the two methods read, bound to the real implementations."""

    _launch_control_step_graph = NewtonSimEngine._launch_control_step_graph
    _run_substeps = NewtonSimEngine._run_substeps

    def __init__(self, wp, substeps=10):
        self._wp = wp
        self.substeps = substeps
        self._model = types.SimpleNamespace(device="cuda:0", gravity=object())
        self._solver = SolverMuJoCo(self._solver_step)
        self._state_0 = types.SimpleNamespace(clear_forces=lambda: None, name="A")
        self._state_1 = types.SimpleNamespace(clear_forces=lambda: None, name="B")
        self._control = types.SimpleNamespace(joint_target_q=object())
        self.python_steps = 0

    def _solver_step(self, *_a):
        if not self._wp.capturing:
            self.python_steps += 1


def test_the_first_step_captures_and_the_next_replays_without_recapturing(monkeypatch):
    monkeypatch.delenv("STRANDS_NEWTON_CUDA_GRAPH", raising=False)
    wp = _Wp()
    eng = _Engine(wp)
    for _ in range(5):
        assert eng._launch_control_step_graph(0.001) is True
    assert (wp.captures, wp.launches, eng.python_steps) == (1, 5, 0)
    assert eng._state_0.name == "A"  # an even substep count ends where it began


def test_an_odd_substep_count_alternates_two_graphs_and_the_state_roles(monkeypatch):
    monkeypatch.delenv("STRANDS_NEWTON_CUDA_GRAPH", raising=False)
    wp = _Wp()
    eng = _Engine(wp, substeps=7)
    names = []
    for _ in range(6):
        eng._launch_control_step_graph(0.001)
        names.append(eng._state_0.name)
    assert names == ["B", "A", "B", "A", "B", "A"]
    assert wp.captures == 2 and wp.launches == 6


@pytest.mark.parametrize(
    "change",
    [
        lambda e: setattr(e._control, "joint_target_q", object()),
        lambda e: setattr(e._model, "gravity", object()),
        lambda e: setattr(e, "_state_0", types.SimpleNamespace(clear_forces=lambda: None, name="C")),
    ],
    ids=["new target array", "new gravity array", "new state"],
)
def test_a_new_buffer_is_a_new_capture(monkeypatch, change):
    monkeypatch.delenv("STRANDS_NEWTON_CUDA_GRAPH", raising=False)
    wp = _Wp()
    eng = _Engine(wp)
    eng._launch_control_step_graph(0.001)
    change(eng)
    eng._launch_control_step_graph(0.001)
    assert wp.captures == 2


def test_a_new_solver_drops_every_graph_of_the_old_one(monkeypatch):
    monkeypatch.delenv("STRANDS_NEWTON_CUDA_GRAPH", raising=False)
    wp = _Wp()
    eng = _Engine(wp, substeps=7)
    eng._launch_control_step_graph(0.001)
    eng._launch_control_step_graph(0.001)
    eng._solver = SolverMuJoCo(eng._solver_step)
    eng._launch_control_step_graph(0.001)
    assert len(eng._step_graphs) == 1


def test_a_new_dt_is_a_new_capture(monkeypatch):
    monkeypatch.delenv("STRANDS_NEWTON_CUDA_GRAPH", raising=False)
    wp = _Wp()
    eng = _Engine(wp)
    eng._launch_control_step_graph(0.001)
    eng._launch_control_step_graph(0.0025)
    assert wp.captures == 2


@pytest.mark.parametrize("reason", ["cpu", "opt-out", "capture fails"])
def test_the_python_loop_runs_when_no_graph_can(monkeypatch, reason):
    monkeypatch.delenv("STRANDS_NEWTON_CUDA_GRAPH", raising=False)
    wp = _Wp(cuda=reason != "cpu", fail_capture=reason == "capture fails")
    if reason == "opt-out":
        monkeypatch.setenv("STRANDS_NEWTON_CUDA_GRAPH", "0")
    eng = _Engine(wp)
    assert eng._launch_control_step_graph(0.001) is False
    assert eng._launch_control_step_graph(0.001) is False
    assert wp.launches == 0
    assert wp.captures == 0
    # A failed capture restores the state roles the loop recorded under.
    assert eng._state_0.name == "A"


def test_a_solver_whose_replay_drifts_steps_from_python(monkeypatch):
    """Kamino's replay drifted from its loop (2.4e-3 rad in 120 steps), so it is not captured."""
    monkeypatch.delenv("STRANDS_NEWTON_CUDA_GRAPH", raising=False)
    wp = _Wp()
    eng = _Engine(wp)
    eng._solver = SolverKamino(eng._solver_step)
    assert eng._launch_control_step_graph(0.001) is False
    assert wp.captures == 0


def test_joint_targets_are_written_into_the_array_the_graph_holds():
    source = importlib.util.find_spec("strands_robots.simulation.newton.simulation").origin
    text = open(source, encoding="utf-8").read()
    body = text[text.index("def _write_targets") : text.index("def _apply_mjcf_servo_gains")]
    assert "joint_target_q.assign(" in body
    assert "joint_target_q = " not in body


_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


def _cuda() -> bool:
    if not _HAS_NEWTON:
        return False
    import warp as wp

    try:
        return wp.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cuda(), reason="needs Newton on a CUDA device")
@pytest.mark.parametrize(("solver", "substeps"), [("mujoco", 10), ("mujoco", 7), ("featherstone", 10)])
def test_the_graph_replay_is_the_python_loop_bit_for_bit(monkeypatch, solver, substeps):
    import numpy as np

    def run(flag: str) -> np.ndarray:
        monkeypatch.setenv("STRANDS_NEWTON_CUDA_GRAPH", flag)
        sim = NewtonSimEngine(solver=solver, substeps=substeps)
        try:
            sim.create_world()
            sim.add_robot("so100")
            out = []
            for k in range(24):
                sim.send_action({"Rotation": 0.4 * np.sin(k / 6), "Pitch": 0.3}, robot_name="so100")
                if k == 6:
                    sim.set_gravity([0, 0, -3.0])
                if k == 12:
                    sim.add_object("cube", shape="box", position=[0.25, 0, 0.05], size=[0.015] * 3)
                if k == 18:
                    sim.set_timestep(1 / 400)
                sim.step(2)
                out.append(sim._state_0.joint_q.numpy()[:6].copy())
            return np.array(out)
        finally:
            sim.destroy()

    graph, loop = run("1"), run("0")
    assert np.abs(loop[-1] - loop[0]).max() > 0.05  # the arm moved
    assert np.array_equal(graph, loop)
