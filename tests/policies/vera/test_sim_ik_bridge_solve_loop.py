"""Dependency-free coverage of the VERA eef-delta -> MuJoCo IK bridge solve loop.

The real-solver accuracy regression in ``test_vera_sim_ik_solver.py``
``importorskip``s on the ``sim-mujoco`` extra (``mink`` + ``mujoco`` +
``qpsolvers``), so on the default clean-install image the whole
:class:`~strands_robots.policies.vera.sim_ik.MinkIKBridge` body is skipped - the
construction, forward kinematics, and the damped-least-squares solve loop.

The bridge only ever calls a small, *duck-typed* slice of ``mink``
(``Configuration`` / ``FrameTask`` / ``PostureTask`` / ``SE3`` / ``solve_ik``),
so the full solve loop is driven here against a fake ``mink`` module that models
a single-step exact solver. No ``mink``, ``mujoco`` or ``qpsolvers`` need be
installed - these contracts execute in plain CI and guard what a clean-install
user hits first:

* construction wires the frame + posture tasks and resolves the QP backend;
* ``ee_pose`` runs forward kinematics through the configuration;
* ``solve`` warm-starts at the seed, iterates ``solve_ik`` -> ``integrate`` and
  *breaks early* once both position and orientation error fall under threshold,
  and otherwise runs the full ``max_iters`` budget.
"""

import sys
import types

import numpy as np
import pytest

from strands_robots.policies.vera import sim_ik


class _FakeTransform:
    """Stand-in for a ``mink`` frame transform; ``as_matrix`` returns the pose."""

    def __init__(self, matrix: np.ndarray):
        self._m = np.asarray(matrix, dtype=float)

    def as_matrix(self) -> np.ndarray:
        return self._m


class _FakeSE3:
    """Minimal ``mink.SE3`` carrying the target homogeneous matrix."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> "_FakeSE3":
        return cls(m)


class _FakeConfiguration:
    """A ``mink.Configuration`` whose EE translation is simply ``q[:3]``.

    Makes forward kinematics deterministic and testable: ``ee_pose(q)`` returns
    an identity-rotation pose translated to the first three joint values. Counts
    ``integrate_inplace`` calls so the solve loop's iteration budget is
    observable.
    """

    def __init__(self, model):
        self.model = model
        self.q = np.zeros(model.nq, dtype=float)
        self.integrate_calls = 0

    def update(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, dtype=float).copy()

    def get_transform_frame_to_world(self, name: str, ftype: str) -> _FakeTransform:
        m = np.eye(4)
        m[:3, 3] = self.q[:3]
        return _FakeTransform(m)

    def integrate_inplace(self, velocity: np.ndarray, dt: float) -> None:
        self.integrate_calls += 1
        self.q = self.q + np.asarray(velocity, dtype=float) * dt


class _FakeFrameTask:
    """Cartesian frame task; ``compute_error`` is the target-minus-achieved delta."""

    def __init__(self, frame_name, frame_type, position_cost, orientation_cost, lm_damping):
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.target = None

    def set_target(self, target: _FakeSE3) -> None:
        self.target = np.asarray(target.matrix, dtype=float)

    def compute_error(self, config: _FakeConfiguration) -> np.ndarray:
        pos_err = self.target[:3, 3] - config.q[:3]
        return np.concatenate([pos_err, np.zeros(3)])  # orientation already aligned.


class _FakePostureTask:
    def __init__(self, model, cost):
        self.target = None

    def set_target(self, q: np.ndarray) -> None:
        self.target = np.asarray(q, dtype=float)


def _solve_ik(config, tasks, dt, solver, damping):
    """A one-step exact solver: velocity drives ``q[:3]`` onto the frame target."""
    frame_task = tasks[0]
    tgt = frame_task.target[:3, 3]
    v = np.zeros(config.model.nq, dtype=float)
    v[:3] = (tgt - config.q[:3]) / dt  # integrate adds v*dt -> reaches target exactly.
    return v


@pytest.fixture
def fake_mink(monkeypatch):
    """Install a fake ``mink`` module plus a stub ``qpsolvers`` for the bridge."""
    mod = types.ModuleType("mink")
    mod.Configuration = _FakeConfiguration
    mod.FrameTask = _FakeFrameTask
    mod.PostureTask = _FakePostureTask
    mod.SE3 = _FakeSE3
    mod.solve_ik = _solve_ik
    monkeypatch.setitem(sys.modules, "mink", mod)

    qp = types.ModuleType("qpsolvers")
    qp.available_solvers = ["quadprog"]
    monkeypatch.setitem(sys.modules, "qpsolvers", qp)
    return mod


def _model(nq: int = 7):
    return types.SimpleNamespace(nq=nq)


def _target_pose(xyz) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = xyz
    return pose


def _make_bridge(fake_mink, **kw):
    return sim_ik.MinkIKBridge(model=_model(), ee_frame_name="gripper", solver="quadprog", **kw)


def test_construction_wires_tasks_and_resolves_solver(fake_mink):
    """The bridge builds its frame + posture tasks and resolves the QP backend."""
    bridge = _make_bridge(fake_mink)

    assert bridge.solver == "quadprog"
    assert bridge.ee_frame_name == "gripper"
    assert isinstance(bridge._frame_task, _FakeFrameTask)
    assert isinstance(bridge._posture_task, _FakePostureTask)
    assert bridge._tasks == [bridge._frame_task, bridge._posture_task]


def test_ee_pose_runs_forward_kinematics_through_configuration(fake_mink):
    """``ee_pose`` returns the configuration's frame transform for the given q."""
    bridge = _make_bridge(fake_mink)
    q = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0], dtype=float)

    pose = bridge.ee_pose(q)

    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[:3, 3], [0.1, 0.2, 0.3], atol=1e-6)


def test_solve_breaks_early_once_under_threshold(fake_mink):
    """The exact one-step solver converges, so the loop breaks after one iter."""
    bridge = _make_bridge(fake_mink, max_iters=20)
    target = _target_pose([0.4, -0.1, 0.25])

    q = bridge.solve(target, q_init=np.zeros(7))

    # Reached the commanded Cartesian target ...
    np.testing.assert_allclose(q[:3], [0.4, -0.1, 0.25], atol=1e-6)
    # ... and stopped after a single integrate rather than burning all 20 iters.
    assert bridge._configuration.integrate_calls == 1


def test_solve_runs_full_iteration_budget_when_never_converged(fake_mink):
    """An impossible threshold means the break never fires; all iters run."""
    bridge = _make_bridge(fake_mink, max_iters=5, pos_threshold=-1.0)
    target = _target_pose([0.4, 0.0, 0.3])

    bridge.solve(target, q_init=np.zeros(7))

    assert bridge._configuration.integrate_calls == 5
