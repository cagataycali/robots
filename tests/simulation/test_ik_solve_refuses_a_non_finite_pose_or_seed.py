"""``MinkIKBridge.solve`` refuses a non-finite pose or seed instead of solving with it.

:class:`~strands_robots.simulation.ik.MinkIKBridge` holds all eight of its
numeric *knobs* to a shared domain in ``__init__`` - the three task costs and the
two thresholds to :func:`~strands_robots.utils.finite_number_error`, ``damping``
to the local wrapper over it, ``max_iters`` to
:func:`~strands_robots.utils.positive_count_error` and ``dt`` to
:func:`~strands_robots.utils.positive_finite_number_error` - and handed the two
*arrays* ``solve`` reads straight through. ``solve_trajectory`` checked the one
thing a wrong-shaped batch would break on, ``poses.shape``, and none of the
values inside it.

Both arrays are applied rather than forwarded: the pose becomes the frame task's
target and the seed becomes the configuration the QP warm-starts from. Measured
against a Panda (``nq=9``) on a reachable 80 mm target whose healthy solve
returns nine finite joints:

* one ``nan`` anywhere in ``target_pose`` returned **9 of 9** joints non-finite,
  as a successful return shaped exactly like a converged solve;
* one ``inf`` in ``target_pose`` did the same;
* one ``nan`` in ``q_init`` did the same;
* one ``inf`` in ``q_init`` did *not* come back at all - it left the QP backend
  itself unable to solve, so the call raised ``mink.exceptions.NoSolutionFound``
  out of a third-party module rather than the ``ValueError`` this method
  documents. The same class of bad value therefore had two exits, one silent and
  one naming neither the method nor the parameter;
* ``solve_trajectory([good, bad])`` returned **9 of 18** non-finite - the first
  waypoint a real configuration and the second entirely NaN, so a caller
  iterating waypoints gets a partially valid trajectory rather than an error.

Both checks together cost 45.8 us against a 7.6 ms solve on that same Panda -
0.60% of one call - so there is no budget argument for leaving them to the
consumer. That is the shipped guard's own cost: :func:`finite_vector_error`
reads each component rather than vectorising the scan, so it is roughly linear
in the array and the pose's sixteen elements dominate the seed's nine.

Two things are deliberately *not* claimed. The bridge does not carry the damage
across calls - ``solve`` re-seeds the configuration from ``q_init`` every time,
so the next healthy solve was already clean before this change; the guard's
placement ahead of the mutation is craft rather than a fix for a measured leak,
and ``TestTheRefusalCostsTheBridgeNothing`` pins it. And ``tracking_error``
stays untouched: it returns ``{"mean_mm": nan, "max_mm": nan}``, which is a
visibly non-finite reading rather than a plausible one.

The behavioural cells drive the real ``solve`` loop against the fake ``mink``
module the sibling suites already use, so nothing here needs ``mink``,
``mujoco`` or ``qpsolvers`` installed.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.ik import MinkIKBridge
from strands_robots.utils import finite_vector_error
from tests.policies.cosmos3.test_sim_ik_bridge_solve_loop import (
    _FakeConfiguration,
    _FakeFrameTask,
    _FakePostureTask,
    _FakeSE3,
    _model,
    _solve_ik,
    _target_pose,
)

#: The two array arguments ``solve`` reads, named locally so these cells are an
#: independent oracle rather than a restatement of the module.
ARRAY_PARAMETERS = ("target_pose", "q_init")

#: A joint the stand-in solver never drives - it moves only ``q[:3]`` onto the
#: frame target - so a seed value here survives the solve and is what shows the
#: seed became the warm-start configuration.
UNDRIVEN_DOF = 4


@pytest.fixture
def fake_mink(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install the sibling suite's fake ``mink`` plus a stub ``qpsolvers``.

    Declared here rather than imported: importing a fixture re-binds the name in
    this module, which ``ruff`` reports as a redefinition for every cell that
    then takes it as a parameter. Only the stand-in classes are shared.
    """
    mod = types.ModuleType("mink")
    mod.Configuration = _FakeConfiguration  # type: ignore[attr-defined]
    mod.FrameTask = _FakeFrameTask  # type: ignore[attr-defined]
    mod.PostureTask = _FakePostureTask  # type: ignore[attr-defined]
    mod.SE3 = _FakeSE3  # type: ignore[attr-defined]
    mod.solve_ik = _solve_ik  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "mink", mod)

    qp = types.ModuleType("qpsolvers")
    qp.available_solvers = ["quadprog"]  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "qpsolvers", qp)
    return mod


def _bridge() -> MinkIKBridge:
    return MinkIKBridge(model=_model(), ee_frame_name="gripper", solver="quadprog")


def _seed(nq: int = 7) -> np.ndarray:
    return np.zeros(nq, dtype=np.float64)


def _solve(bridge: MinkIKBridge, pose: Any, seed: Any) -> np.ndarray:
    """Call ``solve`` through one deliberately untyped funnel.

    The cells below pass a nested list and a non-finite array on purpose, which
    is the whole point of a value domain. Routing them here keeps the call sites
    free of the ``arg-type`` reports a literal would draw.
    """
    return bridge.solve(pose, seed)


def _poisoned(base: np.ndarray, index: tuple[int, ...], value: float) -> np.ndarray:
    out = base.copy()
    out[index] = value
    return out


_POSE_CASES = [
    pytest.param((0, 3), float("nan"), id="nan-in-translation"),
    pytest.param((1, 3), float("inf"), id="inf-in-translation"),
    pytest.param((0, 0), float("nan"), id="nan-in-rotation"),
    pytest.param((2, 2), float("-inf"), id="negative-inf-in-rotation"),
]


class TestThePoseAndSeedAreWhatTheSolverReads:
    """Premise: both arrays reach the solver, which is why a bad value spreads."""

    def test_the_seed_becomes_the_configuration_and_the_pose_the_frame_target(
        self, fake_mink: types.ModuleType
    ) -> None:
        """Both arrays are read through the *solve*, not merely stored.

        Asserted on the returned configuration rather than on either task's
        stored target. ``mink`` names that storage ``PostureTask.target_q`` and
        ``FrameTask.transform_target_to_world`` while the stand-in keeps a
        single ``target``, so reading it would pin the fake's own convention
        rather than the library's. The returned configuration is the surface
        both agree on, and it shows each array *propagating* through the solve
        instead of merely arriving at a setter - which is the premise this
        class exists for.
        """
        bridge = _bridge()
        seed = _seed()
        seed[UNDRIVEN_DOF] = 0.25
        translation = [0.4, 0.0, 0.4]
        solved = _solve(bridge, _target_pose(translation), seed)
        # The pose became the frame target: the solver drives the first three
        # joints onto its translation.
        assert solved[:3] == pytest.approx(translation)
        # The seed became the warm-start configuration: a value on a joint the
        # solver never drives survives into the answer.
        assert solved[UNDRIVEN_DOF] == pytest.approx(0.25)

    def test_the_shared_domain_refuses_a_flat_non_finite_vector(self) -> None:
        clean = finite_vector_error("solve", "q_init", _seed())
        dirty = finite_vector_error("solve", "q_init", _poisoned(_seed(), (2,), float("nan")))
        assert clean is None
        assert dirty is not None and "q_init" in dirty

    def test_the_shared_domain_reads_a_two_dimensional_argument_by_rows(self) -> None:
        """Why the pose is flattened: the domain refuses even a *clean* 4x4."""
        matrix = np.eye(4)
        assert finite_vector_error("solve", "target_pose", matrix) is not None
        assert finite_vector_error("solve", "target_pose", matrix.ravel()) is None


class TestANonFinitePoseIsRefused:
    """Regression: the all-NaN configuration is no longer returned as a solve."""

    @pytest.mark.parametrize(("index", "value"), _POSE_CASES)
    def test_solve_refuses_the_pose(self, fake_mink: types.ModuleType, index: tuple[int, ...], value: float) -> None:
        pose = _poisoned(_target_pose([0.4, 0.0, 0.4]), index, value)
        with pytest.raises(ValueError, match="'target_pose' must contain finite numbers"):
            _solve(_bridge(), pose, _seed())

    def test_the_refusal_names_the_method_and_the_parameter(self, fake_mink: types.ModuleType) -> None:
        pose = _poisoned(_target_pose([0.4, 0.0, 0.4]), (0, 3), float("nan"))
        with pytest.raises(ValueError) as excinfo:
            _solve(_bridge(), pose, _seed())
        text = str(excinfo.value)
        assert text.startswith("solve:")
        assert "'target_pose'" in text
        assert "'q_init'" not in text

    def test_no_configuration_comes_back_non_finite(self, fake_mink: types.ModuleType) -> None:
        """The outcome the refusal replaces: nine of nine joints were NaN."""
        pose = _poisoned(_target_pose([0.4, 0.0, 0.4]), (0, 3), float("nan"))
        with pytest.raises(ValueError):
            _solve(_bridge(), pose, _seed())


class TestANonFiniteSeedIsRefused:
    """Regression: a seed the QP cannot warm-start from is refused by name."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_solve_refuses_the_seed(self, fake_mink: types.ModuleType, value: float) -> None:
        seed = _poisoned(_seed(), (2,), value)
        with pytest.raises(ValueError, match="'q_init' must contain finite numbers"):
            _solve(_bridge(), _target_pose([0.4, 0.0, 0.4]), seed)

    def test_the_refusal_names_the_seed_not_the_pose(self, fake_mink: types.ModuleType) -> None:
        seed = _poisoned(_seed(), (0,), float("nan"))
        with pytest.raises(ValueError) as excinfo:
            _solve(_bridge(), _target_pose([0.4, 0.0, 0.4]), seed)
        text = str(excinfo.value)
        assert "'q_init'" in text
        assert "'target_pose'" not in text


class TestTheTrajectoryPathInheritsTheGuard:
    """Regression: one guard in ``solve`` covers every ``solve_trajectory`` waypoint."""

    def test_a_bad_waypoint_is_refused_rather_than_partially_solved(self, fake_mink: types.ModuleType) -> None:
        good = _target_pose([0.4, 0.0, 0.4])
        bad = _poisoned(good, (0, 3), float("nan"))
        with pytest.raises(ValueError, match="'target_pose' must contain finite numbers"):
            _bridge().solve_trajectory(np.stack([good, bad]), _seed())

    def test_a_bad_seed_is_refused_on_the_first_waypoint(self, fake_mink: types.ModuleType) -> None:
        good = _target_pose([0.4, 0.0, 0.4])
        with pytest.raises(ValueError, match="'q_init' must contain finite numbers"):
            _bridge().solve_trajectory(np.stack([good, good]), _poisoned(_seed(), (1,), float("nan")))


class TestWhatStaysAccepted:
    """Over-reach controls: every legitimate call is unchanged."""

    def test_a_healthy_solve_still_returns_finite_joints(self, fake_mink: types.ModuleType) -> None:
        solved = _solve(_bridge(), _target_pose([0.4, 0.0, 0.4]), _seed())
        assert np.all(np.isfinite(solved))
        assert solved.shape == (7,)

    def test_a_nested_list_pose_and_seed_are_still_accepted(self, fake_mink: types.ModuleType) -> None:
        pose = _target_pose([0.4, 0.0, 0.4]).tolist()
        solved = _solve(_bridge(), pose, _seed().tolist())
        assert np.all(np.isfinite(solved))

    def test_a_large_but_finite_coordinate_is_still_accepted(self, fake_mink: types.ModuleType) -> None:
        """Unreachable is not unusable: the domain bounds finiteness, not range."""
        solved = _solve(_bridge(), _target_pose([1e6, 0.0, 0.0]), _seed())
        assert np.all(np.isfinite(solved))

    def test_the_shape_refusal_still_precedes_the_value_check(self, fake_mink: types.ModuleType) -> None:
        """A wrong-shaped batch keeps naming the shape, not the values inside it."""
        with pytest.raises(ValueError, match=r"poses must be \[N, 4, 4\]"):
            _bridge().solve_trajectory(np.full((2, 3, 3), np.nan), _seed())

    def test_an_empty_trajectory_is_still_zero_length_with_nq_width(self, fake_mink: types.ModuleType) -> None:
        out = _bridge().solve_trajectory(np.empty((0, 4, 4)), _seed())
        assert out.shape == (0, 7)

    def test_a_healthy_trajectory_still_solves_every_waypoint(self, fake_mink: types.ModuleType) -> None:
        poses = np.stack([_target_pose([0.4, 0.0, 0.4]), _target_pose([0.45, 0.0, 0.4])])
        out = _bridge().solve_trajectory(poses, _seed())
        assert out.shape == (2, 7)
        assert np.all(np.isfinite(out))


class TestTheRefusalCostsTheBridgeNothing:
    """The guard precedes the mutation, so a refused solve leaves the bridge usable."""

    def test_a_solve_after_a_refusal_is_identical_to_one_before_it(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        good = _target_pose([0.4, 0.0, 0.4])
        before = _solve(bridge, good, _seed())
        with pytest.raises(ValueError):
            _solve(bridge, _poisoned(good, (0, 3), float("nan")), _seed())
        after = _solve(bridge, good, _seed())
        assert np.array_equal(before, after)

    def test_a_refused_solve_does_not_update_the_configuration(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        _solve(bridge, _target_pose([0.4, 0.0, 0.4]), _seed())
        settled = np.asarray(bridge._configuration.q).copy()
        with pytest.raises(ValueError):
            _solve(bridge, _target_pose([0.4, 0.0, 0.4]), _poisoned(_seed(), (0,), float("nan")))
        assert np.array_equal(np.asarray(bridge._configuration.q), settled)


class TestBothArraysReachTheSharedDomain:
    """Structural: the check is the shared domain, applied to each array, once."""

    def test_solve_consults_the_shared_domain_for_every_array_it_reads(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(MinkIKBridge.solve)))
        guarded = {
            node.args[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "finite_vector_error"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        }
        assert guarded == set(ARRAY_PARAMETERS)

    def test_the_checks_precede_the_state_the_solve_mutates(self) -> None:
        source = textwrap.dedent(inspect.getsource(MinkIKBridge.solve))
        last_guard = source.rindex("finite_vector_error(")
        assert last_guard < source.index("self._configuration.update(")
        assert last_guard < source.index("self._frame_task.set_target(")

    def test_the_pose_is_flattened_for_the_check(self) -> None:
        """A 2-D argument would be refused for the wrong reason; see the premise."""
        source = textwrap.dedent(inspect.getsource(MinkIKBridge.solve))
        line = next(ln for ln in source.splitlines() if 'finite_vector_error("solve", "target_pose"' in ln)
        assert ".ravel()" in line
