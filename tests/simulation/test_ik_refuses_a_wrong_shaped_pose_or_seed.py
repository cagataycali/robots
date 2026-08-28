"""``MinkIKBridge`` refuses a wrong-shaped pose or seed instead of letting the
consumer break on it.

Three methods document a shape and none checked it.
:meth:`~strands_robots.simulation.ik.MinkIKBridge.solve` documents
``target_pose`` as a ``(4, 4)`` homogeneous pose and ``q_init`` as "length
``model.nq``"; :meth:`~strands_robots.simulation.ik.MinkIKBridge.ee_pose`
documents ``qpos`` the same way. Measured against a Panda (``nq=9``) on a
reachable 80 mm target, every wrong shape did raise - the defect is *what* it
raised:

* a pose that is not ``(4, 4)`` reached ``mink.SE3.from_matrix``, whose whole
  shape check is a bare ``assert``, so the caller got an ``AssertionError``
  carrying **no message at all** - for ``(3, 3)``, for a flat sixteen-vector and
  for a ``(2, 4, 4)`` batch alike;
* under ``python -O``, where that assertion is stripped, the same three calls
  raised ``IndexError`` ("index 3 is out of bounds for axis 1 with size 3") or
  ``TypeError`` from ``mju_mat2Quat`` instead - so the exception *type* depended
  on an interpreter flag;
* a wrong-length ``q_init`` raised ``ValueError: could not broadcast input array
  from shape (6,) into shape (9,)`` out of the numpy assignment inside
  ``mink.Configuration.update``, naming neither the parameter nor the class, and
  a length-one seed raised ``mink.exceptions.InvalidTarget`` instead - a third
  type for the same class of caller error;
* ``ee_pose`` gave the same anonymous broadcast message, and it is reachable
  with a caller's own array: ``tracking_error(poses, qpos_traj)`` reads every row
  of ``qpos_traj`` back through it.

None of those is the ``ValueError`` channel these methods document.
:meth:`~strands_robots.simulation.ik.MinkIKBridge.solve_trajectory` meanwhile
already refused a wrong-shaped ``poses`` batch **by name** and then delegated to
``solve`` per waypoint, so one wrong pose was reported two different ways
depending on which entry point a caller used.

Two scope decisions are measured rather than assumed.

The pose's shape cannot be delegated to the shared fixed-length domain:
``pose_vector_error(..., 16)`` accepts a ``(2, 8)`` array, because sixteen
components is what it checks and that is not what
``mink.SE3.from_matrix``'s ``matrix[:3, :3]`` / ``matrix[:3, 3]`` slicing needs.
So the shape is a local check and the component domain still owns the values.

``ee_pose`` already consults the per-component domain for the *values* (#2879),
so what it gains here is the **length**: ``finite_vector_error`` returns ``None``
for a six-element seed on a nine-joint model, and the fixed-length domain that
wraps it refuses the same array by name. Both methods therefore end up on one
member, ``pose_vector_error``, for one contract - "``model.nq`` finite numbers" -
rather than two spellings of half of it.

The behavioural cells drive the real methods against the fake ``mink`` module the
sibling suites already use, so nothing here needs ``mink``, ``mujoco`` or
``qpsolvers`` installed.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types

import numpy as np
import pytest

from strands_robots.simulation.ik import MinkIKBridge
from strands_robots.utils import finite_vector_error, pose_vector_error
from tests.policies.cosmos3.test_sim_ik_bridge_solve_loop import (
    _FakeConfiguration,
    _FakeFrameTask,
    _FakePostureTask,
    _FakeSE3,
    _model,
    _solve_ik,
    _target_pose,
)

#: The fake model's joint count, stated locally so these cells are an
#: independent oracle rather than a restatement of the fixture.
NQ = 7

#: The shape every pose argument must have. Named locally for the same reason:
#: a cell that read it off the module could not tell a changed contract from a
#: changed expectation.
POSE_SHAPE = (4, 4)


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
    return MinkIKBridge(model=_model(NQ), ee_frame_name="gripper", solver="quadprog")


def _seed(nq: int = NQ) -> np.ndarray:
    return np.zeros(nq, dtype=np.float64)


#: Every wrong pose shape, and why each one is worth its own case. A flat
#: sixteen-vector and a ``(2, 8)`` both carry the sixteen components the shared
#: fixed-length domain would count, so they are what show the shape has to be
#: checked as a shape.
WRONG_POSE_SHAPES = (
    pytest.param(np.eye(3), id="three-by-three"),
    pytest.param(np.eye(4).ravel(), id="flat-sixteen-vector"),
    pytest.param(np.zeros((2, 8)), id="two-by-eight-also-sixteen"),
    pytest.param(np.stack([np.eye(4), np.eye(4)]), id="a-batch-not-a-pose"),
    pytest.param(np.zeros((4, 5)), id="four-by-five"),
)

#: Wrong seed lengths on both sides of ``nq``, plus the length-one case that
#: reached a third exception type in the real stack.
WRONG_SEED_LENGTHS = (NQ - 3, NQ + 3, 1)


class TestSolveRefusesAWrongShapedPose:
    """``solve`` names the parameter and the shape instead of asserting."""

    @pytest.mark.parametrize("pose", WRONG_POSE_SHAPES)
    def test_a_pose_that_is_not_four_by_four_is_refused(self, fake_mink: types.ModuleType, pose: np.ndarray) -> None:
        with pytest.raises(ValueError, match=r"solve: 'target_pose' must be a \(4, 4\) homogeneous pose matrix"):
            _bridge().solve(pose, _seed())

    @pytest.mark.parametrize("pose", WRONG_POSE_SHAPES)
    def test_the_refusal_reports_the_shape_that_was_supplied(
        self, fake_mink: types.ModuleType, pose: np.ndarray
    ) -> None:
        """A caller debugging this needs to know what it sent, not only what was wanted."""
        with pytest.raises(ValueError, match=rf"got shape {np.shape(pose)}".replace("(", r"\(").replace(")", r"\)")):
            _bridge().solve(pose, _seed())


class TestSolveRefusesAWrongLengthSeed:
    """``solve`` holds ``q_init`` to the length it documents."""

    @pytest.mark.parametrize("nq", WRONG_SEED_LENGTHS)
    def test_a_seed_that_is_not_nq_long_is_refused(self, fake_mink: types.ModuleType, nq: int) -> None:
        with pytest.raises(ValueError, match=rf"solve: 'q_init' must be a {NQ}-element vector, got {nq}"):
            _bridge().solve(_target_pose([0.1, 0.0, 0.0]), _seed(nq))

    def test_the_trajectory_entry_point_inherits_the_refusal(self, fake_mink: types.ModuleType) -> None:
        """``solve_trajectory`` solves through ``solve``, so one guard covers both."""
        with pytest.raises(ValueError, match=r"solve: 'q_init' must be a 7-element vector"):
            _bridge().solve_trajectory(np.stack([_target_pose([0.1, 0.0, 0.0])]), _seed(NQ - 3))

    def test_a_wrong_shaped_waypoint_is_refused_by_name_through_either_entry_point(
        self, fake_mink: types.ModuleType
    ) -> None:
        """The pair that used to disagree: the batch check, and the per-waypoint one."""
        bridge = _bridge()
        with pytest.raises(ValueError, match=r"poses must be \[N, 4, 4\]"):
            bridge.solve_trajectory(np.stack([np.eye(3)]), _seed())
        with pytest.raises(ValueError, match=r"'target_pose' must be a \(4, 4\) homogeneous pose matrix"):
            bridge.solve(np.eye(3), _seed())


class TestEePoseRefusesAWrongLengthConfiguration:
    """``ee_pose`` holds ``qpos`` to the length it documents, and so does its reader."""

    @pytest.mark.parametrize("nq", WRONG_SEED_LENGTHS)
    def test_a_configuration_that_is_not_nq_long_is_refused(self, fake_mink: types.ModuleType, nq: int) -> None:
        with pytest.raises(ValueError, match=rf"ee_pose: 'qpos' must be a {NQ}-element vector, got {nq}"):
            _bridge().ee_pose(_seed(nq))

    def test_tracking_error_inherits_the_refusal_for_a_wrong_width_row(self, fake_mink: types.ModuleType) -> None:
        """``qpos_traj`` is a caller's array, and every row is read back through ``ee_pose``."""
        with pytest.raises(ValueError, match=r"ee_pose: 'qpos' must be a 7-element vector"):
            _bridge().tracking_error(np.stack([_target_pose([0.1, 0.0, 0.0])]), np.zeros((1, NQ - 3)))


class TestTheShapeCheckCannotBeDelegatedToTheLengthDomain:
    """Premise: the shared fixed-length domain accepts a matrix that is not a pose."""

    def test_sixteen_components_do_not_establish_four_by_four(self) -> None:
        """``pose_vector_error(..., 16)`` accepts a ``(2, 8)``, so it is not the check."""
        assert pose_vector_error("solve", "target_pose", np.zeros((2, 8)).ravel(), 16) is None

    def test_the_shared_domain_does_establish_the_seed_length(self) -> None:
        """The seed is a flat vector, so the shared domain is exactly right for it."""
        assert pose_vector_error("solve", "q_init", _seed(), NQ) is None
        assert pose_vector_error("solve", "q_init", _seed(NQ - 3), NQ) is not None


class TestTheAcceptedShapesStillSolve:
    """Over-reach controls: nothing the contract allows became an error."""

    def test_a_four_by_four_pose_and_an_nq_seed_still_solve(self, fake_mink: types.ModuleType) -> None:
        out = _bridge().solve(_target_pose([0.1, 0.2, 0.3]), _seed())
        assert out.shape == (NQ,)
        assert np.allclose(out[:3], [0.1, 0.2, 0.3])

    def test_an_nq_configuration_still_reads_forward(self, fake_mink: types.ModuleType) -> None:
        pose = _bridge().ee_pose(np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0]))
        assert pose.shape == POSE_SHAPE
        assert np.allclose(pose[:3, 3], [0.4, 0.5, 0.6])

    def test_a_trajectory_of_valid_poses_still_solves(self, fake_mink: types.ModuleType) -> None:
        poses = np.stack([_target_pose([0.1, 0.0, 0.0]), _target_pose([0.2, 0.0, 0.0])])
        out = _bridge().solve_trajectory(poses, _seed())
        assert out.shape == (2, NQ)

    def test_an_empty_batch_still_returns_the_documented_width(self, fake_mink: types.ModuleType) -> None:
        assert _bridge().solve_trajectory(np.empty((0, 4, 4)), _seed()).shape == (0, NQ)

    def test_tracking_error_still_reports_millimetres_for_matching_widths(self, fake_mink: types.ModuleType) -> None:
        report = _bridge().tracking_error(np.stack([_target_pose([0.0, 0.0, 0.0])]), np.stack([_seed()]))
        assert report == {"mean_mm": 0.0, "max_mm": 0.0}


class TestBothMethodsEndOnOneMember:
    """The two methods now apply one contract, not two spellings of half of it.

    ``ee_pose`` already refused a non-finite configuration; what it lacked was the
    length, which the per-component domain does not check. Both halves are pinned
    here so a future edit cannot drop either one back to the narrower member.
    """

    def test_the_component_domain_alone_would_accept_a_wrong_length(self) -> None:
        """Premise: this is exactly the half the narrower member does not cover."""
        assert finite_vector_error("ee_pose", "qpos", _seed(NQ - 3)) is None
        assert pose_vector_error("ee_pose", "qpos", _seed(NQ - 3), NQ) is not None

    def test_a_two_dimensional_configuration_is_still_refused(self, fake_mink: types.ModuleType) -> None:
        """Inherited: the component domain already refuses a column vector.

        Recorded here rather than with the length cells because it holds either
        way - a ``(nq, 1)`` has ``nq`` values and the wider member refuses it for
        its elements, not its length - so it is the shape a reader would reach
        for to argue the length check is redundant, and it is not.
        """
        with pytest.raises(ValueError, match=r"ee_pose: 'qpos' elements must be numbers"):
            _bridge().ee_pose(_seed().reshape(NQ, 1))

    def test_ee_pose_still_refuses_a_non_finite_configuration(self, fake_mink: types.ModuleType) -> None:
        q = _seed()
        q[0] = np.nan
        with pytest.raises(ValueError, match=r"ee_pose: 'qpos' must contain finite numbers"):
            _bridge().ee_pose(q)

    def test_solve_still_refuses_a_non_finite_seed(self, fake_mink: types.ModuleType) -> None:
        q = _seed()
        q[0] = np.nan
        with pytest.raises(ValueError, match=r"solve: 'q_init'"):
            _bridge().solve(_target_pose([0.1, 0.0, 0.0]), q)

    def test_solve_still_refuses_a_non_finite_pose(self, fake_mink: types.ModuleType) -> None:
        pose = _target_pose([0.1, 0.0, 0.0])
        pose[0, 3] = np.inf
        with pytest.raises(ValueError, match=r"solve: 'target_pose'"):
            _bridge().solve(pose, _seed())


class TestTheChecksPrecedeTheStateTheyProtect:
    """Structural: each guard runs before the call it exists to keep clean."""

    def test_the_pose_shape_is_checked_before_its_components(self) -> None:
        """The shape is the more specific diagnosis, so it is reported first."""
        source = textwrap.dedent(inspect.getsource(MinkIKBridge.solve))
        assert source.index("_homogeneous_pose_error(") < source.index('finite_vector_error("solve", "target_pose"')

    def test_every_solve_guard_precedes_the_configuration_update(self) -> None:
        source = textwrap.dedent(inspect.getsource(MinkIKBridge.solve))
        last = max(source.rindex(f"{name}(") for name in ("_homogeneous_pose_error", "pose_vector_error"))
        assert last < source.index("self._configuration.update(")
        assert last < source.index("self._frame_task.set_target(")

    def test_the_ee_pose_guard_precedes_the_configuration_update(self) -> None:
        source = textwrap.dedent(inspect.getsource(MinkIKBridge.ee_pose))
        assert source.index("pose_vector_error(") < source.index("self._configuration.update(")

    def test_each_documented_shape_has_exactly_one_owner(self) -> None:
        """One guard per contract, so the two entry points cannot drift apart."""
        calls: dict[str, list[str]] = {}
        for method in (MinkIKBridge.solve, MinkIKBridge.ee_pose, MinkIKBridge.solve_trajectory):
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            calls[method.__name__] = sorted(
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"_homogeneous_pose_error", "pose_vector_error"}
            )
        assert calls["solve"] == ["_homogeneous_pose_error", "pose_vector_error"]
        assert calls["ee_pose"] == ["pose_vector_error"]
        # solve_trajectory keeps only its own batch-shape check and inherits the rest.
        assert calls["solve_trajectory"] == []
