"""``MinkIKBridge.ee_pose`` refuses a non-finite configuration instead of reporting one.

:class:`~strands_robots.simulation.ik.MinkIKBridge` has exactly two methods that
*apply* a joint configuration to its own state - both call
``self._configuration.update(...)`` - and until now only one of them held that
array to a domain. ``solve`` gained
:func:`~strands_robots.utils.finite_vector_error` for its ``q_init``; ``ee_pose``,
the forward-kinematics reader with 19 call sites across five modules, read the
same kind of array into the same call one line down and checked nothing.

Measured against a Panda (``nq=9``) whose healthy ``ee_pose`` returns a pose with
sixteen finite entries: a single ``nan`` or ``inf`` anywhere in ``qpos`` returned
a ``(4, 4)`` pose with **12 of 16** entries non-finite, under a successful return
shaped exactly like a reachable pose. Three consumers then inherit it:

* a ``norm(ee[:3, 3] - target)`` residual - the shape six call sites in the two
  motion-primitive backends use - comes back ``nan``, and ``nan <= threshold``
  is ``False``, so a convergence test never fires;
* ``tracking_error`` reported ``{"mean_mm": nan, "max_mm": nan}``;
* the closed loop in ``policies/cosmos3/sim_ik.py`` and ``policies/vera/sim_ik.py``
  composes a delta onto this pose and solves for it, so ``solve``'s own guard
  refused one step later naming ``target_pose`` - an argument the caller never
  supplied. The caller supplied ``q0``.

That last one is why this is worth a guard rather than a docstring: the sibling
change that guarded ``solve`` turned a silent wrong answer into a *misattributed*
refusal, and only a guard at the method that reads the caller's own array can
name it.

The check costs 8.466 us against a 3759 us ``solve``, and every in-package
consumer pairs ``ee_pose`` with a solve or a norm, so it is 0.224% of a
closed-loop step. Against a *bare* ``ee_pose`` (21.66 us of pure forward
kinematics) it is 64.2%, which is the honest figure for ``tracking_error``'s
per-row loop; a 1000-step trajectory pays about 8 ms.

The shared domain is used verbatim rather than fronted by a fast
``np.all(np.isfinite(...))`` test, and :class:`TestTheSharedDomainIsNotReimplemented`
measures why: that test *accepts* a 0-d scalar and a 2-D array which the domain
refuses, so a hand-rolled fast path would decline to judge two spellings its own
siblings reject.

Two things are deliberately not claimed. The bridge does not carry the damage
across calls - ``solve`` re-seeds the configuration every time, so the next
healthy solve was already clean before this change; the guard's placement ahead
of the mutation is craft, and :class:`TestTheRefusalCostsTheBridgeNothing` pins
it. And ``tracking_error`` gains its refusal by *inheritance* rather than by a
guard of its own: the sibling change left it alone because a ``nan`` reading is
visibly non-finite, and this change reaches it through the method it calls.
:class:`TestTheConsumersInheritTheRefusal` records that consequence rather than
leaving it to be discovered.

The behavioural cells drive the real ``ee_pose`` against the fake ``mink`` module
the sibling suites already use, so nothing here needs ``mink``, ``mujoco`` or
``qpsolvers`` installed.
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

#: The shared domain both configuration readers consult, named locally so these
#: cells are an independent oracle rather than a restatement of the module.
DOMAIN = finite_vector_error

#: The method name the refusal must carry, stated locally for the same reason.
METHOD = "ee_pose"

#: The parameter name the refusal must carry.
PARAMETER = "qpos"


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


def _config(nq: int = 7) -> np.ndarray:
    return np.zeros(nq, dtype=np.float64)


def _ee_pose(bridge: MinkIKBridge, qpos: Any) -> np.ndarray:
    """Call ``ee_pose`` through one deliberately untyped funnel.

    The cells below pass a 0-d array and a nested list on purpose, which is the
    whole point of a value domain. Routing them here keeps the call sites free of
    the ``arg-type`` reports a literal would draw.
    """
    return bridge.ee_pose(qpos)


def _poisoned(base: np.ndarray, index: int, value: float) -> np.ndarray:
    out = base.copy()
    out[index] = value
    return out


_CONFIG_CASES = [
    pytest.param(0, float("nan"), id="nan-at-first-joint"),
    pytest.param(3, float("inf"), id="inf-at-a-middle-joint"),
    pytest.param(6, float("-inf"), id="negative-inf-at-the-last-joint"),
]


def _source_of(method: str) -> str:
    return textwrap.dedent(inspect.getsource(getattr(MinkIKBridge, method)))


def _methods_that_apply_a_configuration() -> set[str]:
    """Derive every method that writes a configuration into the bridge's state.

    The rule is dataflow rather than naming: a method that calls
    ``self._configuration.update(...)`` has applied the caller's array to the
    state forward kinematics and the QP are evaluated against, so it owns the
    domain for it. A method that merely delegates - ``solve_trajectory`` calls
    ``solve``, ``tracking_error`` calls ``ee_pose`` - inherits the refusal and is
    correctly outside this set.
    """
    module = ast.parse(inspect.getsource(inspect.getmodule(MinkIKBridge)))  # type: ignore[arg-type]
    cls = next(n for n in ast.walk(module) if isinstance(n, ast.ClassDef) and n.name == "MinkIKBridge")
    found: set[str] = set()
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "_configuration"
            ):
                found.add(node.name)
    return found


class TestTheConfigurationIsWhatForwardKinematicsReads:
    """Premise: the array is applied, which is why a bad value reaches the pose."""

    def test_the_returned_pose_is_a_function_of_the_configuration(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        near = _ee_pose(bridge, np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]))
        far = _ee_pose(bridge, np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0]))
        assert not np.array_equal(near, far)
        assert np.allclose(near[:3, 3], [0.1, 0.2, 0.3])

    def test_the_configuration_is_written_into_the_bridge_state(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        _ee_pose(bridge, np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert np.asarray(bridge._configuration.q)[0] == pytest.approx(0.7)

    def test_exactly_two_methods_apply_a_configuration(self) -> None:
        assert _methods_that_apply_a_configuration() == {"ee_pose", "solve"}


class TestANonFiniteConfigurationIsRefused:
    """The regression: a pose is not reported for a configuration that has none."""

    @pytest.mark.parametrize(("index", "value"), _CONFIG_CASES)
    def test_a_poisoned_joint_is_refused(self, fake_mink: types.ModuleType, index: int, value: float) -> None:
        bridge = _bridge()
        with pytest.raises(ValueError, match="must contain finite numbers"):
            _ee_pose(bridge, _poisoned(_config(), index, value))

    def test_an_entirely_non_finite_configuration_is_refused(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        with pytest.raises(ValueError, match="must contain finite numbers"):
            _ee_pose(bridge, np.full(7, float("nan")))

    def test_no_pose_is_returned_for_a_non_finite_configuration(self, fake_mink: types.ModuleType) -> None:
        """The defect's own shape: a successful return carrying a non-finite pose."""
        bridge = _bridge()
        with pytest.raises(ValueError):
            _ee_pose(bridge, _poisoned(_config(), 0, float("nan")))

    def test_the_guard_precedes_the_state_it_would_otherwise_poison(self) -> None:
        source = _source_of("ee_pose")
        assert source.index("finite_vector_error(") < source.index("self._configuration.update(")


class TestTheRefusalNamesTheMethodAndTheParameter:
    """A refusal a caller cannot act on is a dead end."""

    def test_the_refusal_names_the_method(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        with pytest.raises(ValueError) as excinfo:
            _ee_pose(bridge, _poisoned(_config(), 0, float("nan")))
        assert METHOD in str(excinfo.value)

    def test_the_refusal_names_the_parameter(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        with pytest.raises(ValueError) as excinfo:
            _ee_pose(bridge, _poisoned(_config(), 0, float("nan")))
        assert PARAMETER in str(excinfo.value)

    def test_the_refusal_is_the_shared_domain_verbatim(self, fake_mink: types.ModuleType) -> None:
        """Not merely equivalent wording: the same domain, so the two readers agree."""
        bad = _poisoned(_config(), 0, float("nan"))
        bridge = _bridge()
        with pytest.raises(ValueError) as excinfo:
            _ee_pose(bridge, bad)
        assert str(excinfo.value) == DOMAIN(METHOD, PARAMETER, bad)


class TestTheRefusalCostsTheBridgeNothing:
    """A refused call leaves the bridge as it was."""

    def test_the_configuration_is_unchanged_by_a_refusal(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        _ee_pose(bridge, np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]))
        before = np.asarray(bridge._configuration.q).copy()
        with pytest.raises(ValueError):
            _ee_pose(bridge, np.full(7, float("inf")))
        assert np.array_equal(before, np.asarray(bridge._configuration.q))

    def test_a_healthy_call_after_a_refusal_still_reports(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        with pytest.raises(ValueError):
            _ee_pose(bridge, np.full(7, float("nan")))
        pose = _ee_pose(bridge, np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert np.isfinite(pose).all()


class TestTheConsumersInheritTheRefusal:
    """The two in-package consumers now refuse by name instead of reporting nan.

    ``tracking_error`` is reached by inheritance rather than by a guard of its
    own. The sibling change left it alone deliberately - a ``nan`` reading is
    visibly non-finite rather than plausible - and this change reaches it through
    the method it calls per row. Recorded here so the consequence is pinned
    rather than discovered.
    """

    def test_tracking_error_refuses_a_non_finite_solved_trajectory(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        poses = np.stack([_target_pose([0.1, 0.0, 0.0])])
        traj = np.stack([np.full(7, float("nan"))])
        with pytest.raises(ValueError) as excinfo:
            bridge.tracking_error(poses, traj)
        assert PARAMETER in str(excinfo.value)

    def test_the_closed_loop_refuses_at_the_argument_the_caller_supplied(self, fake_mink: types.ModuleType) -> None:
        """The misattribution this fix removes: the seed, not the composed target."""
        bridge = _bridge()
        with pytest.raises(ValueError) as excinfo:
            _ee_pose(bridge, _poisoned(_config(), 0, float("nan")))
        message = str(excinfo.value)
        assert PARAMETER in message
        assert "target_pose" not in message


class TestTheSharedDomainIsNotReimplemented:
    """Why the domain is called verbatim rather than fronted by a fast test.

    ``np.all(np.isfinite(...))`` is cheaper than the shared domain and accepts
    two spellings the domain refuses, so a hand-rolled fast path would silently
    decline to judge inputs its own siblings reject.
    """

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(np.asarray(0.0), id="zero-dimensional-scalar"),
            pytest.param(np.zeros((7, 1)), id="two-dimensional-column"),
        ],
    )
    def test_a_fast_finiteness_test_accepts_what_the_domain_refuses(self, value: np.ndarray) -> None:
        assert bool(np.all(np.isfinite(value))) is True
        assert DOMAIN(METHOD, PARAMETER, value) is not None

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(np.asarray(0.0), id="zero-dimensional-scalar"),
            pytest.param(np.zeros((7, 1)), id="two-dimensional-column"),
        ],
    )
    def test_the_method_refuses_them_too_and_names_the_parameter(
        self, fake_mink: types.ModuleType, value: np.ndarray
    ) -> None:
        """Named, because both spellings already raised for the wrong reason.

        Before the guard, a 0-d scalar reached ``self.q[:3]`` as an
        ``IndexError`` and a 2-D column as ``ValueError: could not broadcast
        input array from shape (3,1) into shape (3,)`` - a refusal that names
        neither the method nor the parameter, and which a bare
        ``pytest.raises(ValueError)`` would have accepted as a pass.
        """
        bridge = _bridge()
        with pytest.raises(ValueError) as excinfo:
            _ee_pose(bridge, value)
        message = str(excinfo.value)
        assert METHOD in message
        assert PARAMETER in message


class TestWhatAFiniteConfigurationStillDoes:
    """Over-reach controls: the guard refuses non-finite values and nothing else."""

    def test_a_finite_configuration_still_reports_a_pose(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        pose = _ee_pose(bridge, _config())
        assert pose.shape == (4, 4)
        assert np.isfinite(pose).all()

    def test_a_large_finite_configuration_is_accepted(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        assert np.isfinite(_ee_pose(bridge, np.full(7, 1e6))).all()

    def test_a_negative_configuration_is_accepted(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        assert np.isfinite(_ee_pose(bridge, np.full(7, -2.5))).all()

    def test_an_integer_configuration_is_accepted(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        assert np.isfinite(_ee_pose(bridge, np.zeros(7, dtype=np.int64))).all()

    def test_a_list_configuration_is_accepted(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        assert np.isfinite(_ee_pose(bridge, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])).all()

    def test_a_finite_solve_is_unchanged(self, fake_mink: types.ModuleType) -> None:
        """The sibling reader keeps its own behaviour; only ee_pose gained a guard."""
        bridge = _bridge()
        solved = bridge.solve(_target_pose([0.2, 0.1, 0.05]), _config())
        assert np.isfinite(solved).all()

    def test_a_finite_tracking_error_is_still_reported(self, fake_mink: types.ModuleType) -> None:
        bridge = _bridge()
        poses = np.stack([_target_pose([0.1, 0.0, 0.0])])
        traj = np.stack([np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
        report = bridge.tracking_error(poses, traj)
        assert np.isfinite(report["mean_mm"])
        assert np.isfinite(report["max_mm"])


class TestEveryConfigurationReaderSharesTheDomain:
    """Derived: a method that applies a configuration must hold it to the domain.

    Stated over the dataflow rather than a list of names, so a third method that
    starts writing into ``self._configuration`` is graded the day it lands
    instead of inheriting an exemption by being absent from a tuple.
    """

    @pytest.mark.parametrize("method", sorted(_methods_that_apply_a_configuration()))
    def test_the_method_consults_the_shared_domain(self, method: str) -> None:
        assert "finite_vector_error(" in _source_of(method)

    @pytest.mark.parametrize("method", sorted(_methods_that_apply_a_configuration()))
    def test_the_guard_precedes_the_update(self, method: str) -> None:
        source = _source_of(method)
        assert source.index("finite_vector_error(") < source.index("self._configuration.update(")

    def test_the_derivation_is_not_vacuous(self) -> None:
        assert len(_methods_that_apply_a_configuration()) >= 2
