"""Every numeric knob on the shared IK bridge is refused when it cannot be honored.

:class:`~strands_robots.simulation.ik.MinkIKBridge` is the one home for the mink
differential-IK solve, re-exported as the public ``MinkIKBridge`` of the
``cosmos3`` and ``vera`` providers and documented as a constructor a caller
builds directly (``docs/policies/cosmos3.md``). It validated two of its
arguments thoroughly - ``commanded_dofs`` per element, with ``bool`` rejected by
name and every index range-checked, and ``solver`` through
:func:`~strands_robots.simulation.ik.resolve_qp_solver` - and handed its eight
numeric knobs straight through.

Every one of those knobs is *applied* rather than forwarded: ``max_iters``
bounds the ``range`` the solve loop iterates, ``dt`` integrates the joint
velocity, ``damping`` and the three task costs weight the QP, and the two
thresholds decide when the loop breaks. So an unusable value produced a
plausible-looking joint configuration rather than an error. Measured against a
Panda on a reachable 80 mm target whose converged residual is 0.761 mm:

* ``max_iters=0`` (also ``False``, also a negative count) made ``range`` empty,
  so ``solve`` ran the solver zero times and returned ``q_init`` unchanged with
  the full 80 mm still to go - a solve that never happened, reported as one.
* ``max_iters=True`` ran exactly one iteration: 14.147 mm.
* ``pos_threshold`` and ``ori_threshold`` both infinite (also both ``True``)
  made the *first* iteration count as converged - 14.147 mm, byte-identical to
  ``max_iters=1``.
* ``dt`` of ``0.0``, ``nan`` or ``inf``, ``damping`` of ``nan``, and any
  infinite task cost each returned an all-NaN configuration.
* ``damping=inf`` produced no joint motion at all - again the full 80 mm, and
  ``damping=True`` weighted the QP a thousandfold heavier than the ``1e-3``
  default for a residual of 29.673 mm.

The remaining spellings failed *late* instead: a fractional, non-finite, string
or ``None`` ``max_iters`` raised ``TypeError`` from ``range`` inside ``solve``,
and a negative ``damping`` raised ``qpsolvers``' "matrix P is not positive
definite" mid-solve - both after the QP backend was resolved, both tasks were
built and the bridge had logged that it was ready.

The behavioural classes drive the real constructor and the real solve loop
against the fake ``mink`` module the sibling suite already uses, so nothing here
needs ``mink``, ``mujoco`` or ``qpsolvers`` installed.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import textwrap
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation import ik as ik_mod
from strands_robots.simulation.ik import MinkIKBridge
from strands_robots.utils import finite_number_error
from tests.policies.cosmos3.test_sim_ik_bridge_solve_loop import (
    _FakeConfiguration,
    _FakeFrameTask,
    _FakePostureTask,
    _FakeSE3,
    _model,
    _solve_ik,
    _target_pose,
)

#: The domain helper each numeric knob is held to, stated here rather than read
#: off the module so these cells are an independent oracle instead of a
#: restatement of the code under test.
EXPECTED_DOMAINS: dict[str, str] = {
    "position_cost": "finite_number_error",
    "orientation_cost": "finite_number_error",
    "posture_cost": "finite_number_error",
    "damping": "_damping_error",
    "max_iters": "positive_count_error",
    "dt": "positive_finite_number_error",
    "pos_threshold": "finite_number_error",
    "ori_threshold": "finite_number_error",
}

#: Spellings the bridge accepted and then answered with the wrong configuration:
#: no solve at all, one iteration where twenty were asked for, or all-NaN joints.
SILENTLY_WRONG: list[tuple[str, dict[str, Any]]] = [
    ("max_iters-zero-runs-the-solver-never", {"max_iters": 0}),
    ("max_iters-False-runs-the-solver-never", {"max_iters": False}),
    ("max_iters-negative-runs-the-solver-never", {"max_iters": -5}),
    ("max_iters-True-runs-one-iteration", {"max_iters": True}),
    ("dt-zero-yields-nan-joints", {"dt": 0.0}),
    ("dt-nan-yields-nan-joints", {"dt": float("nan")}),
    ("dt-inf-yields-nan-joints", {"dt": float("inf")}),
    ("damping-nan-yields-nan-joints", {"damping": float("nan")}),
    ("damping-inf-freezes-the-arm", {"damping": float("inf")}),
    ("damping-True-is-a-thousandfold-heavier-solve", {"damping": True}),
    ("position_cost-inf-yields-nan-joints", {"position_cost": float("inf")}),
    ("orientation_cost-inf-yields-nan-joints", {"orientation_cost": float("inf")}),
    ("posture_cost-inf-yields-nan-joints", {"posture_cost": float("inf")}),
    ("posture_cost-True-overwhelms-the-frame-task", {"posture_cost": True}),
    (
        "both-thresholds-inf-converge-on-iteration-one",
        {"pos_threshold": float("inf"), "ori_threshold": float("inf")},
    ),
    ("both-thresholds-True-converge-on-iteration-one", {"pos_threshold": True, "ori_threshold": True}),
]

#: Spellings that raised from inside ``solve``, after the bridge reported ready.
LATE_FAILURES: list[tuple[str, dict[str, Any]]] = [
    ("max_iters-fractional-breaks-range", {"max_iters": 2.5}),
    ("max_iters-whole-float-breaks-range", {"max_iters": 20.0}),
    ("max_iters-nan-breaks-range", {"max_iters": float("nan")}),
    ("max_iters-inf-breaks-range", {"max_iters": float("inf")}),
    ("max_iters-str-breaks-range", {"max_iters": "20"}),
    ("max_iters-None-breaks-range", {"max_iters": None}),
    ("damping-negative-makes-the-qp-indefinite", {"damping": -1.0}),
]

#: Values a caller can legitimately mean, which must keep reaching the solve.
STAYS_ACCEPTED: list[tuple[str, dict[str, Any]]] = [
    ("the-documented-defaults", {}),
    ("a-raised-iteration-budget", {"max_iters": 200}),
    ("orientation_cost-zero-is-the-position-only-solve", {"orientation_cost": 0.0}),
    ("posture_cost-zero-drops-the-regularizer", {"posture_cost": 0.0}),
    ("damping-zero-is-the-undamped-solve", {"damping": 0.0}),
    ("a-numpy-float-timestep", {"dt": np.float64(1e-2)}),
    ("a-tighter-position-threshold", {"pos_threshold": 1e-4}),
    ("an-unreachable-threshold-runs-the-whole-budget", {"pos_threshold": -1.0}),
    ("a-zero-threshold-runs-the-whole-budget", {"pos_threshold": 0.0}),
]


def _ids(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [row[0] for row in rows]


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


def _build(**overrides: Any) -> MinkIKBridge:
    """Build the bridge through one deliberately untyped funnel.

    The cells below supply values that are off-type on purpose - a string
    iteration count, a ``None`` timestep - which is the whole point of a domain.
    Routing them through ``**overrides`` keeps the call sites free of the
    ``arg-type`` reports a literal would draw.
    """
    return MinkIKBridge(model=_model(), ee_frame_name="gripper", solver="quadprog", **overrides)


def _init_source() -> str:
    return textwrap.dedent(inspect.getsource(MinkIKBridge.__init__))


def _numeric_parameters() -> dict[str, str]:
    """The constructor's ``int`` / ``float`` parameters, read off its signature.

    Derived rather than listed so a ninth numeric knob added later is graded on
    arrival instead of inheriting an exemption by being absent from a tuple.
    """
    found: dict[str, str] = {}
    for name, param in inspect.signature(MinkIKBridge.__init__).parameters.items():
        annotation = param.annotation
        if isinstance(annotation, str) and annotation.replace(" ", "") in {"int", "float"}:
            found[name] = annotation
    return found


def _domained_parameters() -> dict[str, set[str]]:
    """Map each parameter named in a domain call inside ``__init__`` to its domains.

    Reads both spellings the constructor uses: a direct
    ``some_error(value, "name", label)`` call, and a ``for (name, value) in
    ((literal, Name), ...)`` sweep whose loop variable is passed instead.
    """
    tree = ast.parse(_init_source())
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)

    # Loop variable -> the parameter names that flow through it.
    via_loop: dict[str, set[str]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Tuple):
            continue
        if not isinstance(node.target, ast.Tuple) or len(node.target.elts) != 2:
            continue
        value_var = node.target.elts[1]
        if not isinstance(value_var, ast.Name):
            continue
        for pair in node.iter.elts:
            if isinstance(pair, ast.Tuple) and len(pair.elts) == 2 and isinstance(pair.elts[1], ast.Name):
                via_loop.setdefault(value_var.id, set()).add(pair.elts[1].id)

    domains: dict[str, set[str]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name is None or not (name.endswith("_error") or name.endswith("_problems")):
            continue
        first = node.args[0]
        if not isinstance(first, ast.Name):
            continue
        for param in via_loop.get(first.id, {first.id}):
            domains.setdefault(param, set()).add(name)
    return domains


class TestWhatMadeEachValueUnusable:
    """The arithmetic and the loop shape the domains were chosen from.

    Holds on the code either way: these are properties of ``range``, of
    floating-point comparison and of the solve loop, not of the guards.
    """

    def test_an_empty_range_runs_the_solver_never(self) -> None:
        """``0``, ``False`` and a negative count all make the loop body unreachable."""
        assert len(range(0)) == 0
        assert len(range(False)) == 0
        assert len(range(-5)) == 0

    def test_a_true_iteration_budget_is_a_single_iteration(self) -> None:
        """``bool`` is an ``int`` subclass, so ``True`` is a silent budget of one."""
        assert len(range(True)) == 1

    def test_an_infinite_threshold_is_met_by_any_error(self) -> None:
        """So the convergence break fires on the first iteration."""
        residual = 0.08  # metres still to go after one iteration of an 80 mm move.
        for threshold in (float("inf"), True):
            assert residual <= threshold

    def test_an_unreachable_threshold_means_never_break_early(self) -> None:
        """Zero and negative thresholds are the *conservative* direction.

        No residual satisfies them, so the loop runs its whole budget rather
        than stopping short - which is why they are held only to finiteness.
        """
        smallest_residual = 1e-12
        for threshold in (0.0, -1.0):
            assert not smallest_residual <= threshold

    def test_the_solve_loop_reads_the_iteration_budget_and_both_thresholds(self) -> None:
        """The three consumers the domains are chosen from, read off the source."""
        source = textwrap.dedent(inspect.getsource(MinkIKBridge.solve))
        assert "range(self.max_iters)" in source
        assert "self.pos_threshold" in source
        assert "self.ori_threshold" in source

    def test_the_neighbouring_arguments_were_already_checked_thoroughly(self) -> None:
        """``commanded_dofs`` is validated per element, and ``solver`` is resolved.

        The contrast is the finding: two arguments checked exhaustively, eight
        numbers handed through.
        """
        mask_source = textwrap.dedent(inspect.getsource(MinkIKBridge._build_dof_mask))
        assert "isinstance(index, bool)" in mask_source
        assert "range(model.nv)" in mask_source or "nv" in mask_source
        assert "resolve_qp_solver(" in _init_source()


class TestTheSilentlyWrongSolvesAreRefused:
    """Each value that produced a wrong configuration is refused at construction."""

    @pytest.mark.parametrize(("label", "overrides"), SILENTLY_WRONG, ids=_ids(SILENTLY_WRONG))
    def test_the_bridge_refuses_the_value(self, fake_mink: types.ModuleType, label: str, overrides: dict) -> None:
        """A ``ValueError`` naming the parameter, instead of a plausible solve."""
        with pytest.raises(ValueError) as excinfo:
            _build(**overrides)

        text = str(excinfo.value)
        assert "MinkIKBridge" in text
        assert any(param in text for param in overrides), f"{label}: {text!r} names no supplied parameter"

    def test_a_zero_iteration_budget_no_longer_hands_back_the_seed(self, fake_mink: types.ModuleType) -> None:
        """The headline case: ``solve`` ran the solver zero times and reported it.

        Pre-fix the bridge built, ``solve`` iterated an empty ``range``, and the
        caller received ``q_init`` untouched - the full requested displacement
        still to go, with nothing to distinguish it from a converged solve.
        """
        with pytest.raises(ValueError, match=r"max_iters must be a positive integer"):
            _build(max_iters=0)

    def test_a_non_finite_timestep_no_longer_yields_nan_joints(self, fake_mink: types.ModuleType) -> None:
        """``dt`` scales the integrated velocity, so a non-finite one poisons every joint."""
        with pytest.raises(ValueError, match=r"dt must be > 0"):
            _build(dt=float("nan"))


class TestTheLateFailuresBecomeConstructionRefusals:
    """Values that raised from inside ``solve`` are refused before the tasks exist."""

    @pytest.mark.parametrize(("label", "overrides"), LATE_FAILURES, ids=_ids(LATE_FAILURES))
    def test_the_bridge_refuses_the_value(self, fake_mink: types.ModuleType, label: str, overrides: dict) -> None:
        with pytest.raises(ValueError) as excinfo:
            _build(**overrides)

        assert any(param in str(excinfo.value) for param in overrides)

    def test_a_negative_damping_names_the_qp_rule_it_breaks(self, fake_mink: types.ModuleType) -> None:
        """Rather than surfacing as "matrix P is not positive definite" mid-solve."""
        with pytest.raises(ValueError, match=r"damping must be >= 0"):
            _build(damping=-1.0)


class TestEveryNumericParameterReachesADomain:
    """The drift guard: a knob added later cannot skip the check.

    Derived from the constructor's own signature, so this fires on a ninth
    numeric parameter the hour it lands rather than when someone remembers.
    """

    def test_the_signature_still_carries_the_eight_numeric_knobs(self) -> None:
        """Non-vacuity: the derivation reads real parameters off a real signature."""
        assert set(_numeric_parameters()) == set(EXPECTED_DOMAINS)

    def test_every_numeric_parameter_is_asked_of_a_domain(self) -> None:
        domained = _domained_parameters()
        missing = sorted(set(_numeric_parameters()) - set(domained))
        assert not missing, f"numeric parameters reaching no domain: {missing}"

    @pytest.mark.parametrize(("param", "domain"), sorted(EXPECTED_DOMAINS.items()))
    def test_the_parameter_is_held_to_the_domain_its_consumer_needs(self, param: str, domain: str) -> None:
        """Each knob's domain is the one its consumer's accepted set implies."""
        assert domain in _domained_parameters().get(param, set())


class TestTheRefusalPrecedesEverySideEffect:
    """A refused value is reported before anything is resolved, built or logged."""

    def test_the_checks_run_before_the_solver_resolution_and_the_tasks(self) -> None:
        """Read off the constructor's own statement order."""
        source = _init_source()
        first_check = min(
            source.index(f"{domain}(") for domain in set(EXPECTED_DOMAINS.values()) if f"{domain}(" in source
        )
        for later in (
            "resolve_qp_solver(",
            "_build_dof_mask(",
            "mink.Configuration(",
            "mink.FrameTask(",
            "logger.info(",
        ):
            assert first_check < source.index(later), f"{later} runs before the value checks"

    def test_a_refused_value_logs_no_ready_line(
        self, fake_mink: types.ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The bridge must not announce itself ready and then be unusable."""
        with caplog.at_level(logging.INFO, logger=ik_mod.__name__), pytest.raises(ValueError):
            _build(max_iters=0)

        assert [record.getMessage() for record in caplog.records] == []


class TestWhatStaysAccepted:
    """Over-reach controls: nothing a caller can legitimately mean is refused."""

    @pytest.mark.parametrize(("label", "overrides"), STAYS_ACCEPTED, ids=_ids(STAYS_ACCEPTED))
    def test_the_bridge_accepts_the_value(self, fake_mink: types.ModuleType, label: str, overrides: dict) -> None:
        bridge = _build(**overrides)

        assert bridge.solver == "quadprog"

    def test_an_accepted_bridge_still_solves_to_the_target(self, fake_mink: types.ModuleType) -> None:
        """The guards check and pass the value on; they do not reshape the solve."""
        bridge = _build(max_iters=20)
        target = _target_pose([0.4, -0.1, 0.25])

        solved = bridge.solve(target, np.zeros(7, dtype=float))

        np.testing.assert_allclose(solved[:3], [0.4, -0.1, 0.25], atol=1e-6)

    def test_the_position_only_solve_is_still_reachable(self, fake_mink: types.ModuleType) -> None:
        """``orientation_cost=0.0`` is documented for arms with fewer than 6 DOF.

        The costs are held to finiteness rather than to a positive floor for
        exactly this reason.
        """
        bridge = _build(orientation_cost=0.0)

        assert bridge._frame_task is not None


class TestKnobsDeliberatelyLeftToTheirConsumer:
    """What the constructor does not restate, and why.

    ``mink`` refuses a negative task cost by name at task construction, which is
    already a loud refusal from the layer that owns the rule, so only the kind of
    the three costs is checked here. Recording it keeps the next reader from
    reading the asymmetry as an oversight.
    """

    def test_the_costs_are_held_to_finiteness_rather_than_to_a_floor(self) -> None:
        assert _domained_parameters()["position_cost"] == {"finite_number_error"}
        assert math.isfinite(-1.0)  # so the shared domain accepts it and mink refuses it.

    def test_the_thresholds_are_held_to_finiteness_rather_than_to_a_floor(self) -> None:
        """An unreachable threshold is a documented way to force the full budget.

        ``tests/policies/cosmos3/test_sim_ik_bridge_solve_loop.py`` and its
        ``vera`` twin both pass ``pos_threshold=-1.0`` to assert the loop burns
        its whole ``max_iters`` budget when the break never fires. A positive
        floor here would refuse that, and would refuse it for a value that
        cannot produce a wrong answer.
        """
        assert _domained_parameters()["pos_threshold"] == {"finite_number_error"}
        assert _domained_parameters()["ori_threshold"] == {"finite_number_error"}

    def test_the_damping_floor_is_stated_once(self) -> None:
        """One helper owns the ``>= 0`` rule, so the two halves cannot drift."""
        module_source = inspect.getsource(ik_mod)
        assert module_source.count("def _damping_error(") == 1
        assert _domained_parameters()["damping"] == {"_damping_error"}

    def test_the_damping_helper_delegates_the_boolean_refusal(self) -> None:
        """Its bare ``float()`` is reached only for a value the shared domain took.

        ``tests/simulation/test_input_validators_refuse_a_boolean.py`` exempts
        ``_damping_error`` from the boolean census on exactly that basis, and
        names this file as where the delegation is pinned behaviourally. The
        ``damping-True`` cell above is that pin; this one holds the shared
        domain's own refusal it rests on.
        """
        assert finite_number_error(True, "damping", "Ctx") is not None
        assert "finite_number_error(value" in textwrap.dedent(inspect.getsource(ik_mod._damping_error))
