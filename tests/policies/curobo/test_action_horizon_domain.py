"""``CuroboPolicy(action_horizon=...)`` shares the provider chunk-count domain.

``action_horizon`` is the slice width :meth:`CuroboPolicy._next_chunk` takes out
of the cached trajectory - the number of planned waypoints one
:meth:`~CuroboPolicy.get_actions` call hands the execution loop before the next
one is served. That is the same quantity the two lerobot providers accept as
``actions_per_step``, and :func:`strands_robots.policies.base.chunk_count_error`
exists as its shared domain, for the reason its own docstring states: the same
chunk count must not be refused by a local checkpoint and accepted by the server
serving it.

This constructor tested it with a bare ``action_horizon < 1`` instead, which
admits everything that one comparison happens to answer ``False`` for and lets
everything it cannot answer at all escape as whatever ``<`` or the following
``int()`` raised. Measured against the stub planner seam before the fix:

===================== ==========================================================
``action_horizon=``   pre-fix outcome
===================== ==========================================================
``True``              **accepted**, stored as ``1`` - one waypoint per call
``False``             refused
``2.7``               **accepted**, truncated to ``2``
``16.0``              **accepted** where a lerobot provider refuses it
``"16"``              ``TypeError: '<' not supported between ... 'str' and 'int'``
``None``              ``TypeError: '<' not supported between ... 'NoneType' ...``
``float("nan")``      ``ValueError: cannot convert float NaN to integer``
``float("inf")``      ``OverflowError: cannot convert float infinity to integer``
``[4]``               ``TypeError: '<' not supported between ... 'list' and 'int'``
===================== ==========================================================

``True`` accepted while ``False`` is refused is the hole
:func:`~strands_robots.utils.positive_count_error` documents itself as existing
to close - ``bool`` is an ``int`` subclass, so a bare ``value < 1`` reads ``True``
as a silent count of one. Nothing reports it: the policy streams a trajectory it
planned in full one waypoint per call, under a successful construction.

The three exception rows are a separate defect from the two silent ones. The
class docstring's ``Raises`` section promises ``ValueError``, so a caller that
validates its own configuration by catching what this constructor documents
catches none of them.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from typing import Any

import pytest

from strands_robots.policies.curobo import CuroboPolicy

# ---------------------------------------------------------------------------
# Stub planner - local to this module, as in every sibling curobo test file.
# ---------------------------------------------------------------------------


class _StubKinematics:
    def __init__(self) -> None:
        self.tool_frames = ["tool0"]


class _StubResult:
    def __init__(self, ndof: int, horizon: int) -> None:
        self.success = True
        self.status = "ok"
        self.trajectory = [[float(i)] * ndof for i in range(horizon)]


class _StubPlanner:
    """Returns a fixed-length synthetic plan; records nothing else."""

    def __init__(self, ndof: int = 6, horizon: int = 8) -> None:
        self.ndof = ndof
        self.horizon = horizon
        self.kinematics = _StubKinematics()

    def plan_single(self, start_state: object, goal: object) -> _StubResult:
        return _StubResult(self.ndof, self.horizon)

    def plan_single_js(self, start_state: object, goal: object) -> _StubResult:
        return _StubResult(self.ndof, self.horizon)


def _one_chunk(policy: CuroboPolicy) -> list[dict[str, float]]:
    """Plan once and return the waypoints a single ``get_actions`` call yields."""
    return asyncio.run(
        policy.get_actions(
            observation_dict={"observation.state": [0.0] * 6},
            instruction="",
            target_joints={f"joint_{i}": 0.1 for i in range(6)},
        )
    )


# Every value the bare comparison admitted or could not answer, with the reason
# each one is unusable as a slice bound over a planned trajectory.
_REFUSED = [
    pytest.param(0, id="zero-yields-no-waypoint"),
    pytest.param(-3, id="negative"),
    pytest.param(True, id="bool-true-was-a-silent-horizon-of-one"),
    pytest.param(False, id="bool-false"),
    pytest.param(2.7, id="fractional-was-truncated"),
    pytest.param(16.0, id="integral-float-a-lerobot-provider-refuses"),
    pytest.param("16", id="numeric-string"),
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="negative-inf"),
    pytest.param([4], id="list"),
    pytest.param({"horizon": 4}, id="mapping"),
]


class TestAnUnusableHorizonIsRefused:
    """Every value outside the shared chunk-count domain is refused as ``ValueError``."""

    @pytest.mark.parametrize("horizon", _REFUSED)
    def test_it_is_refused(self, horizon: Any) -> None:
        with pytest.raises(ValueError):
            CuroboPolicy(motion_gen=_StubPlanner(), action_horizon=horizon)

    @pytest.mark.parametrize("horizon", _REFUSED)
    def test_the_refusal_names_the_provider_and_the_parameter(self, horizon: Any) -> None:
        """A caller cannot act on a refusal that does not say what to change.

        Pinned for every value, not only the two the old comparison answered,
        because the three that escaped as ``TypeError`` / ``OverflowError`` named
        neither - and one of them, ``nan``, named only the ``int()`` conversion
        that happened to be reading it.
        """
        with pytest.raises(ValueError) as excinfo:
            CuroboPolicy(motion_gen=_StubPlanner(), action_horizon=horizon)
        message = str(excinfo.value)
        assert "curobo" in message
        assert "action_horizon" in message
        assert "must be a positive integer" in message
        assert "Omit it to use the provider default." in message

    @pytest.mark.parametrize("horizon", _REFUSED)
    def test_it_is_a_valueerror_as_the_class_docstring_promises(self, horizon: Any) -> None:
        """The ``Raises`` section is the contract a caller writes ``except`` against.

        ``TypeError`` and ``OverflowError`` are not ``ValueError`` subclasses, so
        a caller validating its own configuration through the documented
        exception caught none of the four values that escaped as one of those.
        """
        with pytest.raises(ValueError):
            CuroboPolicy(motion_gen=_StubPlanner(), action_horizon=horizon)


class TestTheBoolAsymmetryIsGone:
    """``True`` and ``False`` are now answered the same way, and it is refusal.

    Kept as its own case rather than folded into the table above because the
    asymmetry - one silently honored, one refused - is the specific behaviour
    ``positive_count_error`` documents itself as existing to prevent, and a
    parametrized run would not show that the two used to disagree.
    """

    def test_neither_bool_is_a_horizon(self) -> None:
        for value in (True, False):
            with pytest.raises(ValueError, match="must be a positive integer"):
                CuroboPolicy(motion_gen=_StubPlanner(), action_horizon=value)

    def test_true_no_longer_streams_one_waypoint_of_a_full_plan(self) -> None:
        """The consequence, stated as behaviour rather than as a type check.

        Pre-fix this constructed successfully and served 1 of the 8 planned
        waypoints per call. An 8-waypoint plan is not wrong when it is asked
        for; being served a sixteenth of it because a boolean was read as a
        count is, and nothing in the result said so.
        """
        with pytest.raises(ValueError):
            CuroboPolicy(motion_gen=_StubPlanner(horizon=8), action_horizon=True)


class TestAUsableHorizonIsHonoredExactly:
    """The accepted domain is unchanged, and the value is stored uncoerced."""

    @pytest.mark.parametrize("horizon", [1, 2, 4, 16, 64])
    def test_a_positive_int_is_accepted_and_stored_as_given(self, horizon: int) -> None:
        policy = CuroboPolicy(motion_gen=_StubPlanner(), action_horizon=horizon)
        assert policy.action_horizon == horizon
        assert type(policy.action_horizon) is int

    def test_the_default_is_unchanged(self) -> None:
        assert CuroboPolicy(motion_gen=_StubPlanner()).action_horizon == 16

    def test_the_stored_horizon_is_the_slice_width_served_per_call(self) -> None:
        """The accepted path still chunks the plan, which is what the value is for.

        Guards against a refusal being tightened into one that also breaks the
        streaming contract: 4 of an 8-waypoint plan, then the remaining 4.
        """
        policy = CuroboPolicy(motion_gen=_StubPlanner(horizon=8), action_horizon=4)
        assert len(_one_chunk(policy)) == 4
        assert len(_one_chunk(policy)) == 4

    def test_a_horizon_longer_than_the_plan_yields_the_whole_plan(self) -> None:
        """The final chunk is short rather than padded - pinned so the guard
        cannot be read as having introduced a length requirement on the plan."""
        policy = CuroboPolicy(motion_gen=_StubPlanner(horizon=5), action_horizon=16)
        assert len(_one_chunk(policy)) == 5


class TestTheHorizonIsRefusedBeforeAnyPlannerIsBuilt:
    """Ordering, pinned as a property rather than left to the line numbering.

    This was already correct - the check was the constructor's first statement -
    so it is a preservation pin, not a fix. It matters because the alternative
    placement is silently worse: ``_build_motion_gen`` imports cuRobo, allocates
    on the CUDA device and optionally warms the planner up, so a horizon rejected
    after it has run has already paid for a planner that is then discarded.
    """

    def test_no_planner_is_constructed_for_a_refused_horizon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[object] = []

        def _explode(*args: object, **kwargs: object) -> object:
            built.append(args)
            raise AssertionError("_build_motion_gen ran for a horizon that must be refused first")

        monkeypatch.setattr(CuroboPolicy, "_build_motion_gen", _explode)
        with pytest.raises(ValueError, match="action_horizon"):
            CuroboPolicy(robot_config="franka.yml", action_horizon=0, warmup=False)
        assert built == []

    def test_a_bad_horizon_is_named_ahead_of_a_missing_robot_config(self) -> None:
        """Both are refused; the horizon is reported, because it is checked first.

        Pinned so the two refusals cannot be reordered without a decision: the
        missing-config message tells the caller to pass ``robot_config=``, which
        is not the thing wrong with this call.
        """
        with pytest.raises(ValueError, match="action_horizon"):
            CuroboPolicy(action_horizon=0)


class TestNoProviderChunkCountSkipsTheSharedDomain:
    """Structural guard: a fourth provider cannot hand-roll this check.

    The defect this module fixes was not that the comparison was wrong in
    isolation - it was that a shared domain existed, two of the three providers
    accepting a chunk count used it, and the third did not. That asymmetry is
    invisible to any test of a single provider, so it is asserted over the
    package.
    """

    _FAMILY = frozenset(
        {
            "action_horizon",
            "actions_per_step",
            "actions_per_chunk",
            "rtc_execution_horizon",
        }
    )

    @staticmethod
    def _modules_accepting_a_chunk_count() -> dict[str, set[str]]:
        """Map module path -> chunk-count parameters its ``__init__`` accepts."""
        root = pathlib.Path(__file__).resolve().parents[3] / "strands_robots" / "policies"
        assert root.is_dir(), f"policies package not found at {root}"
        found: dict[str, set[str]] = {}
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            params: set[str] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if node.name != "__init__":
                    continue
                args = node.args
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                    if arg.arg in TestNoProviderChunkCountSkipsTheSharedDomain._FAMILY:
                        params.add(arg.arg)
            if params:
                found[str(path.relative_to(root.parent.parent))] = params
        return found

    def test_the_scan_is_not_vacuous(self) -> None:
        """A rename must break this guard loudly rather than empty it.

        Without this, renaming every parameter in the family - or moving the
        providers - leaves a scan that finds nothing and asserts nothing about
        each of zero modules, which passes.
        """
        found = self._modules_accepting_a_chunk_count()
        assert len(found) >= 3, f"expected at least three providers accepting a chunk count, found {found}"
        assert any("curobo" in module for module in found), (
            f"the provider this guard was written for is no longer found by it: {sorted(found)}"
        )

    def test_every_such_provider_routes_through_chunk_count_error(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[3]
        offenders: dict[str, set[str]] = {}
        for module, params in self._modules_accepting_a_chunk_count().items():
            source = (root / module).read_text(encoding="utf-8")
            if "chunk_count_error" not in source:
                offenders[module] = params
        assert not offenders, (
            "these policy providers accept a per-inference chunk count without the shared "
            f"chunk_count_error domain, so the same count they refuse may be accepted by a "
            f"sibling provider: {offenders}"
        )
