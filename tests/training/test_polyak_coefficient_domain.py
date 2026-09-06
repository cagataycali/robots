"""The rate a target network tracks its online one is checked, not just compared.

``tau`` is the Polyak coefficient of the two off-policy backends. Each spends it
in one expression per mirrored critic pair::

    tp.mul_(1.0 - spec.tau).add_(spec.tau * p)

so it does not merely tune the update - it decides whether a separate target
network exists at all, and a target network is what makes an off-policy critic's
bootstrap target stationary enough to regress onto.

Its interval was the *precedent* the two on-policy interval gates were written
against: ``discount_factor_problems`` and ``gae_lambda_problems`` both cite "the
sibling FastSAC preflight already bounds its own interval coefficient this way
(``tau`` must be in ``(0, 1]``)" as the shape they generalize. The check they
cited was a bare local comparison, ``if not 0.0 < spec.tau <= 1.0``, duplicated
verbatim in both backends - so the precedent was weaker than the generalization
it inspired, and it left two holes that generalization had already closed.

``True`` is a silent ``tau`` of **one**, because ``bool`` is an ``int``
subclass - the maximum of the interval, at which the Polyak average degenerates
to ``tp.mul_(0.0).add_(p)`` and the target network becomes a copy of the online
network on every update. Measured on a 60-timestep FastSAC run with
``validate()`` returning ``[]`` and the run reporting success: the largest
online-to-target parameter gap in the exported checkpoint was ``0.0`` exactly,
against ``9.9e-04`` for the default ``tau``, and the checkpoint was
byte-identical to a run that asked for ``tau=1.0``. A flag landed as a request to
switch target networks off, and nothing in the run said so.

A numeric string, ``None`` or a list raised ``TypeError: '<' not supported
between instances of 'float' and 'str'`` out of the comparison itself - from a
``validate`` documented to *return* its problems.

``nan`` and the two infinities were already refused, but by an accident of
Python's chained comparison rather than by a finiteness test: every comparison
against ``nan`` is False and ``inf`` is above the upper bound. They stay refused,
now with a reason that names finiteness.

Neither endpoint changes reading, so nothing that worked becomes an error:
``1.0`` stays accepted as the deliberate hard update and ``0.0`` stays refused
because it freezes the target network at its initialization. That one decision -
zero outside the interval - is what makes this domain half-open where the
on-policy pair's is closed.

Every test reaches the real ``validate`` entry point, so the wiring is covered as
well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import finite_number_error
from tests.training._spec_field_reads import reads_spec_field

# The two backends that maintain a target network and so read the coefficient.
OFF_POLICY = ("fast_sac", "fast_td3")

# Backends with no target network, which must stay silent about the field.
NO_TARGET_BACKENDS = ("ppo", "mock")

# Values a bare interval comparison took or raised on. ``True`` is the silent
# hard update; the strings, ``None`` and the list raised ``TypeError`` out of the
# comparison; ``10**400`` is a finite real with no float64 form.
UNUSABLE_WITH_NO_READING: list[Any] = [True, False, "0.005", "1.0", None, [0.005], {}, 10**400]

# Values outside the interval but with a numeric reading - refused before and
# after, on the interval rather than on the type.
OUTSIDE_THE_INTERVAL: list[Any] = [0.0, -0.005, -1.0, 1.0000001, 2.0, 100.0]

# Non-finite values the chained comparison happened to answer; still refused,
# now for finiteness rather than by accident.
NON_FINITE: list[Any] = [float("nan"), float("inf"), float("-inf")]

# Coefficients with a reading, including both spellings of the endpoint that is
# inside the interval and the scalar types a config or a sweep arrives as.
USABLE: list[Any] = [0.005, 0.5, 1.0, 1, 1e-8, Fraction(1, 200), np.float64(0.01), np.float32(0.005)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/polyak_coefficient_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _tau_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``tau``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"tau"`` substring, so an unrelated problem can neither mask a
    missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: tau ")]


class TestACoefficientWithNoReadingIsReportedRatherThanTakenOrRaisedOn:
    """The two holes the bare comparison left: a silent bool and a raising type."""

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("value", UNUSABLE_WITH_NO_READING)
    def test_it_is_reported_as_a_problem(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.tau = value
        assert _tau_problems(provider, spec), f"{provider} accepted tau={value!r}"

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("value", UNUSABLE_WITH_NO_READING)
    def test_nothing_raises_out_of_a_preflight_that_returns_its_problems(
        self, provider: str, spec: RLTrainSpec, value: Any
    ) -> None:
        """``validate`` answers with a message; a numeric string used to raise TypeError."""
        spec.tau = value
        problems = create_trainer(provider).validate(spec)  # must not raise
        assert any(p.startswith(f"{provider}: tau ") for p in problems), problems

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_a_boolean_is_refused_rather_than_read_as_the_hard_update(self, provider: str, spec: RLTrainSpec) -> None:
        """``True`` aliased ``tau=1.0``, which switches target networks off."""
        spec.tau = True
        assert _tau_problems(provider, spec)
        # The value it aliased is still accepted on its own axis - the refusal is
        # of the spelling, not of the reading.
        spec.tau = 1.0
        assert _tau_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("value", UNUSABLE_WITH_NO_READING + OUTSIDE_THE_INTERVAL + NON_FINITE)
    def test_the_problem_names_the_backend_the_field_and_the_value(
        self, provider: str, spec: RLTrainSpec, value: Any
    ) -> None:
        spec.tau = value
        (problem,) = _tau_problems(provider, spec)
        assert problem.startswith(f"{provider}: tau must be"), problem
        assert repr(value) in problem or str(value) in problem, problem


class TestTheIntervalIsUnchanged:
    """No value that had a reading before is newly refused, and none is newly taken."""

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_coefficient_reports_nothing(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.tau = value
        assert _tau_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", OFF_POLICY)
    @pytest.mark.parametrize("value", OUTSIDE_THE_INTERVAL + NON_FINITE)
    def test_a_coefficient_outside_the_interval_is_still_refused(
        self, provider: str, spec: RLTrainSpec, value: Any
    ) -> None:
        spec.tau = value
        assert _tau_problems(provider, spec)

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_the_default_spec_reports_nothing(self, provider: str, spec: RLTrainSpec) -> None:
        assert spec.tau == 0.005
        assert _tau_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_the_upper_endpoint_is_inside_the_interval(self, provider: str, spec: RLTrainSpec) -> None:
        """One is the deliberate hard update ``tp = p``, not a degenerate spelling."""
        spec.tau = 1.0
        assert _tau_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", OFF_POLICY)
    def test_zero_is_outside_it(self, provider: str, spec: RLTrainSpec) -> None:
        """Zero freezes the target parameters at their initialization for the whole run."""
        spec.tau = 0.0
        assert _tau_problems(provider, spec)


class TestTheDomainIsHalfOpenWhereTheOnPolicyOneIsClosed:
    """The one decision this interval owns, against the gate that cites it."""

    def test_zero_parts_the_two_intervals(self, spec: RLTrainSpec) -> None:
        """``gamma=0`` is a myopic agent; ``tau=0`` is a target network that never moves."""
        spec.gamma = 0.0
        spec.tau = 0.0
        problems = create_trainer("fast_sac").validate(spec)
        assert not [p for p in problems if p.startswith("fast_sac: gamma ")], problems
        assert [p for p in problems if p.startswith("fast_sac: tau ")], problems

    def test_everything_but_the_interval_is_the_shared_signed_domain(self, spec: RLTrainSpec) -> None:
        """Type, bool and finiteness verdicts read identically to every other field's."""
        for value in UNUSABLE_WITH_NO_READING + NON_FINITE:
            spec.tau = value
            expected = finite_number_error(value, "tau", "fast_sac")
            assert expected is not None, value
            assert _tau_problems("fast_sac", spec) == [expected], value


class TestBothOffPolicyBackendsAgree:
    """One rule, two callers - a second copy of it would be free to drift."""

    @pytest.mark.parametrize("value", UNUSABLE_WITH_NO_READING + OUTSIDE_THE_INTERVAL + NON_FINITE + USABLE)
    def test_the_two_backends_report_the_same_verdict(self, spec: RLTrainSpec, value: Any) -> None:
        spec.tau = value
        sac = _tau_problems("fast_sac", spec)
        td3 = _tau_problems("fast_td3", spec)
        assert bool(sac) == bool(td3), (value, sac, td3)
        # Only the backend name differs, since the message is prefixed with it.
        assert [p.replace("fast_sac:", "") for p in sac] == [p.replace("fast_td3:", "") for p in td3]


class TestABackendWithNoTargetNetworkStaysSilent:
    """Per ``TrainSpec`` a backend ignores the fields it does not support."""

    @pytest.mark.parametrize("provider", NO_TARGET_BACKENDS)
    @pytest.mark.parametrize("value", [True, float("nan"), "0.005", 0.0, 2.0])
    def test_it_reports_nothing_about_the_coefficient(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.tau = value
        assert _tau_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_TARGET_BACKENDS)
    def test_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str, spec: RLTrainSpec) -> None:
        """Non-vacuity: the same spec's own learning rate is still refused by them."""
        spec.tau = True
        spec.learning_rate = 0.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems


class TestWhatTheUncheckedCoefficientDidToTheTargetNetwork:
    """Executable premises for the measured harm, without a full training run."""

    def test_a_boolean_coefficient_is_the_maximum_of_the_interval(self) -> None:
        """``bool`` is an ``int`` subclass, so a bare interval test takes ``True``."""
        assert isinstance(True, int)
        assert float(True) == 1.0
        assert 0.0 < True <= 1.0  # noqa: E712 - the bare test this replaces

    def test_the_maximum_makes_the_target_a_copy_of_the_online_network(self) -> None:
        """At ``tau=1`` the Polyak average is ``tp = p`` - no target network at all."""
        import torch

        for tau, expect_gap in ((0.005, True), (1.0, False), (True, False)):
            online = torch.tensor([1.0, 2.0])
            target = torch.zeros(2)
            target.mul_(1.0 - tau).add_(tau * online)
            gap = float((online - target).abs().max())
            assert (gap > 0.0) is expect_gap, (tau, gap)

    def test_zero_leaves_the_target_at_its_initialization(self) -> None:
        """The reason the interval is half-open rather than closed."""
        import torch

        online = torch.tensor([1.0, 2.0])
        target = torch.zeros(2)
        target.mul_(1.0 - 0.0).add_(0.0 * online)
        assert torch.equal(target, torch.zeros(2))


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = (root / "_validate.py").resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_coefficient(source: str) -> bool:
    """Does *source* read ``spec.tau``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("tau",))


def _compares_the_coefficient(source: str) -> bool:
    """Does *source* use ``spec.tau`` as an operand of a comparison?

    That is the bare interval test this gate replaces. Detected on the syntax
    tree rather than in the text so that a docstring *quoting* the removed
    comparison - which the gate's own documentation does, to record what it
    replaced - is not mistaken for a re-implementation of it.
    """

    def is_the_field(node: ast.AST) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "tau"

    return any(
        isinstance(node, ast.Compare) and any(is_the_field(operand) for operand in [node.left, *node.comparators])
        for node in ast.walk(ast.parse(source))
    )


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_polyak_coefficient_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForThePolyakCoefficientDomain:
    """No backend that keeps a target network may skip the domain or re-derive it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.tau`` must route it through the shared gate, so a
    third backend that starts maintaining a target network fails this test until
    it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_coefficient(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.tau without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_coefficient(p.read_text())}
        assert readers == {"fast_sac.py", "fast_td3.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def update(self, spec):\n    return 1.0 - spec.tau\n"
        assert _reads_the_coefficient(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """The bare interval comparison this replaces, in either backend, is a defect."""
        offenders = [p.name for p in _training_modules() if _compares_the_coefficient(p.read_text())]
        assert offenders == [], f"modules compare spec.tau locally: {offenders}"

    def test_the_comparison_scanner_detects_the_bare_test_it_replaces(self) -> None:
        """Non-vacuity, and the reason the scan is on syntax rather than on text.

        The exact comparison removed from both backends must be reported, while
        prose that *quotes* it - as this gate's own docstrings do, to say what
        they replaced - must not be. A substring scan cannot tell those apart:
        it reported ``base.py`` for a docstring citing the defect.
        """
        assert _compares_the_coefficient("def validate(self, spec):\n    return not 0.0 < spec.tau <= 1.0\n")
        assert not _compares_the_coefficient('"""A gate replacing `if not 0.0 < spec.tau <= 1.0`."""\n')
