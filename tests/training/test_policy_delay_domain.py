"""FastTD3's delayed-update cadence is checked before it can silently skip the actor.

TD3's actor and target networks move only every ``policy_delay``-th gradient
step, and the field is consumed as the modulus of the one test that decides
whether they move at all::

    if self._update_count % spec.policy_delay == 0:  # actor + Polyak step

Nothing downstream judges that modulus. Measured on a CPU FastTD3 run of 6
gradient updates: the shipped ``2`` takes 3 actor updates; ``nan`` takes **0**
(``n % nan`` is ``nan``, which compares unequal to everything) while the
critics train normally, so the run reports success and checkpoints a
deployable actor that never took a gradient step; ``2.5`` takes 1 (only the
integer multiples of 2.5 satisfy the test) and ``True`` takes 6 - each a
silently different cadence from the one the caller named; ``0`` raises
``ZeroDivisionError`` and a string or ``None`` a bare ``TypeError``, from
inside the update loop after the env, the networks, the optimizers and the
replay buffer are built.

The domain is the shared strict-``int``
:func:`~strands_robots.utils.positive_count_error` - the rule this package
already applies to every count consumed as a modulus or ``range()`` bound -
with ``1`` first-class: a delay of one is TD3 with the delay disabled, a
configuration rather than a defect. Every test reaches the real ``validate``
entry point, so the wiring is covered as well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

from strands_robots.training import create_trainer
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import positive_count_error
from tests.training._spec_field_reads import reads_spec_field

# The one backend that delays its policy updates.
DELAYED_BACKEND = "fast_td3"

# Backends that never read ``policy_delay``: PPO has no target networks and
# SAC moves its actor every gradient step.
NO_DELAY_BACKENDS = ("ppo", "fast_sac", "mock")

# Values that are not a usable modulus. ``nan`` silently never satisfies the
# cadence test (zero actor updates under success), ``True`` is a silent delay
# of one, a fraction a silently different cadence, ``0`` a ZeroDivisionError
# and the rest raise ``TypeError`` out of ``%`` mid-update.
UNUSABLE: list[Any] = [
    0,
    -1,
    -3,
    True,
    False,
    2.5,
    2.0,
    float("nan"),
    float("inf"),
    float("-inf"),
    "2",
    None,
    [2],
    {},
]

# Delays the modulus honors, ``1`` (delay disabled) included.
USABLE: list[Any] = [1, 2, 3, 10, 10**6]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(
        output_dir="/tmp/policy_delay_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _policy_delay_reports(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``policy_delay``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"policy_delay"`` substring, so an unrelated problem can
    neither mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: policy_delay ")]


class TestTheDelayedBackendRefusesAModulusItCannotHonor:
    """FastTD3 refuses every value the cadence test cannot take as a delay."""

    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, value: Any) -> None:
        spec.policy_delay = value
        assert _policy_delay_reports(DELAYED_BACKEND, spec), f"fast_td3 accepted policy_delay={value!r}"

    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_the_problem_names_the_field_and_the_value(self, spec: RLTrainSpec, value: Any) -> None:
        spec.policy_delay = value
        (problem,) = _policy_delay_reports(DELAYED_BACKEND, spec)
        assert problem.startswith("fast_td3: policy_delay must be a positive integer"), problem
        assert repr(value) in problem, problem

    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_validate_returns_rather_than_raising(self, spec: RLTrainSpec, value: Any) -> None:
        """A ``validate`` documented to return problems must not raise one."""
        spec.policy_delay = value
        assert isinstance(create_trainer(DELAYED_BACKEND).validate(spec), list)


class TestTheUsableDomainIsUntouched:
    """A delay the modulus honors is not newly refused."""

    @pytest.mark.parametrize("value", USABLE, ids=repr)
    def test_a_usable_delay_reports_nothing(self, spec: RLTrainSpec, value: Any) -> None:
        spec.policy_delay = value
        assert _policy_delay_reports(DELAYED_BACKEND, spec) == []

    def test_the_default_spec_reports_nothing(self, spec: RLTrainSpec) -> None:
        """The shipped ``2`` default must not trip the new gate."""
        assert _policy_delay_reports(DELAYED_BACKEND, spec) == []
        assert spec.policy_delay == 2

    def test_a_delay_of_one_is_a_configuration(self, spec: RLTrainSpec) -> None:
        """``1`` disables the delay; disabling a mechanism is not a defect."""
        spec.policy_delay = 1
        assert _policy_delay_reports(DELAYED_BACKEND, spec) == []


class TestABackendWithNoDelayStaysQuiet:
    """A backend that never reads the field must not report on it."""

    @pytest.mark.parametrize("provider", NO_DELAY_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_it_reports_nothing_about_the_delay(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.policy_delay = value
        assert _policy_delay_reports(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_DELAY_BACKENDS)
    def test_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str, spec: RLTrainSpec) -> None:
        """The same spec's *own* learning rate is still refused by those backends."""
        spec.learning_rate = 0.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems


class TestTheGateAddsNothingToTheSharedDomain:
    """The verdict is the shared rule's, so the two cannot drift apart."""

    @pytest.mark.parametrize("value", UNUSABLE + USABLE, ids=repr)
    def test_the_verdict_matches_the_shared_domain(self, spec: RLTrainSpec, value: Any) -> None:
        spec.policy_delay = value
        shared = positive_count_error(value, "policy_delay", DELAYED_BACKEND)
        assert _policy_delay_reports(DELAYED_BACKEND, spec) == ([shared] if shared is not None else [])


class TestTheModulusSilentlySkipsTheActor:
    """The measured consequence: a cadence the test never satisfies trains no actor.

    This is why the field needed a preflight rather than being left to the
    update loop: the critics keep training, the losses are real numbers, and
    the run reports success - only the deployable actor is missing its every
    gradient step.
    """

    def test_a_nan_modulus_never_fires_the_cadence(self) -> None:
        pytest.importorskip("torch")
        assert not any(n % float("nan") == 0 for n in range(1, 1000))

    def test_the_honored_default_fires_it(self) -> None:
        """Control: the shipped delay of 2 fires on every second update."""
        assert [n for n in range(1, 7) if n % 2 == 0] == [2, 4, 6]

    def test_the_cadence_is_the_one_expression_the_field_reaches(self) -> None:
        """Executable premise: the modulus is where the field is consumed."""
        from strands_robots.training.rl.fast_td3 import FastTd3Trainer

        update = inspect.getsource(FastTd3Trainer.update)
        assert "self._update_count % spec.policy_delay == 0" in update


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = (root / "_validate.py").resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_delay(source: str) -> bool:
    """Does *source* read ``spec.policy_delay``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("policy_delay",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_policy_delay_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForThePolicyDelayDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.policy_delay`` must route it through the shared
    gate, so a second backend that starts delaying its policy fails this test
    until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_delay(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.policy_delay without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_delay(p.read_text())}
        assert readers == {"fast_td3.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def update(self):\n    return self.n % spec.policy_delay\n"
        assert _reads_the_delay(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local comparison would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if "spec.policy_delay <" in p.read_text() or "spec.policy_delay >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.policy_delay locally: {offenders}"
