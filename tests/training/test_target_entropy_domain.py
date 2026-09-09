"""FastSAC's entropy temperature is checked where it is aimed, not only where it starts.

``target_entropy`` is the third and last caller-supplied field of FastSAC's
temperature block, and the only one nothing judged. It is the constant the
temperature is optimized *toward* - the one term of the temperature loss a caller
supplies::

    alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

reaching that expression through an unconditional ``float(spec.target_entropy)``
coercion in ``setup``. Its two neighbours - the temperature's starting value and
the rate that moves it - were both put behind a positive-finite domain, and this
field was explicitly left out of both because it is *signed* by construction: it
defaults to ``-num_actions``, so a positive-finite domain would refuse every
reading of it. The domain it does take is the signed counterpart the on-policy
loss weights already read, ``finite_number_error``: any finite real, either sign,
because no endpoint of a target entropy in nats is decidable while a value that is
not a finite real has no reading at all.

Measured on a 40-timestep run, ``validate()`` returning ``[]`` for all of them.
``True`` is a silent sign flip - a target entropy of ``+1.0`` where the field's own
default is ``-6.0`` for this env - and the run reported success while checkpointing
``log_alpha == -0.0018031001091003418`` against the honored run's
``-0.001801646314561367``, so it drove the temperature somewhere else rather than
harmlessly. ``nan`` / ``inf`` / ``-inf`` poison ``alpha``, which scales the entropy
term of both the critic's TD target and the actor loss, and the next rollout raises
``ValueError`` from inside ``torch.distributions.Normal`` about a tensor of ``nan``
policy means - naming that distribution's parameter rather than the field, after
the env, both networks and a full rollout have been built. A list or a dict raises
``TypeError`` out of the ``float()`` coercion in ``setup``.

Two scoping decisions are pinned as well as the domain. ``None`` is exempt rather
than refused: the field is annotated ``float | None`` and ``None`` is the
documented request for the ``-num_actions`` heuristic, unlike the two neighbours
whose ``float`` annotations carry concrete defaults. And the check is *not*
conditioned on ``autotune_alpha`` even though only the tuning branch spends the
value, because the coercion that reads it is unconditional - with tuning off a list
raises the same ``TypeError`` while ``nan`` reaches a successful run that never
spends it, so a check scoped to the tuning branch would let the raising shape
through for the untuned one.

Every test reaches the real ``validate`` entry point, so the wiring is covered as
well as the domain.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training import create_trainer
from strands_robots.training.base import Trainer
from strands_robots.training.rl import RLTrainSpec
from strands_robots.utils import finite_number_error
from tests.training._spec_field_reads import reads_spec_field

# The one backend that optimizes a temperature against a target entropy.
OFF_POLICY = "fast_sac"

# Backends that never read ``target_entropy``.
NO_TARGET_BACKENDS = ("ppo", "fast_td3", "mock")

# Values with no reading as a target entropy. ``True`` / ``False`` are a silent
# ``+1.0`` / ``0.0`` a bare comparison would take; the non-finite values poison
# ``alpha``; the rest raise out of the ``float()`` coercion in ``setup``.
UNUSABLE: list[Any] = [
    True,
    False,
    float("nan"),
    float("inf"),
    float("-inf"),
    "-6",
    "-6.0",
    [-6.0],
    {},
    10**400,
]

# Target entropies with a reading, including the negative default the field is
# documented to substitute and the spellings a config or a probed value arrives as.
USABLE: list[Any] = [-6.0, -6, -3.5, 0.0, 1.5, -1e-3, np.float64(-6.0), np.float32(-2.5)]


@pytest.fixture
def spec() -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the field under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir="/tmp/target_entropy_domain",
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _target_entropy_problems(provider: str, spec: RLTrainSpec) -> list[str]:
    """Problems the real ``validate`` entry point reports about ``target_entropy``.

    Filtered on the shared domains' ``"{context}: {param} "`` message shape rather
    than on a bare ``"target_entropy"`` substring, so an unrelated problem can
    neither mask a missing refusal nor be mistaken for one.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: target_entropy ")]


class TestTheOffPolicyBackendRefusesATargetWithNoReading:
    """FastSAC refuses every value that is not a finite real target entropy."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: RLTrainSpec, value: Any) -> None:
        spec.target_entropy = value
        assert _target_entropy_problems(OFF_POLICY, spec), f"fast_sac accepted target_entropy={value!r}"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_problem_names_the_field_and_the_value(self, spec: RLTrainSpec, value: Any) -> None:
        spec.target_entropy = value
        (problem,) = _target_entropy_problems(OFF_POLICY, spec)
        assert problem.startswith("fast_sac: target_entropy must be"), problem
        assert repr(value) in problem or str(value) in problem, problem

    def test_a_refusal_does_not_hide_the_rest_of_the_temperature_block(self, spec: RLTrainSpec) -> None:
        """All three temperature fields are reported at once, not one round at a time."""
        spec.target_entropy = float("nan")
        spec.init_alpha = 0.0
        spec.alpha_lr = 0.0
        problems = create_trainer(OFF_POLICY).validate(spec)
        for field in ("target_entropy", "init_alpha", "alpha_lr"):
            assert any(p.startswith(f"fast_sac: {field} ") for p in problems), (field, problems)


class TestTheUsableDomainIsUntouched:
    """A finite real target entropy of either sign is not newly refused."""

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_target_reports_nothing(self, spec: RLTrainSpec, value: Any) -> None:
        spec.target_entropy = value
        assert _target_entropy_problems(OFF_POLICY, spec) == []

    def test_the_negative_default_is_inside_the_domain(self, spec: RLTrainSpec) -> None:
        """The substituted ``-num_actions`` is the reading a positive domain would refuse."""
        spec.target_entropy = -6.0
        assert _target_entropy_problems(OFF_POLICY, spec) == []


class TestTheNoneSentinelIsExemptRatherThanRefused:
    """``None`` is the documented request for ``-num_actions``, not a value."""

    def test_the_default_spec_reports_nothing(self, spec: RLTrainSpec) -> None:
        assert spec.target_entropy is None
        assert _target_entropy_problems(OFF_POLICY, spec) == []

    def test_the_field_declares_the_sentinel_in_its_annotation(self) -> None:
        """Executable premise for the exemption, unlike the two neighbouring fields."""
        assert RLTrainSpec.__annotations__["target_entropy"] == "float | None"
        assert RLTrainSpec.__annotations__["init_alpha"] == "float"
        assert RLTrainSpec.__annotations__["alpha_lr"] == "float"

    def test_the_backend_substitutes_the_heuristic_for_it(self) -> None:
        """Executable premise: ``None`` reaches a substitution, not the coercion."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert (
            "float(spec.target_entropy) if spec.target_entropy is not None else -float(self.env.num_actions)" in setup
        )


class TestBothTemperatureBranchesAreCovered:
    """The coercion that reads the field runs whether or not the temperature is tuned.

    Only the tuning branch *spends* the value, but the ``float()`` that reads it is
    unconditional, so a non-real value raises in ``setup`` either way. Scoping the
    check to ``autotune_alpha`` would let that shape through for the untuned branch.
    """

    @pytest.mark.parametrize("autotune", [True, False])
    @pytest.mark.parametrize("value", [[-6.0], {}, "-6"])
    def test_a_non_real_value_is_reported_on_either_branch(self, spec: RLTrainSpec, value: Any, autotune: bool) -> None:
        spec.autotune_alpha = autotune
        spec.target_entropy = value
        assert _target_entropy_problems(OFF_POLICY, spec), (autotune, value)

    def test_the_coercion_is_not_under_the_tuning_branch(self) -> None:
        """Executable premise: the read precedes ``if self.autotune_alpha``."""
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        setup = inspect.getsource(FastSacTrainer.setup)
        assert setup.index("float(spec.target_entropy)") < setup.index("if self.autotune_alpha:")


class TestABackendThatIgnoresTheFieldStaysSilent:
    """Per ``TrainSpec`` a backend ignores the fields it does not support."""

    @pytest.mark.parametrize("provider", NO_TARGET_BACKENDS)
    @pytest.mark.parametrize("value", [float("nan"), True, [-6.0]])
    def test_it_reports_nothing_about_the_target(self, provider: str, spec: RLTrainSpec, value: Any) -> None:
        spec.target_entropy = value
        assert _target_entropy_problems(provider, spec) == []

    @pytest.mark.parametrize("provider", NO_TARGET_BACKENDS)
    def test_silence_is_scoping_rather_than_an_empty_preflight(self, provider: str, spec: RLTrainSpec) -> None:
        """Non-vacuity: the same spec's own learning rate is still refused by them."""
        spec.target_entropy = float("nan")
        spec.learning_rate = 0.0
        problems = create_trainer(provider).validate(spec)
        assert any(p.startswith(f"{provider}: learning_rate ") for p in problems), problems


class TestTheDomainIsTheSharedSignedOne:
    """The verdict is the shared helper's, not a local re-derivation."""

    @pytest.mark.parametrize("value", UNUSABLE + USABLE)
    def test_the_verdict_matches_the_shared_domain(self, spec: RLTrainSpec, value: Any) -> None:
        spec.target_entropy = value
        expected = finite_number_error(value, "target_entropy", OFF_POLICY)
        assert _target_entropy_problems(OFF_POLICY, spec) == ([expected] if expected else [])

    def test_the_domain_is_signed_rather_than_positive(self) -> None:
        """The premise the two neighbouring gates could not express."""
        from strands_robots.utils import positive_finite_number_error

        assert finite_number_error(-6.0, "target_entropy", OFF_POLICY) is None
        assert positive_finite_number_error(-6.0, "target_entropy", OFF_POLICY) is not None


class TestWhatTheUncheckedValueDidToTheTemperature:
    """Executable premises for the measured harm, without a full training run."""

    def test_a_non_finite_target_makes_the_temperature_loss_non_finite(self) -> None:
        """``nan`` reaches ``alpha``, which scales the entropy term of both losses."""
        import torch

        log_alpha = torch.tensor(0.0, requires_grad=True)
        logp = torch.zeros(4, 1)
        for target, finite in ((-6.0, True), (float("nan"), False), (float("inf"), False)):
            alpha_loss = -(log_alpha * (logp + target).detach()).mean()
            assert math.isfinite(float(alpha_loss.detach())) is finite, target

    def test_a_boolean_target_is_a_different_target_than_the_default(self) -> None:
        """``True`` is ``+1.0``, not the ``-num_actions`` the field documents."""
        assert float(True) == 1.0
        assert float(True) != -6.0
        # And a bare sign test would take it: bool is an int subclass.
        assert isinstance(True, int)


# --- one owner for the domain ------------------------------------------------


def _training_modules() -> list[pathlib.Path]:
    """Every training module except the one that owns the gate."""
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = (root / "_validate.py").resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_target(source: str) -> bool:
    """Does *source* read ``spec.target_entropy``, by name or a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("target_entropy",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_target_entropy_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheTargetEntropyDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.target_entropy`` must route it through the shared
    gate, so a second backend that starts tuning a temperature fails this test
    until it does.
    """

    def test_every_module_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = [
            p.name
            for p in _training_modules()
            if _reads_the_target(p.read_text()) and not _calls_the_gate(p.read_text())
        ]
        assert adrift == [], f"modules read spec.target_entropy without the shared gate: {adrift}"

    def test_the_reader_set_is_the_expected_one(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        readers = {p.name for p in _training_modules() if _reads_the_target(p.read_text())}
        assert readers == {"fast_sac.py"}, readers

    def test_the_scanner_detects_a_planted_reader(self) -> None:
        """A module reading the field without the gate is really reported."""
        planted = "def setup(self, spec):\n    return float(spec.target_entropy)\n"
        assert _reads_the_target(planted)
        assert not _calls_the_gate(planted)

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local comparison would drift from the shared rule."""
        offenders = [
            p.name
            for p in _training_modules()
            if "spec.target_entropy <" in p.read_text() or "spec.target_entropy >" in p.read_text()
        ]
        assert offenders == [], f"modules compare spec.target_entropy locally: {offenders}"
