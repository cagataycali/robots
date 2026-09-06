"""Every field-scoped shared-domain guard sees a table-driven read of its field.

Each shared numeric domain on :class:`~strands_robots.training.base.Trainer`
documents a biconditional - a backend that reads the field MUST route it
through the gate, one that ignores it MUST NOT report on it - and the guard for
each domain pins the first half with a scope *derived from the tree*, so that
"a new backend that starts reading the field fails this test until it does".

That promise rests entirely on the guard's notion of "reads the field". A
backend can read a spec field two ways: by name (``spec.seed``) or through a
forwarding table (``getattr(spec, field)`` over a tuple of field names, which
is how a transport-only provider serializes every field it passes on). A scan
keyed on the first form alone certifies a complete sweep while a table-driven
reader sits outside it, and the biconditional is then unenforced for exactly
that backend - silently, because the guard reports a clean tree.

This grades the guards from the outside rather than trusting each to grade
itself: the set of field-scoped guards is discovered structurally, so a new
domain guard is held to the same rule the moment it lands.

That promise is only as wide as the discovery, and a discovery keyed on the
*name* of a helper is not a structural one. The guards spell the helper that
lists the backend modules two ways - ``_trainer_modules`` and
``_training_modules`` - so a rule keyed on either spelling grades the guards
that use it and reports a clean sweep over the rest.
:func:`is_field_scoped_guard` keys on the two properties that make a guard
gradeable instead (it has one reader helper, and its scope is rooted at the
backend tree), and is the one rule both the sweep over the real tree and the
constructed exemplars in :class:`TestTheDiscoveryDoesNotDependOnAHelperName`
consult.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
from typing import Any

import pytest

from strands_robots.training.base import Trainer
from strands_robots.training.sagemaker import _FORWARDED_FIELDS
from tests.training._spec_field_reads import reads_spec_field

# The gates whose scope is a field rather than every backend, mapped to the
# TrainSpec fields each owns. The learning-rate gate is deliberately absent: no
# backend may skip it, so its guard scans Trainer subclasses rather than field
# reads and needs no notion of "reads the field" at all.
FIELD_SCOPED_GATES: dict[str, tuple[str, ...]] = {
    "_checkpoint_cadence_problems": ("save_freq",),
    "_seed_problems": ("seed",),
    "_validation_episodes_problems": ("val_episodes",),
    "_lora_hyperparameter_problems": ("lora_r", "lora_alpha"),
    "_launch_topology_problems": ("num_gpus", "num_nodes"),
    # The RL run-size gate. Its two fields live on ``RLTrainSpec`` and no
    # provider forwards them, so it is graded on the reader scan only.
    "_rl_run_size_problems": ("total_timesteps", "rollout_steps"),
    # The RL replay-count gate. Its three fields live on ``RLTrainSpec`` and no
    # provider forwards them, so it is graded on the reader scan only.
    "_rl_replay_problems": ("buffer_size", "batch_size", "gradient_steps"),
    # The RL-hyperparameter gates. Their fields live on ``RLTrainSpec`` and no
    # provider forwards them today, so they are graded on the reader scan only
    # (see TestTheForwardingProviderIsInScopeForEveryGateItReads, which derives
    # its own scope from what is actually forwarded).
    "_discount_factor_problems": ("gamma",),
    "_gae_lambda_problems": ("lam",),
    "_optimization_epochs_problems": ("num_learning_epochs",),
    "_temperature_learning_rate_problems": ("alpha_lr",),
    "_initial_temperature_problems": ("init_alpha",),
    "_target_entropy_problems": ("target_entropy",),
    "_gradient_clip_problems": ("max_grad_norm",),
    "_loss_weight_problems": ("value_loss_coef", "entropy_coef"),
    "_clip_range_problems": ("clip_param",),
    "_policy_delay_problems": ("policy_delay",),
    # The Polyak-coefficient gate. Its field lives on ``RLTrainSpec`` and no
    # provider forwards it, so it is graded on the reader scan only - and that
    # scan finds two backends rather than one, since both off-policy backends
    # maintain a target network.
    "_polyak_coefficient_problems": ("tau",),
    "_td3_noise_problems": ("exploration_noise_std", "target_noise_std", "target_noise_clip"),
    # The RL checkpoint-interval gate. Its field lives on ``RLTrainSpec`` and no
    # provider forwards it, so it is graded on the reader scan only - and that
    # scan is the *secondary* derivation for this guard, whose primary scope is
    # the BaseRLAlgo hierarchy: PPO inherits the loop that reads the field and
    # never names it.
    "_rl_checkpoint_interval_problems": ("log_interval",),
    # The network-architecture gate. Its one field is a *sequence*, and it is
    # scoped like the learning rate across the RL backends (all three build
    # their actor and critics from it) while still being field-scoped overall,
    # since a supervised backend takes its architecture from the checkpoint.
    "_network_width_problems": ("hidden_dims",),
}


def _scans_the_backend_tree(tree: ast.AST) -> bool:
    """Does the module root a backend scan at the tree that defines ``Trainer``?

    That rooting is what makes a guard's scope *derived from the tree* rather
    than listed, which is the property this meta-guard needs to hold of the
    guards it grades. Detected structurally - a ``inspect.getfile(Trainer)``
    call - rather than through the name of the helper that wraps it, because
    that name is incidental to the property: the guards spell it both
    ``_trainer_modules`` and ``_training_modules``, so a discovery keyed on one
    spelling silently drops every guard that uses the other.
    """
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getfile"
        and any(isinstance(arg, ast.Name) and arg.id == "Trainer" for arg in node.args)
        for node in ast.walk(tree)
    )


def is_field_scoped_guard(source: str) -> bool:
    """Does *source* look like a field-scoped domain guard this must grade?

    Two properties, both structural:

    * it derives its scope from a reader scan - exactly one ``_reads...``
      helper, which is the scan this meta-guard grades and the one
      :func:`_reader_helper` resolves; and
    * that scope is rooted at the backend tree
      (:func:`_scans_the_backend_tree`), which is what makes it derived rather
      than listed.

    Neither property is the *name* of the helper that carries it. That
    distinction is the point: a guard lists the backend modules through a helper
    it spells either ``_trainer_modules`` or ``_training_modules``, so a rule
    keyed on one spelling drops every guard using the other while reporting a
    clean sweep of the rest.

    The guards that pin a domain no backend may skip (learning rate, run size)
    have no reader helper at all - they scan ``Trainer`` subclasses rather than
    field reads - so they do not qualify, which is correct: there is no notion
    of "reads the field" for this meta-guard to grade in them.
    """
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    readers = [n for n in names if n.startswith("_reads")]
    return len(readers) == 1 and _scans_the_backend_tree(tree)


def _guard_modules() -> dict[str, Any]:
    """The field-scoped domain guards, discovered by structure not by name.

    Membership is decided by :func:`is_field_scoped_guard`, so the sweep over
    the real tree and the constructed exemplars in
    :class:`TestTheDiscoveryDoesNotDependOnAHelperName` grade the same rule.
    Discovering the guards rather than listing them means a new domain guard is
    held to this one the moment it lands.
    """
    here = pathlib.Path(__file__).parent
    guards: dict[str, Any] = {}
    for path in sorted(here.glob("test_*_domain.py")):
        if not is_field_scoped_guard(path.read_text()):
            continue
        guards[path.name] = importlib.import_module(f"tests.training.{path.stem}")
    return guards


#: The subset of :data:`FIELD_SCOPED_GATES` whose fields the forwarding provider
#: actually passes on, derived from its table rather than assumed. The
#: RL-hyperparameter gates own ``RLTrainSpec`` fields that no provider forwards,
#: so asserting a forwarded read of them would assert a premise that is false;
#: they are graded on the reader scan alone. ``TestTheForwardingProviderIsInScope``
#: pins this set so a field leaving ``_FORWARDED_FIELDS`` still surfaces here.
FORWARDED_GATES: dict[str, tuple[str, ...]] = {
    gate: forwarded
    for gate, fields in FIELD_SCOPED_GATES.items()
    if (forwarded := tuple(f for f in fields if f in _FORWARDED_FIELDS))
}


def _reader_helper(module: Any) -> Any:
    """The single ``_reads...`` helper a field-scoped guard derives its scope from."""
    helpers = [
        getattr(module, name) for name in dir(module) if name.startswith("_reads") and callable(getattr(module, name))
    ]
    assert len(helpers) == 1, f"{module.__name__} has {len(helpers)} reader helpers"
    return helpers[0]


def _table_driven_reader(field: str) -> str:
    """A backend that reads *field* only through a forwarding table."""
    return f'FIELDS = ("{field}",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in FIELDS]\n'


class TestEveryFieldScopedGuardSeesBothFormsOfARead:
    """The headline: a reader scan must recognize a table-driven read."""

    def test_the_scan_finds_the_field_scoped_guards(self) -> None:
        """Non-vacuity: a scan that matched nothing would grade nothing."""
        assert set(_guard_modules()) == {
            "test_checkpoint_cadence_domain.py",
            "test_clip_range_domain.py",
            "test_discount_factor_domain.py",
            "test_gae_lambda_domain.py",
            "test_gradient_clip_domain.py",
            "test_initial_temperature_domain.py",
            "test_launch_topology_domain.py",
            "test_lora_hyperparameter_domain.py",
            "test_loss_weight_domain.py",
            "test_network_width_domain.py",
            "test_optimization_epochs_domain.py",
            "test_policy_delay_domain.py",
            "test_polyak_coefficient_domain.py",
            "test_rl_run_size_domain.py",
            "test_rl_checkpoint_interval_domain.py",
            "test_rl_replay_domain.py",
            "test_seed_domain.py",
            "test_target_entropy_domain.py",
            "test_td3_noise_domain.py",
            "test_temperature_learning_rate_domain.py",
            "test_validation_episodes_domain.py",
        }

    @pytest.mark.parametrize("guard_name", sorted(_guard_modules()))
    def test_it_sees_a_table_driven_read(self, guard_name: str) -> None:
        module = _guard_modules()[guard_name]
        reads = _reader_helper(module)
        gate = next(g for g in FIELD_SCOPED_GATES if any(g in line for line in inspect.getsource(module).splitlines()))
        for field in FIELD_SCOPED_GATES[gate]:
            assert reads(_table_driven_reader(field)), (
                f"{guard_name} does not see a table-driven read of spec.{field}, "
                "so a backend that forwards the field by name is outside its derived scope"
            )

    @pytest.mark.parametrize("guard_name", sorted(_guard_modules()))
    def test_it_still_sees_a_read_by_name(self, guard_name: str) -> None:
        """The form it already recognized must keep being recognized."""
        module = _guard_modules()[guard_name]
        reads = _reader_helper(module)
        gate = next(g for g in FIELD_SCOPED_GATES if any(g in line for line in inspect.getsource(module).splitlines()))
        for field in FIELD_SCOPED_GATES[gate]:
            assert reads(f"def validate(self, spec):\n    return [spec.{field}]\n")


class TestTheForwardingProviderIsInScopeForEveryGateItReads:
    """The reader the literal-only scans could not see, on the real tree."""

    def test_the_forwarded_gates_are_the_expected_ones(self) -> None:
        """Non-vacuity: a field leaving the table must surface here, not silently.

        :data:`FORWARDED_GATES` is derived, so a field dropped from
        ``_FORWARDED_FIELDS`` removes its gate from the parametrization below
        rather than failing it. Pinning the set is what keeps that visible.
        """
        assert set(FORWARDED_GATES) == {
            "_checkpoint_cadence_problems",
            "_launch_topology_problems",
            "_lora_hyperparameter_problems",
            "_seed_problems",
            "_validation_episodes_problems",
        }

    @pytest.mark.parametrize(("gate", "fields"), sorted(FORWARDED_GATES.items()))
    def test_it_is_discovered_as_a_reader(self, gate: str, fields: tuple[str, ...]) -> None:
        source = pathlib.Path(inspect.getfile(Trainer)).parent.joinpath("sagemaker.py").read_text()
        assert reads_spec_field(source, fields)

    @pytest.mark.parametrize(("gate", "fields"), sorted(FORWARDED_GATES.items()))
    def test_it_routes_that_read_through_the_shared_gate(self, gate: str, fields: tuple[str, ...]) -> None:
        """Being in scope is only useful if the gate is then enforced on it."""
        source = pathlib.Path(inspect.getfile(Trainer)).parent.joinpath("sagemaker.py").read_text()
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert gate in calls, f"sagemaker.py forwards {fields} without calling {gate}"


class TestTheSharedRuleIsPrecise:
    """Both halves of the table form are required, so the rule cannot over-reach."""

    def test_a_field_name_in_a_string_alone_is_not_a_read(self) -> None:
        """A message or a docstring naming the field reads nothing."""
        source = 'def validate(self, spec):\n    return ["seed must be positive"]\n'
        assert not reads_spec_field(source, ("seed",))

    def test_a_getattr_on_spec_for_other_fields_is_not_a_read(self) -> None:
        """Forwarding a table that does not contain the field reads nothing."""
        source = 'FIELDS = ("steps",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in FIELDS]\n'
        assert not reads_spec_field(source, ("seed",))

    def test_a_getattr_on_something_else_is_not_a_read(self) -> None:
        source = 'def validate(self, spec):\n    return [getattr(self, "seed")]\n'
        assert not reads_spec_field(source, ("seed",))

    def test_an_unrelated_module_reads_nothing(self) -> None:
        assert not reads_spec_field("x = 1\n", ("seed",))


#: A guard's reader helper: the scan this meta-guard grades.
_A_READER = 'def _reads_the_thing(source: str) -> bool:\n    return "thing" in source\n'

#: A scope helper rooted at the backend tree, which is what makes a guard's
#: scope derived. The name is a parameter because the name is the thing that
#: must not matter.
_A_ROOTED_SCOPE = (
    "def {helper}() -> list[pathlib.Path]:\n"
    "    root = pathlib.Path(inspect.getfile(Trainer)).parent\n"
    "    return sorted(root.rglob('*.py'))\n"
)

#: A scope that is *listed* rather than derived, under the name the previous
#: rule keyed on - so accepting it would be over-reach, not compatibility.
_A_LISTED_SCOPE = "def _trainer_modules() -> list[str]:\n    return ['ppo.py', 'fast_sac.py']\n"


class TestTheDiscoveryDoesNotDependOnAHelperName:
    """The hole this closed: eight guards were outside the sweep by one identifier.

    Every guard here derives its scope by listing the backend modules, and spells
    the helper that does it either ``_trainer_modules`` or ``_training_modules``.
    Keying discovery on one spelling left the guards using the other ungraded -
    silently, because the sweep reported a clean tree over the ones it could see.
    """

    @pytest.mark.parametrize("helper", ["_trainer_modules", "_training_modules", "_backend_modules"])
    def test_a_guard_qualifies_whichever_way_it_names_its_scope_helper(self, helper: str) -> None:
        assert is_field_scoped_guard(_A_READER + _A_ROOTED_SCOPE.format(helper=helper))

    def test_a_listed_scope_does_not_qualify(self) -> None:
        """Not derived from the tree, so this meta-guard's premise does not hold."""
        assert not is_field_scoped_guard(_A_READER + _A_LISTED_SCOPE)

    def test_a_scope_rooted_somewhere_else_does_not_qualify(self) -> None:
        """ "Rooted at the backend tree" is the property, not "calls ``getfile``".

        The guards resolve two locations that way: the backend tree, and the
        module owning the gate (to exclude it from their own scan). A guard whose
        scope came from the second is scanning a different population, so this
        meta-guard's premise does not hold of it.
        """
        elsewhere = (
            "def _trainer_modules() -> list[pathlib.Path]:\n"
            "    root = pathlib.Path(inspect.getfile(seed_problems)).parent\n"
            "    return sorted(root.rglob('*.py'))\n"
        )
        assert not is_field_scoped_guard(_A_READER + elsewhere)

    def test_a_guard_with_no_reader_helper_does_not_qualify(self) -> None:
        """The learning-rate and run-size shape: no notion of "reads the field"."""
        assert not is_field_scoped_guard(_A_ROOTED_SCOPE.format(helper="_trainer_modules"))

    def test_two_reader_helpers_do_not_qualify(self) -> None:
        """:func:`_reader_helper` resolves exactly one, so two is not gradeable."""
        second = "def _reads_another(source: str) -> bool:\n    return False\n"
        source = _A_READER + second + _A_ROOTED_SCOPE.format(helper="_trainer_modules")
        assert not is_field_scoped_guard(source)

    def test_the_rule_separates_the_exemplars(self) -> None:
        """Non-vacuity: a rule that answered one way would pass some case above."""
        qualifying = _A_READER + _A_ROOTED_SCOPE.format(helper="_training_modules")
        assert {is_field_scoped_guard(qualifying), is_field_scoped_guard(_A_READER + _A_LISTED_SCOPE)} == {True, False}

    def test_every_discovered_guard_registers_its_gate(self) -> None:
        """A discovered guard whose gate is unregistered fails opaquely.

        The gate lookup in the tests above raises ``StopIteration`` rather than
        naming the omission, so the mapping is checked here where it can.
        """
        unregistered = sorted(
            name
            for name, module in _guard_modules().items()
            if not any(g in line for g in FIELD_SCOPED_GATES for line in inspect.getsource(module).splitlines())
        )
        assert unregistered == [], f"guards whose gate is absent from FIELD_SCOPED_GATES: {unregistered}"
