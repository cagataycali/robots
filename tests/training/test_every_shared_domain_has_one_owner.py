"""Every shared training domain has one owner, and one table grades all of them.

Each ``Trainer._*_problems`` gate owns the domain of one or more
:class:`~strands_robots.training.base.TrainSpec` fields, and each documents a
biconditional: a backend that READS such a field must route it through the gate,
and a backend that CALLS the gate must read the field it judges. Both halves are
derived from the source rather than listed, so a new backend that starts reading
a field fails these tests until it routes through the gate.

The per-field ``test_*_domain.py`` files own the *domain* of their field - which
values are refused, what the message says, and the executable premise for why an
unusable value breaks the consumer. What they do not each need is a private copy
of this scan. :data:`DOMAINS` is that scan's one table, a row per gate.

The rule for "no backend re-implements the domain" is graded on the syntax tree
here, where the per-field copies scanned text: a comparison is reported wherever
the field appears in it, so the chained (``0 < spec.x < n``) and right-hand forms
a substring scan cannot see are covered. The comparisons and conversions a
backend legitimately makes - a launcher choosing single- or multi-node, a mode
selector, an API field being serialized - are declared per row, so a new one has
to be declared rather than merely not matching a spelling.

:class:`TestEveryGateIsInTheTable` is the ratchet the per-file copies could not
have: a gate on :class:`Trainer` that neither a row nor
:data:`GATES_GRADED_ELSEWHERE` names is reported, so a new shared gate cannot
ship with nothing grading its readers at all.
"""

from __future__ import annotations

import ast
import functools
import inspect
import pathlib
from dataclasses import dataclass, field

import pytest

from strands_robots.training import _validate
from strands_robots.training.base import Trainer
from tests.training._spec_field_reads import reads_spec_field


@dataclass(frozen=True)
class SharedDomain:
    """One shared gate, the spec fields it owns, and what the tree may do with them.

    Attributes:
        gate: The :class:`Trainer` method every reader must route through.
        fields: The ``TrainSpec`` field names whose domain the gate owns.
        readers: File names of the training modules that read those fields. A
            non-vacuity pin - a mis-rooted scan cannot report a clean sweep over
            an empty tree - and a ratchet: a backend that starts or stops reading
            the field says so here.
        comparisons: Comparisons of a field against a literal that are a
            consumer's own question rather than a copy of the domain, each with
            the reason it is not an offender.
        conversions: Conversions of a field (``int``/``float``/``len``/...) that
            serialize or build with the value after the gate has admitted it.
        gates_without_reading: Modules that call the gate for a field they do not
            themselves read, e.g. because an inherited loop is the reader.
    """

    gate: str
    fields: tuple[str, ...]
    readers: frozenset[str]
    comparisons: frozenset[str] = field(default_factory=frozenset)
    conversions: frozenset[str] = field(default_factory=frozenset)
    gates_without_reading: frozenset[str] = field(default_factory=frozenset)


def _domain(
    gate: str,
    fields: tuple[str, ...],
    *readers: str,
    comparisons: tuple[str, ...] = (),
    conversions: tuple[str, ...] = (),
) -> SharedDomain:
    return SharedDomain(
        gate=gate,
        fields=fields,
        readers=frozenset(readers),
        comparisons=frozenset(comparisons),
        conversions=frozenset(conversions),
    )


#: One row per shared gate, in ``Trainer``'s own declaration order.
DOMAINS: tuple[SharedDomain, ...] = (
    _domain("_rl_replay_problems", ("buffer_size", "batch_size", "gradient_steps"), "fast_sac.py", "fast_td3.py"),
    _domain(
        "_launch_topology_problems",
        ("num_gpus", "num_nodes"),
        "cosmos3.py",
        "groot.py",
        "lerobot.py",
        "sagemaker.py",
        # "Is this a topology I have to launch differently" - a launcher choice,
        # asked after the gate has established the counts are usable.
        comparisons=("spec.num_gpus > 1", "spec.num_gpus <= 1", "spec.num_nodes > 1"),
        # The count as SageMaker's InstanceCount API field.
        conversions=("int(spec.num_nodes)",),
    ),
    _domain(
        "_seed_problems",
        ("seed",),
        "cosmos3.py",
        "lerobot.py",
        "sagemaker.py",
        "ppo.py",
        "fast_sac.py",
        "fast_td3.py",
    ),
    _domain(
        "_checkpoint_cadence_problems",
        ("save_freq",),
        "cosmos3.py",
        "groot.py",
        "lerobot.py",
        "sagemaker.py",
        # The eval_steps fallback selector: which cadence to forward, not whether
        # the cadence is usable.
        comparisons=("spec.save_freq > 0",),
    ),
    _domain(
        "_validation_episodes_problems",
        ("val_episodes",),
        "lerobot.py",
        "sagemaker.py",
        # Whether the held-out count fits the dataset actually on disk - a
        # relation against another quantity, which the gate cannot see.
        comparisons=("0 < spec.val_episodes < effective",),
    ),
    _domain("_resume_problems", ("resume",), "groot.py", "lerobot.py", "sagemaker.py"),
    _domain("_streaming_problems", ("streaming",), "lerobot.py", "sagemaker.py"),
    _domain("_observation_normalization_problems", ("normalize_obs",), "ppo.py", "fast_sac.py", "fast_td3.py"),
    _domain("_advantage_normalization_problems", ("normalize_advantage",), "ppo.py"),
    _domain("_temperature_autotune_problems", ("autotune_alpha",), "fast_sac.py"),
    _domain("_lora_hyperparameter_problems", ("lora_r", "lora_alpha"), "lerobot.py", "sagemaker.py"),
    _domain("_discount_factor_problems", ("gamma",), "ppo.py", "fast_sac.py", "fast_td3.py"),
    _domain("_gae_lambda_problems", ("lam",), "ppo.py"),
    _domain("_optimization_epochs_problems", ("num_learning_epochs",), "ppo.py"),
    _domain("_temperature_learning_rate_problems", ("alpha_lr",), "fast_sac.py"),
    _domain("_initial_temperature_problems", ("init_alpha",), "fast_sac.py"),
    _domain(
        "_target_entropy_problems",
        ("target_entropy",),
        "fast_sac.py",
        # The admitted value as the entropy target the loss is built with.
        conversions=("float(spec.target_entropy)",),
    ),
    _domain("_gradient_clip_problems", ("max_grad_norm",), "ppo.py"),
    _domain("_loss_weight_problems", ("value_loss_coef", "entropy_coef"), "ppo.py"),
    _domain("_clip_range_problems", ("clip_param",), "ppo.py"),
    _domain("_policy_delay_problems", ("policy_delay",), "fast_td3.py"),
    _domain(
        "_td3_noise_problems",
        ("exploration_noise_std", "target_noise_std", "target_noise_clip"),
        "fast_td3.py",
    ),
    _domain("_network_width_problems", ("hidden_dims",), "ppo.py", "fast_sac.py", "fast_td3.py"),
    _domain("_polyak_coefficient_problems", ("tau",), "fast_sac.py", "fast_td3.py"),
)

#: Gates whose owner guard is a different shape, graded by the file named here
#: rather than by a row above. Each derives its scope from something other than
#: "the modules that read the field", so a row would restate or weaken it.
GATES_GRADED_ELSEWHERE: dict[str, str] = {
    # Scoped to the CONCRETE Trainer subclasses rather than to field reads: the
    # field has a default, so every backend is a reader.
    "_learning_rate_problems": "tests/training/test_learning_rate_domain.py",
    "_run_size_problems": "tests/training/test_run_size_domain.py",
    # Scoped to the RL class hierarchy, because PPO reads the field by inheriting
    # the loop that consumes it rather than by naming it.
    "_rl_run_size_problems": "tests/training/test_rl_run_size_domain.py",
    "_rl_checkpoint_interval_problems": "tests/training/test_rl_checkpoint_interval_domain.py",
    # A relation between two fields, so the offending shape is a comparison of
    # one field against the other rather than against a literal.
    "_rl_warmup_batch_problems": "tests/training/test_learning_starts_count_domain.py",
    "_rl_warmup_reachability_problems": "tests/training/test_learning_starts_count_domain.py",
    # Graded against torch's own device vocabulary, not against a reader set.
    "_spec_device_problems": "tests/training/test_rl_trainer_device_domain.py",
    # Not a field domain: the remote-code / credential posture of a whole spec.
    "_security_problems": "tests/training/test_call_time_dependency_preflight.py",
}


@functools.cache
def _trainer_sources() -> dict[str, str]:
    """Source of every training module except the one that owns the gates.

    Rooted at the module that defines :class:`Trainer`, so the scan cannot
    silently point at the wrong tree, and read once for the whole table instead
    of once per field. The gates' own module is excluded by derivation rather
    than by name: it reads every field as their owner, not as a consumer.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(_validate)).resolve()
    return {
        path.name: path.read_text()
        for path in sorted(root.rglob("*.py"))
        if path.name != "__init__.py" and path.resolve() != owner
    }


def _is_the_field(node: ast.AST, fields: tuple[str, ...]) -> bool:
    """Is *node* a ``spec.<field>`` access for one of *fields*?"""
    return isinstance(node, ast.Attribute) and node.attr in fields and getattr(node.value, "id", None) == "spec"


def calls_the_gate(source: str, gate: str) -> bool:
    """Does *source* route through ``self.<gate>(...)``?"""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == gate
        for node in ast.walk(ast.parse(source))
    )


def comparisons_against_a_literal(source: str, fields: tuple[str, ...]) -> list[str]:
    """Comparisons in *source* that judge one of *fields* against a number.

    Graded on the syntax tree, and on every operand of the comparison rather than
    on its left-hand side alone, so ``0 < spec.tau <= 1.0`` and ``1 > spec.tau``
    are reported like ``spec.tau < 1``. Prose that quotes such a comparison is
    not, which is why the scan is not on text.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if any(_is_the_field(operand, fields) for operand in operands) and any(
            isinstance(operand, ast.Constant) and isinstance(operand.value, int | float) for operand in operands
        ):
            found.append(ast.unparse(node))
    return found


def conversions_of_the_field(source: str, fields: tuple[str, ...]) -> list[str]:
    """Calls in *source* that convert or measure one of *fields*.

    ``int(spec.x)`` after the gate is a consumer serializing an admitted value;
    the same call instead of the gate is a local type test. Both are reported, so
    the row says which one it is.
    """
    builtins = ("int", "float", "len", "bool", "type", "isinstance")
    return [
        ast.unparse(node)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in builtins
        and node.args
        and _is_the_field(node.args[0], fields)
    ]


_IDS = [domain.gate.strip("_").removesuffix("_problems") for domain in DOMAINS]


@pytest.mark.parametrize("domain", DOMAINS, ids=_IDS)
class TestTheReadersAndTheGateAreTheSameSet:
    """The biconditional, both ways, for every shared domain in the table."""

    @staticmethod
    def _readers(domain: SharedDomain) -> set[str]:
        return {name for name, source in _trainer_sources().items() if reads_spec_field(source, domain.fields)}

    def test_the_reader_set_is_the_expected_one(self, domain: SharedDomain) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep over nothing."""
        assert self._readers(domain) == set(domain.readers)

    def test_every_reader_routes_through_the_gate(self, domain: SharedDomain) -> None:
        adrift = sorted(
            name
            for name, source in _trainer_sources().items()
            if reads_spec_field(source, domain.fields) and not calls_the_gate(source, domain.gate)
        )
        assert adrift == [], f"modules read {domain.fields} without {domain.gate}: {adrift}"

    def test_no_module_gates_a_field_it_does_not_read(self, domain: SharedDomain) -> None:
        """The other half: a gate is not a free extra check on an unread field."""
        over_reaching = sorted(
            name
            for name, source in _trainer_sources().items()
            if calls_the_gate(source, domain.gate) and not reads_spec_field(source, domain.fields)
        )
        assert over_reaching == sorted(domain.gates_without_reading), (
            f"modules call {domain.gate} without reading {domain.fields}: {over_reaching}"
        )

    def test_the_only_comparisons_are_the_declared_ones(self, domain: SharedDomain) -> None:
        """Any other comparison against a literal is a re-implemented domain."""
        found = {
            comparison
            for source in _trainer_sources().values()
            for comparison in comparisons_against_a_literal(source, domain.fields)
        }
        assert found == set(domain.comparisons)

    def test_the_only_conversions_are_the_declared_ones(self, domain: SharedDomain) -> None:
        """Any other conversion is a local type test the gate already makes."""
        found = {
            conversion
            for source in _trainer_sources().values()
            for conversion in conversions_of_the_field(source, domain.fields)
        }
        assert found == set(domain.conversions)


class TestTheScannersDetectAPlantedDefect:
    """A scanner that silently matched nothing would look like a clean tree."""

    @pytest.mark.parametrize("domain", DOMAINS, ids=_IDS)
    def test_a_reader_that_names_the_field_is_reported(self, domain: SharedDomain) -> None:
        planted = f"def validate(self, spec):\n    return [] if spec.{domain.fields[0]} else []\n"
        assert reads_spec_field(planted, domain.fields)
        assert not calls_the_gate(planted, domain.gate)

    @pytest.mark.parametrize("domain", DOMAINS, ids=_IDS)
    def test_a_reader_that_forwards_the_field_by_name_is_reported(self, domain: SharedDomain) -> None:
        """The form a transport-only provider takes: no attribute names the field."""
        planted = (
            f"FIELDS = ({domain.fields[0]!r},)\n"
            "def validate(self, spec):\n    return [getattr(spec, f) for f in FIELDS]\n"
        )
        assert reads_spec_field(planted, domain.fields)
        assert not calls_the_gate(planted, domain.gate)

    @pytest.mark.parametrize(
        "planted",
        [
            "spec.tau <= 0",
            "spec.tau>1",
            "1 > spec.tau",
            "not 0.0 < spec.tau <= 1.0",
        ],
    )
    def test_a_comparison_in_any_position_is_reported(self, planted: str) -> None:
        source = f"def validate(self, spec):\n    return ['bad'] if {planted} else []\n"
        assert comparisons_against_a_literal(source, ("tau",))

    def test_prose_quoting_a_comparison_is_not_reported(self) -> None:
        """The reason the scan is on syntax: a docstring may cite the defect it replaced."""
        assert comparisons_against_a_literal('"""A gate replacing `if not 0.0 < spec.tau <= 1.0`."""\n', ("tau",)) == []

    def test_a_comparison_of_another_field_is_not_reported(self) -> None:
        source = "def validate(self, spec):\n    return ['bad'] if spec.gamma > 1 else []\n"
        assert comparisons_against_a_literal(source, ("lam",)) == []

    def test_a_local_type_test_is_reported(self) -> None:
        source = "def validate(self, spec):\n    return [] if isinstance(spec.save_freq, int) else ['bad']\n"
        assert conversions_of_the_field(source, ("save_freq",)) == ["isinstance(spec.save_freq, int)"]


class TestEveryGateIsInTheTable:
    """No shared gate ships with nothing grading its readers.

    The roster is read off :class:`Trainer` rather than listed, so a gate added
    there is reported here until a row owns it or :data:`GATES_GRADED_ELSEWHERE`
    names the file that grades it another way.
    """

    @staticmethod
    def _gates_on_the_trainer() -> set[str]:
        return {name for name in vars(Trainer) if name.endswith("_problems")}

    def test_every_gate_is_owned_by_a_row_or_named_elsewhere(self) -> None:
        ungraded = sorted(self._gates_on_the_trainer() - {d.gate for d in DOMAINS} - set(GATES_GRADED_ELSEWHERE))
        assert ungraded == [], f"shared gates with no owner guard: {ungraded}"

    def test_no_row_names_a_gate_the_trainer_does_not_have(self) -> None:
        """A renamed gate leaves a row pointing at nothing, which would pass vacuously."""
        named = {d.gate for d in DOMAINS} | set(GATES_GRADED_ELSEWHERE)
        assert sorted(named - self._gates_on_the_trainer()) == []

    def test_the_file_named_for_each_other_shape_exists(self) -> None:
        missing = sorted(path for path in GATES_GRADED_ELSEWHERE.values() if not pathlib.Path(path).exists())
        assert missing == []

    def test_each_gate_appears_once(self) -> None:
        gates = [d.gate for d in DOMAINS]
        assert sorted(gates) == sorted(set(gates))
