"""The three RL posture flags on an RLTrainSpec are checked, never read by truthiness.

``normalize_obs``, ``normalize_advantage`` and ``autotune_alpha`` each select a
*posture* - wrap both observation streams in ``EmpiricalNormalization`` or feed
them raw; standardize advantages per batch or use them as computed; build a
temperature optimizer or hold ``alpha`` at ``init_alpha`` - and every RL backend
read the flags it consumes by truthiness while its ``validate`` graded every
numeric knob around them. Every non-empty string is truthy, so the spellings a
caller reaches for when opting out (``"false"``, ``"no"``, ``"0"``) selected the
affirmative posture, and ``0``, ``None`` and ``""`` selected the negative one
without being a declared spelling of it. Measured on ``b4c3ea8``:

* ``validate`` on ``ppo``, ``fast_sac`` and ``fast_td3`` reported nothing for
  ``normalize_obs`` set to any of ``"false"``, ``"no"``, ``"0"``, ``0`` or
  ``None``, and ``setup()`` reads it as ``... if spec.normalize_obs else None``
  on all three, so the first three built the normalizers the caller declined;
* ``ppo`` reported nothing for the same values of ``normalize_advantage``, which
  ``update()`` reads at two sites as ``if spec.normalize_advantage:``;
* ``fast_sac`` reported nothing for the same values of ``autotune_alpha``, which
  ``setup()`` stores and branches on - so ``"false"`` built the temperature
  optimizer ``tests/training/test_rl_fast_sac.py`` pins that ``False`` must not,
  and ``0`` held the temperature fixed without being a spelling of ``False``;
* the flag also *gates* the ``alpha_lr`` check, and it was read there by
  truthiness too: ``autotune_alpha="false", alpha_lr=-1.0`` was refused as
  ``alpha_lr`` - the rate of an optimizer the caller had asked not to build -
  while the flag whose misread selected that branch was named nowhere.

The fields are held to the shared :func:`~strands_robots.utils.boolean_flag_error`
domain through three field-scoped gates on
:class:`~strands_robots.training.base.Trainer`, in the same biconditional as the
numeric domains beside them: a backend that reads a field routes it through the
gate, and a backend that ignores the field reports nothing about it. Three gates
rather than one because the readers differ - all three backends read
``normalize_obs``, PPO alone reads ``normalize_advantage`` and SAC alone reads
``autotune_alpha``.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.training._validate import (
    advantage_normalization_problems,
    observation_normalization_problems,
    temperature_autotune_problems,
)
from strands_robots.training.base import Trainer
from strands_robots.training.factory import create_trainer
from strands_robots.training.rl.base_algo import RLTrainSpec
from tests.training._spec_field_reads import reads_spec_field

# The spellings a caller reaches for when opting out (every one truthy), two
# truthy numbers, and the falsy values that are not a declared spelling of the
# negative posture either.
NOT_A_BOOLEAN = [
    pytest.param("false", id="str-false"),
    pytest.param("no", id="str-no"),
    pytest.param("0", id="str-zero"),
    pytest.param(1, id="int-one"),
    pytest.param(0, id="int-zero"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(None, id="none"),
    pytest.param([], id="empty-list"),
]

# Both python spellings plus the numpy booleans the shared domain also accepts.
A_BOOLEAN = [
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(np.True_, id="np-true"),
    pytest.param(np.False_, id="np-false"),
]

# Which backend reads which field, by provider name. The one-owner scan at the
# bottom derives the same sets from the tree, so a backend that starts reading a
# field is graded on arrival.
READERS: dict[str, tuple[str, ...]] = {
    "normalize_obs": ("ppo", "fast_sac", "fast_td3"),
    "normalize_advantage": ("ppo",),
    "autotune_alpha": ("fast_sac",),
}
RL_PROVIDERS = ("ppo", "fast_sac", "fast_td3")
IGNORERS: dict[str, tuple[str, ...]] = {
    field: tuple(p for p in RL_PROVIDERS if p not in readers) for field, readers in READERS.items()
}
GATES: dict[str, str] = {
    "normalize_obs": "_observation_normalization_problems",
    "normalize_advantage": "_advantage_normalization_problems",
    "autotune_alpha": "_temperature_autotune_problems",
}
READER_CELLS = [(field, provider) for field, providers in READERS.items() for provider in providers]
IGNORER_CELLS = [(field, provider) for field, providers in IGNORERS.items() for provider in providers]


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> RLTrainSpec:
    """An otherwise-valid RL spec, so only the flag under test is exercised."""
    return RLTrainSpec(  # type: ignore[return-value]
        output_dir=str(tmp_path / "out"),
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _problems_about(provider: str, spec: RLTrainSpec, field: str) -> list[str]:
    """``validate`` problems naming *field* in the shared domain's own shape.

    Matched as ``"{provider}: {field} "`` rather than the bare word, so an
    unrelated problem quoting the field name can neither mask nor fake a verdict.
    """
    return [p for p in create_trainer(provider).validate(spec) if p.startswith(f"{provider}: {field} ")]


def _set(spec: RLTrainSpec, **fields: Any) -> RLTrainSpec:
    """Assign posture values through one funnel.

    The values under test are deliberately outside the declared ``bool``, which
    is the point; routing the assignment through ``setattr`` states that once
    rather than suppressing it at every site.
    """
    for name, value in fields.items():
        setattr(spec, name, value)
    return spec


class TestEveryReaderRefusesAPostureItCanOnlyMisread:
    """A backend that branches on a flag refuses every value it could only misread."""

    @pytest.mark.parametrize(("field", "provider"), READER_CELLS)
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_the_flag_is_refused(self, spec: RLTrainSpec, field: str, provider: str, value: Any) -> None:
        assert _problems_about(provider, _set(spec, **{field: value}), field), f"{provider} accepted {field}={value!r}"

    @pytest.mark.parametrize(("field", "provider"), READER_CELLS)
    def test_the_problem_names_the_backend_the_field_and_the_value(
        self, spec: RLTrainSpec, field: str, provider: str
    ) -> None:
        (problem,) = _problems_about(provider, _set(spec, **{field: "false"}), field)
        assert problem.startswith(f"{provider}: {field} must be a boolean, got 'false'"), problem

    @pytest.mark.parametrize(("field", "provider"), READER_CELLS)
    def test_a_refused_flag_is_a_problem_not_an_exception(self, spec: RLTrainSpec, field: str, provider: str) -> None:
        """``validate`` is documented pure and read-only; a refusal is a return value."""
        problems = create_trainer(provider).validate(_set(spec, **{field: "false"}))
        assert isinstance(problems, list)


class TestAUsableBooleanIsUntouched:
    """The controls: every declared spelling passes, and passes unchanged."""

    @pytest.mark.parametrize(("field", "provider"), READER_CELLS)
    @pytest.mark.parametrize("value", A_BOOLEAN)
    def test_it_reports_nothing(self, spec: RLTrainSpec, field: str, provider: str, value: Any) -> None:
        assert _problems_about(provider, _set(spec, **{field: value}), field) == []

    def test_the_defaults_report_nothing(self, spec: RLTrainSpec) -> None:
        for provider in RL_PROVIDERS:
            for field in READERS:
                assert _problems_about(provider, spec, field) == []


class TestABackendThatIgnoresTheFieldReportsNothing:
    """The other half of the biconditional.

    A backend that never reads a flag has nothing to misread, and refusing a
    value the requested backend ignores would send a caller to fix a field that
    changes nothing about their run.
    """

    @pytest.mark.parametrize(("field", "provider"), IGNORER_CELLS)
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_it_reports_nothing(self, spec: RLTrainSpec, field: str, provider: str, value: Any) -> None:
        assert _problems_about(provider, _set(spec, **{field: value}), field) == []


class TestTheRefusalPrecedesTheCheckTheFlagGates:
    """``autotune_alpha`` decides whether ``alpha_lr`` is read at all.

    A posture guard placed *after* the numeric guard still refuses, but it names
    the value the misread posture selected rather than the posture - sending the
    caller to correct a rate their opt-out says is never read. The order is the
    point, so it is pinned on its own.
    """

    def test_an_opt_out_beside_an_unusable_rate_is_refused_as_the_flag(self, spec: RLTrainSpec) -> None:
        problems = create_trainer("fast_sac").validate(_set(spec, autotune_alpha="false", alpha_lr=-1.0))
        assert [p for p in problems if p.startswith("fast_sac: autotune_alpha ")], problems
        assert not [p for p in problems if p.startswith("fast_sac: alpha_lr ")], problems

    def test_a_real_opt_in_beside_an_unusable_rate_is_still_the_rate_refusal(self, spec: RLTrainSpec) -> None:
        problems = create_trainer("fast_sac").validate(_set(spec, autotune_alpha=True, alpha_lr=-1.0))
        assert [p for p in problems if p.startswith("fast_sac: alpha_lr ")], problems
        assert not [p for p in problems if p.startswith("fast_sac: autotune_alpha ")], problems

    def test_a_real_opt_out_beside_an_unusable_rate_reports_nothing(self, spec: RLTrainSpec) -> None:
        """With tuning off the rate is never read, so nothing about it is owed."""
        assert create_trainer("fast_sac").validate(_set(spec, autotune_alpha=False, alpha_lr=-1.0)) == []

    def test_the_flag_problem_is_reported_ahead_of_the_rate_problem(self, spec: RLTrainSpec) -> None:
        """When both are wrong, the flag whose misread selects the branch is named first."""
        problems = create_trainer("fast_sac").validate(_set(spec, autotune_alpha=np.True_, alpha_lr=-1.0))
        assert [p for p in problems if p.startswith("fast_sac: alpha_lr ")], problems
        problems = create_trainer("fast_sac").validate(_set(spec, autotune_alpha=1, alpha_lr=-1.0))
        assert problems and problems[0].startswith("fast_sac: autotune_alpha "), problems


class TestTheGatesAreUsableOnTheirOwn:
    """The module-level functions carry the context they are given, and only their field."""

    @pytest.mark.parametrize(
        ("gate", "field"),
        [
            (observation_normalization_problems, "normalize_obs"),
            (advantage_normalization_problems, "normalize_advantage"),
            (temperature_autotune_problems, "autotune_alpha"),
        ],
    )
    def test_it_reports_the_context_it_was_given(self, spec: RLTrainSpec, gate: Any, field: str) -> None:
        (problem,) = gate(_set(spec, **{field: "no"}), context="acme")
        assert problem.startswith(f"acme: {field} must be a boolean, got 'no'"), problem

    def test_each_gate_reads_only_its_own_field(self, spec: RLTrainSpec) -> None:
        _set(spec, normalize_obs="false", normalize_advantage="false", autotune_alpha="false")
        assert [p.split(" must")[0] for p in observation_normalization_problems(spec, context="acme")] == [
            "acme: normalize_obs"
        ]
        assert [p.split(" must")[0] for p in advantage_normalization_problems(spec, context="acme")] == [
            "acme: normalize_advantage"
        ]
        assert [p.split(" must")[0] for p in temperature_autotune_problems(spec, context="acme")] == [
            "acme: autotune_alpha"
        ]


def _trainer_modules() -> list[pathlib.Path]:
    """Every trainer module, minus the one that defines the shared gates.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The gates' own module reads the fields as
    their owner, not as a consumer, and is excluded by derivation rather than by
    name.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(observation_normalization_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _calls_the_gate(source: str, gate: str) -> bool:
    """Does *source* route through ``self.<gate>(...)``?"""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == gate
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForEachPostureFlag:
    """No backend may skip a gate for a field it reads, and none may skip a read.

    The set of backends in scope is derived from the tree rather than listed: a
    module that reads ``spec.normalize_obs`` (by name or through a forwarding
    table) must call ``_observation_normalization_problems``, and likewise for
    the other two, so a backend that starts reading a field fails this test
    until it does.
    """

    @pytest.mark.parametrize(
        ("field", "expected_readers"),
        [
            ("normalize_obs", {"ppo.py", "fast_sac.py", "fast_td3.py"}),
            ("normalize_advantage", {"ppo.py"}),
            ("autotune_alpha", {"fast_sac.py"}),
        ],
    )
    def test_the_scan_finds_the_readers(self, field: str, expected_readers: set[str]) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        readers = {p.name for p in _trainer_modules() if reads_spec_field(p.read_text(), (field,))}
        assert readers == expected_readers

    @pytest.mark.parametrize(("field", "gate"), sorted(GATES.items()))
    def test_every_reader_routes_through_the_gate(self, field: str, gate: str) -> None:
        adrift = sorted(
            p.name
            for p in _trainer_modules()
            if reads_spec_field(source := p.read_text(), (field,)) and not _calls_the_gate(source, gate)
        )
        assert adrift == [], f"modules reading spec.{field} without {gate}: {adrift}"

    @pytest.mark.parametrize(("field", "gate"), sorted(GATES.items()))
    def test_no_backend_gates_a_field_it_does_not_read(self, field: str, gate: str) -> None:
        """The other half of the biconditional: a gate is not a free extra check."""
        over_reaching = sorted(
            p.name
            for p in _trainer_modules()
            if _calls_the_gate(source := p.read_text(), gate) and not reads_spec_field(source, (field,))
        )
        assert over_reaching == [], f"modules calling {gate} without reading spec.{field}: {over_reaching}"

    def test_the_rate_gate_is_consulted_after_the_flag_gate_it_depends_on(self) -> None:
        """Ordering, pinned structurally on the one module that consults both."""
        source = pathlib.Path(inspect.getfile(create_trainer("fast_sac").__class__)).read_text()
        calls = [
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("_temperature_autotune_problems", "_temperature_learning_rate_problems")
        ]
        assert calls == ["_temperature_autotune_problems", "_temperature_learning_rate_problems"], calls

    def test_the_scanner_detects_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def setup(self, spec):\n    self.norm = Norm() if spec.normalize_obs else None\n"
        assert reads_spec_field(planted, ("normalize_obs",))
        assert not _calls_the_gate(planted, "_observation_normalization_problems")

    def test_the_class_of_readers_reads_the_field_in_the_test_table(self) -> None:
        """The hand-written reader tuples above agree with the derived ones."""
        by_module = {"ppo.py": "ppo", "fast_sac.py": "fast_sac", "fast_td3.py": "fast_td3"}
        for field, listed in READERS.items():
            readers = {p.name for p in _trainer_modules() if reads_spec_field(p.read_text(), (field,))}
            assert {by_module[name] for name in readers} == set(listed), field
