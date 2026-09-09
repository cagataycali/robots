"""``resume`` and ``streaming`` on a TrainSpec are checked, never read by truthiness.

Each of the two selects a *posture* - continue the run under ``output_dir`` or
start it fresh; stream the dataset or materialize it - and every backend that
read either did so by truthiness. Every non-empty string is truthy, so the
spellings a caller reaches for when opting out (``"false"``, ``"no"``, ``"0"``)
selected the affirmative posture, and ``0``, ``None`` and ``""`` selected the
negative one without being a declared spelling of it. Measured on ``d2f23d8``:

* ``TrainSpec(resume="false")`` with a checkpoint under ``output_dir`` made
  LeRobot's ``build_config`` return the checkpoint's own ``train_config.json``
  in place of a config built from the spec, so the fresh run that was asked for
  became a resume of a previous configuration and the spec's ``steps``,
  ``global_batch_size`` and ``save_freq`` never reached it - the defect
  ``tests/tools/test_lerobot_train_flag_domain.py`` closed on the sibling tool,
  still reachable through the provider-agnostic ``train_policy`` one layer up;
* ``TrainSpec(streaming="false", val_episodes=2)`` was refused by LeRobot's
  ``validate`` with "set streaming=False to keep the validation split", a remedy
  the caller had already spelled - the flag *gates* the pair check, so a
  misread posture was reported as the option it selected rather than as itself;
* GR00T emitted ``--resume_from_checkpoint`` for ``resume="false"``, and
  SageMaker forwarded the raw string as a hyperparameter.

The fields are held to the shared :func:`~strands_robots.utils.boolean_flag_error`
domain through two field-scoped gates on :class:`~strands_robots.training.base.Trainer`,
in the same biconditional as the numeric domains beside them: a backend that
reads the field routes it through the gate, and a backend that ignores the field
reports nothing about it. Two gates rather than one because the readers differ -
GR00T reads ``resume`` and ignores ``streaming``.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.tools.train_policy import train_policy
from strands_robots.training._validate import resume_problems, streaming_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from strands_robots.training.sagemaker import SagemakerTrainer
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

# Which backend reads which field, by name or through a forwarding table. The
# one-owner scan at the bottom derives the same sets from the tree, so a
# backend that starts reading a field is graded on arrival.
READS_RESUME = (LerobotTrainer, Gr00tTrainer, SagemakerTrainer)
READS_STREAMING = (LerobotTrainer, SagemakerTrainer)
IGNORES_RESUME = (MockTrainer, Cosmos3Trainer)
IGNORES_STREAMING = (MockTrainer, Cosmos3Trainer, Gr00tTrainer)


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose posture flags are the only thing under test.

    ``validate`` may report unrelated problems (Cosmos wants a recipe TOML,
    SageMaker an S3 URI); every assertion below filters for the field name, so
    an unrelated problem can neither mask nor fake a verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
    )


def _problems_about(trainer: Trainer, spec: TrainSpec, field: str) -> list[str]:
    """``validate`` problems naming *field* in the shared domain's own shape.

    Matched as ``": <field> "`` rather than the bare word: pytest derives
    ``tmp_path`` from the test name, so an unrelated problem quoting a path can
    contain the field name too.
    """
    return [p for p in trainer.validate(spec) if f": {field} " in p]


def _set(spec: TrainSpec, **fields: Any) -> TrainSpec:
    """Assign posture values through one funnel.

    The values under test are deliberately outside the declared ``bool``, which
    is the point; routing the assignment through ``setattr`` states that once
    rather than suppressing it at every site.
    """
    for name, value in fields.items():
        setattr(spec, name, value)
    return spec


def _run_tool(**kwargs: Any) -> dict[str, Any]:
    """Call the agent tool through one funnel, for the same reason as _set."""
    return dict(train_policy(**kwargs))


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(item.get("text", "") for item in envelope.get("content", []))


class TestEveryReaderRefusesAPostureItCanOnlyMisread:
    """Each backend that reads a field refuses every non-boolean, by name."""

    @pytest.mark.parametrize("trainer_cls", READS_RESUME)
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_resume_is_refused(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.resume = value
        problems = _problems_about(trainer_cls(), spec, "resume")
        assert problems, f"{trainer_cls.__name__} accepted resume={value!r}"
        assert any("resume must be a boolean" in p for p in problems), problems

    @pytest.mark.parametrize("trainer_cls", READS_STREAMING)
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_streaming_is_refused(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.streaming = value
        problems = _problems_about(trainer_cls(), spec, "streaming")
        assert problems, f"{trainer_cls.__name__} accepted streaming={value!r}"
        assert any("streaming must be a boolean" in p for p in problems), problems

    @pytest.mark.parametrize("trainer_cls", READS_RESUME)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        trainer = trainer_cls()
        _set(spec, resume="false")
        assert any(p.startswith(f"{trainer.provider_name}: resume ") for p in _problems_about(trainer, spec, "resume"))

    @pytest.mark.parametrize("trainer_cls", READS_RESUME)
    def test_a_refused_flag_is_a_problem_not_an_exception(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """``validate`` returns problems; a non-boolean must not raise out of it."""
        _set(spec, resume=[1], streaming={"on": True})
        problems = trainer_cls().validate(spec)  # must not raise
        assert any(": resume " in p for p in problems), problems


class TestAUsableBooleanIsUntouched:
    """Both python spellings and the numpy booleans pass every reader."""

    @pytest.mark.parametrize("trainer_cls", READS_RESUME)
    @pytest.mark.parametrize("value", A_BOOLEAN)
    def test_resume(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.resume = value
        assert _problems_about(trainer_cls(), spec, "resume") == []

    @pytest.mark.parametrize("trainer_cls", READS_STREAMING)
    @pytest.mark.parametrize("value", A_BOOLEAN)
    def test_streaming(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.streaming = value
        assert _problems_about(trainer_cls(), spec, "streaming") == []


class TestABackendThatIgnoresTheFieldReportsNothing:
    """A backend must not report on a field it never reads.

    :class:`TrainSpec` documents that a backend "reads the fields it supports
    and ignores the rest", so each gate is scoped to that field's readers.
    """

    @pytest.mark.parametrize("trainer_cls", IGNORES_RESUME)
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_resume(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.resume = value
        assert _problems_about(trainer_cls(), spec, "resume") == []

    @pytest.mark.parametrize("trainer_cls", IGNORES_STREAMING)
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_streaming(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.streaming = value
        assert _problems_about(trainer_cls(), spec, "streaming") == []


class TestTheRefusalPrecedesTheCheckTheFlagGates:
    """LeRobot's streaming / val_episodes pair check branches on the flag.

    A posture guard placed *after* that check still refuses, but names the
    option the misread posture selected rather than the flag - "set
    streaming=False" at a caller who had spelled exactly that. So the gate is
    consulted first and the pair is read only when it reports nothing.
    """

    @pytest.fixture
    def local_dataset(self, spec: TrainSpec) -> TrainSpec:
        """A readable local ``meta/info.json``, so the pair check is reachable."""
        meta = pathlib.Path(spec.dataset_root) / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text('{"total_episodes": 10, "total_tasks": 1}', encoding="utf-8")
        return spec

    def test_an_opt_out_beside_a_split_is_refused_as_the_flag(self, local_dataset: TrainSpec) -> None:
        _set(local_dataset, streaming="false", val_episodes=2)
        problems = LerobotTrainer().validate(local_dataset)
        assert any("streaming must be a boolean" in p for p in problems), problems
        # Measured on d2f23d8: the pair refusal, whose remedy the caller had
        # already followed.
        assert not any("set streaming=False" in p for p in problems), problems

    def test_a_real_opt_in_beside_a_split_is_still_the_pair_refusal(self, local_dataset: TrainSpec) -> None:
        """Control: the guard narrows nothing the pair check already refused."""
        local_dataset.streaming = True
        local_dataset.val_episodes = 2
        problems = LerobotTrainer().validate(local_dataset)
        assert any("set streaming=False" in p for p in problems), problems
        assert not any("must be a boolean" in p for p in problems), problems


class TestTheProviderAgnosticToolReportsTheRefusal:
    """``train_policy`` reaches the gate through ``validate`` on every spec action."""

    @pytest.mark.parametrize("action", ["validate", "train", "export"])
    @pytest.mark.parametrize("field", ["resume", "streaming"])
    def test_a_spec_action_is_refused_by_name(self, tmp_path: pathlib.Path, action: str, field: str) -> None:
        envelope = _run_tool(
            action=action,
            provider="lerobot_local",
            dataset_root=str(tmp_path / "ds"),
            output_dir=str(tmp_path / "out"),
            **{field: "false"},
        )
        assert envelope["status"] == "error"
        assert f"{field} must be a boolean" in _text(envelope), _text(envelope)

    def test_an_action_that_builds_no_spec_is_not_refused_for_one(self) -> None:
        envelope = _run_tool(action="list", resume="false", streaming="no")
        assert envelope["status"] == "success"
        assert "must be a boolean" not in _text(envelope)


class TestTheGatesAreUsableOnTheirOwn:
    """Each gate's own contract, independent of any backend."""

    def test_resume_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        _set(spec, resume="false")
        [problem] = resume_problems(spec, context="acme")
        assert problem.startswith("acme: resume must be a boolean, got 'false'.")

    def test_streaming_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        _set(spec, streaming=0)
        [problem] = streaming_problems(spec, context="acme")
        assert problem.startswith("acme: streaming must be a boolean, got 0.")

    def test_each_gate_reads_only_its_own_field(self, spec: TrainSpec) -> None:
        _set(spec, resume="false")
        assert streaming_problems(spec, context="acme") == []
        _set(spec, resume=False, streaming="false")
        assert resume_problems(spec, context="acme") == []

    def test_the_defaults_report_nothing(self, spec: TrainSpec) -> None:
        assert resume_problems(spec, context="acme") == []
        assert streaming_problems(spec, context="acme") == []


def _trainer_modules() -> list[pathlib.Path]:
    """Every trainer module, minus the one that defines the shared gates.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The gates' own module reads the fields as
    their owner, not as a consumer, and is excluded by derivation rather than by
    name.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(resume_problems)).resolve()
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
    module that reads ``spec.resume`` (by name or through a forwarding table)
    must call ``_resume_problems``, and likewise for ``streaming``, so a backend
    that starts reading either field fails this test until it does.
    """

    @pytest.mark.parametrize(
        ("field", "expected_readers"),
        [
            ("resume", {"groot.py", "lerobot.py", "sagemaker.py"}),
            ("streaming", {"lerobot.py", "sagemaker.py"}),
        ],
    )
    def test_the_scan_finds_the_readers(self, field: str, expected_readers: set[str]) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        readers = {p.name for p in _trainer_modules() if reads_spec_field(p.read_text(), (field,))}
        assert readers == expected_readers

    @pytest.mark.parametrize(("field", "gate"), [("resume", "_resume_problems"), ("streaming", "_streaming_problems")])
    def test_every_reader_routes_through_the_gate(self, field: str, gate: str) -> None:
        adrift = sorted(
            p.name
            for p in _trainer_modules()
            if reads_spec_field(source := p.read_text(), (field,)) and not _calls_the_gate(source, gate)
        )
        assert adrift == [], f"modules reading spec.{field} without {gate}: {adrift}"

    @pytest.mark.parametrize(("field", "gate"), [("resume", "_resume_problems"), ("streaming", "_streaming_problems")])
    def test_no_backend_gates_a_field_it_does_not_read(self, field: str, gate: str) -> None:
        """The other half of the biconditional: a gate is not a free extra check."""
        over_reaching = sorted(
            p.name
            for p in _trainer_modules()
            if _calls_the_gate(source := p.read_text(), gate) and not reads_spec_field(source, (field,))
        )
        assert over_reaching == [], f"modules calling {gate} without reading spec.{field}: {over_reaching}"

    def test_the_scanner_detects_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def validate(self, spec):\n    return ['--resume'] if spec.resume else []\n"
        assert reads_spec_field(planted, ("resume",))
        assert not _calls_the_gate(planted, "_resume_problems")

    def test_the_class_of_readers_reads_the_field_in_the_test_table(self) -> None:
        """The hand-written reader tuples above agree with the derived ones."""
        by_module = {"lerobot.py": LerobotTrainer, "groot.py": Gr00tTrainer, "sagemaker.py": SagemakerTrainer}
        for field, listed in (("resume", READS_RESUME), ("streaming", READS_STREAMING)):
            readers = {p.name for p in _trainer_modules() if reads_spec_field(p.read_text(), (field,))}
            assert {by_module[name] for name in readers} == set(listed), field
