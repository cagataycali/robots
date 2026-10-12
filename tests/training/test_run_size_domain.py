"""The run size a spec asks for is one shared positive-count domain.

:attr:`~strands_robots.training.base.TrainSpec.steps` and
:attr:`~strands_robots.training.base.TrainSpec.global_batch_size` are the two
factors of how much training a spec asks for, and every supervised backend
reads both straight into a discrete consumer: ``steps`` bounds the optimizer
loop (lerobot iterates ``range(step, cfg.steps)``) and ``global_batch_size``
becomes a ``DataLoader`` batch size / a ``--global_batch_size`` flag.

Before this contract each backend re-implemented ``if spec.steps <= 0`` and
none checked ``global_batch_size`` at all, so the four copies agreed with each
other and shared one hole: a comparison admits every value that is not
comparably non-positive. ``True`` became a silent run of one optimizer step,
a fractional or non-finite value reached ``range()`` and raised there - after
the dataset and the model were already loaded - and a non-numeric value raised
out of the comparison itself, from a :meth:`Trainer.validate` documented to
*return* problems rather than raise.

Which backends refuse which values, in which words, is the table in
``tests/training/test_train_spec_field_domains.py``. The tests below keep every
backend on the one gate.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import pytest

from strands_robots.training._validate import run_size_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.isaaclab import IsaacLabTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from strands_robots.training.rsl_rl import RslRlTrainer
from strands_robots.training.sagemaker import SagemakerTrainer
from tests._package_ast import parse_source, walk_tree

# The two factors of the run size, and the values no backend can honor. Split
# by failure mode so a test can say which contract each value belongs to.
RUN_SIZE_FIELDS = ("steps", "global_batch_size")

# Non-positive: already refused before this change (by four separate copies of
# the same comparison). Kept so the fix cannot regress what worked.

# Slipped through a comparison, then misbehaved in the consumer.

# Raised out of the comparison itself, i.e. out of validate().


# Every backend that reads the run size. The RL trainers are deliberately
# absent - see TestTheRLTrainersIgnoreAFieldTheyDoNotRead.
# RslRlTrainer is RL but reads the run size (steps = PPO iterations, global_batch_size = worlds).
SUPERVISED_TRAINERS = (MockTrainer, Cosmos3Trainer, LerobotTrainer, RslRlTrainer, SagemakerTrainer)

# Backends that read ``steps`` (as an iteration count) and never
# ``global_batch_size``: Isaac Lab's rsl_rl batch is num_envs x rollout length,
# set by the task. They grade ``steps`` with the same count domain, per field.
STEPS_ONLY_TRAINERS = (IsaacLabTrainer,)


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose non-numeric fields are all fine, so only the run size varies."""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10, "fps": 30}))
    out = tmp_path / "out"
    out.mkdir()
    return TrainSpec(
        dataset_root=str(root),
        base_model="lerobot/act",
        output_dir=str(out),
        embodiment="so101",
    )


def _training_modules() -> list[pathlib.Path]:
    """Top-level training backend modules (the ``rl`` subpackage is excluded).

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    return sorted(p for p in root.glob("*.py") if p.name != "__init__.py")


def _concrete_trainer_validators(source: str) -> dict[str, ast.FunctionDef]:
    """Map ``ClassName -> its validate() node`` for each ``Trainer`` subclass."""
    found: dict[str, ast.FunctionDef] = {}
    for node in parse_source(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(isinstance(b, ast.Name) and b.id == "Trainer" for b in node.bases):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "validate":
                found[node.name] = item
    return found


def _calls_run_size_gate(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_run_size_problems"
        for n in ast.walk(fn)
    )


def _local_run_size_comparisons(source: str) -> list[str]:
    """Comparisons of a run-size field against a numeric literal.

    That is the shape of a re-implemented domain (``if spec.steps <= 0``). A
    comparison against another *field* is a different question and is allowed.
    """
    hits: list[str] = []
    for node in walk_tree(parse_source(source)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr in RUN_SIZE_FIELDS):
            continue
        if any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) for c in node.comparators):
            hits.append(ast.unparse(node))
    return hits


class TestOneOwnerForTheRunSizeDomain:
    """A fifth backend cannot ship with its own copy of the rule."""

    def test_every_supervised_backend_routes_through_the_shared_gate(self) -> None:
        adrift: dict[str, list[str]] = {}
        seen: set[str] = set()
        for path in _training_modules():
            for cls_name, fn in _concrete_trainer_validators(path.read_text()).items():
                seen.add(cls_name)
                if not _calls_run_size_gate(fn):
                    adrift.setdefault(path.name, []).append(cls_name)
        steps_only = {t.__name__ for t in STEPS_ONLY_TRAINERS}
        adrift = {name: [c for c in classes if c not in steps_only] for name, classes in adrift.items()}
        adrift = {name: classes for name, classes in adrift.items() if classes}
        assert not adrift, f"validate() does not call _run_size_problems: {adrift}"
        # Non-vacuity: the scan really reached every backend, not an empty tree.
        assert seen == {t.__name__ for t in SUPERVISED_TRAINERS + STEPS_ONLY_TRAINERS}, seen

    def test_no_backend_re_implements_the_domain(self) -> None:
        copies = {p.name: hits for p in _training_modules() if (hits := _local_run_size_comparisons(p.read_text()))}
        assert not copies, f"run size compared against a literal instead of the shared gate: {copies}"

    def test_the_scanners_detect_a_planted_copy(self) -> None:
        """An empty result must mean clean sources, not a scanner matching nothing."""
        planted = (
            "class RogueTrainer(Trainer):\n"
            "    def validate(self, spec):\n"
            "        problems = []\n"
            "        if spec.steps <= 0:\n"
            "            problems.append('bad')\n"
            "        return problems\n"
        )
        validators = _concrete_trainer_validators(planted)
        assert set(validators) == {"RogueTrainer"}
        assert not _calls_run_size_gate(validators["RogueTrainer"])
        assert _local_run_size_comparisons(planted) == ["spec.steps <= 0"]


class TestTheGateIsUsableOnItsOwn:
    """The shared gate is a plain function - the trainers only bind a context."""

    def test_it_reports_both_factors_at_once(self, spec: TrainSpec) -> None:
        spec.steps = 0
        spec.global_batch_size = -1
        problems = run_size_problems(spec, context="probe")
        assert len(problems) == 2
        assert all(p.startswith("probe: ") for p in problems)

    def test_a_usable_spec_reports_nothing(self, spec: TrainSpec) -> None:
        assert run_size_problems(spec, context="probe") == []
