"""The optimizer learning rate is one shared positive-finite domain.

:attr:`~strands_robots.training.base.TrainSpec.learning_rate` is the one
numeric on a spec that decides whether a run *learns* rather than how much work
it does, and unlike the run-size fields **every** backend reads it: the three
supervised trainers assign it to their config's optimizer field (LeRobot
``policy.optimizer_lr``, GR00T ``FinetuneConfig.learning_rate``, Cosmos
``optimizer.lr``) and the RL trainers pass it straight to
``torch.optim.Adam(..., lr=...)``. ``rl/base_algo.py`` calls it one of the
"universal" spec fields.

Before this contract no backend checked it, while each bounded its neighbours:
``FastSacTrainer.validate`` compares eight sibling numerics against literals
(``total_timesteps``, ``rollout_steps``, ``num_envs``, ``buffer_size``,
``batch_size``, ``gradient_steps``, ``tau``, ``learning_starts``) and skipped
the one that decides whether any of that work updates a weight, and the
supervised backends gate the two run-size factors and skipped it too.

The two ends of the domain fail *silently*, which is why this is a preflight
rather than something a backend can be left to notice:

* ``0`` (and ``False``) runs the full ``steps`` x ``global_batch_size`` of work
  and updates no weight - the checkpoint equals the initialisation and the run
  reports success. That is the pathology the run-size gate exists to prevent,
  reached by a different route and at full cost.
* ``inf`` diverges on the first step, so the checkpoint is all ``NaN``, again
  under a successful result.
* ``True`` is a silent learning rate of ``1.0``.

A negative or ``nan`` value *is* refused - by ``torch.optim.Adam`` itself - but
only once the dataset and model are loaded, past the point
:meth:`Trainer.validate` documents itself as running before.

Which backends refuse which values, in which words, is the table in
``tests/training/test_train_spec_field_domains.py``. The tests below ground the
refused values in what a real optimizer does with them and keep every backend on
the one gate.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
from typing import Any

import pytest

from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from strands_robots.training.rsl_rl import RslRlTrainer
from strands_robots.training.sagemaker import SagemakerTrainer
from tests._package_ast import parse_source, walk_tree

# Values no backend can honor, split by how each one failed before the gate.

# Ran the whole schedule and updated nothing, under a successful result.
SILENT_NO_OP = (0, 0.0, False)

# Completed and wrote a checkpoint of NaN, also under a successful result.
SILENT_DIVERGENCE = (float("inf"),)

# Honored as 1.0 - four orders of magnitude above a fine-tuning preset.
SILENT_MISREAD = (True,)

# Refused by the optimizer, but only after the dataset and model were loaded.
LOUD_BUT_LATE = (-1e-3, float("nan"))

# Never reached an optimizer at all: a config field of the wrong type.


SUPERVISED_TRAINERS = (MockTrainer, Cosmos3Trainer, LerobotTrainer, RslRlTrainer, SagemakerTrainer)
RL_TRAINER_NAMES = ("FastSacTrainer", "FastTd3Trainer", "PpoTrainer")
# RL transports: the run happens in another interpreter, which receives the rate as an override.
RL_TRANSPORT_NAMES = ("IsaacLabTrainer",)
ALL_TRAINER_NAMES = tuple(t.__name__ for t in SUPERVISED_TRAINERS) + RL_TRAINER_NAMES + RL_TRANSPORT_NAMES


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose other fields are all fine, so only the rate varies."""
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


class TestTheRefusedValuesAreOnesTheConsumerCannotHonor:
    """Ground the domain in what a real optimizer does with each value."""

    @staticmethod
    def _run(rate: Any, steps: int = 20) -> tuple[float, bool]:
        """Train a 1-layer model for ``steps`` - returns (max weight delta, any NaN)."""
        import torch

        torch.manual_seed(0)
        model = torch.nn.Linear(4, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        inputs, targets = torch.randn(8, 4), torch.randn(8, 1)
        for _ in range(steps):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(inputs), targets).backward()
            optimizer.step()
        after = model.weight.detach()
        return float((after - before).abs().max()), bool(torch.isnan(after).any())

    def test_a_zero_rate_completes_the_run_and_updates_nothing(self) -> None:
        pytest.importorskip("torch")
        for value in SILENT_NO_OP:
            delta, has_nan = self._run(value)
            assert delta == 0.0, f"lr={value!r} moved the weights by {delta}"
            assert not has_nan, f"{value!r}"

    def test_a_usable_rate_does_move_the_weights(self) -> None:
        """Non-vacuity: the probe can tell learning from no learning."""
        pytest.importorskip("torch")
        delta, has_nan = self._run(1e-3)
        assert delta > 0.0
        assert not has_nan

    def test_an_infinite_rate_completes_the_run_with_nan_weights(self) -> None:
        pytest.importorskip("torch")
        for value in SILENT_DIVERGENCE:
            _, has_nan = self._run(value)
            assert has_nan, f"lr={value!r} was expected to diverge"

    def test_a_boolean_rate_is_read_as_one(self) -> None:
        pytest.importorskip("torch")
        for value in SILENT_MISREAD:
            assert self._run(value)[0] == pytest.approx(self._run(1.0)[0]), f"{value!r}"

    def test_a_negative_or_nan_rate_is_refused_by_the_optimizer(self) -> None:
        """Refused, but only once a model exists to build an optimizer over."""
        torch = pytest.importorskip("torch")
        for value in LOUD_BUT_LATE:
            with pytest.raises(ValueError, match="[Ll]earning rate"):
                torch.optim.Adam(torch.nn.Linear(2, 1).parameters(), lr=value)

    def test_the_backend_config_layer_does_not_catch_it_either(self) -> None:
        """lerobot's own optimizer config accepts every unusable rate."""
        pytest.importorskip("lerobot")
        from lerobot.optim.optimizers import AdamWConfig

        for value in (0.0, -1e-3, float("nan"), float("inf")):
            assert AdamWConfig(lr=value).lr == value or value != value


def _trainer_sources() -> dict[str, str]:
    """``filename -> source`` for every backend module in the tree."""
    return {p.name: p.read_text() for p in _trainer_modules()}


def _trainer_modules() -> list[pathlib.Path]:
    """Every backend module, INCLUDING the ``rl`` subpackage.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The rl subpackage is in scope here
    (unlike for the run-size gate) precisely because there is no backend that
    may skip this rule.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py")


def _trainer_class_defs(sources: dict[str, str]) -> dict[str, ast.ClassDef]:
    """Every class in the backend tree that descends from :class:`Trainer`.

    The base chain is followed to a fixpoint rather than matched against one
    name: the RL trainers reach :class:`Trainer` through ``BaseRLAlgo``, so a
    scanner keyed on ``class X(Trainer)`` alone would silently skip the two
    backends that hand the rate straight to ``torch.optim.Adam``.
    """
    classes: dict[str, ast.ClassDef] = {}
    for source in sources.values():
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node

    descendants = {"Trainer"}
    grew = True
    while grew:
        grew = False
        for name, node in classes.items():
            if name in descendants:
                continue
            if any(isinstance(b, ast.Name) and b.id in descendants for b in node.bases):
                descendants.add(name)
                grew = True
    return {n: c for n, c in classes.items() if n in descendants - {"Trainer"}}


def _concrete_trainer_validators(sources: dict[str, str]) -> dict[str, ast.FunctionDef]:
    """Map ``ClassName -> its validate() node`` for each concrete backend.

    A class that does not define ``validate`` of its own (the abstract
    ``BaseRLAlgo``) has no gate to call and is not a backend.
    """
    found: dict[str, ast.FunctionDef] = {}
    for name, node in _trainer_class_defs(sources).items():
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "validate":
                found[name] = item
    return found


def _calls_learning_rate_gate(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_learning_rate_problems"
        for n in ast.walk(fn)
    )


def _local_learning_rate_comparisons(source: str) -> list[str]:
    """Comparisons of ``learning_rate`` against a numeric literal.

    That is the shape of a re-implemented domain (``if spec.learning_rate <= 0``).
    """
    hits: list[str] = []
    for node in walk_tree(parse_source(source)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == "learning_rate"):
            continue
        if any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) for c in node.comparators):
            hits.append(ast.unparse(node))
    return hits


class TestOneOwnerForTheLearningRateDomain:
    """A seventh backend cannot ship without the rule."""

    def test_every_backend_routes_through_the_shared_gate(self) -> None:
        validators = _concrete_trainer_validators(_trainer_sources())
        adrift = sorted(n for n, fn in validators.items() if not _calls_learning_rate_gate(fn))
        assert not adrift, f"validate() does not call _learning_rate_problems: {adrift}"
        # Non-vacuity: the scan really reached every backend, not an empty tree.
        assert set(validators) == set(ALL_TRAINER_NAMES), sorted(validators)

    def test_no_backend_re_implements_the_domain(self) -> None:
        copies = {p.name: hits for p in _trainer_modules() if (hits := _local_learning_rate_comparisons(p.read_text()))}
        assert not copies, f"learning_rate compared against a literal instead of the shared gate: {copies}"

    def test_the_scanners_detect_a_planted_copy(self) -> None:
        """An empty result must mean clean sources, not a scanner matching nothing."""
        planted = (
            "class RogueTrainer(Trainer):\n"
            "    def validate(self, spec):\n"
            "        problems = []\n"
            "        if spec.learning_rate <= 0:\n"
            "            problems.append('bad')\n"
            "        return problems\n"
        )
        validators = _concrete_trainer_validators({"planted.py": planted})
        assert set(validators) == {"RogueTrainer"}
        assert not _calls_learning_rate_gate(validators["RogueTrainer"])
        assert _local_learning_rate_comparisons(planted) == ["spec.learning_rate <= 0"]
