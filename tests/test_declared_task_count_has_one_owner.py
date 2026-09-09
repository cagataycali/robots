"""The task count a dataset's ``meta/info.json`` declares has ONE verdict.

``total_tasks`` decides whether a global held-out episode COUNT is expressible
at all. lerobot applies a validation split as one ``eval_split`` FRACTION and
holds out ``ceil(episodes_in_task * eval_split)`` from every task independently
(``lerobot.datasets.factory.make_train_eval_datasets``), so a single fraction
reproduces a requested global count only on a single-task dataset.
:func:`~strands_robots.utils.validation_split_error` is the guard that refuses
the request rather than quietly reserving a different number of episodes.

Two surfaces read the header and hand it to that guard: the ``lerobot_train``
tool and :class:`~strands_robots.training.lerobot.LerobotTrainer`. Both used to
convert it first - ``total if isinstance(total, int) and not isinstance(total,
bool) else 0`` - and 0 is the value the guard honors as "no task count
recorded", so every declaration outside a bare ``int`` arrived as the absent
case. A three-task dataset whose header spelled its count ``3.0`` or ``"3"``
passed the guard written to refuse exactly that dataset, and lerobot then took
the ceiling once per task.

The declaration is now graded by :func:`~strands_robots.utils.declared_count`,
the one owner every reader of a LeRobot header count already shares, and a
header that declares something which is not a count is a THIRD outcome - refused
on its own terms rather than collapsed into the absent case.

Fixtures are raw ``info.json`` text (not ``json.dumps``, which cannot write
``1e400`` or ``NaN`` from a Python value), so these tests need no dataset files
beyond the metadata.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import pytest

import strands_robots
from strands_robots.tools.lerobot_train import _read_total_tasks, build_train_command
from strands_robots.utils import validation_split_error

#: Requested held-out episode count, and the dataset the request is made against.
VAL_EPISODES = 2
TOTAL_EPISODES = 10
#: Episodes per task of a three-task, ten-episode dataset. lerobot's per-task
#: ceiling holds out 3 of these for the fraction that reserves exactly 2
#: globally, which is why the request is refused rather than reinterpreted.
TASK_SIZES = (3, 3, 4)

#: Every spelling of a three-task count that is not a usable task count. ``3.0``
#: and ``"3"`` are "right number, wrong type" - the pair a writer plausibly
#: produces and the pair that read as an ABSENT header. The rest are not three at
#: all, and were honored as single-task just the same.
UNUSABLE_DECLARATIONS = ["3.0", '"3"', "2.5", "true", "1e400", "NaN", "-3"]

#: Declarations that mean "this dataset records no task count", which lerobot's
#: own field spells 0. These are honored as single-task and must stay so.
NO_COUNT_DECLARATIONS = ["0", "1", "null", None]


def _dataset(root: Path, declaration: str | None) -> str:
    """Write a ten-episode dataset whose header declares ``declaration`` verbatim.

    Args:
        root: Dataset root to create.
        declaration: Raw JSON text for ``total_tasks``, or ``None`` to omit the
            key entirely (the "no header" case).

    Returns:
        The dataset root as a string, ready for either surface under test.
    """
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    body = f'"total_episodes": {TOTAL_EPISODES}'
    if declaration is not None:
        body += f', "total_tasks": {declaration}'
    (meta / "info.json").write_text("{" + body + "}")
    return str(root)


def _spec_problems(root: str) -> list[str]:
    """``LerobotTrainer.validate`` problems for a ``val_episodes`` request."""
    pytest.importorskip("lerobot")
    from strands_robots.training.base import TrainSpec
    from strands_robots.training.lerobot import LerobotTrainer

    spec = TrainSpec(dataset_root=root, output_dir=str(Path(root) / "out"), val_episodes=VAL_EPISODES, steps=10)
    return LerobotTrainer(device="cpu").validate(spec)


def _emitted_split(cmd: list[str]) -> float | None:
    """The single ``--dataset.eval_split`` value in ``cmd``, or None."""
    hits = [c.split("=", 1)[1] for c in cmd if c.startswith("--dataset.eval_split=")]
    assert len(hits) <= 1, f"eval_split emitted {len(hits)} times: {hits}"
    return float(hits[0]) if hits else None


class TestADeclarationThatIsNotACountIsRefused:
    """The third outcome: neither a usable count nor an absent header."""

    @pytest.mark.parametrize("declaration", UNUSABLE_DECLARATIONS)
    def test_the_train_tool_refuses_it_by_name(self, declaration: str, tmp_path: Path) -> None:
        root = _dataset(tmp_path / "ds", declaration)
        with pytest.raises(ValueError, match="not a task count"):
            build_train_command(dataset_root=root, policy_type="act", val_episodes=VAL_EPISODES)

    @pytest.mark.parametrize("declaration", UNUSABLE_DECLARATIONS)
    def test_the_trainspec_backend_reports_the_same_problem(self, declaration: str, tmp_path: Path) -> None:
        root = _dataset(tmp_path / "ds", declaration)
        assert any("not a task count" in problem for problem in _spec_problems(root))

    def test_the_refusal_renders_the_declaration_and_the_remedy(self) -> None:
        error = validation_split_error(VAL_EPISODES, "3", "lerobot_train", passthrough_param="extra_flags")
        assert error is not None
        assert error.startswith("lerobot_train: ")
        assert "total_tasks='3'" in error, error
        assert "extra_flags={'dataset.eval_split': 0.1" in error


class TestTheDeclarationReachesTheGuardUnconverted:
    """A reader that converts first hands the guard a value it cannot grade."""

    @pytest.mark.parametrize("declaration", [*UNUSABLE_DECLARATIONS, "3", "0"])
    def test_the_tool_reader_returns_what_the_header_carried(self, declaration: str, tmp_path: Path) -> None:
        root = _dataset(tmp_path / "ds", declaration)
        declared = _read_total_tasks(root)
        # Compared by repr so 3.0 is not indistinguishable from 3, and True not
        # from 1: those collapses are the reason the guard could not see them.
        assert repr(declared) == repr(_json_value(declaration))

    def test_no_header_is_the_absent_case(self, tmp_path: Path) -> None:
        assert _read_total_tasks(_dataset(tmp_path / "ds", None)) is None

    def test_a_dataset_without_metadata_is_the_absent_case(self, tmp_path: Path) -> None:
        assert _read_total_tasks(str(tmp_path / "nothing-here")) is None


class TestBothSurfacesSpendTheOneVerdict:
    """No surface may grade this header for itself."""

    def test_exactly_the_two_known_surfaces_consult_the_guard(self) -> None:
        package = Path(strands_robots.__file__).parent
        callers = set()
        for module in package.rglob("*.py"):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("validation_split_error"):
                    callers.add(module.relative_to(package).as_posix())
        assert callers == {"tools/lerobot_train.py", "training/lerobot.py"}, callers

    @pytest.mark.parametrize("declaration", [*UNUSABLE_DECLARATIONS, *NO_COUNT_DECLARATIONS, "3"])
    def test_the_two_surfaces_reach_the_same_verdict(self, declaration: str | None, tmp_path: Path) -> None:
        root = _dataset(tmp_path / "ds", declaration)
        try:
            build_train_command(dataset_root=root, policy_type="act", val_episodes=VAL_EPISODES)
        except ValueError as exc:
            tool_refused: str | None = str(exc)
        else:
            tool_refused = None
        spec_refused = [p for p in _spec_problems(root) if "task count" in p or "reserved exactly" in p]
        assert (tool_refused is None) == (not spec_refused), (declaration, tool_refused, spec_refused)


class TestTheHonoredCasesAreUnchanged:
    """A recorded count of 0/1, and no header at all, still take the split."""

    @pytest.mark.parametrize("declaration", NO_COUNT_DECLARATIONS)
    def test_a_split_is_still_emitted(self, declaration: str | None, tmp_path: Path) -> None:
        root = _dataset(tmp_path / "ds", declaration)
        cmd = build_train_command(dataset_root=root, policy_type="act", val_episodes=VAL_EPISODES)
        fraction = _emitted_split(cmd)
        assert fraction is not None
        # Single task, so lerobot's one ceiling reserves exactly what was asked.
        assert math.ceil(TOTAL_EPISODES * fraction) == VAL_EPISODES

    def test_a_truthful_multi_task_count_is_still_refused_as_such(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path / "ds", "3")
        with pytest.raises(ValueError, match="cannot be reserved exactly"):
            build_train_command(dataset_root=root, policy_type="act", val_episodes=VAL_EPISODES)

    def test_the_multi_task_refusal_still_names_the_count(self) -> None:
        error = validation_split_error(VAL_EPISODES, 3, "ctx", passthrough_param="extra")
        assert error is not None and "dataset with 3 tasks" in error

    def test_the_fraction_a_three_task_dataset_would_have_taken_overshoots(self) -> None:
        """Why the request is refused rather than reinterpreted, in episodes."""
        from strands_robots.utils import validation_split_fraction

        fraction = validation_split_fraction(VAL_EPISODES, TOTAL_EPISODES)
        held_out = sum(math.ceil(size * fraction) for size in TASK_SIZES)
        assert held_out != VAL_EPISODES


def _json_value(declaration: str) -> Any:
    """The Python value ``json.load`` produces for raw JSON text."""
    import json

    return json.loads(declaration)
