"""The RL pages must cover every trainer that ships and every field it reads.

``strands_robots.training.rl`` ships three trainers - ``ppo``, ``fast_sac`` and
``fast_td3`` - and ``create_trainer`` resolves all three. Two surfaces name the
third one already (``docs/getting-started/installation.md`` advertises
``FastTd3Trainer`` under the ``[rl]`` extra, and ``docs/policies/rl.md``
describes "the three backends"), so a training page that documents two of them
is drift, not scope: a reader who follows the install line has no page for the
trainer it installed, and no domain for the four fields only that trainer reads
(``policy_delay``, ``exploration_noise_std``, ``target_noise_std``,
``target_noise_clip``).

Three rules, each graded against the tree rather than a copied roster:

1. every trainer registered under ``training.rl`` has a row in the hub page's
   component table and a section of its own;
2. every RL-only ``RLTrainSpec`` field a trainer *reads* is named somewhere on
   the reference page, so it is discoverable at all;
3. a reference row's "Graded by" cell names every trainer that reads the field
   it keys. ``both`` cannot: with three trainers it is a count that does not say
   which two, and it stood on rows every trainer reads.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots.training.base import TrainSpec
from strands_robots.training.rl import RLTrainSpec

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_RL_PACKAGE = _REPO_ROOT / "strands_robots" / "training" / "rl"
_HUB = _REPO_ROOT / "docs" / "training" / "rl.md"
_REFERENCE = _REPO_ROOT / "docs" / "training" / "rl-reference.md"

#: Trainer name -> the class the hub page must name, and the backend module it
#: lives in. Read back from ``create_trainer`` below, so a fourth trainer fails
#: the roster cell rather than passing unnoticed.
_TRAINERS = {
    "ppo": ("PpoTrainer", "ppo"),
    "fast_sac": ("FastSacTrainer", "fast_sac"),
    "fast_td3": ("FastTd3Trainer", "fast_td3"),
}

#: How the reference's "Graded by" cell spells a set of trainers. A collective
#: spelling that names no trainer at all ("each backend") is a wildcard: the row
#: says the domain is per-backend and the domain column carries the detail.
_COLLECTIVE = {
    "all three": frozenset(_TRAINERS),
    "off-policy": frozenset({"fast_sac", "fast_td3"}),
}
_WILDCARD = ("each backend", "the backend that reads it")
_NAMED = {"PPO": "ppo", "FastSAC": "fast_sac", "FastTD3": "fast_td3"}


def _rl_only_fields() -> frozenset[str]:
    """``RLTrainSpec`` fields that are not inherited supervised ``TrainSpec`` ones."""
    return frozenset({f.name for f in dataclasses.fields(RLTrainSpec)}) - {
        f.name for f in dataclasses.fields(TrainSpec)
    }


def _spec_reads() -> dict[str, frozenset[str]]:
    """Map each RL-only field to the trainers that read ``spec.<field>``.

    Harvested from the sources rather than listed, so a field a new backend
    starts reading is graded without touching this test. A read in a shared
    module (``base_algo``'s own train loop, the env adapters) is a read by
    every trainer, because every trainer runs that code.
    """
    rl_only = _rl_only_fields()
    backends = {module: name for name, (_, module) in _TRAINERS.items()}
    reads: dict[str, set[str]] = {}
    for module in sorted(_RL_PACKAGE.glob("*.py")):
        owners = {backends[module.stem]} if module.stem in backends else set(_TRAINERS)
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "spec"
                and node.attr in rl_only
            ):
                reads.setdefault(node.attr, set()).update(owners)
    return {field: frozenset(owners) for field, owners in reads.items()}


def _reference_rows() -> list[tuple[frozenset[str], str]]:
    """The reference's field table as ``(fields keyed by the row, graded-by cell)``."""
    rows: list[tuple[frozenset[str], str]] = []
    for line in _REFERENCE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue
        fields = frozenset(re.findall(r"`([a-z_]+)`", cells[0]))
        rows.append((fields, cells[2]))
    return rows


def _graded_by(cell: str) -> frozenset[str] | None:
    """The trainers a "Graded by" cell names, or ``None`` when it is a wildcard."""
    if any(spelling in cell for spelling in _WILDCARD):
        return None
    for spelling, trainers in _COLLECTIVE.items():
        if spelling in cell:
            return trainers
    return frozenset(name for spelling, name in _NAMED.items() if spelling in cell)


def test_create_trainer_resolves_exactly_the_documented_roster() -> None:
    """The roster above is the trainers that exist, so the rules below are complete."""
    from strands_robots.training import create_trainer

    for name, (class_name, _) in _TRAINERS.items():
        assert type(create_trainer(name)).__name__ == class_name, name
    shipped = {module.stem for module in _RL_PACKAGE.glob("*.py")}
    assert shipped >= {module for _, module in _TRAINERS.values()}
    unlisted = {
        module
        for module in shipped
        if module not in {m for _, m in _TRAINERS.values()}
        and "class " in (_RL_PACKAGE / f"{module}.py").read_text(encoding="utf-8")
        and "BaseRLAlgo)" in (_RL_PACKAGE / f"{module}.py").read_text(encoding="utf-8")
    }
    assert not unlisted, f"a trainer this test does not grade: {sorted(unlisted)}"


@pytest.mark.parametrize(("trainer", "class_name"), [(n, c) for n, (c, _) in _TRAINERS.items()])
def test_the_hub_page_documents_every_shipped_trainer(trainer: str, class_name: str) -> None:
    """A trainer with no table row and no section is one the install line sells blind."""
    page = _HUB.read_text(encoding="utf-8")
    table_rows = [line for line in page.splitlines() if line.startswith("| [`") or line.startswith("| `")]
    assert any(f"`{class_name}`" in row for row in table_rows), (
        f"{class_name} (create_trainer({trainer!r})) has no row in the component table of {_HUB.name}"
    )
    headings = {line.lstrip("# ").strip().lower() for line in page.splitlines() if line.startswith("## ")}
    assert any(class_name.lower().replace("trainer", "") in heading.replace(" ", "") for heading in headings), (
        f"{class_name} has no section in {_HUB.name}; headings are {sorted(headings)}"
    )
    assert f'create_trainer("{trainer}")' in page, f"{_HUB.name} never spells create_trainer({trainer!r})"


def test_every_field_a_trainer_reads_is_named_on_the_reference_page() -> None:
    """A field with no mention anywhere on the reference page cannot be discovered."""
    reads = _spec_reads()
    assert len(reads) >= 25, f"harvest went thin: only {len(reads)} RL-only fields read"
    reference = _REFERENCE.read_text(encoding="utf-8")
    missing = sorted(field for field in reads if f"`{field}`" not in reference)
    assert not missing, f"{_REFERENCE.name} names none of {missing}, read by " + ", ".join(
        f"{field} ({', '.join(sorted(reads[field]))})" for field in missing
    )


def test_every_reference_row_names_the_trainers_that_read_its_fields() -> None:
    """An attribution narrower than the readers sends a caller to the wrong domain."""
    reads = _spec_reads()
    rows = _reference_rows()
    assert len(rows) >= 15, f"row harvest went thin: {len(rows)} rows"
    graded_any = 0
    wrong: list[str] = []
    for fields, cell in rows:
        named = _graded_by(cell)
        if named is None:
            continue
        for field in sorted(fields):
            readers = reads.get(field)
            if readers is None:
                continue  # a SimEnv argument or an evaluate() knob, not a spec field
            graded_any += 1
            if not readers <= named:
                wrong.append(f"{field}: read by {sorted(readers)}, graded by {cell!r} -> {sorted(named)}")
    assert graded_any >= 10, f"no attribution was graded ({graded_any} field/row pairs)"
    assert not wrong, "reference rows attribute a field to fewer trainers than read it:\n" + "\n".join(wrong)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("all three", frozenset(_TRAINERS)),
        ("off-policy", frozenset({"fast_sac", "fast_td3"})),
        ("PPO", frozenset({"ppo"})),
        ("FastSAC", frozenset({"fast_sac"})),
        ("FastTD3", frozenset({"fast_td3"})),
        ("each backend", None),
        ("the backend that reads it", None),
        ("both", frozenset()),  # names no trainer, so it can satisfy no reader
    ],
)
def test_the_graded_by_reader_expands_each_spelling(cell: str, expected: frozenset[str] | None) -> None:
    """The falsification of the rule above: a cell naming nobody must grade as nobody."""
    assert _graded_by(cell) == expected
