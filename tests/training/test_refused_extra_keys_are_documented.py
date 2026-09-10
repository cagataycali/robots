"""Every ``extra`` key a training preflight refuses by name is named in the docs.

``TrainSpec.extra`` is an open dict: most keys are forwarded to lerobot's config
tree and a key that names no field is dropped with a warning. A handful are
different - the trainers read them by name and ``validate()`` / ``build_config``
refuse them with a message that spells the key, e.g.::

    extra['sample_weighting'] does not support field(s) ['kapa'];
    accepted keys are ['epsilon', 'head_mode', 'kappa', 'progress_path', 'type']

A caller who reads that message needs somewhere to look up what the accepted
fields mean, and the Training overview is that page. A refusal naming a key the
docs never mention is a dead end: the message proves the knob exists and the
documentation denies it.

The population is derived from the refusal strings themselves rather than
listed here, so a new refusal that names a new ``extra`` key joins the guard on
the commit that adds it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_DIR = _REPO_ROOT / "strands_robots" / "training"
_OVERVIEW = _REPO_ROOT / "docs" / "training" / "overview.md"

# ``extra['key']`` / ``extra["key"]`` inside a message, dotted keys included.
_EXTRA_KEY_IN_MESSAGE = re.compile(r"""extra\[['"]([a-z_.]+)['"]\]""")


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Identity of every docstring constant in ``tree``.

    Docstrings describe a key; they do not refuse one. Only runtime message
    strings establish that a caller can be sent to a key by name.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            ids.add(id(first.value))
    return ids


def _refused_extra_keys() -> dict[str, list[str]]:
    """Map each ``extra`` key named in a refusal message to its sites."""
    found: dict[str, list[str]] = {}
    for path in sorted(_TRAINING_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstring_ids(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in skip:
                continue
            for key in _EXTRA_KEY_IN_MESSAGE.findall(node.value):
                found.setdefault(key, []).append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
    return found


_REFUSED = _refused_extra_keys()


def test_the_walk_finds_the_structured_extra_dicts() -> None:
    """The scan reads real refusals, so an empty walk cannot pass the guard."""
    assert len(_REFUSED) >= 5, _REFUSED
    # The two dicts with their own field allowlists - the keys most in need of a
    # documented vocabulary, because the refusal names fields, not just the key.
    assert "reward_model" in _REFUSED
    assert "sample_weighting" in _REFUSED


@pytest.mark.parametrize("key", sorted(_REFUSED))
def test_refused_extra_key_is_documented(key: str) -> None:
    """A key a preflight refuses by name is named in the Training overview."""
    overview = _OVERVIEW.read_text(encoding="utf-8")
    assert key in overview, (
        f"extra['{key}'] is refused by name at {_REFUSED[key]} but never appears in "
        f"{_OVERVIEW.relative_to(_REPO_ROOT)}. A caller sent to a key by a refusal "
        "message has nowhere to look up what it accepts; document the key or stop "
        "naming it in the message."
    )
