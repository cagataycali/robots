"""Every robot page says which checkpoints ran on it, and every number has a source.

The site used to let a reader go Home -> Start -> a robot page and never meet a
policy. Each generated robot page now ends with "Policies that ran on this
robot": a table read from ``docs/hooks/data/checkpoints.json`` when a checkpoint
was run on that robot first-hand, and otherwise the sentence that none has been
verified, with the recording page linked. This module grades the data file and
the pages it produces.

* Every robot the data file names is in the registry, so a renamed robot does
  not silently drop its rows.
* Every row carries a checkpoint, a provider, where it ran, what happened and a
  ``source`` that names the script, docs fence, issue or pull request the numbers
  came from. A number without a source is how a benchmark claim starts.
* A row's provider is a registered provider, or names the pull request that adds
  it: the page may not imply a provider that a reader cannot install.
* Every generated page carries the section, and a robot without rows carries the
  honest sentence rather than nothing.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
DATA = DOCS / "hooks" / "data" / "checkpoints.json"
PROVIDERS = REPO / "strands_robots" / "registry" / "policies.json"
REGISTRY = REPO / "strands_robots" / "registry" / "robots.json"

_REQUIRED = ("checkpoint", "kind", "provider", "where", "result", "source")
_SOURCE_SHAPE = re.compile(r"#\d{3,}|check_fences\.py|\.py\b|\.log\b|PR #\d+", re.I)
_HEADING = "## Policies verified on this robot"
_NONE_LINE = "No checkpoint verified on this robot yet."


def _hook():
    spec = importlib.util.spec_from_file_location("docs_hooks_robot_pages_ckpt", DOCS / "hooks" / "robot_pages.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data() -> dict[str, list[dict[str, str]]]:
    return json.loads(DATA.read_text(encoding="utf-8"))["robots"]


def _registry() -> set[str]:
    return set(json.loads(REGISTRY.read_text(encoding="utf-8"))["robots"])


def _providers() -> set[str]:
    return set(json.loads(PROVIDERS.read_text(encoding="utf-8"))["providers"])


def test_the_data_file_names_robots_the_registry_ships() -> None:
    unknown = sorted(set(_data()) - _registry())
    assert not unknown, f"checkpoints.json names robots the registry does not: {unknown}"


def test_some_robot_has_a_verified_checkpoint() -> None:
    """Non-vacuity: the table exists for at least the SO-101."""
    assert any(_data().values()), (
        "checkpoints.json lists no verified checkpoint; the section would say 'none' everywhere"
    )
    assert "so101" in _data()


@pytest.mark.parametrize(
    ("robot", "index", "row"),
    [(robot, i, row) for robot, rows in _data().items() for i, row in enumerate(rows)],
    ids=lambda v: v if isinstance(v, str) else (str(v) if isinstance(v, int) else v.get("checkpoint", "?")[:40]),
)
def test_every_row_is_complete_and_sourced(robot: str, index: int, row: dict[str, str]) -> None:
    missing = [k for k in _REQUIRED if not str(row.get(k, "")).strip()]
    assert not missing, f"checkpoints.json {robot}[{index}] lacks {missing}"
    assert _SOURCE_SHAPE.search(row["source"]), (
        f"checkpoints.json {robot}[{index}] source {row['source']!r} names no script, fence, log, issue or PR"
    )
    provider = row["provider"]
    assert provider.split(" ", 1)[0] in _providers() or re.search(r"PR #\d+", provider), (
        f"checkpoints.json {robot}[{index}] provider {provider!r} is neither registered nor tied to a PR"
    )


def test_every_row_avoids_a_score_it_cannot_back() -> None:
    """A result says what the rollout did; 'success rate' is a benchmark word this file does not use."""
    offenders = [
        f"{robot}: {row['checkpoint']}"
        for robot, rows in _data().items()
        for row in rows
        if re.search(r"success rate|\bSOTA\b|state[- ]of[- ]the[- ]art", row["result"], re.I)
    ]
    assert not offenders, f"checkpoints.json rows claim a score the source does not measure: {offenders}"


@pytest.mark.parametrize("name", sorted(_registry()))
def test_every_generated_page_states_its_verified_checkpoints(name: str) -> None:
    page = _hook().robot_page(name)
    assert _HEADING in page, f"robots/{name}.md has no '{_HEADING}' section"
    body = page.split(_HEADING, 1)[1]
    rows = _data().get(name, [])
    if rows:
        for row in rows:
            assert row["checkpoint"] in body, f"robots/{name}.md omits the verified checkpoint {row['checkpoint']!r}"
        assert _NONE_LINE not in body
    else:
        assert _NONE_LINE in body, f"robots/{name}.md has no verified checkpoint yet does not say so"
        assert "learn/data/record.md" in body, f"robots/{name}.md does not tell the reader how to record one"


def test_the_committed_pages_are_the_generator_output() -> None:
    """The pages are committed; a stale one would show a section the hook no longer writes."""
    hook = _hook()
    stale = [
        name
        for name in hook.registry()
        if (DOCS / "robots" / f"{name}.md").read_text(encoding="utf-8") != hook.robot_page(name)
    ]
    assert not stale, f"docs/robots pages differ from docs/hooks/robot_pages.py output; rerun it: {stale}"
