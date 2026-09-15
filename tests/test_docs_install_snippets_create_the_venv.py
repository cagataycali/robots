"""Repo hygiene: an install entry point creates the venv before it installs.

``uv pip install`` installs into the *active* virtual environment and refuses
when there is none::

    error: No virtual environment found; run `uv venv` to create an
    environment, or pass `--system` to install into a non-virtual environment

A page whose first ``uv pip install`` command is not preceded by ``uv venv``
therefore fails on its first line for a reader working in a fresh shell, and the
reader is told nothing about why. The entry-point pages below are the ones read
before an environment exists, so each has to create one first.

Only commands inside fenced code blocks are graded: prose may name
``uv pip install`` while explaining the rule, and that mention is not a step a
reader runs.

Feature pages (``docs/mesh.md``, ``docs/policies/*.md``, ...) are out of scope:
they layer one extra onto an environment the reader already has, and
``docs/getting-started/installation.md`` states the rule once for all of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pages a reader reaches before they have an environment.
ENTRY_POINTS = (
    "README.md",
    "docs/index.md",
    "docs/getting-started/installation.md",
    "docs/getting-started/quickstart.md",
    "docs/contributing.md",
)


def _fenced_command_lines(text: str) -> list[str]:
    """Return the lines inside ``` fenced blocks, in page order.

    Prose is dropped, so a sentence naming a command is not read as a step the
    reader runs. Fence info strings (``bash``, ``python``) are dropped too.
    """
    lines: list[str] = []
    in_fence = False
    for raw in text.splitlines():
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            lines.append(raw.strip())
    return lines


@pytest.mark.parametrize("relpath", ENTRY_POINTS)
def test_install_entry_point_creates_the_venv_first(relpath: str) -> None:
    """The first ``uv pip install`` command on the page follows a ``uv venv``."""
    page = REPO_ROOT / relpath
    assert page.exists(), f"missing install entry point: {relpath}"
    commands = _fenced_command_lines(page.read_text(encoding="utf-8"))

    first_install = next((i for i, line in enumerate(commands) if "uv pip install" in line), None)
    assert first_install is not None, (
        f"{relpath} is listed as an install entry point but no fenced block "
        "runs 'uv pip install'; drop it from ENTRY_POINTS or restore the snippet."
    )

    assert any("uv venv" in line for line in commands[: first_install + 1]), (
        f"{relpath} runs 'uv pip install' before any 'uv venv'. uv's pip "
        "interface installs into the active virtual environment only and exits "
        "with 'No virtual environment found' when none is active, so a reader "
        "in a fresh shell fails on the first line. Create the environment "
        "first: 'uv venv --python 3.12 && source .venv/bin/activate'."
    )
