"""The backend tables on the docs site list every built-in simulation backend.

``docs/concepts/backends.md`` and ``docs/learn/simulation/index.md`` each carry
a table with one row per simulation backend. Both are hand-written, and both
fell a backend behind the code: ``mjlab`` shipped in ``_BUILTIN_BACKENDS``
while the tables still listed three, so a reader starting on either page never
learned it existed. This pin reads the first cell of each row and holds the set
to the factory's registry, in both directions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.simulation.factory import _BUILTIN_BACKENDS

_DOCS = Path(__file__).resolve().parent.parent / "docs"

#: page -> how a row's first cell spells a backend, keyed to the backend name.
_TABLES: dict[str, dict[str, str]] = {
    "concepts/backends.md": {"MuJoCo": "mujoco", "Newton": "newton", "Isaac Sim": "isaac", "mjlab": "mjlab"},
    "learn/simulation/index.md": {name: name for name in _BUILTIN_BACKENDS},
}


def _first_cells(page: str) -> list[str]:
    """The first cell of every row of the table headed ``| backend |``."""
    lines = (_DOCS / page).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| backend |"))
    cells = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cell = line.split("|")[1].strip()
        cells.append(re.sub(r"\[(.+?)\]\(.+?\)", r"\1", cell).strip("`"))
    return cells


@pytest.mark.parametrize("page", sorted(_TABLES))
def test_the_backend_table_lists_every_builtin(page: str) -> None:
    spelled = _TABLES[page]
    listed = sorted(spelled.get(cell, cell) for cell in _first_cells(page))
    assert listed == sorted(_BUILTIN_BACKENDS), (
        f"docs/{page} backend table lists {listed}; strands_robots.simulation.factory._BUILTIN_BACKENDS "
        f"ships {sorted(_BUILTIN_BACKENDS)}"
    )
