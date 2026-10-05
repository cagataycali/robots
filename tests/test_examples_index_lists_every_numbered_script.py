"""``examples/README.md`` indexes every numbered script under one number each.

A new reader picks a walkthrough from the README index and then runs it by
number. On ``972df275`` the index had drifted from the directory three ways:
the intro sold ``01_*``..``15_*`` while the scripts ran to ``18_*``,
``18_so101_pick_and_lift.py`` had no row, and two scripts shared the ``17_``
prefix (``17_pour_task.py`` and ``17_judge_recorded_episodes.py``), so the
index carried two rows numbered 17. One rule covers all three.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_ROW = re.compile(r"^\| (\d\d) \| \[`(\d\d_\w+\.py)`\]", re.MULTILINE)
_INTRO = re.compile(r"numbered `(\d\d)_\*`\.\.`(\d\d)_\*`")


def _index_problems(readme: str, scripts: list[str]) -> list[str]:
    """Return every way ``readme`` disagrees with the numbered ``scripts``."""
    problems: list[str] = []
    by_number: dict[str, list[str]] = {}
    for name in scripts:
        by_number.setdefault(name[:2], []).append(name)
    problems += [f"prefix {n}_ is shared by {sorted(v)}" for n, v in by_number.items() if len(v) > 1]
    rows = _ROW.findall(readme)
    problems += [f"row {n} links {f}" for n, f in rows if not f.startswith(f"{n}_")]
    listed = [f for _, f in rows]
    problems += [f"{f} has no index row" for f in scripts if f not in listed]
    problems += [f"{f} is indexed but not on disk" for f in listed if f not in scripts]
    intro = _INTRO.search(readme)
    span = (min(by_number), max(by_number)) if by_number else ("", "")
    if intro is None or intro.groups() != span:
        problems.append(f"intro range {intro.groups() if intro else None} is not {span}")
    return problems


def test_the_index_matches_the_directory() -> None:
    scripts = sorted(p.name for p in _EXAMPLES.glob("[0-9][0-9]_*.py"))
    assert _index_problems((_EXAMPLES / "README.md").read_text(encoding="utf-8"), scripts) == []


_GOOD = "The numbered `01_*`..`02_*` scripts\n| 01 | [`01_a.py`](01_a.py) |\n| 02 | [`02_b.py`](02_b.py) |\n"


@pytest.mark.parametrize(
    ("readme", "scripts", "problem"),
    [
        (_GOOD.replace("`02_*`", "`01_*`"), ["01_a.py", "02_b.py"], "intro range"),
        (_GOOD, ["01_a.py", "02_b.py", "03_c.py"], "03_c.py has no index row"),
        (_GOOD, ["01_a.py", "02_b.py", "02_c.py"], "prefix 02_ is shared"),
        (_GOOD.replace("| 02 |", "| 01 |"), ["01_a.py", "02_b.py"], "row 01 links 02_b.py"),
    ],
    ids=["stale-intro", "missing-row", "shared-prefix", "misnumbered-row"],
)
def test_a_planted_drift_is_reported(readme: str, scripts: list[str], problem: str) -> None:
    assert _index_problems(_GOOD, ["01_a.py", "02_b.py"]) == []
    assert any(problem in p for p in _index_problems(readme, scripts))
