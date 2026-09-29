"""Every doc path the Isaac backend points a user at exists.

Two runtime refusals - a worker-thread call with no main-thread pump, in
``_require_main_thread_or_pump`` and ``set_joint_positions`` - told the user to
"see docs/reference/simulation/isaac.md for the agent-driven shape". No such
file exists (the page is ``docs/learn/simulation/isaac.md``), and the page that
does exist did not describe the shape either. The page itself cited
``docs-old/reference/simulation/isaac-parity.md``, a directory the repository
does not have.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DOC = _ROOT / "docs" / "learn" / "simulation" / "isaac.md"
_PATH = re.compile(r"\bdocs(?:-old)?/[\w./-]+\.md\b")


def _cited(paths: list[pathlib.Path]) -> list[tuple[pathlib.Path, str]]:
    return [(p, m) for p in paths for m in _PATH.findall(p.read_text(encoding="utf-8"))]


def test_the_isaac_backend_cites_only_docs_that_exist() -> None:
    sources = sorted((_ROOT / "strands_robots" / "simulation" / "isaac").rglob("*.py"))
    sources += [_DOC, *sorted((_ROOT / "examples" / "isaac_gs").rglob("*.py"))]
    missing = [f"{p.relative_to(_ROOT)} -> {m}" for p, m in _cited(sources) if not (_ROOT / m).is_file()]
    assert not missing, "\n".join(missing)


def test_the_page_describes_the_agent_driven_shape_the_refusals_point_at() -> None:
    text = _DOC.read_text(encoding="utf-8")
    section = text.split("## Threading", 1)[1].split("\n## ", 1)[0]
    assert "run_pump_forever(stop_event=" in section
    assert "run_on_main(" in section
