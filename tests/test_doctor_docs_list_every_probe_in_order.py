"""The doctor page's probe table lists exactly :data:`strands_robots.doctor.CHECKS`, in run order.

``docs/start/doctor.md`` is the one page that describes each probe. It once
stopped at 15 rows while the runtime printed 16 (``IoT Direct``, between
``Mesh`` and ``Sim Test``), so a reader could not place a row every report
prints. ``docs/reference/cli.md`` links here instead of keeping its own list.
Other pages name "the probes" without a count: ``start/install.md`` and
``start/index.md`` still said "fifteen" after :data:`CHECKS` reached 17.
"""

from __future__ import annotations

import re
from pathlib import Path

from strands_robots.doctor import CHECKS

_PAGE = Path(__file__).resolve().parents[1] / "docs" / "start" / "doctor.md"
_DOCS = Path(__file__).resolve().parents[1] / "docs"
_COUNTED = re.compile(
    r"\b(\d+|[a-z]+teen|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)\s+probes\b", re.I
)
_HEADER = "| row | what is checked | not PASS when |"


def test_the_probe_table_lists_every_probe_in_run_order():
    table = _PAGE.read_text(encoding="utf-8").split(_HEADER, 1)[1].split("\n\n", 1)[0]
    rows = [line.split("|")[1].strip() for line in table.strip().splitlines()[1:]]
    assert rows == [label for label, _ in CHECKS]


def test_no_docs_page_states_a_probe_count():
    hits = [
        f"{page.relative_to(_DOCS)}: {m.group(0)}"
        for page in sorted(_DOCS.rglob("*.md"))
        for m in _COUNTED.finditer(page.read_text(encoding="utf-8"))
    ]
    assert hits == []
