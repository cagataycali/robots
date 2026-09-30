"""The doctor page's probe table lists exactly :data:`strands_robots.doctor.CHECKS`, in run order.

``docs/start/doctor.md`` is the one page that describes each probe. It once
stopped at 15 rows while the runtime printed 16 (``IoT Direct``, between
``Mesh`` and ``Sim Test``), so a reader could not place a row every report
prints. ``docs/reference/cli.md`` links here instead of keeping its own list.
"""

from __future__ import annotations

from pathlib import Path

from strands_robots.doctor import CHECKS

_PAGE = Path(__file__).resolve().parents[1] / "docs" / "start" / "doctor.md"
_HEADER = "| row | what is checked | not PASS when |"


def test_the_probe_table_lists_every_probe_in_run_order():
    table = _PAGE.read_text(encoding="utf-8").split(_HEADER, 1)[1].split("\n\n", 1)[0]
    rows = [line.split("|")[1].strip() for line in table.strip().splitlines()[1:]]
    assert rows == [label for label, _ in CHECKS]
