"""Repro: docs/hooks/extras.py emits blank purpose cells for 6 extras.

The ``{{extras:table}}`` token on docs/start/install.md expands via
docs/hooks/extras.py:extras_table(). The hook reads every entry from
``[project.optional-dependencies]`` in pyproject.toml, but renders the
purpose column from a hand-maintained ``_PURPOSE`` dict:

    rows.append(f"| `{name}` | {_package_names(specs)} | {_PURPOSE.get(name, '')} |")
                                                           ^^^^^^^^^^^^^^^^^^^^^^^
                                                           blank when key is missing

Six capability extras are declared in pyproject but absent from _PURPOSE,
so the published install page shows them with blank "purpose" cells. A
reader scanning the table sees:

    | `voice`    | `strands-agents[bidi]`   |     |
    | `xarm`     | `xarm-python-sdk`        |     |
    | `spot`     | `bosdyn-client`          |     |
    | `rby1`     | `rby1-sdk`               |     |
    | `holosoma` | `onnxruntime`, `huggingface_hub` |  |
    | `flux3`    |                          |     |

and reasonably concludes they are placeholders or deprecated.

The sibling test tests/test_docs_all_extra_membership.py grades only the
``[all]`` row's membership column; nothing grades the purpose column, so
the drift ships silently.

Run:
    cd $REPO && python3 bugbash_repros/extras_table_blank_purpose_repro.py

Expected (current, broken): "6 blank cells: ['flux3', 'holosoma', 'xarm', 'rby1', 'spot', 'voice']"
Expected (after fix):       "0 blank cells"
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from docs.hooks.extras import extras_table


def main() -> int:
    table = extras_table()
    blank: list[str] = []
    for row in table.split("\n")[2:]:  # skip header + separator
        cells = row.split("|")
        if len(cells) < 4:
            continue
        extra = cells[1].strip().strip("`")
        purpose = cells[3].strip()
        if not purpose:
            blank.append(extra)

    print(f"{len(blank)} blank cells: {blank}")
    if blank:
        print()
        print("Rows a reader sees on strands-labs.github.io/robots/start/install/:")
        for row in table.split("\n"):
            cells = row.split("|")
            if len(cells) >= 4 and cells[1].strip().strip("`") in blank:
                print(f"  {row}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
