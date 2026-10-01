"""The committed ``static/app.js`` and ``static/vendor/react.js`` are one build.

Vite splits React into a vendor chunk and gives its exports minified names
(``r``, ``j``, ``R``, ``a``); ``app.js`` imports exactly those. A rebuilt
``app.js`` committed against the previous vendor chunk imports a name the chunk
does not export, and the browser throws ``SyntaxError: The requested module
'./vendor/react.js' does not provide an export named ...`` before a single line
of the SPA runs - a blank dashboard, shipped inside the wheel. No frontend
helper test sees it because they exercise functions, not the module graph.
This grader reads both files and holds the pair together.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard" / "static"

_IMPORT = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"\./vendor/react\.js"')
_EXPORT = re.compile(r"export\s*\{([^}]*)\}")


def _names(spec: str, *, imported: bool) -> set[str]:
    out = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(" as ")]
        out.add(parts[0] if imported else parts[-1])
    return out


def test_every_react_name_app_js_imports_is_exported_by_the_committed_vendor_chunk() -> None:
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    vendor = (STATIC / "vendor" / "react.js").read_text(encoding="utf-8")
    imports = _IMPORT.findall(app)
    assert imports, "app.js no longer imports ./vendor/react.js - update this grader with the new chunk name"
    imported = set().union(*(_names(spec, imported=True) for spec in imports))
    exported = set().union(*(_names(spec, imported=False) for spec in _EXPORT.findall(vendor)))
    assert exported, "vendor/react.js has no export statement the grader can read"
    missing = sorted(imported - exported)
    assert not missing, (
        f"app.js imports {missing} from ./vendor/react.js but the committed chunk exports only "
        f"{sorted(exported)}: the two files come from different builds. Re-run `npm run build` in "
        "strands_robots/dashboard/frontend and commit app.js AND vendor/react.js together."
    )
