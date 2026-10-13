"""Ensure examples/ never imports from lerobot directly.

Examples are the first thing a new user reads. They must demonstrate the
strands_robots abstraction, not bypass it. Any ``from lerobot`` or
``import lerobot`` at the top-level of an example file is a documentation
failure: it teaches users to skip the SDK and wire lerobot manually.

Examples now live in topic subfolders (vla/, wbc/, training/, ...), so this
walks the tree recursively; every example script is checked wherever it lives.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# Regex matching bare lerobot imports (top-level or inside functions).
# Matches: "from lerobot", "import lerobot"
_LEROBOT_IMPORT_RE = re.compile(r"^\s*(from\s+lerobot|import\s+lerobot)", re.MULTILINE)


def _example_scripts() -> list[Path]:
    """Collect .py files anywhere under examples/ (recursively)."""
    if not _EXAMPLES_DIR.is_dir():
        return []
    return sorted(_EXAMPLES_DIR.rglob("*.py"))


def test_no_direct_lerobot_import() -> None:
    """Top-level example scripts must not import lerobot directly."""
    scripts = _example_scripts()
    assert scripts, f"no example script under {_EXAMPLES_DIR}; the sweep graded nothing"
    offenders = [
        f"{script.relative_to(_EXAMPLES_DIR)}: {matches}"
        for script in scripts
        if (matches := _LEROBOT_IMPORT_RE.findall(script.read_text(encoding="utf-8")))
    ]
    assert not offenders, (
        f"example(s) import lerobot directly: {offenders}. "
        f"Examples should use strands_robots.Robot / strands_robots.policies.create_policy "
        f"instead. Move lerobot usage behind the strands_robots abstraction."
    )
