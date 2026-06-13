"""Verify that top-level examples use only strands_robots imports for robot/policy.

The examples/ directory is the first thing new users see. If an example
imports from lerobot directly (for robot construction, cameras, or policy
creation), it undermines the abstraction and teaches users to bypass
strands_robots.

The examples/lerobot/ subdirectory is explicitly excluded — it exists to
show hub-to-hardware integration and is allowed to use lerobot directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Subdirectories that are allowed to use raw lerobot imports
ALLOWED_SUBDIRS = {"lerobot"}


def _top_level_example_files() -> list[Path]:
    """Collect .py files in examples/ excluding allowed subdirs."""
    if not EXAMPLES_DIR.exists():
        return []
    files = []
    for p in EXAMPLES_DIR.rglob("*.py"):
        # Skip files in allowed subdirectories
        relative = p.relative_to(EXAMPLES_DIR)
        if relative.parts and relative.parts[0] in ALLOWED_SUBDIRS:
            continue
        files.append(p)
    return files


_LEROBOT_IMPORT_RE = re.compile(r"^\s*(from\s+lerobot\b|import\s+lerobot\b)", re.MULTILINE)


@pytest.mark.parametrize(
    "example_file",
    _top_level_example_files(),
    ids=lambda p: str(p.relative_to(EXAMPLES_DIR)),
)
def test_no_raw_lerobot_imports(example_file: Path):
    """Top-level examples must not import from lerobot directly.

    Use strands_robots.Robot() for robot construction and
    strands_robots.create_policy() for policy loading instead.
    """
    content = example_file.read_text(encoding="utf-8")
    matches = _LEROBOT_IMPORT_RE.findall(content)
    assert not matches, (
        f"{example_file.name} imports lerobot directly: {matches}. "
        f"Top-level examples should use only strands_robots imports. "
        f"Move lerobot-specific examples to examples/lerobot/ if needed."
    )
