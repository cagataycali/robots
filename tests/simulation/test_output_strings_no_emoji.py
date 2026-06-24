"""Regression: the simulation package must not embed emoji in source strings.

AGENTS.md forbids emoji in user-facing strings (tool-result ``text`` payloads,
log messages, error messages): agents read these strings programmatically, and
emoji are tokenizer noise that render inconsistently across terminals.

These ``content`` ``text`` payloads and ``logger`` calls historically carried
decorative glyphs (floppy disk, brain, dart, chart, etc.) plus their orphan
``U+FE0F`` variation selectors. This test scans the simulation source modules
for any pictograph / dingbat / symbol-emoji codepoint (and stray variation
selectors) so the regression cannot silently reappear.

It deliberately does NOT require pure ASCII: the modules legitimately use math
typography (``+/-``, multiplication sign, arrows) in comments and numeric
output. Only emoji codepoints are rejected.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots.simulation as sim_pkg

# Emoji / pictograph / dingbat / symbol ranges plus variation selectors.
# Intentionally excludes the Mathematical Operators arrows (U+2190-21FF base
# arrows are allowed in comments) but DOES include emoji-presentation arrows
# such as U+25B6 (play) and the U+FE00-FE0F variation selectors that turn a
# plain glyph into an emoji.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # supplemental symbols, pictographs, emoticons
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U00002300-\U000023ff"  # technical (stopwatch, hourglass, etc.)
    "\U00002b00-\U00002bff"  # arrows/stars emoji block
    "\U000025a0-\U000025ff"  # geometric shapes (play/stop emoji bases)
    "\U0000fe00-\U0000fe0f"  # variation selectors (orphan emoji markers)
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "]"
)

_PACKAGE_DIR = Path(sim_pkg.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_simulation_sources_discovered() -> None:
    """Guard: the scan actually walked a non-trivial set of modules."""
    sources = _python_sources()
    assert len(sources) > 5
    names = {p.name for p in sources}
    assert {"policy_runner.py", "base.py", "benchmark.py"} <= names


def test_no_emoji_in_simulation_sources() -> None:
    """No simulation module may embed emoji codepoints or variation selectors."""
    offenders: list[str] = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _EMOJI.finditer(line):
                cp = match.group()
                offenders.append(
                    f"{path.relative_to(_PACKAGE_DIR.parent)}:{lineno}: U+{ord(cp[0]):04X} {line.strip()[:80]!r}"
                )
    assert not offenders, "emoji found in simulation sources:\n" + "\n".join(offenders)
