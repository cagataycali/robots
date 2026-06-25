"""Regression: no ``strands_robots`` module may embed emoji in source strings.

AGENTS.md forbids emoji in code, logs, and error messages: agents read these
strings programmatically, emoji are tokenizer noise, and they render
inconsistently (or as mojibake) across terminals and log pipelines. The
prohibition is package-wide, not specific to one subpackage - a stray glyph in
a tool banner, an RPC ``print`` log line, a registry docstring, or an inline
code comment is just as harmful as one in the simulation engine.

This scan walks every Python module under the ``strands_robots`` package and
rejects any pictograph / dingbat / symbol-emoji codepoint (plus orphan
``U+FE00-FE0F`` variation selectors). It deliberately does NOT require pure
ASCII: modules legitimately use math typography (``+/-``, multiplication sign,
base arrows) in comments and numeric output. Only emoji codepoints are
rejected.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

# Emoji / pictograph / dingbat / symbol ranges plus variation selectors.
# Intentionally excludes the Mathematical Operators arrows (U+2190-21FF base
# arrows are allowed in comments) but DOES include emoji-presentation arrows
# such as U+25B6 (play) and the U+FE00-FE0F variation selectors that turn a
# plain glyph into an emoji. The regional-indicator (flag) block
# U+1F1E6-1F1FF is not listed separately: it already falls inside the
# U+1F000-1FAFF range above, and CodeQL flags the duplicate as overlapping.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # supplemental symbols, pictographs, emoticons, flags
    "\U00002600-\U000027bf"  # misc symbols + dingbats (includes U+2713 check mark)
    "\U00002300-\U000023ff"  # technical (stopwatch, hourglass, stop/play, etc.)
    "\U00002b00-\U00002bff"  # arrows/stars emoji block
    "\U000025a0-\U000025ff"  # geometric shapes (play/stop emoji bases)
    "\U0000fe00-\U0000fe0f"  # variation selectors (orphan emoji markers)
    "]"
)

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_package_sources_discovered() -> None:
    """Guard: the scan actually walked the whole package, not one subtree."""
    sources = _python_sources()
    # The package spans many subpackages; a healthy scan sees dozens of modules
    # across simulation, tools, registry, benchmarks, device_connect, mesh, etc.
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"simulation", "tools", "registry", "benchmarks", "device_connect"} <= rel_dirs


def test_no_emoji_in_package_sources() -> None:
    """No ``strands_robots`` module may embed emoji codepoints or variation selectors."""
    offenders: list[str] = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _EMOJI.finditer(line):
                cp = match.group()
                offenders.append(
                    f"{path.relative_to(_PACKAGE_DIR.parent)}:{lineno}: U+{ord(cp[0]):04X} {line.strip()[:80]!r}"
                )
    assert not offenders, "emoji found in strands_robots sources:\n" + "\n".join(offenders)
