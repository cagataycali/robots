"""Regression: no graded surface may embed an emoji or a Unicode dash.

AGENTS.md forbids emoji in code, logs and error messages, and mandates plain
ASCII punctuation in the strings a reader receives. Both glyph classes fail the
same way: agents read these strings programmatically, so the glyphs are
tokenizer noise, and they render inconsistently (or as mojibake) across
terminals and log pipelines. An ASCII hyphen ``-`` carries the em dash's meaning
everywhere.

One rule table, one surface table, one scan. The two rules were two modules
scanning two different sets of surfaces, and the asymmetry was the defect rather
than the duplication: the emoji rule graded the package, the test tree and
``changelog.d/*.md``, while the dash rule graded the package alone. Measured on
``2a4707a``, with the package clean of both: 28 em dashes in 9 test modules, and
127 in 42 changelog fragments. A fragment is the surface that matters most,
because ``scripts/assemble_changelog.py --apply`` folds its body *verbatim* into
``CHANGELOG.md`` and then ``unlink``s it - so every one of those 127 would have
landed in the released notes on the next release, behind a fully green run, on a
rule the package had already been swept for.

A surface is a row, so a rule cannot read one set of files and its sibling
another; adding a rule grades every surface, and adding a surface grades every
rule. The emoji rule deliberately does NOT require pure ASCII: modules
legitimately use math typography (``+/-``, the multiplication sign, base arrows)
in comments and numeric output, and :mod:`tests.test_log_strings_are_ascii` is
the narrower guard that holds the *log* strings to ASCII.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

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

# Figure dash, en dash, em dash, horizontal bar. ASCII hyphen-minus (U+002D) is
# the sanctioned replacement and is, of course, allowed.
_UNICODE_DASH = re.compile("[\u2012-\u2015]")

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent
_TESTS_DIR = Path(__file__).resolve().parent
_FRAGMENT_DIR = Path(__file__).resolve().parents[1] / "changelog.d"


def _python_sources() -> list[Path]:
    """Every module of the shipped package."""
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _test_sources() -> list[Path]:
    """Every module of the test tree.

    A half-applied glyph sweep is easy to spot in production code but slips
    through here: prod stops emitting a glyph, yet an assertion (or a debug
    ``print``) still pins it.
    """
    return sorted(p for p in _TESTS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _fragment_sources() -> list[Path]:
    """Every changelog news fragment, whose body is folded verbatim into the log.

    Every ``*.md`` in the directory is scanned, with no reserved-name exemption.
    ``README.md`` documents the fragment convention and is the file a
    contributor copies a skeleton out of, so holding it to the same bar is the
    point rather than an oversight.
    """
    return sorted(_FRAGMENT_DIR.glob("*.md"))


@dataclass(frozen=True)
class _Rule:
    """A forbidden glyph class and the remedy its failure names."""

    label: str
    pattern: re.Pattern[str]
    remedy: str


@dataclass(frozen=True)
class _Surface:
    """A set of files graded by every rule.

    :param label: How the surface is named in a failure.
    :param sources: The files to read.
    :param root: The directory failures report paths relative to the parent of.
    :param required_subdirs: Subdirectories a healthy walk must reach, or
        ``None`` when a file count is not a usable guard - see
        :func:`test_the_fragment_directory_still_resolves`.
    """

    label: str
    sources: Callable[[], list[Path]]
    root: Path
    required_subdirs: frozenset[str] | None


_RULES = (
    _Rule("emoji", _EMOJI, "drop the glyph"),
    _Rule("unicode dash", _UNICODE_DASH, "use the ASCII hyphen '-'"),
)

_SURFACES = (
    _Surface(
        "package",
        _python_sources,
        _PACKAGE_DIR,
        frozenset({"simulation", "tools", "registry", "drivers", "device_connect"}),
    ),
    _Surface("tests", _test_sources, _TESTS_DIR, frozenset({"simulation", "policies", "drivers"})),
    _Surface("changelog fragments", _fragment_sources, _FRAGMENT_DIR, None),
)

_WALKED_SURFACES = [pytest.param(s, id=s.label.replace(" ", "-")) for s in _SURFACES if s.required_subdirs]
_ALL_SURFACES = [pytest.param(s, id=s.label.replace(" ", "-")) for s in _SURFACES]
_ALL_RULES = [pytest.param(r, id=r.label.replace(" ", "-")) for r in _RULES]


@pytest.mark.parametrize("surface", _WALKED_SURFACES)
def test_the_walked_surface_reaches_its_whole_tree(surface: _Surface) -> None:
    """Guard: a walk must reach the whole tree, not one subtree of it."""
    sources = surface.sources()
    assert len(sources) > 50, f"{surface.label}: walked {len(sources)} files"
    reached = {p.relative_to(surface.root).parts[0] for p in sources if p.parent != surface.root}
    assert surface.required_subdirs is not None
    assert surface.required_subdirs <= reached, f"{surface.label}: reached {sorted(reached)}"


def test_the_fragment_directory_still_resolves() -> None:
    """Guard: the scan points at the fragment directory rather than at nothing.

    Deliberately *not* a file-count assertion, unlike the walked surfaces.
    ``assemble_changelog.py --apply`` ``unlink``s every fragment it consumes, so
    an empty ``changelog.d/`` is the legitimate state of the tree immediately
    after a release, and a ``len(...) > N`` guard would turn ``main`` red on the
    release commit itself. The failure actually worth catching is the scan
    walking a path the fragments have moved out of; the convention doc lives
    alongside them, so its presence is what proves the path still resolves.
    """
    assert _FRAGMENT_DIR.is_dir(), f"fragment directory is missing: {_FRAGMENT_DIR}"
    assert (_FRAGMENT_DIR / "README.md").is_file(), (
        f"{_FRAGMENT_DIR / 'README.md'} is absent - the fragment directory was moved or renamed "
        "and this scan is now walking the wrong path"
    )


@pytest.mark.parametrize("rule", _ALL_RULES)
@pytest.mark.parametrize("surface", _ALL_SURFACES)
def test_no_file_on_the_surface_carries_the_glyph(surface: _Surface, rule: _Rule) -> None:
    """Every rule reads every surface: neither table may skip a cell."""
    offenders: list[str] = []
    for path in surface.sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in rule.pattern.finditer(line):
                codepoint = ord(match.group()[0])
                offenders.append(
                    f"{path.relative_to(surface.root.parent)}:{lineno}: U+{codepoint:04X} {line.strip()[:80]!r}"
                )
    assert not offenders, f"{rule.label} found on the {surface.label} surface ({rule.remedy}):\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    ("rule", "rejected", "allowed"),
    [
        # A pictograph, and the math typography the emoji rule leaves alone.
        pytest.param(_RULES[0], "recording \U0001f6a8 stopped", "torque +/- 0.5 -> 1.0 (2x)", id="emoji"),
        # An em dash, and the ASCII hyphen that replaces it.
        pytest.param(_RULES[1], "a refusal \u2014 not a warning", "a refusal - not a warning", id="unicode-dash"),
    ],
)
def test_the_pattern_reads_the_glyph_and_not_its_ascii_replacement(rule: _Rule, rejected: str, allowed: str) -> None:
    """Non-vacuity: each pattern flags a planted glyph and clears the remedy."""
    assert rule.pattern.search(rejected), f"{rule.label} pattern missed {rejected!r}"
    assert not rule.pattern.search(allowed), f"{rule.label} pattern flagged {allowed!r}"
