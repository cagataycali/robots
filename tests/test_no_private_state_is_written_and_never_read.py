"""Private instance state must have a reader.

An attribute that is assigned and never read is one of two things, and both
cost the reader of the code: dead state that documents an intent the package
abandoned, or a value collected for a purpose that never happens. The second
is the expensive one - ``_record_converge`` below reads an operator-facing
environment variable into an attribute nothing consults, so the knob named in
a runbook silently does nothing, and the only way to find that out is to read
every line of the class.

The invariant here is narrow on purpose: an assignment to ``self._name``
anywhere in :mod:`strands_robots` needs at least one reader. A reader is

* a load of the same attribute anywhere in the package - ``self._name`` or
  ``other._name``, since a mixin's state is routinely read by its host;
* a string literal equal to the name anywhere in the package, which covers
  the ``getattr(self, "_name", None)`` shape the mesh and rendering paths use
  to stay safe on an object built through ``__new__``;
* any mention of the name under ``tests/`` or ``tests_integ/``.

Dunder attributes are excluded (``__dict__`` and friends are not state), and
so is anything in :data:`WRITE_ONLY_BY_DESIGN`, which is a ledger with a
reason per entry rather than a place to park a new offender.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "strands_robots"
TEST_TREES = (REPO_ROOT / "tests", REPO_ROOT / "tests_integ")

# Attributes that are written and never read, and correctly so. One reason per
# entry; an entry without a reason that survives review is a bug in review.
WRITE_ONLY_BY_DESIGN: dict[str, str] = {
    # asyncio drops a task whose only reference is the event loop's weak set,
    # cancelling the Device Connect runtime mid-flight. The attribute IS the
    # strong reference - reading it would not make it any more correct.
    "_background_task": "strong reference that keeps an asyncio task from being garbage collected",
    # Isaac's RTX recording path reads SO101_RECORD_CONVERGE into this and
    # never converges on it, unlike its _idle_converge sibling. Whether the
    # knob should gain a consumer or be dropped needs an RTX device to answer,
    # so it is pinned here rather than guessed at. Tracked in #3578.
    "_record_converge": "tracked in #3578: the RTX convergence knob has no consumer yet",
}


def _package_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _scan() -> tuple[dict[str, list[str]], set[str], int]:
    """Return (write-only attribute -> locations, all attributes seen, file count)."""
    files = _package_files()
    sources = {path: path.read_text(encoding="utf-8") for path in files}
    package_text = "\n".join(sources.values())
    test_text = "\n".join(
        path.read_text(encoding="utf-8") for tree in TEST_TREES if tree.is_dir() for path in tree.rglob("*.py")
    )

    writes: dict[str, list[str]] = defaultdict(list)
    read: set[str] = set()
    seen: set[str] = set()
    for path, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Attribute):
                continue
            name = node.attr
            if not name.startswith("_") or name.startswith("__"):
                continue
            seen.add(name)
            if isinstance(node.ctx, ast.Store):
                writes[name].append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node.ctx, ast.Load):
                read.add(name)

    unread: dict[str, list[str]] = {}
    for name, locations in writes.items():
        if name in read or name in WRITE_ONLY_BY_DESIGN:
            continue
        quoted = re.compile(f"""["']{re.escape(name)}["']""")
        if quoted.search(package_text):
            continue  # reached dynamically via getattr/setattr/hasattr
        if re.search(rf"\b{re.escape(name)}\b", test_text):
            continue
        unread[name] = locations
    return unread, seen, len(files)


def test_every_private_attribute_written_in_the_package_has_a_reader() -> None:
    unread, _, _ = _scan()
    assert not unread, "Private state written and never read:\n" + "\n".join(
        f"  {name}: {', '.join(locations)}" for name, locations in sorted(unread.items())
    )


def test_the_scan_reaches_the_package() -> None:
    """A scan that walked nothing would pass the check above vacuously."""
    _, seen, file_count = _scan()
    assert file_count > 200, f"only {file_count} package files parsed"
    assert len(seen) > 500, f"only {len(seen)} private attributes seen"


def test_every_exemption_states_a_reason() -> None:
    for name, reason in WRITE_ONLY_BY_DESIGN.items():
        assert len(reason.split()) >= 5, f"{name} needs a reason, not a label: {reason!r}"
