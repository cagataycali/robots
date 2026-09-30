"""A configuration table must not print a ``STRANDS_MESH`` default that opts in.

Mesh is opt-in: ``strands_robots.robot._mesh_env_opt_in`` turns it on only for
``true``/``1``/``yes``, so a bare ``Robot()`` is quiet and never spins up
Zenoh, ACL or e-stop machinery. ``tests/mesh/test_mesh_wiring.py`` pins that
behaviour.

A table that prints an opt-in spelling in its Default column therefore tells a
reader mesh is already running, and hides the one spelling that actually
enables it - the reader's only documented knob ("set it to ``false``") is a
no-op for the state they are actually in.

These tests grade the shipped tables against the resolver itself rather than
against a hand-copied list of spellings, so widening the accepted spellings
later cannot silently invalidate the guard.

Two table shapes ship. The hand-written switch tables (``docs/learn/mesh/index.md``)
carry a ``values`` column that marks the default inline (``unset (default)``), and
``docs/reference/configuration.md`` renders one generated row per variable through
``docs/hooks/env_vars.py`` with a ``default`` column of its own. Both are read as
the reader sees them: the default cell is found by its header (or by the inline
``(default)`` marker), and the configuration page's one-sentence boolean rule
("A boolean variable accepts ``1``, ``true``, ``yes``") counts as that page's
opt-in spelling, since it sits directly above the table.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.robot import _mesh_env_opt_in
from tests._docs_hooks import docs_hook

_ENV = "STRANDS_MESH"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_REFERENCE = _REPO_ROOT / "docs" / "reference" / "configuration.md"
#: The configuration page states the accepted boolean spellings once, above the table.
_BOOLEAN_RULE = re.compile(r"boolean variable accepts ((?:`[A-Za-z0-9]+`,\s*)*(?:and\s*)?`[A-Za-z0-9]+`)")
#: A value marked as the default inline: ``unset (default)``.
_INLINE_DEFAULT = re.compile(r"([^,;]+?)\s*\(default\)")

# The guard is only meaningful while it still reaches the shipped tables. If a
# rename or a reformat drops them all, fail loudly instead of reporting clean.
_MINIMUM_DOCUMENTED_ROWS = 2


def _rendered_configuration_page() -> str:
    """The configuration page with ``{{env_vars}}`` expanded by the shipped hook, ``<code>`` folded to backticks."""
    module = docs_hook("env_vars")
    source = _CONFIG_REFERENCE.read_text(encoding="utf-8")
    rendered = module.on_page_markdown(source, page=None, config=None, files=None)
    assert rendered != source, "configuration.md carries no {{env_vars}} token for the hook to expand"
    return rendered.replace("<code>", "`").replace("</code>", "`")


def _page_texts() -> list[tuple[str, str]]:
    """Every user-facing page as the reader sees it, keyed by repository-relative path."""
    texts: list[tuple[str, str]] = []
    for doc in [_REPO_ROOT / "README.md", *sorted((_REPO_ROOT / "docs").rglob("*.md"))]:
        if not doc.exists():
            continue
        relative = str(doc.relative_to(_REPO_ROOT))
        text = _rendered_configuration_page() if doc == _CONFIG_REFERENCE else doc.read_text(encoding="utf-8")
        texts.append((relative, text))
    return texts


def _cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _default_cell(header: list[str] | None, cells: list[str]) -> str:
    """The row's default: the ``default`` column when the table has one, else the ``(default)``-marked value.

    Falls back to the last cell, the shape the older tables used, when neither is present.
    """
    if header is not None:
        lowered = [cell.lower() for cell in header]
        if "default" in lowered:
            return cells[lowered.index("default")]
    marked = [match.group(1).strip() for cell in cells[1:] for match in _INLINE_DEFAULT.finditer(cell)]
    if marked:
        return ", ".join(marked)
    return cells[-1]


def _documented_rows() -> list[tuple[str, int, str, str]]:
    """Return every markdown table row whose first cell is exactly ``STRANDS_MESH``.

    Returns:
        A list of ``(relative_path, line_number, description_cell, default_cell)``
        tuples. ``description_cell`` is every cell of the row except the default,
        joined, plus the page's boolean rule sentence when it states one. Rows
        that bundle several variables into one cell are skipped: this guard grades
        the single-variable rows a reader copies a value from.
    """
    rows: list[tuple[str, int, str, str]] = []
    for relative, text in _page_texts():
        rule = _BOOLEAN_RULE.search(text)
        page_spellings = rule.group(1) if rule else ""
        header: list[str] | None = None
        for lineno, line in enumerate(text.splitlines(), 1):
            cells = _cells(line)
            if cells is None:
                header = None
                continue
            if header is None:
                header = cells
                continue
            if len(cells) < 3 or cells[0] != f"`{_ENV}`":
                continue
            default = _default_cell(header, cells)
            description = " ".join(cell for cell in cells[1:] if cell != default)
            rows.append((relative, lineno, f"{description} {page_spellings}".strip(), default))
    return rows


class TestTheResolverOnlyEverOptsIn:
    """The oracle the documented defaults are graded against."""

    @pytest.mark.parametrize("raw", ["true", "TRUE", "True", "1", "yes", " yes "])
    def test_an_opt_in_spelling_turns_mesh_on(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert _mesh_env_opt_in() is True

    @pytest.mark.parametrize("raw", ["", " ", "false", "0", "no", "off", "on", "enabled", "maybe"])
    def test_every_other_value_leaves_mesh_off(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert _mesh_env_opt_in() is False

    def test_an_unset_env_leaves_mesh_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        assert _mesh_env_opt_in() is False


class TestTheDocumentedDefaultLeavesMeshOff:
    """Grade every shipped configuration table against the resolver."""

    def test_the_guard_still_reaches_the_shipped_tables(self) -> None:
        rows = _documented_rows()
        assert len(rows) >= _MINIMUM_DOCUMENTED_ROWS, (
            f"expected at least {_MINIMUM_DOCUMENTED_ROWS} configuration rows keyed to "
            f"{_ENV}, found {len(rows)}: {rows}. The extractor no longer reaches the "
            "shipped tables, so a clean result below would prove nothing."
        )

    def test_no_table_prints_an_opt_in_spelling_as_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        offenders: list[str] = []
        for path, lineno, _description, default in _documented_rows():
            for token in re.findall(r"[A-Za-z0-9]+", default):
                monkeypatch.setenv(_ENV, token)
                if _mesh_env_opt_in():
                    offenders.append(
                        f"{path}:{lineno} prints {token!r} as the default for {_ENV}, but "
                        f"{_ENV}={token} opts IN. A bare Robot() has mesh OFF, so this row "
                        "tells a reader mesh is already running and hides the opt-in."
                    )
        assert not offenders, "documented default contradicts the resolver:\n" + "\n".join(offenders)

    def test_every_table_still_names_a_spelling_that_opts_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Correcting the default must not remove the only way to turn mesh on."""
        silent: list[str] = []
        for path, lineno, description, default in _documented_rows():
            opts_in = False
            for token in re.findall(r"`([^`]+)`", f"{description} {default}"):
                monkeypatch.setenv(_ENV, token)
                opts_in = opts_in or _mesh_env_opt_in()
            if not opts_in:
                silent.append(
                    f"{path}:{lineno} documents {_ENV} without naming any value that "
                    "turns mesh on, so the opt-in is undiscoverable from this row."
                )
        assert not silent, "\n".join(silent)
