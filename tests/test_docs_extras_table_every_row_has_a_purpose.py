"""The install page's extras table carries a purpose cell for every extra.

The ``{{extras:table}}`` macro on ``docs/start/install.md`` expands via
``docs/hooks/extras.py``. The hook iterates
``[project.optional-dependencies]`` from ``pyproject.toml`` and renders the
third ("purpose") column from a hand-maintained ``_PURPOSE`` dict with a
``.get(name, '')`` fallback. When an extra is declared in pyproject but
absent from ``_PURPOSE`` the row ships with a blank purpose cell on the
published site: a reader scanning the table sees ``| voice | ... | |`` and
reasonably treats it as a placeholder or deprecated extra.

The sibling ``test_docs_all_extra_membership.py`` grades only the
membership column of the ``[all]`` row, so this drift ships silently.
This test closes the loop: every extra declared in pyproject must have a
non-blank purpose cell in the rendered table.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tests._docs_hooks import docs_hook

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_ROW = re.compile(r"^\|\s*`([a-z0-9][a-z0-9-]*)`\s*\|([^|]*)\|([^|]*)\|\s*$")


def _rendered_table() -> dict[str, str]:
    """Map every rendered row's extra name to its purpose cell (stripped)."""
    hook = docs_hook("extras")
    table = hook.extras_table()
    cells: dict[str, str] = {}
    for line in table.splitlines():
        match = _ROW.match(line)
        if match:
            cells[match.group(1)] = match.group(3).strip()
    return cells


def _declared() -> set[str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return set(data["project"]["optional-dependencies"])


class TestEveryExtraCarriesAPurposeCell:
    """Every extra in pyproject renders with a non-blank purpose on the install page."""

    def test_the_table_renders_a_row_for_every_declared_extra(self) -> None:
        """The hook's iteration order is pyproject's, so every declared extra appears.

        Grades the derivation, not the hand-written map: a reader can only
        read a purpose for an extra the hook actually renders a row for.
        """
        rendered = _rendered_table()
        missing = sorted(_declared() - set(rendered))
        assert not missing, (
            f"docs/hooks/extras.py: the rendered table omits {missing}; the install page "
            f"should name every entry of pyproject's [project.optional-dependencies]."
        )

    def test_no_rendered_row_has_a_blank_purpose_cell(self) -> None:
        """A blank third cell renders as whitespace on the site, so the reader learns nothing.

        The hook's ``_PURPOSE.get(name, '')`` falls through to an empty string
        when an extra is declared in pyproject but absent from the map; this
        rule names the extras currently missing a row so the fix is one map
        entry per extra.
        """
        rendered = _rendered_table()
        blank = sorted(name for name, purpose in rendered.items() if not purpose)
        assert not blank, (
            f"docs/hooks/extras.py: {blank} ship with a blank purpose cell on the install page. "
            f"Add a one-clause row to _PURPOSE for each; the hook's ``.get(name, '')`` fallback "
            f"otherwise emits ``| extra | ... | |`` and a reader treats the extra as a placeholder."
        )


class TestTheGuardIsNotVacuous:
    """The rules above have something to grade: the declared set and the rendered set are real."""

    def test_the_declaration_is_non_empty(self) -> None:
        declared = _declared()
        assert "sim-mujoco" in declared, "pyproject must declare the default sim extra; this guard would be blind without it"
        assert len(declared) > 10, f"only {len(declared)} extras declared; the rules above grade very little"

    def test_a_fabricated_blank_row_would_be_caught(self) -> None:
        """A row the hook renders with an empty third cell is reported by the second rule.

        This grades the rule, not the shipped table: we build a row the hook
        would have emitted for a hypothetical extra not in ``_PURPOSE`` and
        verify the row-level regex classifies the third cell as blank.
        """
        stale_row = "| `imaginary` | `imaginary-pkg` |  |"
        match = _ROW.match(stale_row)
        assert match is not None, "row regex must accept the hook's own output shape, or the second rule is blind"
        assert match.group(3).strip() == "", "a blank third cell must read as empty, or the second rule misses blanks"

    def test_a_populated_row_is_not_reported(self) -> None:
        """A row with a real purpose clause reads as non-blank, so a corrected row is accepted."""
        fine_row = "| `sim-mujoco` | `mujoco`, `robot_descriptions` | MuJoCo simulation, offscreen rendering, IK |"
        match = _ROW.match(fine_row)
        assert match is not None
        assert match.group(3).strip() != "", "a populated purpose must register as non-blank"
