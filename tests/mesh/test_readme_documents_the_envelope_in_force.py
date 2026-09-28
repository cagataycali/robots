"""README's teleop envelope rows describe the bound the module enforces.

The mesh input envelope is documented in the configuration table, which the
tree calls the single source of truth for operators. It began life in README;
in the new docs tree it is ``docs/reference/configuration.md``, rendered from
the package's own reads by ``docs/hooks/env_vars.py``. Both of its rows
carried the retired radian defaults - ``12.566`` "(4pi)" and ``25.133``
"(8pi)" - after the bounds themselves were converted to frame units, and the
value row still named the unit as ``(radians)``. An operator sizing a fleet
from the table would therefore compute against a bound two orders of
magnitude away from the one in force, and pick the wrong direction to tune:
under a frame-unit default a smaller unit is *narrowed* to, not widened from.

Nothing graded the pair, so the drift was invisible to every check on the
pull request that introduced it (#2598). These tests read the numbers out of
the table and compare them to the constants themselves, so a later retune
cannot leave the documentation behind.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

from strands_robots.mesh import security

_ROOT = pathlib.Path(__file__).resolve().parents[2]
README = _ROOT / "docs" / "reference" / "configuration.md"  # env-var matrix
HOOK = _ROOT / "docs" / "hooks" / "env_vars.py"

#: Env var -> the module constant its documented default must agree with.
DOCUMENTED_DEFAULTS = {
    "STRANDS_MESH_INPUT_VALUE_ABS": security.DEFAULT_INPUT_VALUE_ABS,
    "STRANDS_MESH_INPUT_SLEW_ABS": security.DEFAULT_INPUT_SLEW_ABS,
}


def _matrix_lines() -> list[str]:
    """The configuration page with ``{{env_vars}}`` expanded by the shipped hook.

    The hook writes variables as ``<code>VAR</code>``; the text is folded to the
    backticks a hand-written row uses so the row rules read either spelling.
    """
    spec = importlib.util.spec_from_file_location("docs_hooks_env_vars", HOOK)
    assert spec is not None and spec.loader is not None
    module = sys.modules.get(spec.name) or importlib.util.module_from_spec(spec)
    if spec.name not in sys.modules:
        sys.modules[spec.name] = module  # dataclasses in the hook resolve their module here
        spec.loader.exec_module(module)
    source = README.read_text(encoding="utf-8")
    rendered = module.on_page_markdown(source, page=None, config=None, files=None)
    assert rendered != source, "configuration.md carries no {{env_vars}} token for the hook to expand"
    return rendered.replace("<code>", "`").replace("</code>", "`").splitlines()


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _row(var: str) -> str:
    """Return the single README table row documenting ``var``."""
    rows = [line for line in _matrix_lines() if line.lstrip().startswith("|") and f"`{var}`" == _cells(line)[0]]
    assert len(rows) == 1, f"expected exactly one README table row for {var}, found {len(rows)}"
    return rows[0]


def _default_cell(var: str) -> str:
    """The cell under the table's ``default`` header on ``var``'s row.

    The hand-written matrix kept the default last; the generated one puts it
    before the meaning column. The header row above the row decides.
    """
    lines = _matrix_lines()
    row = _row(var)
    index = lines.index(row)
    separator = next((i for i in range(index - 1, -1, -1) if re.match(r"^\s*\|[\s:|-]*---", lines[i])), None)
    header = _cells(lines[separator - 1]) if separator else []
    lowered = [cell.lower() for cell in header]
    column = lowered.index("default") if "default" in lowered else -1
    return _cells(row)[column]


def _documented_default(var: str) -> float:
    """Return the numeric default the README row states for ``var``."""
    default_cell = _default_cell(var)
    quoted = re.findall(r"`([^`]+)`", default_cell)
    assert quoted, f"README states no value for {var}'s default: {default_cell!r}"
    return float(quoted[0].replace(",", "").replace("_", ""))


@pytest.mark.parametrize(("var", "constant"), sorted(DOCUMENTED_DEFAULTS.items()))
def test_readme_documents_the_default_in_force(var: str, constant: float) -> None:
    documented = _documented_default(var)
    # The table rounds deliberately - it wrote `25.133` for 8pi - so compare at
    # the precision the table offers rather than demanding the full repr.
    assert documented == pytest.approx(constant, rel=1e-3), (
        f"README documents {var} as {documented} but the module enforces {constant}"
    )


@pytest.mark.parametrize("var", sorted(DOCUMENTED_DEFAULTS))
def test_readme_names_the_unit_the_frames_carry(var: str) -> None:
    """The row states the unit, because the number alone does not imply it.

    ``720`` is a plausible bound in degrees and an implausible one in radians,
    so a row that gives the magnitude without the unit leaves the operator to
    guess which of the two the validator compares against.
    """
    row = _row(var).lower()
    assert "frame unit" in row, f"README's {var} row does not name the unit the frames carry: {row!r}"
