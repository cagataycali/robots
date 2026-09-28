"""Docs hygiene: every ``Robot(...)`` on the bimanual robot pages constructs.

The bimanual family ships as ``docs/robots/bimanual/index.md`` (cards and a
table, no fence of its own) plus one generated ``docs/robots/<name>.md`` per
robot, and each robot page opens with the reader's first ``Robot()`` call for
that two-arm rig. Every such line must be constructible on a clean install or
say what stands in the way. The registry is the oracle:

* a sim line (no ``mode="real"``) must name an entry with an ``asset`` block:
  ``bi_openarm`` declares hardware only, so ``Robot("bi_openarm")`` refuses
  with "registered for real hardware only";
* an asset with ``auto_download: false`` is never fetched, so the fence must
  name the ``<dir>/<model_xml>`` the reader places by hand: ``trossen_wxai``
  listed as a plain sim line refuses with "model file is not on disk";
* a ``mode="real"`` line must name an entry with a ``hardware`` block.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = REPO_ROOT / "docs" / "robots"
FAMILY_PAGE = ROBOTS_DIR / "bimanual" / "index.md"
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"

_FENCE_RE = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)


def _registry() -> dict[str, dict]:
    data = json.loads(ROBOTS_JSON.read_text())
    return data.get("robots", data)


def _bimanual_pages() -> list[Path]:
    """The family index plus the generated page of every bimanual robot."""
    names = sorted(n for n, e in _registry().items() if e.get("category") == "bimanual")
    return [FAMILY_PAGE, *(ROBOTS_DIR / f"{n}.md" for n in names)]


def _robot_calls() -> list[tuple[str, str, dict[str, ast.expr], str]]:
    """Every ``Robot("<name>", ...)`` call as (page, name, keywords, fence source)."""
    calls: list[tuple[str, str, dict[str, ast.expr], str]] = []
    for page in _bimanual_pages():
        if not page.exists():
            continue
        for fence in _FENCE_RE.findall(page.read_text()):
            for node in ast.walk(ast.parse(fence)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Robot"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    keywords = {k.arg: k.value for k in node.keywords if k.arg}
                    calls.append((page.name, node.args[0].value, keywords, fence))
    return calls


def _is_real(keywords: dict[str, ast.expr]) -> bool:
    mode = keywords.get("mode")
    return isinstance(mode, ast.Constant) and mode.value == "real"


def test_every_bimanual_robot_has_a_generated_page() -> None:
    missing = [p.name for p in _bimanual_pages() if not p.exists()]
    assert not missing, f"bimanual pages the registry promises but the tree lacks: {missing}"


def test_the_pages_name_at_least_one_sim_and_one_hardware_rig() -> None:
    calls = _robot_calls()
    assert any(not _is_real(kw) for _, _, kw, _ in calls)
    assert any(_is_real(kw) for _, _, kw, _ in calls)


@pytest.mark.parametrize(
    ("page", "name", "keywords", "fence"),
    _robot_calls(),
    ids=lambda v: v if isinstance(v, str) and not v.endswith(".md") else "",
)
def test_every_robot_call_names_a_route_the_registry_ships(
    page: str, name: str, keywords: dict[str, ast.expr], fence: str
) -> None:
    entry = _registry()[name]
    if _is_real(keywords):
        assert "hardware" in entry, f"{page}: Robot({name!r}, mode='real') names an entry with no hardware route"
        return
    asset = entry.get("asset")
    assert asset, f"{page}: Robot({name!r}) is written as a sim line but the entry declares no asset (hardware only)"
    if asset.get("auto_download", True) is False:
        placement = f"{asset['dir']}/{asset['model_xml']}"
        assert placement in fence, (
            f"{page}: Robot({name!r}) has auto_download=false, so the fence must tell the reader to place {placement}"
        )
