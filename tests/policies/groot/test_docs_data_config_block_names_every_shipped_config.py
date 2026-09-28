"""The GR00T page's data configs section must name every shipped config.

``docs/learn/policies/groot.md`` carries a ``## Data configs`` section whose
"Shipped names:" sentence is the reader's catalog of ``data_config=`` values,
each as a code span. ``strands_robots/policies/groot/data_configs.json`` is the
vocabulary owner, so the section is graded against it rather than against a
copied list: a config shipped there and absent from the page is a value the
reader cannot discover, and a name on the page that the JSON does not ship
sends the reader to a config that does not resolve. Aliases are graded the
same way against the JSON's ``aliases`` map.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PAGE = _REPO_ROOT / "docs" / "learn" / "policies" / "groot.md"
_DATA_CONFIGS = _REPO_ROOT / "strands_robots" / "policies" / "groot" / "data_configs.json"
_HEADING = "## Data configs"


def _section() -> str:
    """The data configs section, up to the next level-2 heading."""
    text = _PAGE.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(_HEADING)}\s*\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"docs/learn/policies/groot.md has no {_HEADING!r} section"
    return match.group(1)


def _names_after(label: str) -> list[str]:
    """Code spans in the sentence that starts with ``label`` (``Shipped names:``, ``Aliases:``)."""
    match = re.search(rf"{re.escape(label)}\s*(.*?)\.(?:\s|$)", _section(), flags=re.DOTALL)
    assert match is not None, f"the {_HEADING!r} section has no {label!r} sentence"
    return re.findall(r"`([^`]+)`", match.group(1))


def _shipped() -> dict:
    return json.loads(_DATA_CONFIGS.read_text(encoding="utf-8"))


def test_every_shipped_data_config_is_named_on_the_page() -> None:
    configs = sorted(_shipped()["configs"])
    named = _names_after("Shipped names:")
    missing = [config for config in configs if config not in named]
    assert missing == [], f"data_configs.json ships {missing} but the groot.md Data configs section does not name them"


def test_every_name_on_the_page_is_a_shipped_data_config() -> None:
    configs = set(_shipped()["configs"])
    unknown = [name for name in _names_after("Shipped names:") if name not in configs]
    assert unknown == [], f"groot.md names data configs that data_configs.json does not ship: {unknown}"


def test_the_aliases_on_the_page_are_the_shipped_aliases() -> None:
    aliases = set(_shipped().get("aliases", {}))
    named = set(_names_after("Aliases:"))
    assert named == aliases, f"groot.md aliases {sorted(named)} differ from data_configs.json {sorted(aliases)}"


def test_the_scan_reads_a_real_catalog() -> None:
    """Non-vacuity: the JSON ships many configs and the page names many."""
    assert len(_shipped()["configs"]) >= 20
    assert len(_names_after("Shipped names:")) >= 20
