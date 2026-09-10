"""mkdocs hook: numbers the docs never type by hand.

Every ``{{n:key}}`` token in a page is replaced at build time with a value read
from the tree, so a count can no longer drift from the code it describes
(index.md said 68 robots, robots/index.md said 73, the registry held 73).

Keys (all derived, none configured):

* ``robots``            entries in ``strands_robots/registry/robots.json``
* ``robots_decade``     that number rounded down to a ten, for "70+" prose
* ``categories``        distinct ``category`` values in the registry
* ``<category>``        robots in that category, e.g. ``{{n:arm}}``,
                        ``{{n:humanoid}}``, ``{{n:mobile_manip}}``
* ``hardware``          robots with a ``hardware`` block (drivable for real)
* ``aliases``           alias strings across the registry
* ``tools``             ``@tool`` decorators under ``strands_robots/``
* ``sim_backends``      simulation backend packages (mujoco, newton, isaac)
* ``policy_providers``  packages under ``strands_robots/policies/``

An unknown key fails the build (``mkdocs build --strict`` treats the hook's
``log.warning`` as an error), so a typo cannot ship as literal ``{{n:...}}``.
Reads only the filesystem: no ``strands_robots`` import, so the hook works in
the docs venv without the package's optional extras.
"""

from __future__ import annotations

import collections
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.numbers")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_TOKEN = re.compile(r"\{\{\s*n:([a-z_]+)\s*\}\}")
_SIM_NON_BACKENDS = frozenset({"task_objects"})


@lru_cache(maxsize=1)
def numbers() -> dict[str, int]:
    """Derive every number once per build."""
    robots = json.loads((_PKG / "registry" / "robots.json").read_text(encoding="utf-8"))["robots"]
    categories = collections.Counter(spec["category"] for spec in robots.values())
    tools = sum(len(re.findall(r"^\s*@tool\b", path.read_text(encoding="utf-8"), re.M)) for path in _PKG.rglob("*.py"))
    sim_backends = sorted(
        p.name
        for p in (_PKG / "simulation").iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name not in _SIM_NON_BACKENDS
    )
    providers = sorted(p.name for p in (_PKG / "policies").iterdir() if p.is_dir() and not p.name.startswith("_"))
    out: dict[str, int] = {
        "robots": len(robots),
        "robots_decade": len(robots) // 10 * 10,
        "categories": len(categories),
        "hardware": sum(1 for spec in robots.values() if spec.get("hardware")),
        "aliases": sum(len(spec.get("aliases", ())) for spec in robots.values()),
        "tools": tools,
        "sim_backends": len(sim_backends),
        "policy_providers": len(providers),
    }
    for category, count in categories.items():
        out[category] = count
    return out


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Replace every ``{{n:key}}`` in ``markdown``; warn on an unknown key."""
    values = numbers()

    def _one(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            log.warning("%s: unknown numbers key {{n:%s}} (known: %s)", page_path, key, ", ".join(sorted(values)))
            return match.group(0)
        return str(values[key])

    return _TOKEN.sub(_one, markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    return substitute(markdown, page.file.src_path)
