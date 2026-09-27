"""mkdocs hook: the reference lane indexes itself.

The nav carries the pages a reader browses - Start, Robots - and one pointer at
this lane. The ninety-odd reference pages are reached from
``reference/index.md`` instead, which means that page has to name every one of
them: a page nothing links to is reachable only by search, and a hand-written
list of ninety links is a second place to forget a page.

So the list is generated. ``{{reference_index}}`` is replaced by every page
under ``docs/reference/`` grouped by its directory, each entry carrying the
page's own ``# `` title, so adding a page under that tree publishes it with no
edit to the index. ``tests/test_docs_two_lane_architecture.py`` grades the
rendered list against the tree.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(f"mkdocs.hooks.{__name__}")

#: The directory this hook indexes, relative to the docs root.
LANE = "reference"

#: The heading each directory is published under. A directory absent here is
#: a directory nobody named yet - the hook refuses rather than guessing, so a
#: new subtree is announced on purpose.
GROUPS: dict[str, str] = {
    "": "The library",
    "simulation": "Simulation",
    "hardware": "Real hardware",
    "policies": "Policies",
    "ros2": "ROS 2",
    "data": "Data",
    "training": "Training",
    "inference": "Inference",
    "security": "Security",
    "examples": "Examples",
}

_TOKEN = re.compile(r"^\s*\{\{\s*reference_index\s*\}\}\s*$", re.MULTILINE)
_TITLE = re.compile(r"^# (?P<title>.+)$", re.MULTILINE)


def _docs_dir() -> Path:
    """The ``docs/`` directory, resolved from this file rather than the cwd."""
    return Path(__file__).resolve().parent.parent


def title_of(page: Path) -> str:
    """The page's own first-level heading, or its stem when it carries none."""
    match = _TITLE.search(page.read_text(encoding="utf-8"))
    return match["title"].strip() if match else page.stem.replace("-", " ")


def pages() -> list[Path]:
    """Every reference page except the index itself, in path order."""
    lane = _docs_dir() / LANE
    return sorted(p for p in lane.rglob("*.md") if p.name != "index.md")


def render() -> str:
    """The generated block: one section per directory, one row per page."""
    lane = _docs_dir() / LANE
    grouped: dict[str, list[Path]] = {}
    for page in pages():
        group = page.parent.relative_to(lane).as_posix()
        group = "" if group == "." else group
        if group not in GROUPS:
            raise KeyError(
                f"docs/{LANE}/{group}/ has no heading in docs/hooks/reference_index.py "
                f"GROUPS; name the group there so its pages are published."
            )
        grouped.setdefault(group, []).append(page)
    lines: list[str] = []
    for group, heading in GROUPS.items():
        if group not in grouped:
            continue
        lines.append(f"## {heading}\n")
        for page in grouped[group]:
            href = page.relative_to(lane).as_posix()
            lines.append(f"- [{title_of(page)}]({href})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Replace a ``{{reference_index}}`` line with the generated index."""
    if not _TOKEN.search(markdown):
        return markdown
    log.debug("%s: indexing %d reference pages", page_path, len(pages()))
    return _TOKEN.sub(lambda _: render(), markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """Render the index into one page, the hook entry point mkdocs calls."""
    return substitute(markdown, page.file.src_path)
