"""Mermaid fences carry structure, never colour: the theme paints them (``docs/assets/mermaid.js``).

A fence that writes a hex colour, a ``style`` / ``linkStyle`` / ``classDef`` line or an
``%%{init}%%`` directive would look right in one palette and wrong in the other, and would be
the one diagram on the site outside its tokens. The one colour a fence may ask for is the
accent, on the one node the picture is about: ``:::accent``. More than one is decoration.
The wiring is graded too: the CDN script and the local wrapper in ``mkdocs.yml``, the
superfences custom fence that hands the source to the wrapper, and the wrapper re-running on
Material's ``document$`` and on a palette toggle. The first fences that prove the convention
land with the pages that need a sequence or a state diagram.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_MKDOCS = _REPO / "mkdocs.yml"
_WRAPPER = _DOCS / "assets" / "mermaid.js"

_FENCE = re.compile(r"^```+\s*mermaid[^\n]*\n(.*?)^```+\s*$", re.M | re.S)
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_STYLE_LINE = re.compile(r"^\s*(style|linkStyle|classDef)\b", re.M)
_INIT = re.compile(r"%%\{\s*init")
_ACCENT = re.compile(r":::accent\b")


def _fences() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for page in sorted(_DOCS.rglob("*.md")):
        if "hooks" in page.parts:
            continue
        for match in _FENCE.finditer(page.read_text(encoding="utf-8")):
            found.append((page, match.group(1)))
    return found


def test_no_fence_carries_a_colour_or_a_style_line() -> None:
    offenders = []
    for page, body in _fences():
        rel = page.relative_to(_REPO)
        if hex_colour := _HEX.search(body):
            offenders.append(f"{rel}: hex colour {hex_colour.group(0)}")
        if style_line := _STYLE_LINE.search(body):
            offenders.append(f"{rel}: {style_line.group(1)} line")
        if _INIT.search(body):
            offenders.append(f"{rel}: %%{{init}}%% directive")
    assert offenders == [], (
        f"mermaid fences that paint themselves: {offenders}. The theme in docs/assets/mermaid.js is the only palette; "
        "tag the one node the picture is about with :::accent and leave the rest to it."
    )


def test_at_most_one_accent_per_fence() -> None:
    offenders = [
        f"{page.relative_to(_REPO)}: {len(_ACCENT.findall(body))} :::accent tags"
        for page, body in _fences()
        if len(_ACCENT.findall(body)) > 1
    ]
    assert offenders == [], f"one green per drawing: {offenders}"


def test_no_dash_in_a_fence_label() -> None:
    offenders = [str(page.relative_to(_REPO)) for page, body in _fences() if "\u2013" in body or "\u2014" in body]
    assert offenders == [], f"em or en dash in a mermaid label: {offenders}"


def test_the_wiring_hands_every_fence_to_the_wrapper() -> None:
    yml = _MKDOCS.read_text(encoding="utf-8")
    assert re.search(r"mermaid@10/dist/mermaid\.min\.js", yml), (
        "mkdocs.yml extra_javascript lacks the mermaid@10 CDN script"
    )
    assert "assets/mermaid.js" in yml, "mkdocs.yml extra_javascript lacks docs/assets/mermaid.js"
    fence = re.search(r"custom_fences:\s*\n\s*-\s*name:\s*mermaid\s*\n\s*class:\s*(\S+)\s*\n\s*format:\s*(\S+)", yml)
    assert fence, "pymdownx.superfences declares no mermaid custom fence"
    holder_class, fmt = fence.group(1), fence.group(2)
    assert fmt.endswith("fence_code_format"), "the fence must keep its source as code for the wrapper to render"
    js = _WRAPPER.read_text(encoding="utf-8")
    assert f"pre.{holder_class}" in js, f"docs/assets/mermaid.js does not look for pre.{holder_class}"
    assert "document$" in js and "data-md-color-scheme" in js, (
        "the wrapper must re-render on document$ and on a palette toggle"
    )
    for token in ("--sr-mono", "--sr-fg", "--sr-muted", "--sr-accent", "--md-default-bg-color"):
        assert token in js, f"the wrapper reads its palette from the page; {token} is missing"
    assert "prefers-reduced-motion" in js or "animate" not in js.lower(), (
        "a wrapper that animates must read prefers-reduced-motion"
    )
