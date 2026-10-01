#!/usr/bin/env python3
"""Render every ``mermaid`` fence under docs/ headless and fail on one that does not draw.

``tests/test_docs_mermaid_fences.py`` grades what a fence may contain; this script grades that
it renders: Mermaid 10 in a Playwright Chromium, the same build the site loads, with the site's
own wrapper conventions (an appended ``classDef accent`` for flowcharts and state diagrams). A
fence with a syntax error, or one that produces an empty SVG, is a red row. Run it next to
``check_fences.py`` at the end of a docs change::

    python3 docs/hooks/check_mermaid.py            # every fence
    python3 docs/hooks/check_mermaid.py learn/mesh/safety-and-estop.md

Needs ``playwright`` with chromium installed (``python -m playwright install chromium``) and the
network once, for the CDN script mkdocs.yml names.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
MKDOCS = DOCS.parent / "mkdocs.yml"
_FENCE = re.compile(r"^```+\s*mermaid[^\n]*\n(.*?)^```+\s*$", re.M | re.S)
ACCENT_CLASS = "\nclassDef accent fill:rgba(2,164,53,0.12),stroke:#007a3d,color:#007a3d,stroke-width:1.5px\n"


def cdn_script() -> str:
    """The mermaid@10 script mkdocs.yml loads, so the check renders with the site's own build."""
    match = re.search(r"(https://\S+mermaid@10/dist/mermaid\.min\.js)", MKDOCS.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("mkdocs.yml names no mermaid@10 CDN script")
    return match.group(1)


def fences(only: list[str]) -> list[tuple[str, int, str]]:
    """Every mermaid fence as (page, index, source), over *only* or the whole docs/ tree."""
    rows: list[tuple[str, int, str]] = []
    pages = [DOCS / p for p in only] if only else sorted(p for p in DOCS.rglob("*.md") if "hooks" not in p.parts)
    for page in pages:
        for index, match in enumerate(_FENCE.finditer(page.read_text(encoding="utf-8")), start=1):
            rows.append((str(page.relative_to(DOCS)), index, match.group(1)))
    return rows


def render_all(rows: list[tuple[str, int, str]]) -> list[str]:
    """Render every row in one headless Chromium; return one line per fence that did not draw."""
    from playwright.sync_api import sync_playwright

    problems: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_content(f'<html><body><script src="{cdn_script()}"></script></body></html>')
        page.wait_for_function("typeof mermaid !== 'undefined'", timeout=60_000)
        page.evaluate("mermaid.initialize({startOnLoad: false, theme: 'base', securityLevel: 'strict'})")
        for number, (rel, index, body) in enumerate(rows):
            src = body + ACCENT_CLASS if re.match(r"\s*(flowchart|graph|stateDiagram(-v2)?)\b", body) else body
            try:
                svg = page.evaluate(
                    "async ([id, src]) => { const out = await mermaid.render(id, src); return out.svg; }",
                    [f"m{number}", src],  # one id per render: mermaid refuses to draw twice into the same id
                )
            except Exception as exc:  # noqa: BLE001 - the browser's message is the finding
                problems.append(f"{rel} #{index}: {str(exc).splitlines()[0][:160]}")
                continue
            if "<svg" not in svg or "Syntax error" in svg or len(svg) < 200:
                problems.append(f"{rel} #{index}: empty or errored svg")
            else:
                print(f"PASS  {rel} #{index}  {len(svg)} bytes")
        browser.close()
    return problems


def main(argv: list[str]) -> int:
    """Render the fences named on the command line (or all of them); exit 1 on any red row."""
    rows = fences(argv)
    if not rows:
        print("no mermaid fences under docs/")
        return 0
    problems = render_all(rows)
    for line in problems:
        print("FAIL ", line)
    print(f"{len(rows)} mermaid fence(s): {len(rows) - len(problems)} pass, {len(problems)} fail")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
