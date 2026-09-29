"""preview.py: screenshot every exported drawing (both schemes) into a directory for a human look.

    python3 preview.py ~/.tiny/docs-revamp-20260929/shots [d01_what_is ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_SVG = Path(__file__).resolve().parent.parent.parent / "assets" / "drawings"


def main(argv: list[str]) -> None:
    dest = Path(argv[0]).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    names = argv[1:]
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1240, "height": 600}, device_scale_factor=2)
        for svg_path in sorted(OUT_SVG.glob("*.svg")):
            stem, scheme = svg_path.name.rsplit(".", 2)[0], svg_path.name.rsplit(".", 2)[1]
            if names and stem not in names:
                continue
            bg = "#ffffff" if scheme == "paper" else "#000000"
            svg = svg_path.read_text(encoding="utf-8").replace("<svg ", '<svg style="width:1240px" ', 1)
            pg.set_content(f'<body style="margin:0;background:{bg}"><div style="width:1240px">{svg}</div></body>')
            pg.wait_for_timeout(250)
            pg.screenshot(path=str(dest / f"{stem}.{scheme}.png"), full_page=True)
            print(dest / f"{stem}.{scheme}.png")
        b.close()


if __name__ == "__main__":
    main(sys.argv[1:])
