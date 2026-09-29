"""render.py: export every docs/drawings/*.excalidraw to docs/assets/drawings/<id>.{paper,dark}.svg.

Uses Excalidraw's own exporter (@excalidraw/utils, installed next to this file with ``npm i``)
inside a headless Chromium, so the SVG is exactly what excalidraw.com would export. The export
is then restyled by ``restyle.py``: docs fonts embedded, hex values mapped for the dark scheme,
``role="img"`` and a ``<title>`` added, the fixed pixel size dropped in favour of the viewBox.

    python3 -m playwright install chromium   # npm i of @excalidraw/utils happens on first run, in .cache/
    python3 render.py            # all drawings
    python3 render.py d01_what_is   # one
    python3 render.py --check    # exit 1 when a committed SVG differs from its source
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from restyle import restyle

HERE = Path(__file__).resolve().parent
SRC = HERE.parent  # docs/drawings
REPO = HERE.parent.parent.parent
OUT = REPO / "docs" / "assets" / "drawings"
# The npm project lives OUTSIDE docs/ on purpose: the docs graders walk every *.md under docs/,
# and a node_modules tree there would be graded as site pages.
NODE = REPO / ".cache" / "drawings-node"
PORT = 8793
EXCALIDRAW_UTILS = "0.1.5"


def ensure_node_modules() -> None:
    if (NODE / "node_modules" / "@excalidraw" / "utils").exists():
        return
    NODE.mkdir(parents=True, exist_ok=True)
    (NODE / "package.json").write_text(
        json.dumps({"name": "strands-robots-drawings", "private": True,
                    "dependencies": {"@excalidraw/utils": EXCALIDRAW_UTILS}}, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(["npm", "i", "--silent"], cwd=NODE, check=True)
    (NODE / "render_page.html").write_text((HERE / "render_page.html").read_text(encoding="utf-8"), encoding="utf-8")


def export_all(files: list[Path]) -> dict[str, str]:
    ensure_node_modules()
    (NODE / "render_page.html").write_text((HERE / "render_page.html").read_text(encoding="utf-8"), encoding="utf-8")
    srv = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
        cwd=NODE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out: dict[str, str] = {}
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{PORT}/render_page.html")
            page.wait_for_function("window.ready === true")
            for f in files:
                scene = json.loads(f.read_text(encoding="utf-8"))
                out[f.stem] = page.evaluate("s => window.renderScene(s)", scene)
            browser.close()
    finally:
        srv.terminate()
    return out


def main(argv: list[str]) -> int:
    check = "--check" in argv
    names = [a for a in argv if not a.startswith("--")]
    files = sorted(SRC.glob("*.excalidraw"))
    if names:
        files = [f for f in files if f.stem in names]
    OUT.mkdir(parents=True, exist_ok=True)
    raw = export_all(files)
    stale: list[str] = []
    for f in files:
        scene = json.loads(f.read_text(encoding="utf-8"))
        alt = (scene.get("strandsDrawing") or {}).get("alt", f.stem)
        for scheme in ("paper", "dark"):
            svg = restyle(raw[f.stem], scheme=scheme, title=alt)
            target = OUT / f"{f.stem}.{scheme}.svg"
            if check:
                if not target.exists() or target.read_text(encoding="utf-8") != svg:
                    stale.append(str(target.relative_to(OUT.parent.parent.parent)))
                continue
            target.write_text(svg, encoding="utf-8")
            print(f"{target.relative_to(OUT.parent.parent.parent)}  {len(svg) // 1024} KB")
    if check:
        if stale:
            print("stale (re-run docs/drawings/_tools/render.py):\n  " + "\n  ".join(stale))
            return 1
        print(f"{len(files) * 2} drawings match their sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
