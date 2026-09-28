#!/usr/bin/env python3
"""Render every streamable robot's catalog thumbnail with the site's own viewer.

The thumbnail is the 3D view itself, captured with a transparent background, so the
card and the poster show it on the page's stage colour in both themes.

    python3 docs/hooks/render_thumbnails.py [--site URL] [names...]

Needs a built site being served (``mkdocs build -d /tmp/site && python3 -m http.server
8765 -d /tmp/site``), ``pip install playwright pillow`` and ``playwright install
chromium``. Writes docs/assets/img/robots/<name>.webp (800x600, lossy, alpha kept).
Idempotent; pass names to redo a few.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets" / "img" / "robots"
MANIFEST = HERE.parent / "assets" / "viewer" / "robots.json"

MOUNT = """(n) => {
  document.querySelectorAll("robot-viewer").forEach((v) => v.remove());
  // Only the viewer may paint: the capture omits the background, so nothing else may show through.
  document.documentElement.style.background = "transparent";
  document.body.style.background = "transparent";
  for (const el of document.body.children) el.style.visibility = "hidden";
  const v = document.createElement("robot-viewer");
  v.setAttribute("name", n);
  v.setAttribute("compact", "");
  v.style.cssText = "width:800px;height:600px;display:block;position:fixed;top:0;left:0;z-index:99999;border:0;border-radius:0;margin:0;background:transparent";
  document.body.prepend(v);
  v.load();
}"""
READY = "() => ['ready', 'error'].includes(document.querySelector('robot-viewer')?._state)"
HIDE = """() => { const v = document.querySelector('robot-viewer'); for (const s of ['.joints', '.code', '.chrome']) v.shadowRoot.querySelector(s).hidden = true; if (v._grid) v._grid.visible = false; }"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", default="http://127.0.0.1:8765/")
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    from PIL import Image
    from playwright.sync_api import sync_playwright

    robots = json.loads(MANIFEST.read_text())["robots"]
    names = args.names or sorted(r["name"] for r in robots.values() if r.get("viewer"))
    OUT.mkdir(parents=True, exist_ok=True)
    ok, bad = 0, []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1000, "height": 800})
        page.on("pageerror", lambda e: print("  pageerror", str(e)[:120]))
        page.goto(args.site, wait_until="load")
        for name in names:
            t0 = time.monotonic()
            try:
                # A fresh document per robot: the WASM heap is not returned between models
                # and a long sweep otherwise ends in "Could not allocate memory".
                page.goto(args.site, wait_until="load")
                page.evaluate("customElements.whenDefined('robot-viewer')")
                page.evaluate(MOUNT, name)
                page.wait_for_function(READY, timeout=180_000)
                if page.evaluate("document.querySelector('robot-viewer')._state") != "ready":
                    raise RuntimeError(page.evaluate("document.querySelector('robot-viewer').shadowRoot.querySelector('.status')?.textContent"))
                page.evaluate(HIDE)
                page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
                png = page.screenshot(clip={"x": 0, "y": 0, "width": 800, "height": 600}, omit_background=True, animations="disabled", timeout=120_000)
                Image.open(io.BytesIO(png)).convert("RGBA").save(OUT / f"{name}.webp", quality=85, method=6)
                ok += 1
                print(f"ok   {name} {time.monotonic() - t0:.1f}s", flush=True)
            except Exception as exc:  # one robot failing must not stop the sweep
                bad.append(name)
                print(f"FAIL {name} {str(exc)[:160]}", flush=True)
        browser.close()
    print(f"{ok} ok, {len(bad)} failed" + (": " + ", ".join(bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
