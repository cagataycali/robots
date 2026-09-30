#!/usr/bin/env python3
"""Drawings for the strands-robots docs, in the docs site's own visual language.

Each drawing is a module in docs/drawings/scenes/ that exposes `def scene() -> Scene`; this file
only knows how to emit SVG for two palettes (paper | dark) with the site's tokens
(docs/stylesheets/extra.css): Space Grotesk text, JetBrains Mono labels, one green accent, 1px
borders, 8px radius, no shadows, no gradients, fonts embedded as data: woff2. The committed SVGs
under docs/assets/drawings are the output of this file; a grader re-renders every scene and
refuses a drift between the two, so the source of truth is the scene module, never the SVG.

    python3 scene.py --all                       # every scene -> docs/assets/drawings/<id>.{paper,dark}.svg
    python3 scene.py d01_what_is                 # one scene (module name or scene id)
    python3 scene.py --all --png                 # also <id>.{paper,dark}.png at 2x next to the SVGs (playwright)
    python3 scene.py --all --png --shots DIR     # move the PNGs into DIR (they are not committed)
    python3 scene.py --all --check DOCS_DIR      # exit 1 on an identifier-like label (4+ chars) absent
                                                 # from DOCS_DIR/**/*.md, or on an em or en dash
    python3 scene.py --all --verify              # exit 1 when a committed SVG differs from its scene
    python3 scene.py --all --out DIR             # write the SVGs somewhere else

Adding a drawing: create scenes/dNN_slug.py with `from scene import Scene` and a `scene()` function
that returns a Scene built from box / chip / chips / arrow / down / section / para / footnote calls.
Coordinates are plain numbers on a 1200 wide canvas; the height is the Scene's `h` argument. Keep
one accent element per drawing, orthogonal wires with the label beside the wire, a footnote, and
the canvas filled (a WARN line reports more than a fifth empty).
"""
from __future__ import annotations

import base64
import html
import importlib.util
import os
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
SCENES_DIR = HERE.parent / "scenes"
OUT = HERE.parent.parent / "assets" / "drawings"

# Fonts: the Google latin subsets committed under docs/assets/fonts (SCENE_FONT_DIR overrides them).
_FONT_CANDIDATES = [
    pathlib.Path(os.environ["SCENE_FONT_DIR"]) if os.environ.get("SCENE_FONT_DIR") else None,
    HERE.parent.parent / "assets" / "fonts",
]
FONT_DIR = next((p for p in _FONT_CANDIDATES if p and (p / "JetBrainsMono-500.woff2").exists()),
                HERE.parent.parent / "assets" / "fonts")


# ------------------------------------------------------------------ tokens (extra.css)
# Verified 2026-09-30 against ~/robots-docs-revamp/docs/stylesheets/extra.css:
#   paper  bg #ffffff (--md-default-bg-color) fg #000000 (--sr-black) text #3d3c3c (--sr-charcoal-2)
#          muted #767373 accent #007a3d soft rgba(2,164,53,0.12) chip #f4f4f4 (--sr-chip-bg)
#          pill border #999696 (--sr-pill-border)
#   dark   bg #000000 fg #ffffff text #b6b6b6 (--md-default-fg-color--light) muted #999696
#          accent #00cc60 soft rgba(0,204,96,0.14) chip #28292a (--sr-charcoal-1)
#          pill border #3d3c3c (--sr-charcoal-2)
# No hex differs. Two values are the drawing's own and have no CSS counterpart: `wire` and `layer`
# are the fg at 55% and 28% alpha. extra.css also has --sr-pill-bg (#ffffff paper / charcoal-1 dark)
# for pill buttons; chips here use --sr-chip-bg on purpose, the same choice the pr-story drawings made.
PALETTES = {
    "paper": dict(bg="#ffffff", fg="#000000", text="#3d3c3c", muted="#767373", accent="#007a3d",
                  soft="rgba(2,164,53,0.12)", border="#000000", chip="#f4f4f4", pill="#999696",
                  wire="rgba(0,0,0,0.55)", layer="rgba(0,0,0,0.28)"),
    "dark": dict(bg="#000000", fg="#ffffff", text="#b6b6b6", muted="#999696", accent="#00cc60",
                 soft="rgba(0,204,96,0.14)", border="#ffffff", chip="#28292a", pill="#3d3c3c",
                 wire="rgba(255,255,255,0.55)", layer="rgba(255,255,255,0.28)"),
}
RADIUS = 8
MONO = "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace"
GROT = "'Space Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif"
# average advance per em, used for wrapping and chip widths (measured against the rendered PNGs)
EM_GROT, EM_MONO = 0.52, 0.60
WIDTH = 1200


def _font_face(family: str, weight: int, file: str) -> str:
    p = FONT_DIR / file
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f"@font-face{{font-family:'{family}';font-weight:{weight};font-display:block;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}")


FONTS = "".join([
    _font_face("JetBrains Mono", 500, "JetBrainsMono-500.woff2"),
    _font_face("Space Grotesk", 400, "SpaceGrotesk-400.woff2"),
    _font_face("Space Grotesk", 500, "SpaceGrotesk-500.woff2"),
])


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def wrap(text: str, width: float, size: float, em: float = EM_GROT) -> list[str]:
    """Greedy word wrap by estimated advance."""
    per = size * em
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) * per <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_width(s: str, size: float, em: float) -> float:
    return len(s) * size * em


# ------------------------------------------------------------------ the DSL
class Scene:
    """One drawing. `labels` collects every string drawn, for --check."""

    def __init__(self, id: str, title: str, lead: str, alt: str, h: int = 640):
        self.id, self.title, self.lead, self.alt, self.w, self.h = id, title, lead, alt, WIDTH, h
        self.parts: list[str] = []
        self.labels: list[str] = [title, lead]
        self.maxy = 78.0  # lowest ink so far, for the fill check

    def _ink(self, y: float) -> None:
        self.maxy = max(self.maxy, y)

    # text -------------------------------------------------------------
    def text(self, x, y, s, cls="grot text", size=12.5, anchor="start", weight=None, spacing=None):
        self.labels.append(s)
        self._ink(y)
        extra = f' font-weight="{weight}"' if weight else ""
        extra += f' letter-spacing="{spacing}"' if spacing else ""
        self.parts.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                          f'text-anchor="{anchor}"{extra}>{esc(s)}</text>')

    def section(self, x, y, s, accent=False):
        """Mono uppercase small caption with 0.05em letter-spacing, the site's section-name voice."""
        self.text(x, y, s.upper(), cls="mono " + ("accent" if accent else "muted"), size=10.5, spacing="0.05em")

    def para(self, x, y, s, width, size=12.5, cls="grot text", lh=1.35):
        lines = wrap(s, width, size)
        for i, line in enumerate(lines):
            self.text(x, y + i * size * lh, line, cls=cls, size=size)
        return y + len(lines) * size * lh

    def footnote(self, y, s, size=12.5, width=1080):
        """Centred muted Grotesk line(s) at the bottom of the canvas."""
        lines = wrap(s, width, size)
        for i, line in enumerate(lines):
            self.text(self.w / 2, y + i * size * 1.35, line, cls="grot muted", size=size, anchor="middle")
        return y + len(lines) * size * 1.35

    # boxes ------------------------------------------------------------
    def box(self, x, y, w, h, title=None, sub=None, accent=False, dashed=False, size=15, subsize=12.5):
        """Card with a LEFT-aligned mono title and a Grotesk sub-line, wrapped inside the card."""
        cls = "card accent-card" if accent else ("card layer" if dashed else "card")
        self.parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{RADIUS}"/>')
        self._ink(y + h)
        ty = y + 26
        if title:
            self.text(x + 14, ty, title, cls="mono fg", size=size, weight=500)
            ty += 21
        if sub:
            self.para(x + 14, ty - 2, sub, w - 28, size=subsize)
        return x, y, w, h

    def chip(self, x, y, s, size=11.5, accent=False):
        w = text_width(s, size, EM_MONO) + 20
        cls = "chip accent-chip" if accent else "chip"
        self.parts.append(f'<g class="{cls}"><rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" rx="12"/>'
                          f'<text x="{x + 10:.1f}" y="{y + 16.3}" font-size="{size}">{esc(s)}</text></g>')
        self.labels.append(s)
        self._ink(y + 24)
        return x + w + 8

    def chips(self, x, y, items, accent=None):
        for s in items:
            x = self.chip(x, y, s, accent=(s == accent))
        return x

    # arrows -----------------------------------------------------------
    def arrow(self, pts, accent=False, dashed=False, head=True, label=None, label_dx=6, label_dy=-6,
              label_anchor="start"):
        """Orthogonal polyline through pts=[(x,y),...], 1.5px, small FILLED triangle at the end.

        head=True puts the triangle at the last point, head="both" at both ends, head=False none.

        The label sits beside the FIRST segment's midpoint, offset by label_dx / label_dy; pass a
        negative label_dx with label_anchor="end" to put it on the left of a vertical wire.
        """
        cls = "wire" + (" accent-wire" if accent else "") + (" dashed" if dashed else "")
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
        hid = "head-accent" if accent else "head"
        marker = ""
        if head:  # True = head at the end; "both" = a head at each end; False = bare wire
            marker += f' marker-end="url(#{hid})"'
        if head == "both":
            marker += f' marker-start="url(#{hid})"'
        self.parts.append(f'<path class="{cls}" d="{d}"{marker}/>')
        self._ink(max(y for _, y in pts))
        if label:
            mx, my = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
            self.text(mx + label_dx, my + label_dy, label, cls="mono muted", size=10.5, anchor=label_anchor)

    def down(self, x, y1, y2, label=None, **kw):
        self.arrow([(x, y1), (x, y2)], label=label, **kw)

    def up(self, x, y1, y2, label=None, **kw):
        self.arrow([(x, y1), (x, y2)], label=label, **kw)

    # output -----------------------------------------------------------
    def empty_fraction(self) -> float:
        """Share of the canvas height below the lowest ink; the brief allows one fifth."""
        return max(0.0, (self.h - self.maxy) / self.h)

    def svg(self, theme: str) -> str:
        p = PALETTES[theme]
        css = f"""
{FONTS}
:root{{--bg:{p['bg']};--fg:{p['fg']};--text:{p['text']};--muted:{p['muted']};--accent:{p['accent']};
--soft:{p['soft']};--border:{p['border']};--chip:{p['chip']};--pill:{p['pill']};--wire:{p['wire']};--layer:{p['layer']}}}
.bg{{fill:var(--bg)}}
.mono{{font-family:{MONO};font-weight:500}} .grot{{font-family:{GROT};font-weight:400}}
.fg{{fill:var(--fg)}} .text{{fill:var(--text)}} .muted{{fill:var(--muted)}} .accent{{fill:var(--accent)}}
.card{{fill:var(--bg);stroke:var(--border);stroke-width:1}}
.card.layer{{stroke:var(--layer);stroke-dasharray:4 4}}
.card.accent-card{{fill:var(--soft);stroke:var(--accent);stroke-width:1.5}}
.chip rect{{fill:var(--chip);stroke:var(--pill);stroke-width:1}} .chip text{{font-family:{MONO};font-weight:500;fill:var(--text)}}
.chip.accent-chip rect{{fill:var(--soft);stroke:var(--accent)}} .chip.accent-chip text{{fill:var(--accent)}}
.wire{{fill:none;stroke:var(--wire);stroke-width:1.5}} .wire.dashed{{stroke-dasharray:3 5}}
.wire.accent-wire{{stroke:var(--accent)}}
#head path{{fill:var(--wire)}} #head-accent path{{fill:var(--accent)}}
"""
        head = ('<marker id="{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse"><path d="M0 0.5 L10 5 L0 9.5 Z"/></marker>')
        body = "\n".join(self.parts)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-labelledby="t">\n<title id="t">{esc(self.alt)}</title>\n'
                f'<style>{css}</style>\n<defs>{head.format(id="head")}{head.format(id="head-accent")}</defs>\n'
                f'<rect class="bg" width="{self.w}" height="{self.h}"/>\n'
                f'<text class="mono fg" x="600" y="50" font-size="24" text-anchor="middle" letter-spacing="0.05em">{esc(self.title)}</text>\n'
                f'<text class="grot text" x="600" y="78" font-size="14.5" text-anchor="middle">{esc(self.lead)}</text>\n'
                f'{body}\n</svg>\n')


# ------------------------------------------------------------------ the registry
def load_scenes() -> dict[str, "Scene"]:
    """Import every scenes/*.py (not starting with _) and call its scene(); keyed by module name."""
    found: dict[str, Scene] = {}
    if not SCENES_DIR.is_dir():
        return found
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    for path in sorted(SCENES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"scenes.{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if not hasattr(mod, "scene"):
            print(f"skip {path.name}: no scene() function", file=sys.stderr)
            continue
        found[path.stem] = mod.scene()
    return found


# ------------------------------------------------------------------ png + check
def render_png(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """2x screenshots of the SVGs with playwright chromium; <id>.paper.svg -> <id>.paper.png."""
    from playwright.sync_api import sync_playwright
    pngs = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for p in paths:
            svg = p.read_text(encoding="utf-8")
            w, h = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups()
            page = b.new_page(viewport={"width": int(w), "height": int(h)}, device_scale_factor=2)
            page.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
            page.wait_for_timeout(150)
            png = p.with_suffix(".png")
            page.screenshot(path=str(png), full_page=False)
            page.close()
            pngs.append(png)
        b.close()
    return pngs


_WORD = re.compile(r"[A-Za-z0-9_]+")


def docs_vocabulary(docs: pathlib.Path) -> set[str]:
    vocab: set[str] = set()
    for md in docs.rglob("*.md"):
        vocab.update(w.lower() for w in _WORD.findall(md.read_text(encoding="utf-8", errors="ignore")))
    return vocab


def _looks_like_an_identifier(word: str) -> bool:
    """A token a drawing must not invent: snake_case, a digit or a CamelCase name.

    Plain prose ("underneath", "grows") and the uppercased section captions are the drawing's own
    voice and are not graded; the names it draws (run_policy, EmbodimentMap, STRANDS_MESH_OVERRIDE_CODE,
    so101) must exist somewhere in the docs.
    """
    if "_" in word or any(ch.isdigit() for ch in word):
        return True
    return word[:1].isupper() and not word.isupper()


def check_labels(scenes: dict[str, Scene], docs: pathlib.Path) -> list[str]:
    """Every identifier-like label token of 4+ chars must occur in docs/**/*.md; no em or en dash."""
    vocab = docs_vocabulary(docs)
    problems: list[str] = []
    for name, sc in scenes.items():
        seen: set[str] = set()
        for lab in sc.labels:
            if "\u2013" in lab or "\u2014" in lab:
                problems.append(f"{name}: DASH in {lab!r}")
            for w in _WORD.findall(lab):
                lw = w.lower()
                if len(lw) >= 4 and lw not in vocab and lw not in seen and _looks_like_an_identifier(w):
                    seen.add(lw)
                    problems.append(f"{name}: name not in docs: {w}")
    return problems


def verify_svgs(scenes: dict[str, Scene], out: pathlib.Path) -> list[str]:
    """Names of committed SVGs that differ from what their scene renders now (or are missing)."""
    drift: list[str] = []
    for sc in scenes.values():
        for theme in ("paper", "dark"):
            p = out / f"{sc.id}.{theme}.svg"
            if not p.is_file():
                drift.append(f"{p.name}: missing")
            elif p.read_text(encoding="utf-8") != sc.svg(theme):
                drift.append(f"{p.name}: differs from scenes/{sc.id}.py")
    return drift


def main(argv: list[str]) -> int:
    def opt(flag: str) -> str | None:
        if flag in argv:
            i = argv.index(flag)
            return argv[i + 1] if i + 1 < len(argv) else None
        return None

    out = pathlib.Path(opt("--out") or OUT)
    shots = opt("--shots")
    check = opt("--check")
    skip = {opt("--out"), opt("--shots"), opt("--check")}
    if "--verify" in argv and "--png" in argv:
        print("--verify does not render PNGs", file=sys.stderr)
    only = [a for a in argv if not a.startswith("--") and a not in skip]

    scenes = load_scenes()
    if not scenes:
        print(f"no scenes in {SCENES_DIR}", file=sys.stderr)
        return 2
    if "--all" not in argv:
        picked = {k: v for k, v in scenes.items() if k in only or v.id in only}
        if not picked:
            print("known scenes:", ", ".join(f"{k} ({v.id})" for k, v in scenes.items()), file=sys.stderr)
            return 2
        scenes = picked

    if "--verify" in argv:
        drift = verify_svgs(scenes, out)
        for line in drift:
            print("DRIFT", line)
        print(f"verify: {len(drift)} drifted file(s) under {out}")
        return 1 if drift else 0

    out.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    rc = 0
    for name, sc in scenes.items():
        for theme in ("paper", "dark"):
            p = out / f"{sc.id}.{theme}.svg"
            p.write_text(sc.svg(theme), encoding="utf-8")
            written.append(p)
            print(p.name, p.stat().st_size)
        empty = sc.empty_fraction()
        if empty > 0.2:
            print(f"WARN {name}: {empty:.0%} of the canvas is empty (h={sc.h}, ink to y={sc.maxy:.0f})")
    if "--png" in argv:
        pngs = render_png(written)
        for png in pngs:
            print(png.name, png.stat().st_size)
        if shots:
            d = pathlib.Path(shots)
            d.mkdir(parents=True, exist_ok=True)
            for png in pngs:
                shutil.move(str(png), d / png.name)
                print("->", d / png.name)
    if check:
        problems = check_labels(scenes, pathlib.Path(check))
        for line in problems:
            print("CHECK", line)
        print(f"check: {len(problems)} problem(s) against {check}")
        rc = 1 if problems else 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
