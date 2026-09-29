"""excal.py: build Excalidraw scenes in the Strands Robots docs style.

Every drawing under ``docs/drawings/`` is produced by a ``d<NN>_<slug>.py`` script in this
directory that uses this module, so any drawing can be rebuilt with ``python3 build.py``.
The ``.excalidraw`` file is the source of truth (editable on excalidraw.com or in the VS Code
Excalidraw extension); ``render.py`` exports it to SVG with Excalidraw's own exporter and
``restyle.py`` maps the palette and fonts onto the docs tokens for the paper and dark schemes.

House style (docs/stylesheets/extra.css tokens)
  * roughness 0, strokeWidth 1, rounded rectangles (roundness type 3), solid fills, no shadows
  * ink for boxes and arrows, muted for captions, ONE accent element per drawing
    (the gate, the stop, the checkpoint)
  * words in fontFamily 2 (Helvetica -> Space Grotesk on export), code in fontFamily 3
    (Cascadia -> JetBrains Mono on export)
  * no em or en dashes anywhere in a label; every label word must exist in docs/**/*.md

The scene is authored in the PAPER palette. ``restyle.py`` rewrites the hex values for the
dark scheme, so a drawing never has two sources.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent  # docs/drawings/

# Paper palette (docs/stylesheets/extra.css [data-md-color-scheme="paper"]). The dark
# counterparts live in restyle.py so both files derive from one table.
INK = "#000000"
TEXT = "#3d3c3c"
MUTED = "#767373"
ACCENT = "#007a3d"
ACCENT_SOFT = "#e6f6e7"  # --sr-surface-green
CHIP = "#f4f4f4"
PILL = "#999696"
BG = "#ffffff"

LINE_H = 1.25
# average glyph advance as a fraction of fontSize, per Excalidraw fontFamily
CHAR_W = {2: 0.52, 3: 0.60, 1: 0.55, 5: 0.55}
PAD = 10
WORDS = 2  # Helvetica in the source, Space Grotesk after restyle
CODE = 3  # Cascadia in the source, JetBrains Mono after restyle


def text_size(s: str, size: float, family: int = WORDS) -> tuple[float, float]:
    lines = s.split("\n")
    w = max(len(line) for line in lines) * size * CHAR_W.get(family, 0.55)
    h = len(lines) * size * LINE_H
    return w, h


class Drawing:
    """One Excalidraw scene. Coordinates are absolute pixels; the exporter adds padding."""

    def __init__(self, slug: str, alt: str, width: int = 1200, seed: int = 7):
        self.slug = slug
        self.alt = alt
        self.width = width
        self.E: list[dict] = []
        self._i = 0
        self.rng = random.Random(seed)
        self.accent_used = 0

    # ------------------------------------------------------------ primitives
    def _id(self, p: str) -> str:
        return f"{self.slug}-{p}-{len(self.E)}"

    def _base(self, t: str, x: float, y: float, w: float, h: float, **k) -> dict:
        self._i += 1
        e = dict(
            id=k.pop("id", self._id(t)),
            type=t,
            x=x,
            y=y,
            width=w,
            height=h,
            angle=0,
            strokeColor=k.pop("stroke", INK),
            backgroundColor=k.pop("bg", "transparent"),
            fillStyle="solid",
            strokeWidth=k.pop("sw", 1),
            strokeStyle=k.pop("ss", "solid"),
            roughness=0,
            opacity=100,
            groupIds=[],
            frameId=k.pop("frame", None),
            index="a" + format(self._i, "05d"),
            roundness=k.pop("roundness", {"type": 3}),
            seed=self.rng.randint(1, 2**31),
            version=1,
            versionNonce=self.rng.randint(1, 2**31),
            isDeleted=False,
            boundElements=[],
            updated=1759000000000,
            link=None,
            locked=False,
        )
        e.update(k)
        self.E.append(e)
        return e

    def text(self, x, y, s, size=18, color=TEXT, align="left", family=WORDS, container=None, w=None):
        tw, th = text_size(s, size, family)
        if w is not None:
            tw = w
        return self._base(
            "text",
            x,
            y,
            tw,
            th,
            stroke=color,
            roundness=None,
            text=s,
            fontSize=size,
            fontFamily=family,
            textAlign=align,
            verticalAlign="middle" if container else "top",
            containerId=container,
            originalText=s,
            autoResize=True,
            lineHeight=LINE_H,
        )

    def caption(self, x, y, s, size=14, color=MUTED, family=CODE):
        """A small mono caption (the site's section-label idiom)."""
        return self.text(x, y, s, size=size, color=color, family=family)

    def box(
        self,
        x,
        y,
        w,
        h,
        label,
        *,
        kind="plain",
        size=17,
        family=WORDS,
        align="center",
        ss="solid",
        sub=None,
        sub_size=13,
    ):
        """A 1px rounded box. kind: plain (white), chip (grey fill), accent (green border + soft
        fill; the one highlight per drawing), code (mono label, chip fill)."""
        stroke, bg, color = INK, BG, INK
        if kind == "chip":
            stroke, bg, color = PILL, CHIP, TEXT
        elif kind == "accent":
            stroke, bg, color = ACCENT, ACCENT_SOFT, ACCENT
            self.accent_used += 1
        elif kind == "code":
            stroke, bg, color, family = PILL, CHIP, INK, CODE
        b = self._base("rectangle", x, y, w, h, stroke=stroke, bg=bg, ss=ss, sw=1)
        if label:
            # label and sub are FREE text elements placed inside the box (not bound), so each
            # keeps its own size and colour; Excalidraw would give a bound text one size.
            _, th = text_size(label, size, family)
            if sub:
                _, sh = text_size(sub, sub_size, WORDS)
                top = y + (h - th - sh - 2) / 2
                self.text(x + PAD, top, label, size=size, color=color, align=align, family=family, w=w - 2 * PAD)
                self.text(x + PAD, top + th + 2, sub, size=sub_size, color=MUTED if kind != "accent" else color,
                          align=align, family=WORDS, w=w - 2 * PAD)
            else:
                self.text(x + PAD, y + (h - th) / 2, label, size=size, color=color, align=align, family=family,
                          w=w - 2 * PAD)
        return b

    def region(self, x, y, w, h, label=None, ss="dashed"):
        """A transparent dashed outline that groups things, with a mono caption at its top left."""
        r = self._base("rectangle", x, y, w, h, stroke=PILL, bg="transparent", ss=ss, sw=1, roundness={"type": 3})
        if label:
            self.caption(x + 12, y - 20, label)
        return r

    # ------------------------------------------------------------ arrows
    @staticmethod
    def edge(b, side, off=0.0):
        cx, cy = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        ox = off * b["width"] / 2
        oy = off * b["height"] / 2
        return {
            "r": (b["x"] + b["width"], cy + oy),
            "l": (b["x"], cy + oy),
            "t": (cx + ox, b["y"]),
            "b": (cx + ox, b["y"] + b["height"]),
        }[side]

    def arrow(self, a, sa, b, sb, label=None, *, color=INK, ss="solid", via=None, size=13, both=False,
              off_a=0.0, off_b=0.0, label_dy=-10, label_dx=0, family=WORDS):
        """Bound arrow from side sa of a to side sb of b. via = absolute waypoints (orthogonal)."""
        x1, y1 = self.edge(a, sa, off_a)
        x2, y2 = self.edge(b, sb, off_b)
        pts = [[0, 0]] + [[px - x1, py - y1] for px, py in (via or [])] + [[x2 - x1, y2 - y1]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ar = self._base(
            "arrow", x1, y1, max(xs) - min(xs), max(ys) - min(ys), stroke=color, ss=ss, sw=1,
            roundness=None, points=pts, lastCommittedPoint=None,
            startBinding=None,
            endBinding=None,
            startArrowhead="arrow" if both else None, endArrowhead="arrow", elbowed=False,
        )
        if label:
            if len(pts) > 2:
                mid = pts[len(pts) // 2]
            else:
                mid = [(pts[0][0] + pts[-1][0]) / 2, (pts[0][1] + pts[-1][1]) / 2]
            tw, th = text_size(label, size, family)
            # a free text beside the arrow (not bound) so the exporter does not draw a
            # white label box across the line
            self.text(x1 + mid[0] - tw / 2 + label_dx, y1 + mid[1] - th / 2 + label_dy, label, size=size,
                      color=MUTED, align="center", family=family)
        return ar

    def path(self, pts, label=None, *, color=INK, ss="solid", size=13, label_at=None, label_dy=-10, both=False):
        """An arrow along absolute waypoints pts=[(x, y), ...]; label sits at label_at (index of a
        segment midpoint, default the middle segment)."""
        x1, y1 = pts[0]
        rel = [[px - x1, py - y1] for px, py in pts]
        xs = [q[0] for q in rel]
        ys = [q[1] for q in rel]
        ar = self._base(
            "arrow", x1, y1, max(xs) - min(xs), max(ys) - min(ys), stroke=color, ss=ss, sw=1,
            roundness=None, points=rel, lastCommittedPoint=None, startBinding=None, endBinding=None,
            startArrowhead="arrow" if both else None, endArrowhead="arrow", elbowed=False,
        )
        if label:
            i = label_at if label_at is not None else (len(pts) - 1) // 2
            mx = (pts[i][0] + pts[i + 1][0]) / 2
            my = (pts[i][1] + pts[i + 1][1]) / 2
            tw, th = text_size(label, size)
            self.text(mx - tw / 2, my - th / 2 + label_dy, label, size=size, color=MUTED, align="center")
        return ar

    def line(self, x1, y1, x2, y2, color=PILL, ss="solid", sw=1):
        return self._base(
            "line", x1, y1, abs(x2 - x1), abs(y2 - y1), stroke=color, ss=ss, sw=sw, roundness=None,
            points=[[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint=None, startBinding=None,
            endBinding=None, startArrowhead=None, endArrowhead=None,
        )

    def lane(self, x, y, items, *, w=190, h=64, gap=50, direction="h", size=16, labels=None):
        """A row (or column) of boxes joined by arrows. items = [(label, kind)] or [(label, kind, sub)]."""
        out = []
        for i, it in enumerate(items):
            label, kind = it[0], it[1]
            sub = it[2] if len(it) > 2 else None
            bx = x + i * (w + gap) if direction == "h" else x
            by = y if direction == "h" else y + i * (h + gap)
            b = self.box(bx, by, w, h, label, kind=kind, size=size, sub=sub)
            if out:
                lab = labels[i - 1] if labels else None
                if direction == "h":
                    self.arrow(out[-1], "r", b, "l", lab)
                else:
                    self.arrow(out[-1], "b", b, "t", lab)
            out.append(b)
        return out

    # ------------------------------------------------------------ output
    def bounds(self):
        xs = [e["x"] for e in self.E]
        ys = [e["y"] for e in self.E]
        xe = [e["x"] + e["width"] for e in self.E]
        ye = [e["y"] + e["height"] for e in self.E]
        return min(xs), min(ys), max(xe), max(ye)

    def labels(self) -> list[str]:
        return [e["text"] for e in self.E if e["type"] == "text"]

    def save(self) -> Path:
        for s in self.labels():
            for bad in ("\u2014", "\u2013"):
                if bad in s:
                    raise SystemExit(f"{self.slug}: dash in label {s!r}")
        if self.accent_used > 1:
            raise SystemExit(f"{self.slug}: {self.accent_used} accent boxes; the house style allows one")
        doc = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.E,
            "appState": {"gridSize": 20, "gridStep": 5, "gridModeEnabled": False, "viewBackgroundColor": BG},
            "files": {},
            # our own metadata; excalidraw.com ignores unknown top-level keys
            "strandsDrawing": {"alt": self.alt},
        }
        path = OUT_DIR / f"{self.slug}.excalidraw"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, ensure_ascii=True)
        print(f"wrote {os.path.relpath(path, OUT_DIR.parent.parent)}: {len(self.E)} elements")
        return path
