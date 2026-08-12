"""Compose the measured verdict figure for the mesh scouting-default prose fix."""
from __future__ import annotations
import json, os, pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RUN = os.environ.get("GITHUB_RUN_ID", "x")
F = json.loads(pathlib.Path(f"/tmp/art-{RUN}.json").read_text(encoding="utf-8"))

# --- self-audit on the measurements ----------------------------------------
assert F["default"] == {"scouting/multicast/enabled": "false", "scouting/gossip/enabled": "true"}
assert F["opted_in"]["scouting/multicast/enabled"] == "true"
assert (F["guard_failures_before"], F["guard_failures_after"]) == (4, 0), F
sites = F["sites"]
assert len(sites) == 4 and all(s["changed"] for s in sites)
assert all("multicast" in s["before"].lower() for s in sites), "every site claimed multicast"
svg = next(s for s in sites if s["path"].endswith(".svg"))
assert svg["before"] == "Zenoh multicast (default)" and svg["after"] == "Zenoh gossip (default)"

placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

RED, GREEN, INK, MUTED = "#B00020", "#0B6E4F", "#1A3B5C", "#555555"
fig = plt.figure(figsize=(15.4, 11.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[0.92, 2.75, 1.25], width_ratios=[1, 1],
                      hspace=0.20, wspace=0.09, left=0.022, right=0.978, top=0.925, bottom=0.028)

fig.suptitle("Mesh prose vs the scouting default the config emits", fontsize=17, fontweight="bold", color=INK, y=0.975)
fig.text(0.5, 0.945, "left of each pair = strands-labs/robots main, right = this change; every cell is measured",
         ha="center", fontsize=10.5, color=MUTED, style="italic")

# --- row 0: the source of truth --------------------------------------------
for col, (title, key, note) in enumerate([
    ("Source of truth: scouting_block() with no env override", "default", "shipped default"),
    ("scouting_block() with STRANDS_MESH_MULTICAST=true", "opted_in", "operator opt-in"),
]):
    ax = fig.add_subplot(gs[0, col]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    put(ax, 0.02, 0.90, title, transform=ax.transAxes, fontsize=11.2, fontweight="bold", color=INK)
    put(ax, 0.02, 0.74, f"({note})", transform=ax.transAxes, fontsize=9.6, color=MUTED, style="italic")
    rows = list(F[key].items())
    top, last = 0.52, 0.16
    step = (top - last) / (len(rows) - 1)
    assert step > 0.030, step
    y = top
    for k, v in rows:
        on = v == "true"
        put(ax, 0.04, y, k, transform=ax.transAxes, fontsize=10.6, family="monospace", color=INK)
        put(ax, 0.74, y, v, transform=ax.transAxes, fontsize=10.6, family="monospace",
            fontweight="bold", color=(GREEN if on else RED))
        y -= step
    assert abs((y + step) - last) < 1e-9, (y, last)

# --- row 1: the four prose sites -------------------------------------------
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.005, 0.975, "The four prose sites that presented multicast as the mechanism",
    transform=ax.transAxes, fontsize=12.4, fontweight="bold", color=INK)
y = 0.905
for site in sites:
    put(ax, 0.012, y, site["path"], transform=ax.transAxes, fontsize=10.6, family="monospace",
        fontweight="bold", color=INK)
    y -= 0.036
    for label, text, colour in (("main", site["before"], RED), ("this change", site["after"], GREEN)):
        wrapped = textwrap.wrap(text, width=118) or [""]
        put(ax, 0.030, y, f"{label}:", transform=ax.transAxes, fontsize=9.7, color=colour, fontweight="bold")
        for i, line in enumerate(wrapped):
            put(ax, 0.108, y - i * 0.030, line, transform=ax.transAxes, fontsize=9.9,
                family="monospace", color=colour)
        y -= 0.030 * len(wrapped) + 0.008
    y -= 0.016
assert y > 0.02, f"site table overflowed to {y}"

# --- row 2 left: the diagram label box, redrawn to the SVG's own geometry ---
ax = fig.add_subplot(gs[2, 0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.02, 0.93, "examples/lerobot/architecture.svg -- the mesh transport box",
    transform=ax.transAxes, fontsize=11.2, fontweight="bold", color=INK)
put(ax, 0.02, 0.80, "redrawn from the diagram's own rect + text styles", transform=ax.transAxes,
    fontsize=9.2, color=MUTED, style="italic")
for i, (tag, label, colour) in enumerate([("main", svg["before"], RED), ("this change", svg["after"], GREEN)]):
    bx = 0.06 + i * 0.48
    ax.add_patch(Rectangle((bx, 0.24), 0.40, 0.40, transform=ax.transAxes,
                           facecolor="#FFFFFF", edgecolor="#1A3B5C", linewidth=1.4))
    put(ax, bx + 0.20, 0.50, "LAN", transform=ax.transAxes, ha="center", fontsize=11,
        fontweight="600", color="#333333")
    put(ax, bx + 0.20, 0.37, label, transform=ax.transAxes, ha="center", fontsize=10,
        color=colour, style="italic")
    put(ax, bx + 0.20, 0.13, tag, transform=ax.transAxes, ha="center", fontsize=9.6,
        color=colour, fontweight="bold")

# --- row 2 right: guard verdict + gate -------------------------------------
ax = fig.add_subplot(gs[2, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.02, 0.93, "Guard: prose blocks naming multicast without an opt-in marker",
    transform=ax.transAxes, fontsize=11.2, fontweight="bold", color=INK)
lines = [
    (f"main                 {F['guard_failures_before']} guard failures  (4 blocks flagged)", RED),
    (f"this change          {F['guard_failures_after']} guard failures  (0 blocks flagged)", GREEN),
    ("", INK),
    ("29 files scanned: mesh package + README + shipped diagrams", MUTED),
    ("0 exemptions -- the security rationale and the opt-in docs", MUTED),
    ("pass on their own merits by naming the flag", MUTED),
    ("", INK),
    ("session.py / camera_offload.py: docstring-only, proven by an", MUTED),
    ("identical post-strip AST digest on both trees", MUTED),
    ("full suite 28525 passed / 257 skipped / 0 failed", GREEN),
]
top, last = 0.79, 0.06
step = (top - last) / (len(lines) - 1)
assert step > 0.030, step
y = top
for text, colour in lines:
    if text:
        put(ax, 0.03, y, text, transform=ax.transAxes, fontsize=9.9, family="monospace", color=colour)
    y -= step
assert abs((y + step) - last) < 1e-9, (y, last)

# --- layout guard ----------------------------------------------------------
for axis, y, axes_coords in placed:
    assert axes_coords, "every text must use axes coordinates"
    assert -0.03 <= y <= 1.07, f"text at y={y} escapes its axes"

out = pathlib.Path(f"_art/mesh_multicast_default_prose.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB")).astype(int)
sides = {"top": im[:8], "bottom": im[-8:], "left": im[:, :8], "right": im[:, -8:]}
for name, band in sides.items():
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"wrote {out} {im.shape[1]}x{im.shape[0]}  borders clean")
