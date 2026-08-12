"""Compose the artifact: the redrawn diagram band + the measured guard census."""
from __future__ import annotations
import base64, io, json, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
F = json.loads((ROOT / "_art" / "facts.json").read_text())
assert F["tree"] == str(ROOT)

# ---- assert every number this figure will draw, against the measurement ----
N_BEFORE_NAMING = len(F["before"]["naming"]); N_BEFORE_BAD = len(F["before"]["unmarked"])
N_AFTER_NAMING = len(F["after"]["naming"]);  N_AFTER_BAD = len(F["after"]["unmarked"])
assert (N_BEFORE_NAMING, N_BEFORE_BAD) == (10, 2), (N_BEFORE_NAMING, N_BEFORE_BAD)
assert (N_AFTER_NAMING, N_AFTER_BAD) == (9, 0), (N_AFTER_NAMING, N_AFTER_BAD)
bad_files = sorted(u["file"] for u in F["before"]["unmarked"])
assert bad_files == ["examples/lerobot/architecture.svg", "strands_robots/mesh/iot/camera_offload.py"], bad_files
# every compliant block is compliant for a stated reason
for b in F["after"]["naming"]:
    assert b["why"], b

placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

FW, FH = 15.6, 11.4
fig = plt.figure(figsize=(FW, FH), dpi=124)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 1, height_ratios=[2.30, 2.15, 0.60], hspace=0.20,
                      left=0.028, right=0.972, top=0.925, bottom=0.028)

fig.suptitle(
    "strands-labs/robots #2192  -  the mesh discovery posture is gossip-only; two places still said multicast",
    fontsize=14.5, fontweight="600", color="#1A3B5C", y=0.982)
fig.text(0.5, 0.951,
         "Thor, headless. No SVG rasterizer on this host, so the layer-5 band is REDRAWN from the diagram's own "
         "<rect>/<text> geometry and its own CSS classes.",
         ha="center", fontsize=9.6, color="#555", style="italic")

# ------------------------------------------------------- row 1: the diagram band
axb = fig.add_subplot(gs[0]); axb.axis("off")
axb.set_xlim(70, 930); axb.set_ylim(0, 260); axb.invert_yaxis()
SCALE = (FW * 0.944 * 72) / (930 - 70)          # points per SVG user unit

logo_m = re.search(r'<image x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" href="data:image/png;base64,([^"]+)"',
                   (ROOT / "examples" / "lerobot" / "architecture.svg").read_text())
logo = np.array(Image.open(io.BytesIO(base64.b64decode(logo_m.group(5)))).convert("RGBA")) if logo_m else None
LX, LY, LW, LH = (float(logo_m.group(i)) for i in (1, 2, 3, 4)) if logo_m else (0, 0, 0, 0)

def draw_band(side: str, y_off: float, tag: str, tag_color: str) -> None:
    d = F[f"band_{side}"]
    css = d["css"]
    put(axb, 74, y_off + 6, tag, fontsize=11.5, fontweight="700", color=tag_color, va="center")
    axb.add_patch(Rectangle((80, y_off + 20), 840, 100, facecolor=css.get("mesh-fill", "#E8F1F8"),
                            edgecolor="#1A3B5C", linewidth=1.5, zorder=1))
    for r in d["rects"]:
        if r["h"] != 50:
            continue
        dashed = "stroke-dasharray" in r["attrs"]
        edge = "#FF6B35" if "accent-stroke" in r["attrs"] else "#1A3B5C"
        axb.add_patch(Rectangle((r["x"], r["y"] - 660 + y_off + 20), r["w"], r["h"], facecolor="#FFFFFF",
                                edgecolor=edge, linewidth=2 if edge == "#FF6B35" else 1,
                                linestyle=(0, (4, 3)) if dashed else "solid", zorder=2))
    if logo is not None:
        axb.imshow(logo, extent=(LX, LX + LW, LY - 660 + y_off + 20 + LH, LY - 660 + y_off + 20), zorder=3)
    for t in d["texts"]:
        ha = "center" if 'text-anchor="middle"' in t["attrs"] else "left"
        if "title-text" in t["attrs"]:
            size, color, weight, style = 14, "#1A3B5C", "600", "normal"
        elif "body-text" in t["attrs"]:
            size, color, weight, style = 11, "#333333", "600", "normal"
        else:
            size, color, weight, style = 10, "#555555", "normal", "italic"
        changed = "multicast" in t["body"].lower()
        put(axb, t["x"], t["y"] - 660 + y_off + 20, t["body"], fontsize=size * SCALE, color=color,
            fontweight=weight, style=style, ha=ha, va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=("#FFE1D6" if side == "before" else "#DDF3E4"),
                      edgecolor=tag_color, linewidth=1.1) if changed else None)

draw_band("before", 0, "main  -  claims multicast is the LAN default", "#B3261E")
draw_band("after", 130, "this PR  -  names the real posture", "#0F7B3C")

# ------------------------------------------------------- row 2: the census table
axt = fig.add_subplot(gs[1]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
put(axt, 0.5, 1.045, "Every block in scope that mentions multicast, and why it is (or is not) compliant",
    ha="center", fontsize=12.2, fontweight="600", color="#1A3B5C", transform=axt.transAxes)

COLS = [0.012, 0.335, 0.470, 0.700]
HEAD = ["file", "block", "why compliant (main)", "verdict"]
TOP, LAST = 0.955, 0.075
rows = []
for b in F["before"]["naming"]:
    why = " + ".join(b["why"]) if b["why"] else "-- nothing states the default --"
    rows.append((b["file"], b["label"], why, "ok" if b["why"] else "UNMARKED  ->  fixed by this PR"))
STEP = (TOP - LAST) / (len(rows) - 1)   # last row lands exactly on LAST
assert STEP > 0.030, STEP
for cx, h in zip(COLS, HEAD, strict=True):
    put(axt, cx, TOP + 0.052, h, fontsize=9.9, fontweight="700", color="#1A3B5C", transform=axt.transAxes)
axt.plot([0.008, 0.992], [TOP + 0.030] * 2, color="#1A3B5C", lw=1.2, transform=axt.transAxes)
y = TOP
for f, label, why, verdict in rows:
    bad = verdict.startswith("UNMARKED")
    col = "#B3261E" if bad else "#333333"
    if bad:
        axt.add_patch(Rectangle((0.006, y - 0.019), 0.988, 0.040, facecolor="#FDECEA",
                                edgecolor="none", transform=axt.transAxes, zorder=0))
    put(axt, COLS[0], y, f, fontsize=8.7, color=col, family="monospace", transform=axt.transAxes)
    put(axt, COLS[1], y, label, fontsize=8.7, color=col, family="monospace", transform=axt.transAxes)
    put(axt, COLS[2], y, why, fontsize=8.7, color=col, transform=axt.transAxes)
    put(axt, COLS[3], y, verdict, fontsize=8.7, color=col,
        fontweight="700" if bad else "normal", transform=axt.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

put(axt, 0.012, 0.018,
    f"main: {N_BEFORE_NAMING} blocks name multicast, {N_BEFORE_BAD} unmarked   |   "
    f"this PR: {N_AFTER_NAMING} name it, {N_AFTER_BAD} unmarked   |   "
    "no exclusion list: mesh.core's 'Multicast scouting is ON' warning names the flag in the same block",
    fontsize=9.4, color="#0F7B3C", fontweight="600", transform=axt.transAxes)

# ------------------------------------------------------- row 3: gate
axg = fig.add_subplot(gs[2]); axg.axis("off"); axg.set_xlim(0, 1); axg.set_ylim(0, 1)
axg.add_patch(Rectangle((0.004, 0.02), 0.992, 0.96, facecolor="#F4F7FA", edgecolor="#1A3B5C",
                        linewidth=1.0, transform=axg.transAxes))
GTOP, GLAST = 0.80, 0.16
glines = [
    "pre-fix (source at upstream/main, guard present):  3 failed / 37 passed  -  both stale sites named verbatim in the failure text",
    "full suite @ 83cc5272 + this PR:  28720 passed / 258 skipped / 0 failed  (658s)   |   ruff check + format clean (1191 files)",
    "mypy: 14 errors, ALL in examples/isaac_gs, 0 outside  -  error set byte-identical to a pristine-base worktree (environmental)",
    "camera_offload.py is docstring-only: docstring-stripped AST digest 9e50daa4fcc617eb before AND after.  architecture.svg: 1 of 24 <text> labels changed.",
]
GSTEP = (GTOP - GLAST) / (len(glines) - 1)
assert GSTEP > 0.10, GSTEP
gy = GTOP
for ln in glines:
    put(axg, 0.016, gy, ln, fontsize=9.2, color="#333333", family="monospace", transform=axg.transAxes)
    gy -= GSTEP
assert abs((gy + GSTEP) - GLAST) < 1e-9

# ---------------------------------- layout guards
for ax, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.10, f"axes-fraction text out of frame: y={yv}"
    else:
        lo, hi = sorted(ax.get_ylim())
        assert lo - 0.05 * (hi - lo) <= yv <= hi + 0.05 * (hi - lo), f"data text out of frame: y={yv}"

OUT = ROOT / "_art" / "discovery_posture_prose.png"
fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.array(Image.open(OUT).convert("RGB"))
h, w, _ = im.shape
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {w}x{h}  border clean, {len(placed)} text placements checked")
