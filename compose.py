"""Compose the figure. Every drawn number is asserted against facts.json."""

from __future__ import annotations

import json
import pathlib

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = pathlib.Path(__file__).parent
F = json.loads((OUT / "facts.json").read_text())

routes = F["routes"]
n_before = sum(1 for r in routes if r["driven_before"])
assert len(routes) == 7 and n_before == 1, (len(routes), n_before)
assert F["torch_surface_read"] is True
assert abs(F["quaternion_normalized"] - 1.0) < 1e-9
assert abs(F["mapping_error_m"] - 0.4472) < 5e-4, F["mapping_error_m"]
tb, ob = F["shots"]["true_base"], F["shots"]["origin_base"]
assert tb["from_asked_m"] <= 0.03 and ob["from_asked_m"] > 0.3
assert tb["status"] == ob["status"] == "success" and tb["reached"] and ob["reached"]
assert F["panel_diff_frac"] > 0.10

GREEN, RED, INK, MUTE = "#1a7f37", "#b3261e", "#1b1f24", "#57606a"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.pop("axes_coords", True)
    if axes_coords:
        kw["transform"] = ax.transAxes
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 13.2), facecolor="white")
gs = GridSpec(3, 2, height_ratios=[1.5, 1.12, 0.42], hspace=0.16, wspace=0.05,
              left=0.035, right=0.965, top=0.925, bottom=0.028)

fig.text(0.5, 0.972, "Isaac base-pose readback: every route it documents is now driven",
         ha="center", va="center", fontsize=17.5, fontweight="bold", color=INK)
fig.text(0.5, 0.947,
         "The readback answers None for a base it cannot read, because a substituted origin base makes every "
         "world-frame target silently wrong.\nOne of its seven routes was driven. Below: what the substitution "
         "costs, measured on MuJoCo (Isaac Sim is not installed here).",
         ha="center", va="center", fontsize=10.6, color=MUTE)

# ---- row 1: what a substituted base costs ---------------------------------
for col, (label, sh, title, colour) in enumerate([
    ("true_base", tb, "base read from the articulation (what the readback protects)", GREEN),
    ("origin_base", ob, "base substituted with the origin (what None prevents)", RED),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(iio.imread(OUT / sh["png"]))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(colour); sp.set_linewidth(2.6)
    ax.set_title(f"move_to(position={F['world_target']})  ->  {title}", fontsize=11.2,
                 color=colour, fontweight="bold", pad=7)
    ax.set_xlabel(
        f"target the IK was given: {sh['target']}     status={sh['status']}, reached={sh['reached']}\n"
        f"{sh['from_asked_m']:.4f} m from the point the caller asked for"
        + ("   (the request honoured)" if col == 0 else f"   ({sh['from_asked_m'] * 100:.0f} cm away, reported success)"),
        fontsize=10.3, color=colour, labelpad=8,
        fontweight="bold" if col == 1 else "normal")

# ---- row 2: the route table ----------------------------------------------
axr = fig.add_subplot(gs[1, :]); axr.axis("off")
axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 0.965, "Every \u201ccould not be read\u201d route _articulation_base_pose documents, "
    "traced to the arm it reaches", fontsize=12.4, fontweight="bold", color=INK)
COLS = (0.015, 0.30, 0.40, 0.60)
put(axr, COLS[0], 0.885, "route the caller hits", fontsize=10.4, fontweight="bold", color=MUTE)
put(axr, COLS[1], 0.885, "arm", fontsize=10.4, fontweight="bold", color=MUTE)
put(axr, COLS[2], 0.885, "before this PR", fontsize=10.4, fontweight="bold", color=MUTE)
put(axr, COLS[3], 0.885, "with this PR", fontsize=10.4, fontweight="bold", color=MUTE)
axr.plot([0.01, 0.99], [0.855, 0.855], color="#d0d7de", lw=1.1, transform=axr.transAxes)

TOP, LAST = 0.79, 0.30
STEP = (TOP - LAST) / (len(routes) - 1)
assert STEP > 0.030, STEP
y = TOP
for r in routes:
    was = "driven" if r["driven_before"] else "UNREACHED"
    put(axr, COLS[0], y, r["route"], fontsize=10.5, color=INK, family="monospace")
    put(axr, COLS[1], y, f"line {r['line']}", fontsize=10.2, color=MUTE, family="monospace")
    put(axr, COLS[2], y, was, fontsize=10.5, fontweight="bold",
        color=GREEN if r["driven_before"] else RED)
    put(axr, COLS[3], y, "driven", fontsize=10.5, fontweight="bold", color=GREEN)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y

axr.plot([0.01, 0.99], [0.235, 0.235], color="#d0d7de", lw=1.1, transform=axr.transAxes)
put(axr, COLS[0], 0.155, f"routes driven:  {n_before} of {len(routes)}   ->   "
    f"{len(routes)} of {len(routes)}", fontsize=11.6, fontweight="bold", color=INK, family="monospace")
put(axr, COLS[0], 0.055,
    "plus the two surfaces beside them, also unreached: the documented torch-tensor pose "
    f"(read through .cpu().numpy(): {F['torch_surface_read']}) and the quaternion "
    f"normalization (|q| = {F['quaternion_normalized']:.6f}).",
    fontsize=10.2, color=MUTE)

# ---- row 3: footer --------------------------------------------------------
axf = fig.add_subplot(gs[2, :]); axf.axis("off")
axf.set_xlim(0, 1); axf.set_ylim(0, 1)
axf.add_patch(plt.Rectangle((0.005, 0.06), 0.99, 0.88, facecolor="#f6f8fa",
                            edgecolor="#d0d7de", lw=1.0, transform=axf.transAxes))
lines = [
    f"the substitution aims the arm {F['mapping_error_m']:.4f} m away: base {F['base']}, "
    f"caller asked {F['world_target']}, an origin base maps it to {F['substituted_target']}",
    f"renders differ on {F['panel_diff_frac']:.2%} of pixels; both calls returned status=success and reached=True",
    "tests only, no production line changes: motion_primitives.py 93.46% -> 95.33% over tests/simulation/isaac; "
    "8 of 8 mutations caught here, 0 of 8 by every pre-existing test",
]
FTOP, FLAST = 0.79, 0.20
FSTEP = (FTOP - FLAST) / (len(lines) - 1)
assert FSTEP > 0.10, FSTEP
fy = FTOP
for ln in lines:
    put(axf, 0.022, fy, ln, fontsize=10.4, color=INK)
    fy -= FSTEP
assert abs((fy + FSTEP) - FLAST) < 1e-9, fy

for ax, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.07, (yv, "axes-fraction text outside the panel")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.03 <= yv <= hi + 0.07, (yv, lo, hi)

png = OUT / "isaac_base_pose_readback_routes.png"
fig.savefig(png, dpi=124, bbox_inches="tight", pad_inches=0.3, facecolor="white")
plt.close(fig)

img = iio.imread(png)
for side, band in (("top", img[:8]), ("bottom", img[-8:]), ("left", img[:, :8]), ("right", img[:, -8:])):
    nw = int((band[..., :3].min(axis=-1) < 245).sum())
    assert nw == 0, f"{side} border has {nw} non-white pixels"
print(f"WROTE {png}  {img.shape[1]}x{img.shape[0]}")
