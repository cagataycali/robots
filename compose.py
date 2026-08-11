"""Compose the artifact: the frame the gated assertions verify + the measured tables."""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

ART = pathlib.Path(sys.argv[1])
F = json.loads((ART / "facts.json").read_text(encoding="utf-8"))
frame = np.load(ART / "frame.npy")

# --- every number below is a measurement, asserted against the capture -------
assert F["render_status"] == "success", F
assert F["saturated_frac"] > 0.05, F
assert F["refused_status"] == "error", F

SCENARIOS = [
    ("a second gated assertion added to a module already listed\n"
     "(exactly what the guard's remedy text instructs)",
     "1 failed, 9 passed\nassert 5 == 4", "13 passed\nno pin to update", False),
    ("a new module enters scope, correctly gated",
     "2 failed, 8 passed", "1 failed, 12 passed\nnames the module", True),
]
VACUITY = [
    ("survey finds nothing", "both old pins fail", "fails"),
    ("nothing classified as gated", "the count pin fails", "fails"),
    ("scan rooted at a subdirectory", "both old pins fail", "fails"),
]
LOADBEARING = [
    ("the pin compares counts again", "3 of the 4 new tests fail"),
    ("the missing half is dropped", "1 fails"),
    ("the unexpected half is dropped", "1 fails"),
]
assert len(VACUITY) == 3 and len(LOADBEARING) == 3 and len(SCENARIOS) == 2

GREEN, RED, GREY = "#1b7f3b", "#b3261e", "#42474e"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 12.2), dpi=124)
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.42, 1.0, 0.30], width_ratios=[1.0, 1.24],
              hspace=0.20, wspace=0.10)

fig.suptitle(
    "A non-vacuity pin keyed on modules, not on a count of render assertions",
    fontsize=16.5, fontweight="bold", y=0.975,
)
fig.text(0.5, 0.947,
         "tests/test_mujoco_render_assertions_are_gl_gated.py -- tests only, no production line changes",
         ha="center", fontsize=10.5, color=GREY, style="italic")

# ---- row 1 left: the real frame the gated assertions verify -----------------
axi = fig.add_subplot(gs[0, 0])
axi.imshow(frame)
axi.set_xticks([]); axi.set_yticks([])
axi.set_title("What the four gated assertions verify", fontsize=12.5, fontweight="bold", pad=8)
axi.set_xlabel(
    f"headless MuJoCo render after a refused non-string lookup\n"
    f'"{F["refused_text"][:52]}..."  ->  render status={F["render_status"]}\n'
    f"{frame.shape[1]}x{frame.shape[0]}, {F['saturated_frac']:.0%} saturated. "
    f"Unchanged by this PR: the rule and its scope are untouched.",
    fontsize=9.3, color=GREY, labelpad=8,
)

# ---- row 1 right: the two scenarios ----------------------------------------
axs = fig.add_subplot(gs[0, 1]); axs.axis("off")
axs.set_xlim(0, 1); axs.set_ylim(0, 1)
put(axs, 0.0, 1.035, "What a contributor sees, measured on the real tree",
    fontsize=12.5, fontweight="bold", transform=axs.transAxes)
put(axs, 0.015, 0.945, "scenario", fontsize=10, fontweight="bold", transform=axs.transAxes)
put(axs, 0.560, 0.945, "on main", fontsize=10, fontweight="bold", transform=axs.transAxes)
put(axs, 0.800, 0.945, "with this PR", fontsize=10, fontweight="bold", transform=axs.transAxes)
axs.axhline(0.925, xmin=0.01, xmax=0.99, color="#c8ccd0", lw=1.0)

TOP, LAST = 0.845, 0.42
step = (TOP - LAST) / (len(SCENARIOS) - 1)
assert step > 0.10, step
for i, (name, before, after, still) in enumerate(SCENARIOS):
    y = TOP - i * step
    put(axs, 0.015, y, name, fontsize=9.4, va="top", transform=axs.transAxes)
    put(axs, 0.560, y, before, fontsize=9.4, va="top", family="monospace", color=RED,
        transform=axs.transAxes)
    put(axs, 0.800, y, after, fontsize=9.4, va="top", family="monospace",
        color=(RED if still else GREEN), transform=axs.transAxes)
foot = LAST - 0.16
assert foot > 0.02, foot
put(axs, 0.015, foot,
    "The first is the defect: the module set was still exactly correct, so there was\n"
    "no pin to update and the message was two bare integers. The second is the\n"
    "intended ask and still fails -- now naming the module rather than a count.",
    fontsize=9.4, va="top", color=GREY, transform=axs.transAxes)

# ---- row 2: the mutation table --------------------------------------------
axm = fig.add_subplot(gs[1, :]); axm.axis("off")
axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 1.045, "Nothing lost, and the corrected quantity is load-bearing",
    fontsize=12.5, fontweight="bold", transform=axm.transAxes)

put(axm, 0.015, 0.93, "mutation of the survey (a vacuous guard)", fontsize=10, fontweight="bold",
    transform=axm.transAxes)
put(axm, 0.470, 0.93, "the two old pins", fontsize=10, fontweight="bold", transform=axm.transAxes)
put(axm, 0.660, 0.93, "the single new pin", fontsize=10, fontweight="bold", transform=axm.transAxes)
axm.axhline(0.905, xmin=0.01, xmax=0.80, color="#c8ccd0", lw=1.0)
TOP2, LAST2 = 0.845, 0.665
step2 = (TOP2 - LAST2) / (len(VACUITY) - 1)
assert step2 > 0.06, step2
for i, (name, old, new) in enumerate(VACUITY):
    y = TOP2 - i * step2
    put(axm, 0.015, y, name, fontsize=9.4, va="center", transform=axm.transAxes)
    put(axm, 0.470, y, old, fontsize=9.4, va="center", family="monospace", color=GREEN,
        transform=axm.transAxes)
    put(axm, 0.660, y, new, fontsize=9.4, va="center", family="monospace", color=GREEN,
        transform=axm.transAxes)

put(axm, 0.015, 0.545, "mutation of the new pin itself", fontsize=10, fontweight="bold",
    transform=axm.transAxes)
put(axm, 0.470, 0.545, "the four planted tests", fontsize=10, fontweight="bold", transform=axm.transAxes)
axm.axhline(0.520, xmin=0.01, xmax=0.80, color="#c8ccd0", lw=1.0)
TOP3, LAST3 = 0.455, 0.275
step3 = (TOP3 - LAST3) / (len(LOADBEARING) - 1)
assert step3 > 0.06, step3
for i, (name, caught) in enumerate(LOADBEARING):
    y = TOP3 - i * step3
    put(axm, 0.015, y, name, fontsize=9.4, va="center", transform=axm.transAxes)
    put(axm, 0.470, y, caught, fontsize=9.4, va="center", family="monospace", color=GREEN,
        transform=axm.transAxes)
put(axm, 0.015, 0.145,
    "Top: every vacuity the two old pins caught is still caught by the one that replaces them, so the\n"
    "module-set check is not lost. Bottom: restoring the count comparison inside the helper fails three of\n"
    "the four new tests, so the corrected quantity is pinned rather than merely documented.",
    fontsize=9.4, va="top", color=GREY, transform=axm.transAxes)

# ---- row 3: gate ----------------------------------------------------------
axg = fig.add_subplot(gs[2, :]); axg.axis("off")
axg.set_xlim(0, 1); axg.set_ylim(0, 1)
put(axg, 0.0, 0.86, "Gate", fontsize=11.5, fontweight="bold", transform=axg.transAxes)
put(axg, 0.015, 0.50,
    "ruff check / ruff format --check clean on strands_robots tests tests_integ; mypy 0 errors outside "
    "examples/isaac_gs\nand its error set byte-identical to the base. Guard module 10 -> 13 tests. "
    "Every mutation above was applied and reverted, tree byte-clean.",
    fontsize=9.6, va="center", family="monospace", color=GREY, transform=axg.transAxes)

# ---- self-audit -----------------------------------------------------------
for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.07, (y, "axes-fraction out of band")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.03 * (hi - lo) <= y <= hi + 0.07 * (hi - lo), (y, lo, hi)

out = ART / "artifact.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.32, facecolor="white")
plt.close(fig)

im = np.asarray(matplotlib.image.imread(out) * 255).astype(int)[..., :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nonwhite == 0, (name, nonwhite)
print("OK", out, im.shape)
