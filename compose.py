"""Compose the measured figure. Every cell is read from the capture dump."""

from __future__ import annotations

import json, os, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

RUN = os.environ["GITHUB_RUN_ID"]
D = json.loads(pathlib.Path(f"/tmp/art-{RUN}.json").read_text())
SURF, MUT, COV = D["surfaces"], D["mutations"], D["coverage"]

# --- self-audit on the measured data -----------------------------------------
assert COV["before"]["line_541"] == "missing" and COV["after"]["line_541"] == "covered"
assert COV["before"]["missing"] - COV["after"]["missing"] == 1
assert len(SURF) == 4 and len(MUT) == 3
for m in MUT:
    assert m["existing_failed"] == 0 and m["existing_passed"] == 539, m
    assert m["new_failed"] > 0, m
RAISERS = [k for k, v in SURF.items() if v["channel"].startswith("raises")]
assert len(RAISERS) == 3
for k in RAISERS:
    assert set(SURF[k]["verdicts"].values()) == {"refused"}, (k, SURF[k]["verdicts"])
VALUES = list(SURF[RAISERS[0]]["verdicts"])

placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

GREEN, RED, GREY, BLUE = "#1a7f37", "#b3261e", "#6e7781", "#0b5fa5"
fig = plt.figure(figsize=(15.6, 11.0), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.32, 0.86, 0.50], hspace=0.30,
                      left=0.045, right=0.972, top=0.915, bottom=0.035)

fig.suptitle("LIBERO max_steps: the domain's rejecting half, and what verified it", fontsize=16, y=0.972)
fig.text(0.5, 0.940, "Tests only -- no production line changes. The verdicts below are identical on both "
         "trees; what changes is whether anything exercised them.",
         ha="center", fontsize=10.2, color=GREY, style="italic")

# ---- row 1: per-surface verdict grid ----------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.02, "1. What each public surface does with an unusable horizon (measured)",
    fontsize=12.6, fontweight="bold", transform=ax.transAxes)
x0, xw = 0.235, 0.073
for j, v in enumerate(VALUES):
    put(ax, x0 + j * xw + xw / 2, 0.90, f"max_steps=\n{v}", ha="center", va="center",
        fontsize=8.6, family="monospace", color=BLUE, transform=ax.transAxes)
put(ax, 0.895, 0.90, "channel", ha="center", va="center", fontsize=9.4, fontweight="bold",
    transform=ax.transAxes)
rows = list(SURF.items())
TOP, LAST = 0.735, 0.20
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.10, step
for i, (name, info) in enumerate(rows):
    y = TOP - i * step
    put(ax, 0.0, y, name, fontsize=10.6, family="monospace", va="center", transform=ax.transAxes)
    for j, v in enumerate(VALUES):
        verdict = info["verdicts"][v]
        short = "refused" if verdict == "refused" else verdict
        colour = GREEN if verdict == "refused" else "#8a6d00"
        ax.add_patch(Rectangle((x0 + j * xw + 0.004, y - 0.052), xw - 0.008, 0.104,
                               transform=ax.transAxes, facecolor=colour, alpha=0.15, lw=0))
        put(ax, x0 + j * xw + xw / 2, y, short.replace(", ", ",\n"), ha="center", va="center",
            fontsize=7.6, family="monospace", color=colour, transform=ax.transAxes)
    put(ax, 0.895, y, info["channel"], ha="center", va="center", fontsize=8.8,
        color=(GREEN if info["channel"].startswith("raises") else "#8a6d00"), transform=ax.transAxes)
ctrl = SURF["load_libero_suite(...)"]["control"]
put(ax, 0.0, 0.055, f"Control -- a usable horizon is untouched:  {ctrl}   |   "
    f"the three adapter surfaces store 250 verbatim.", fontsize=9.6, color=GREY,
    family="monospace", transform=ax.transAxes)

# ---- row 2: mutation matrix -------------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.06, "2. Would a regression be caught? (each mutation applied to the guard, then both suites run)",
    fontsize=12.6, fontweight="bold", transform=ax2.transAxes)
put(ax2, 0.44, 0.86, "these new tests", ha="center", fontsize=10.4, fontweight="bold", transform=ax2.transAxes)
put(ax2, 0.76, 0.86, "the 539 pre-existing libero tests", ha="center", fontsize=10.4, fontweight="bold",
    transform=ax2.transAxes)
TOP2, LAST2 = 0.62, 0.18
step2 = (TOP2 - LAST2) / (len(MUT) - 1)
for i, m in enumerate(MUT):
    y = TOP2 - i * step2
    put(ax2, 0.0, y, f"M{i + 1}  {m['label']}", fontsize=10.4, va="center", transform=ax2.transAxes)
    ax2.add_patch(Rectangle((0.325, y - 0.062), 0.23, 0.124, transform=ax2.transAxes,
                            facecolor=GREEN, alpha=0.16, lw=0))
    put(ax2, 0.44, y, f"CAUGHT  ({m['new_failed']} failed)", ha="center", va="center",
        fontsize=9.8, color=GREEN, fontweight="bold", family="monospace", transform=ax2.transAxes)
    ax2.add_patch(Rectangle((0.635, y - 0.062), 0.25, 0.124, transform=ax2.transAxes,
                            facecolor=RED, alpha=0.16, lw=0))
    put(ax2, 0.76, y, f"BLIND  (all {m['existing_passed']} pass)", ha="center", va="center",
        fontsize=9.8, color=RED, fontweight="bold", family="monospace", transform=ax2.transAxes)
put(ax2, 0.0, 0.035, "M2 is the shape a structural \"the guard is called\" pin cannot see: the call stays, only "
    "the refusal is discarded.", fontsize=9.6, color=GREY, style="italic", transform=ax2.transAxes)

# ---- row 3: coverage ---------------------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 1.02, "3. Coverage of the refusal branch (adapter.py, tests/benchmarks/libero)",
    fontsize=12.6, fontweight="bold", transform=ax3.transAxes)
lines = [
    ("before", f"adapter.py:541  raise ValueError(error)   ->  {COV['before']['line_541'].upper():8}"
               f"   missing={COV['before']['missing']}   {COV['before']['pct']}%", RED),
    ("after", f"adapter.py:541  raise ValueError(error)   ->  {COV['after']['line_541'].upper():8}"
              f"   missing={COV['after']['missing']}   {COV['after']['pct']}%", GREEN),
]
for i, (_arm, text, colour) in enumerate(lines):
    put(ax3, 0.0, 0.70 - i * 0.30, text, fontsize=10.6, family="monospace", color=colour,
        transform=ax3.transAxes)
put(ax3, 0.0, 0.06, "tests/benchmarks/libero:  539 -> 592 passed (+53).  No policy, simulation, rendering, "
    "recording or asset behaviour changes, so the artifact is this measurement rather than a rollout.",
    fontsize=9.8, color=GREY, transform=ax3.transAxes)

for a, y, axes_coords in placed:
    if axes_coords:
        assert -0.05 <= y <= 1.10, (y, "axes-fraction text outside the panel")
    else:
        lo_, hi_ = a.get_ylim()
        assert lo_ - 0.03 <= y <= hi_ + 0.07, (y, lo_, hi_)

out = pathlib.Path(f"/tmp/max_steps_domain-{RUN}.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(plt.imread(out)[:, :, :3] * 255).astype(int)
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=-1) > 12).sum())
    assert n == 0, (side, n)
print("OK", out, im.shape)
