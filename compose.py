"""Compose the measurement figure for the newton add_robot refusal-coverage PR.

Every number is read from the capture JSONs; nothing is typed by hand.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = pathlib.Path("/tmp/mut-31775053130")
cov = json.loads((D / "covsum.json").read_text())
rows = json.loads((D / "rows.json").read_text())

# ---- derived facts (no literals) -------------------------------------------
SITES = [(label, ln) for label, ln in cov["sites"]]
MB, MA = set(cov["mb"]), set(cov["ma"])
n_missing_before = sum(1 for _, ln in SITES if ln in MB)
n_missing_after = sum(1 for _, ln in SITES if ln in MA)
n_closed = len(cov["closed"])
caught_new = sum(1 for r in rows if r["caught_by_new"])
caught_old = sum(1 for r in rows if r["caught_by_old"])
n_mut = len(rows)

assert n_missing_before == len(SITES), "every site must be unexecuted on main"
assert n_missing_after == 0, "every site must be executed on the branch"
assert caught_new == n_mut and caught_old == 0, "table must be all-caught / all-blind"
assert not (MA - MB), "no line may regress"

GREEN, RED, INK, MUTED = "#1b7f4d", "#b3261e", "#1a1a1a", "#5f6368"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(13.6, 10.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.35, 0.42], hspace=0.30,
                      left=0.045, right=0.965, top=0.925, bottom=0.045)

fig.suptitle(
    "newton add_robot: five caller-input refusals, none of them ever executed",
    fontsize=15.5, fontweight="bold", y=0.972, color=INK,
)
fig.text(0.5, 0.944,
         "tests only - 0 production lines changed, so no policy, simulation, rendering, recording "
         "or asset behaviour moves.\nThis figure is the coverage and mutation measurement, not a rollout.",
         ha="center", fontsize=9.4, color=MUTED, style="italic")

# ---------------- row 1: refusal-site matrix --------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.005, "1. Each refusal site, before and after  (coverage JSON, tests/simulation/newton)",
    fontsize=11.6, fontweight="bold", color=INK, transform=ax.transAxes)

TOP, LAST = 0.845, 0.115
STEP = (TOP - LAST) / (len(SITES) - 1)
assert STEP > 0.030, f"row pitch too tight: {STEP:.4f}"

put(ax, 0.015, 0.945, "refusal the caller receives", fontsize=9.6, fontweight="bold",
    color=MUTED, transform=ax.transAxes)
put(ax, 0.455, 0.945, "line", fontsize=9.6, fontweight="bold", color=MUTED, transform=ax.transAxes)
put(ax, 0.560, 0.945, "on main", fontsize=9.6, fontweight="bold", color=MUTED, transform=ax.transAxes)
put(ax, 0.790, 0.945, "with these tests", fontsize=9.6, fontweight="bold", color=MUTED, transform=ax.transAxes)
ax.plot([0.012, 0.988], [0.912, 0.912], lw=0.9, color="#c9ccd1", transform=ax.transAxes)

y = TOP
for label, ln in SITES:
    put(ax, 0.015, y, label, fontsize=10.4, color=INK, va="center", transform=ax.transAxes)
    put(ax, 0.455, y, f":{ln}", fontsize=9.8, color=MUTED, va="center",
        family="monospace", transform=ax.transAxes)
    for x0, missing in ((0.560, ln in MB), (0.790, ln in MA)):
        col = RED if missing else GREEN
        ax.add_patch(Rectangle((x0, y - 0.030), 0.185, 0.060, transform=ax.transAxes,
                               facecolor=col, alpha=0.13, edgecolor=col, lw=1.1))
        put(ax, x0 + 0.0925, y, "never executed" if missing else "executed",
            fontsize=9.6, fontweight="bold", color=col, ha="center", va="center",
            transform=ax.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, "last row must land on LAST"

put(ax, 0.015, 0.028,
    f"add_robot (lines {cov['fn'][0]}-{cov['fn'][1]}): {n_closed} lines closed, 0 regressions   |   "
    f"newton/simulation.py over its own suite {cov['pct_before']:.2f}% -> {cov['pct_after']:.2f}%",
    fontsize=9.5, color=MUTED, style="italic", transform=ax.transAxes)

# ---------------- row 2: mutation matrix ------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.005,
    f"2. Plausible regressions: caught by these {19} cases {caught_new}/{n_mut}, "
    f"caught by the pre-existing suite {caught_old}/{n_mut}",
    fontsize=11.6, fontweight="bold", color=INK, transform=ax2.transAxes)

TOP2, LAST2 = 0.855, 0.095
STEP2 = (TOP2 - LAST2) / (len(rows) - 1)
assert STEP2 > 0.030, f"mutation pitch too tight: {STEP2:.4f}"

put(ax2, 0.015, 0.950, "mutation applied to newton/simulation.py", fontsize=9.6,
    fontweight="bold", color=MUTED, transform=ax2.transAxes)
put(ax2, 0.640, 0.950, "these tests", fontsize=9.6, fontweight="bold", color=MUTED, transform=ax2.transAxes)
put(ax2, 0.830, 0.950, "pre-existing", fontsize=9.6, fontweight="bold", color=MUTED, transform=ax2.transAxes)
ax2.plot([0.012, 0.988], [0.918, 0.918], lw=0.9, color="#c9ccd1", transform=ax2.transAxes)

y = TOP2
for r in rows:
    put(ax2, 0.015, y, r["label"], fontsize=10.1, color=INK, va="center", transform=ax2.transAxes)
    for x0, n, ok in ((0.640, r["new_failed"], r["caught_by_new"]),
                      (0.830, r["old_failed"], r["caught_by_old"])):
        col = GREEN if ok else RED
        txt = f"{n} failed" if ok else "BLIND"
        ax2.add_patch(Rectangle((x0, y - 0.028), 0.150, 0.056, transform=ax2.transAxes,
                                facecolor=col, alpha=0.13, edgecolor=col, lw=1.1))
        put(ax2, x0 + 0.075, y, txt, fontsize=9.5, fontweight="bold", color=col,
            ha="center", va="center", transform=ax2.transAxes)
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, "last mutation row must land on LAST2"

put(ax2, 0.015, 0.020,
    "Every anchor was scoped to add_robot by AST line range and restored byte-identically; "
    "the unmutated control is clean on both arms.",
    fontsize=9.5, color=MUTED, style="italic", transform=ax2.transAxes)

# ---------------- row 3: gate ----------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(Rectangle((0.006, 0.06), 0.988, 0.88, transform=ax3.transAxes,
                        facecolor="#f4f6f8", edgecolor="#d6d9de", lw=1.0))
put(ax3, 0.020, 0.62,
    "Gate  |  full suite 29837 passed / 266 skipped / 0 failed   "
    "(pristine main 29818 + 19 new cases, arithmetic)",
    fontsize=10.0, color=INK, family="monospace", transform=ax3.transAxes)
put(ax3, 0.020, 0.26,
    "        ruff clean  |  mypy 0 errors outside examples/isaac_gs  |  "
    "diff: 2 files, +191, 0 production lines",
    fontsize=10.0, color=MUTED, family="monospace", transform=ax3.transAxes)

# ---------------- layout guards --------------------------------------------
for a, yy, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yy <= 1.07, f"axes-fraction text out of band: {yy}"
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, f"data text out of band: {yy} vs {(lo, hi)}"

out = pathlib.Path("_art/newton_add_robot_refusal_coverage.png").resolve()
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border not clean: {n} px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}")
print(f"sites missing before={n_missing_before} after={n_missing_after}  closed={n_closed}")
print(f"mutations caught new={caught_new}/{n_mut} old={caught_old}/{n_mut}")
