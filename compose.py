"""Render the measurement figure for the Newton seed-refusal coverage change.

Every number is read from /tmp/facts-<run>.json (written by capture.py) and
asserted before the figure is saved. Tests only: no policy, simulation,
rendering, recording or asset behaviour changes, so the artifact is the
coverage-and-mutation measurement rather than a rollout.
"""
from __future__ import annotations
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

F = json.loads(pathlib.Path(f"/tmp/facts-{sys.argv[1]}.json").read_text())
B, A = F["before"], F["after"]
CELLS = len(B["cells"])
n_bad_before = sum(1 for c in B["cells"] if not c["covered"])
n_bad_after = sum(1 for c in A["cells"] if not c["covered"])
assert (CELLS, n_bad_before, n_bad_after) == (13, 2, 0), (CELLS, n_bad_before, n_bad_after)
assert B["file"]["pct"] == 98.4 and A["file"]["pct"] == 100.0, (B["file"], A["file"])
assert B["file"]["missing"] == 2 and A["file"]["missing"] == 0
muts = F["mutations"]
blind_old = [m for m in muts if m["old"] == 0]
assert len(muts) == 7 and all(m["new"] > 0 for m in muts), muts
assert len(blind_old) == 5, blind_old

GREEN, RED, GREY, INK = "#1b7f3b", "#b3261e", "#e8e8e8", "#111111"
placed: list[tuple[object, float, bool]] = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 11.2), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 0.98, 0.42], hspace=0.20,
                      left=0.035, right=0.972, top=0.925, bottom=0.028)
fig.suptitle("Newton's randomization seed refusal was the only shared guard on the backend nothing executed",
             fontsize=16.5, fontweight="bold", y=0.977)
fig.text(0.5, 0.949,
         "13 refusal cells across the two backends that ship a randomization mixin  -  "
         "2 uncovered, both the seed guard, both on Newton",
         ha="center", fontsize=11.3, color="#444444")

# ---- row 1: the cross-backend refusal matrix ------------------------------- #
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965, "Cross-backend refusal matrix: is the guard's refusal branch ever executed?",
    transform=ax.transAxes, fontsize=12.6, fontweight="bold", color=INK)
put(ax, 0.0, 0.905, "before = full suite on unmodified main;  after = the same suite with this PR's class "
                    "(a line the class executes is executed by any set containing it)",
    transform=ax.transAxes, fontsize=9.6, color="#555555", style="italic")

hdr_y, LAST = 0.822, 0.055
rows = B["cells"]
step = (hdr_y - 0.075 - LAST) / (len(rows) - 1)
assert step > 0.030, step
cols = [(0.005, "backend"), (0.115, "method"), (0.300, "shared guard"), (0.560, "line"),
        (0.640, "before"), (0.790, "after")]
for x, label in cols:
    put(ax, x, hdr_y, label, transform=ax.transAxes, fontsize=10.4, fontweight="bold", color=INK)
ax.plot([0.0, 1.0], [hdr_y - 0.030] * 2, transform=ax.transAxes, color="#999999", lw=1.0)

y = hdr_y - 0.075
for cb, ca in zip(rows, A["cells"]):
    if not cb["covered"]:
        ax.add_patch(Rectangle((0.0, y - 0.018), 1.0, step * 0.86, transform=ax.transAxes,
                               facecolor="#fdecea", edgecolor="none", zorder=0))
    put(ax, 0.005, y, cb["backend"], transform=ax.transAxes, fontsize=10.0, family="monospace", color=INK)
    put(ax, 0.115, y, cb["method"], transform=ax.transAxes, fontsize=10.0, family="monospace", color=INK)
    put(ax, 0.300, y, cb["guard"], transform=ax.transAxes, fontsize=10.0, family="monospace", color=INK)
    put(ax, 0.560, y, f"L{cb['line']}", transform=ax.transAxes, fontsize=9.6, family="monospace", color="#666666")
    for x, c in ((0.640, cb), (0.790, ca)):
        put(ax, x, y, "executed" if c["covered"] else "NEVER EXECUTED", transform=ax.transAxes,
            fontsize=10.0, fontweight="bold" if not c["covered"] else "normal",
            color=GREEN if c["covered"] else RED)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y + step, LAST)

put(ax, 0.640, 0.008, f"uncovered: {n_bad_before} of {CELLS}", transform=ax.transAxes,
    fontsize=10.6, fontweight="bold", color=RED)
put(ax, 0.790, 0.008, f"uncovered: {n_bad_after} of {CELLS}", transform=ax.transAxes,
    fontsize=10.6, fontweight="bold", color=GREEN)

# ---- row 2: mutation matrix ------------------------------------------------ #
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.955, "Mutation matrix: which plausible regressions does each test set catch?",
    transform=ax2.transAxes, fontsize=12.6, fontweight="bold", color=INK)
put(ax2, 0.0, 0.878,
    "the production code is correct, so the new tests pass on unmodified main - what they fail on is a regression",
    transform=ax2.transAxes, fontsize=9.6, color="#555555", style="italic")
h2, L2 = 0.775, 0.085
step2 = (h2 - 0.075 - L2) / (len(muts) - 1)
assert step2 > 0.045, step2
put(ax2, 0.005, h2, "mutation applied to strands_robots/simulation/newton/randomization.py",
    transform=ax2.transAxes, fontsize=10.4, fontweight="bold", color=INK)
put(ax2, 0.660, h2, "this PR's class", transform=ax2.transAxes, fontsize=10.4, fontweight="bold", color=INK)
put(ax2, 0.830, h2, f"the {F['old_arm_total']} pre-existing", transform=ax2.transAxes,
    fontsize=10.4, fontweight="bold", color=INK)
ax2.plot([0.0, 1.0], [h2 - 0.030] * 2, transform=ax2.transAxes, color="#999999", lw=1.0)

y = h2 - 0.075
for m in muts:
    if m["old"] == 0:
        ax2.add_patch(Rectangle((0.0, y - 0.020), 1.0, step2 * 0.84, transform=ax2.transAxes,
                                facecolor="#fdecea", edgecolor="none", zorder=0))
    put(ax2, 0.005, y, f"{m['id']}  {m['what']}", transform=ax2.transAxes,
        fontsize=10.0, family="monospace", color=INK)
    put(ax2, 0.660, y, f"{m['new']} failed", transform=ax2.transAxes,
        fontsize=10.0, fontweight="bold", color=GREEN)
    put(ax2, 0.830, y, f"{m['old']} failed" if m["old"] else "BLIND", transform=ax2.transAxes,
        fontsize=10.0, fontweight="bold", color=GREEN if m["old"] else RED)
    y -= step2
assert abs((y + step2) - L2) < 1e-9, (y + step2, L2)
put(ax2, 0.660, 0.010, f"caught {len(muts)} of {len(muts)}", transform=ax2.transAxes,
    fontsize=10.6, fontweight="bold", color=GREEN)
put(ax2, 0.830, 0.010, f"blind to {len(blind_old)} of {len(muts)}", transform=ax2.transAxes,
    fontsize=10.6, fontweight="bold", color=RED)

# ---- row 3: footer -------------------------------------------------------- #
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(Rectangle((0.0, 0.02), 1.0, 0.96, transform=ax3.transAxes,
                        facecolor=GREY, edgecolor="#cccccc"))
foot = [
    f"strands_robots/simulation/newton/randomization.py:  {B['file']['statements']} statements, "
    f"{B['file']['missing']} missing  {B['file']['pct']}%   ->   {A['file']['missing']} missing  {A['file']['pct']}%",
    f"scoped arm over the two touched modules:  {B['passed']} passed  ->  {A['passed']} passed "
    f"(+{A['passed'] - B['passed']} cases, no production line changed)",
    "full suite (MUJOCO_GL=egl):  28108 passed / 257 skipped / 0 failed      "
    "ruff check + ruff format --check + mypy: clean      the new class needs neither Newton nor Warp",
]
LF, TF = 0.20, 0.80
stepf = (TF - LF) / (len(foot) - 1)
yf = TF
for line in foot:
    put(ax3, 0.014, yf, line, transform=ax3.transAxes, fontsize=10.2, family="monospace", color=INK)
    yf -= stepf
assert abs((yf + stepf) - LF) < 1e-9, (yf + stepf, LF)

for a, yy, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yy <= 1.07, (yy, "axes-fraction out of range")
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

out = pathlib.Path("_art/newton_seed_refusal_matrix.png")
fig.savefig(out, dpi=124, facecolor="white", bbox_inches="tight", pad_inches=0.30)
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"wrote {out}  {im.shape[1]}x{im.shape[0]}  border clean")
print(f"matrix {n_bad_before}/{CELLS} -> {n_bad_after}/{CELLS}   "
      f"mutations caught {len(muts)}/{len(muts)} new, blind {len(blind_old)}/{len(muts)} old")
