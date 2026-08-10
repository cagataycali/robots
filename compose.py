"""Render the measured figure. Every cell comes from /tmp/art-facts.json."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.loads(pathlib.Path("/tmp/art-facts.json").read_text())
prov, muts, counts = F["providers"], F["mutations"], F["counts"]

GREEN, RED, GREY, INK = "#1b7f4b", "#b3261e", "#6b7280", "#111827"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


n_missing_before = sum(1 for v in prov.values() if not v["before"])
n_missing_after = sum(1 for v in prov.values() if not v["after"])
caught_new = sum(1 for m in muts if m["new_failed"] > 0)
caught_old = sum(1 for m in muts if m["old_failed"] > 0)
assert (n_missing_before, n_missing_after) == (7, 0), (n_missing_before, n_missing_after)
assert (caught_new, caught_old) == (len(muts), 1), (caught_new, caught_old)
assert counts["before_tests"] == 35 and counts["after_tests"] == 64, counts

fig = plt.figure(figsize=(15.4, 11.2), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.02, 1.28, 0.30], hspace=0.20,
                      left=0.035, right=0.978, top=0.925, bottom=0.028)

fig.suptitle(
    "robot_state_keys: the refusal the AST classifier could not see",
    fontsize=17, fontweight="bold", y=0.978, color=INK,
)
fig.text(
    0.5, 0.948,
    "set_robot_state_keys names the actuators a policy emits actions for. "
    "Nine surfaces validate the list through one shared domain;\n"
    "a classifier pinned that each CALLS it, and only MockPolicy and RemotePolicy were ever driven behaviourally.",
    ha="center", va="top", fontsize=10.4, color=GREY,
)

# ---------------- panel 1: coverage of the raise line, per provider ----------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.045, "1  Did the provider's refusal ever fire?  (coverage of its `raise ValueError(error)` line, whole suite)",
    transform=ax.transAxes, fontsize=12.4, fontweight="bold", color=INK, va="bottom")
cols = (0.015, 0.335, 0.475, 0.660, 0.845)
hdr = ("provider surface", "raise line", "on main", "with this change", "")
TOP, LAST = 0.90, 0.10
rows = list(prov.items())
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.045, step
for x, h in zip(cols, hdr, strict=True):
    put(ax, x, 0.985, h, transform=ax.transAxes, fontsize=10.1, fontweight="bold", color=GREY)
y = TOP
for label, v in rows:
    put(ax, cols[0], y, f"{v['path'].replace('strands_robots/policies/', '')}::set_robot_state_keys",
        transform=ax.transAxes, fontsize=10.4, family="monospace", color=INK)
    put(ax, cols[1], y, f"L{v['line']}", transform=ax.transAxes, fontsize=10.4, family="monospace", color=GREY)
    put(ax, cols[2], y, "never fired" if not v["before"] else "fired",
        transform=ax.transAxes, fontsize=10.6, fontweight="bold", color=RED if not v["before"] else GREEN)
    put(ax, cols[3], y, "fired" if v["after"] else "never fired",
        transform=ax.transAxes, fontsize=10.6, fontweight="bold", color=GREEN if v["after"] else RED)
    put(ax, cols[4], y, "constructed and driven directly", transform=ax.transAxes, fontsize=9.6, color=GREY)
    y -= step
assert abs((y + step) - LAST) < 1e-9, y
put(ax, cols[0], LAST - 0.075,
    f"refusals never exercised: {n_missing_before} of {len(prov)}  ->  {n_missing_after} of {len(prov)}",
    transform=ax.transAxes, fontsize=11.0, fontweight="bold", color=INK)

# ---------------- panel 2: mutation matrix ----------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.035, "2  Which suite notices a broken refusal?  (each row: the guard mutated, both halves re-run)",
    transform=ax2.transAxes, fontsize=12.4, fontweight="bold", color=INK, va="bottom")
c2 = (0.015, 0.360, 0.495, 0.700)
for x, h in zip(c2, ("mutation applied to the provider", "provider", "the 35 pre-existing tests", "the 29 tests added here"), strict=True):
    put(ax2, x, 0.975, h, transform=ax2.transAxes, fontsize=10.1, fontweight="bold", color=GREY)
TOP2, LAST2 = 0.885, 0.145
step2 = (TOP2 - LAST2) / (len(muts) - 1)
assert step2 > 0.045, step2
y = TOP2
for m in muts:
    put(ax2, c2[0], y, m["style"], transform=ax2.transAxes, fontsize=10.4, family="monospace", color=INK)
    put(ax2, c2[1], y, m["provider"], transform=ax2.transAxes, fontsize=10.4, family="monospace", color=GREY)
    old_ok, new_ok = m["old_failed"] > 0, m["new_failed"] > 0
    put(ax2, c2[2], y,
        f"caught ({m['old_failed']} failed)" if old_ok else f"BLIND - all {m['old_passed']} pass",
        transform=ax2.transAxes, fontsize=10.5, fontweight="bold", color=GREEN if old_ok else RED)
    put(ax2, c2[3], y, f"caught ({m['new_failed']} failed)" if new_ok else "blind",
        transform=ax2.transAxes, fontsize=10.5, fontweight="bold", color=GREEN if new_ok else RED)
    y -= step2
assert abs((y + step2) - LAST2) < 1e-9, y
put(ax2, c2[0], LAST2 - 0.070,
    f"mutations caught: pre-existing {caught_old} of {len(muts)}   |   added here {caught_new} of {len(muts)}",
    transform=ax2.transAxes, fontsize=11.0, fontweight="bold", color=INK)
put(ax2, c2[0], LAST2 - 0.125,
    "The classifier catches only a guard deleted outright. A body that keeps the name_list_error(...) call and drops the raise,\n"
    "or re-words the message locally, satisfies it unchanged - which is why the seven refusals above had never fired.",
    transform=ax2.transAxes, fontsize=9.9, color=GREY, va="top")

# ---------------- panel 3: footer ----------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
foot = [
    f"module: {counts['before_tests']} tests -> {counts['after_tests']} tests   (+{counts['after_tests'] - counts['before_tests']}, no production line changed)",
    "the table of surfaces is derived from the classifier's own set, so a tenth provider fails a test rather than joining the structurally-only half",
    "every value above was measured in this run; tests-only, so no policy, simulation, rendering, recording or asset behaviour changes",
]
FT, FL = 0.80, 0.14
fstep = (FT - FL) / (len(foot) - 1)
assert fstep > 0.030, fstep
yy = FT
for line in foot:
    put(ax3, 0.015, yy, line, transform=ax3.transAxes, fontsize=10.0, color=INK if yy == FT else GREY,
        family="monospace" if yy == FT else "sans-serif")
    yy -= fstep

for a, yv, is_axes in placed:
    if is_axes:
        assert -0.14 <= yv <= 1.10, f"axes-coord text out of band: {yv}"
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, f"data-coord text out of band: {yv}"

out = pathlib.Path("/tmp/state_key_refusal_coverage.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print("OK", out, Image.open(out).size)
