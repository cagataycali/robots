"""Compose the measured figure; every cell read from the capture JSON."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
F = json.loads(pathlib.Path(f"/tmp/art-{ROOT.name}.json").read_text())
assert F["tree"] == str(ROOT), f"capture ran against {F['tree']}"

# --- self-audit: every claim the figure makes -----------------------------
eps = F["entry_points"]
assert len(eps) == 3
assert [e["driven_before"] for e in eps] == [True, False, True], "the middle row is the hole"
assert F["stub"]["status"] == "success" and F["stub"]["validates"] is False
assert F["ordering_refused"] == {"status": "error", "live_stopped": False}
assert F["ordering_accepted"] == {"status": "success", "live_stopped": True, "replaced": True}
muts = F["mutations"]
assert len(muts) == 5
blind = [m for m in muts if m["old"] == 0]
assert len(blind) == 4, f"expected 4 blind rows, got {len(blind)}"
assert all(m["new"] > 0 for m in muts), "every mutation must be caught by the new class"
assert F["coverage"]["missing_before"] - F["coverage"]["missing_after"] == 1

GREEN, RED, AMBER = "#1b7f3b", "#b3261e", "#8a6100"
placed: list[tuple[object, float, bool]] = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.pop("axes_coords", True)
    if axes_coords:
        kw["transform"] = ax.transAxes
        placed.append((ax, y, True))
    t = ax.text(x, y, s, **kw)
    return t

fig = plt.figure(figsize=(15.4, 11.2), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 0.62], hspace=0.24,
                      left=0.035, right=0.972, top=0.925, bottom=0.035)
fig.suptitle(
    "Robot.start_teleop_publish: the third entry point the rate-domain suite names, and never drove",
    fontsize=15.5, fontweight="bold", y=0.972,
)

# ---------------- row 1: the three entry points --------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.035, "One rate domain, three entry points -- what each does with hz=0",
    fontsize=12.5, fontweight="bold")
cols = [0.005, 0.30, 0.755, 0.895]
put(ax, cols[0], 0.895, "entry point", fontsize=10.5, fontweight="bold", color="#333")
put(ax, cols[1], 0.895, "outcome for hz=0", fontsize=10.5, fontweight="bold", color="#333")
put(ax, cols[2], 0.895, "driven before", fontsize=10.5, fontweight="bold", color="#333")
put(ax, cols[3], 0.895, "role of the guard", fontsize=10.5, fontweight="bold", color="#333")
ax.plot([0, 1], [0.865, 0.865], color="#999", lw=1.0, transform=ax.transAxes, clip_on=False)

TOP1, LAST1 = 0.735, 0.315
step1 = (TOP1 - LAST1) / (len(eps) - 1)
assert step1 > 0.130, step1
for i, e in enumerate(eps):
    y = TOP1 - i * step1
    hole = not e["driven_before"]
    if hole:
        ax.add_patch(plt.Rectangle((0.0, y - 0.085), 1.0, 0.185, transform=ax.transAxes,
                                   facecolor="#fdecea", edgecolor=RED, lw=1.4, zorder=0))
    put(ax, cols[0], y, e["surface"], fontsize=11.5, family="monospace",
        fontweight="bold" if hole else "normal")
    put(ax, cols[1], y, e["outcome"][:58], fontsize=10.2, family="monospace", color="#222")
    put(ax, cols[2], y, "yes" if e["driven_before"] else "NO",
        fontsize=12, fontweight="bold", color=GREEN if e["driven_before"] else RED)
    put(ax, cols[3], y, e["note"], fontsize=9.6, color="#444", style="italic")

put(ax, 0.005, 0.155,
    "Why the middle row was invisible: the teleoperate(publish=True) tests reach the mesh publisher through\n"
    f"{F['stub']['class']}, a stand-in that records the call and returns\n"
    f"status={F['stub']['status']!r} without validating anything -- so the real method's refusal had never executed.",
    fontsize=10.4, family="monospace", color="#333", va="top")

# ---------------- row 2: mutation matrix --------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.04,
    f"Plausible regressions vs the {F['suite_arms']['preexisting_cases']} pre-existing cases over the same modules",
    fontsize=12.5, fontweight="bold")
c2 = [0.005, 0.545, 0.735]
put(ax2, c2[0], 0.905, "mutation of start_teleop_publish", fontsize=10.5, fontweight="bold", color="#333")
put(ax2, c2[1], 0.905, f"new class ({F['suite_arms']['new_cases']} cases)", fontsize=10.5, fontweight="bold", color="#333")
put(ax2, c2[2], 0.905, f"pre-existing ({F['suite_arms']['preexisting_cases']} cases)", fontsize=10.5, fontweight="bold", color="#333")
ax2.plot([0, 1], [0.875, 0.875], color="#999", lw=1.0, transform=ax2.transAxes, clip_on=False)

TOP2, LAST2 = 0.760, 0.155
step2 = (TOP2 - LAST2) / (len(muts) - 1)
assert step2 > 0.100, step2
for i, m in enumerate(muts):
    y = TOP2 - i * step2
    is_blind = m["old"] == 0
    put(ax2, c2[0], y, m["label"], fontsize=11.0, family="monospace")
    put(ax2, c2[1], y, f"{m['new']} failed", fontsize=11.0, family="monospace",
        fontweight="bold", color=GREEN)
    put(ax2, c2[2], y, "0 failed  <- BLIND" if is_blind else f"{m['old']} failed",
        fontsize=11.0, family="monospace", fontweight="bold", color=RED if is_blind else AMBER)
put(ax2, c2[0], 0.045,
    f"{len(blind)} of {len(muts)} are invisible to the suite as it stands. M2 is the ordering regression the "
    "method's own comment warns about.",
    fontsize=10.4, color="#333", style="italic")

# ---------------- row 3: measured facts ---------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 1.06, "Measured", fontsize=12.5, fontweight="bold")
cov = F["coverage"]
facts = [
    ("a refused rate leaves a live publisher running",
     f"status={F['ordering_refused']['status']}, live publisher stopped="
     f"{F['ordering_refused']['live_stopped']}", GREEN),
    ("an accepted rate does replace and stop it (the mirror)",
     f"status={F['ordering_accepted']['status']}, stopped="
     f"{F['ordering_accepted']['live_stopped']}, replaced={F['ordering_accepted']['replaced']}", GREEN),
    (f"{cov['file']} line {cov['line']} (the refusal)",
     f"missing {cov['missing_before']} -> {cov['missing_after']} over the same four test files", GREEN),
    ("production lines changed",
     "0 -- tests and one changelog fragment only", GREEN),
]
TOP3, LAST3 = 0.760, 0.115
step3 = (TOP3 - LAST3) / (len(facts) - 1)
assert step3 > 0.150, step3
for i, (label, value, colour) in enumerate(facts):
    y = TOP3 - i * step3
    put(ax3, 0.005, y, label, fontsize=11.0, color="#333")
    put(ax3, 0.475, y, value, fontsize=11.0, family="monospace", fontweight="bold", color=colour)

for ax_, y, is_axes in placed:
    assert -0.03 <= y <= 1.10, f"text at y={y} outside the axes"

OUT = ROOT / "_art/teleop_publish_rate_domain.png"
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(OUT).convert("RGB"))
h, w, _ = im.shape
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((band < 250).any(axis=-1).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"OK {OUT}  {w}x{h}  blind={len(blind)}/{len(muts)}  step1={step1:.3f} step2={step2:.3f} step3={step3:.3f}")
