"""Compose the artifact. Every drawn number is read from the measured dumps."""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

RUN = sys.argv[1]
A = json.loads(pathlib.Path(f"/tmp/art-base-{RUN}.json").read_text())    # upstream/main
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RUN}.json").read_text())  # this PR
M = json.loads(pathlib.Path(f"/tmp/art-mut-{RUN}.json").read_text())

assert A["tree"] != B["tree"], "both dumps came from the same tree"
LIMIT = A["ungated"]["limit"]
# The defect, as measured on upstream/main.
assert A["ungated"]["dispatched"] == 2 and A["ungated"]["slots_after"] == LIMIT + 1
assert A["ungated"]["exceeds_limit"] is True
assert A["ungated"]["statuses"] == ["success", "success"]
# Fixed on this branch.
assert B["ungated"]["dispatched"] == 1 and B["ungated"]["slots_after"] == LIMIT
assert B["ungated"]["exceeds_limit"] is False
assert B["ungated"]["statuses"] == ["error", "success"]
# The approved path was already atomic: unchanged, which is the no-regression half.
for k in ("dispatched", "slots_after", "statuses", "exceeds_limit"):
    assert A["approved"][k] == B["approved"][k], k
# Wording: main blamed an approval; this branch does not.
assert "approval" in A["approved"]["refusal"]
assert "approval" not in B["approved"]["refusal"] and "approval" not in B["ungated"]["refusal"]
assert A["has_record_only"] and not B["has_record_only"]

rows = M["rows"]
CAUGHT = sum(1 for r in rows if r["new_failed"])
BLIND = sum(1 for r in rows if not r["old_failed"])
assert (CAUGHT, BLIND, len(rows)) == (6, 4, 6)
OLDN = M["control"]["old_passed"]
assert M["control"]["new_failed"] == 0 and M["control"]["old_failed"] == 0

GREEN, RED, GREY = "#1a7f37", "#b3261e", "#57606a"
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.0, 12.4), dpi=124)
gs = gridspec.GridSpec(3, 2, height_ratios=[1.05, 0.86, 1.05], hspace=0.40, wspace=0.20,
                       left=0.055, right=0.975, top=0.925, bottom=0.035)

LBL = ["main\napproved", "main\nungated", "this PR\napproved", "this PR\nungated"]
CELLS = [A["approved"], A["ungated"], B["approved"], B["ungated"]]

# --- panel 1: window occupancy -------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
vals = [c["slots_after"] for c in CELLS]
cols = [RED if c["exceeds_limit"] else GREEN for c in CELLS]
bars = ax.bar(LBL, vals, color=cols, width=0.6)
ax.axhline(LIMIT, ls="--", lw=2.0, color=RED)
put(ax, 3.42, LIMIT + 0.07, f"configured cap = {LIMIT}", color=RED, fontsize=10.5,
    ha="right", va="bottom", fontweight="bold")
for b, v in zip(bars, vals, strict=True):
    put(ax, b.get_x() + b.get_width() / 2, v + 0.06, str(v), ha="center", va="bottom",
        fontsize=13, fontweight="bold")
ax.set_ylim(0, LIMIT + 1.6)
ax.set_ylabel("entries in the 60s window\nafter the race", fontsize=10.5)
ax.set_title("Two concurrent emergency_stop calls, one slot free", fontsize=12, fontweight="bold", pad=9)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

# --- panel 2: dispatches that reached the mesh ---------------------------------
ax2 = fig.add_subplot(gs[0, 1])
d = [c["dispatched"] for c in CELLS]
cols2 = [RED if v > 1 else GREEN for v in d]
bars2 = ax2.bar(LBL, d, color=cols2, width=0.6)
ax2.axhline(1, ls="--", lw=2.0, color=GREY)
put(ax2, 3.42, 1.06, "1 slot was free, so 1 is permitted", color=GREY, fontsize=10.5,
    ha="right", va="bottom", fontweight="bold")
for b, v in zip(bars2, d, strict=True):
    put(ax2, b.get_x() + b.get_width() / 2, v + 0.04, str(v), ha="center", va="bottom",
        fontsize=13, fontweight="bold")
ax2.set_ylim(0, 2.6)
ax2.set_ylabel("fleet-wide e-stop broadcasts\nthat reached the mesh", fontsize=10.5)
ax2.set_title("Both calls dispatched where only one had a slot", fontsize=12, fontweight="bold", pad=9)
ax2.grid(axis="y", alpha=0.25)
ax2.set_axisbelow(True)

# --- row 2: verdict ledger -----------------------------------------------------
axl = fig.add_subplot(gs[1, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 1.055, "Measured verdicts  (STRANDS_MESH_HITL_ACTIONS: 'emergency_stop' = approved path, 'none' = ungated path)",
    fontsize=12, fontweight="bold", transform=axl.transAxes)
COLS = [0.0, 0.115, 0.30, 0.44, 0.585, 0.73]
HEAD = ["tree", "gate path", "statuses", "dispatched", f"window / cap {LIMIT}", "over the cap?"]
TOP, LAST = 0.90, 0.20
for x, h in zip(COLS, HEAD, strict=True):
    put(axl, x, TOP, h, fontsize=10.5, fontweight="bold", color=GREY, transform=axl.transAxes)
STEP = (TOP - 0.10 - LAST) / (len(CELLS) - 1)
assert STEP > 0.030, STEP
y = TOP - 0.10
for lbl, c in zip(LBL, CELLS, strict=True):
    tree, path = lbl.split("\n")
    bad = c["exceeds_limit"]
    col = RED if bad else GREEN
    if bad:
        axl.add_patch(plt.Rectangle((-0.006, y - 0.045), 1.01, 0.115, transform=axl.transAxes,
                                    facecolor=RED, alpha=0.09, zorder=0))
    cellvals = [tree, path, " + ".join(c["statuses"]), str(c["dispatched"]),
                f"{c['slots_after']}", "YES - cap exceeded" if bad else "no"]
    for x, v, i in zip(COLS, cellvals, range(6), strict=True):
        put(axl, x, y, v, fontsize=11, transform=axl.transAxes,
            color=col if i >= 2 else "black", fontweight="bold" if i >= 3 else "normal",
            family="monospace" if i in (3, 4) else None)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y
put(axl, 0.0, 0.055,
    "The approved path was already atomic - its four columns are identical on both trees. "
    "Narrowing the gate is exactly the configuration in which the cap is the only bound left.",
    fontsize=10.2, color=GREY, style="italic", transform=axl.transAxes)

# --- row 3: mutation matrix ----------------------------------------------------
axm = fig.add_subplot(gs[2, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 1.045, "Mutation table  -  every plausible regression, against two arms",
    fontsize=12, fontweight="bold", transform=axm.transAxes)
MC = [0.0, 0.615, 0.775]
for x, h in zip(MC, ["regression introduced into robot_mesh",
                     "new module (20 cases)", f"pre-existing suite ({OLDN} cases)"], strict=True):
    put(axm, x, 0.925, h, fontsize=10.5, fontweight="bold", color=GREY, transform=axm.transAxes)
allrows = rows + [M["control"]]
TOPM, LASTM = 0.815, 0.135
STEPM = (TOPM - LASTM) / (len(allrows) - 1)
assert STEPM > 0.030, STEPM
ym = TOPM
for r in allrows:
    ctl = r["label"].startswith("unmutated")
    blind = (not ctl) and r["old_failed"] == 0
    if blind:
        axm.add_patch(plt.Rectangle((-0.006, ym - 0.030), 1.01, 0.082, transform=axm.transAxes,
                                    facecolor=RED, alpha=0.08, zorder=0))
    put(axm, MC[0], ym, r["label"], fontsize=10.6, transform=axm.transAxes,
        color=GREY if ctl else "black", style="italic" if ctl else "normal")
    put(axm, MC[1], ym, f"{r['new_failed']} failed", fontsize=10.6, family="monospace",
        transform=axm.transAxes, color=GREEN if r["new_failed"] or ctl else RED,
        fontweight="bold" if r["new_failed"] else "normal")
    tail = "   <- BLIND" if blind else ""
    put(axm, MC[2], ym, f"{r['old_failed']} failed{tail}", fontsize=10.6, family="monospace",
        transform=axm.transAxes, color=RED if blind else GREY,
        fontweight="bold" if blind else "normal")
    ym -= STEPM
assert abs((ym + STEPM) - LASTM) < 1e-9, ym
put(axm, 0.0, 0.045,
    f"{CAUGHT} of {len(rows)} caught by the new module; {BLIND} of {len(rows)} invisible to all {OLDN} "
    "pre-existing robot_mesh cases. M3 and M6 are caught by both arms: M6 breaks the "
    "declined-approval property the pre-existing suite already owns, and M3 stops the ungated path reserving at all.",
    fontsize=10.2, color=GREY, style="italic", transform=axm.transAxes)

fig.suptitle("robot_mesh: a rate-limit slot must be reserved atomically on both gate paths",
             fontsize=14.5, fontweight="bold", y=0.977)

for ax_, yy, is_axes in placed:
    lo, hi = (-0.03, 1.07) if is_axes else ax_.get_ylim()
    assert lo - 0.05 <= yy <= hi + 0.07, f"text at y={yy} outside {lo}..{hi}"

out = pathlib.Path(f"/tmp/artifact-{RUN}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(matplotlib.image.imread(out) * 255).astype(int)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band - 255).sum(2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}")
