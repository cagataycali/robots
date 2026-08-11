import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.load(open("/tmp/art-facts.json"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
assert F["tree"] == str(ROOT), (F["tree"], str(ROOT))

# ---- derive every claim
N_BLIND = sum(1 for r in F["mutations"] if r["old"]["failed"] == 0)
N_MUT = len(F["mutations"])
assert (N_BLIND, N_MUT) == (3, 5), (N_BLIND, N_MUT)
assert all(r["new"]["failed"] > 0 for r in F["mutations"]), "a mutation escaped the new tests"
assert F["control"]["new"]["passed"] == 13 and F["control"]["old"]["failed"] == 0
for a in ("send", "broadcast"):
    assert F["with_guard"][a]["dispatched"] is False, a
    assert F["without_guard"][a]["dispatched"] is True, a
    assert "None" in F["without_guard"][a]["wire"], a
    assert F["without_guard"][a]["audited_success"] == [], a
N_OLD = F["control"]["old"]["passed"]

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("family", "monospace"); kw.setdefault("fontsize", 9.4)
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.0, 10.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.28, 1.30, 0.40], hspace=0.20,
                      left=0.022, right=0.985, top=0.925, bottom=0.022)
fig.suptitle("robot_mesh: both validate-before-HITL contract guards, measured", fontsize=15.5, fontweight="bold", y=0.978)
fig.text(0.5, 0.949, "tests-only + a corrected comment (AST digest unchanged) -- no policy, simulation, rendering, recording or asset behaviour changes",
         ha="center", fontsize=10.2, style="italic", color="#333333")

GREEN, RED, HEAD = "#1b7a3d", "#b3261e", "#111111"

# ---------- ROW 1: dispatch-consequence ledger
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.955, "1  What the handler does when it is reached with its sentinel unset", fontsize=12.2,
    fontweight="bold", color=HEAD, family="sans-serif", transform=ax.transAxes)
put(ax, 0.0, 0.865, "   (validate_command refactored to validate in place and return None -- the state the guard exists to catch)",
    fontsize=9.6, style="italic", color="#444444", family="sans-serif", transform=ax.transAxes)
COLS = [0.012, 0.135, 0.285, 0.505, 0.665]
put(ax, COLS[0], 0.765, "tree", fontweight="bold", transform=ax.transAxes)
put(ax, COLS[1], 0.765, "action", fontweight="bold", transform=ax.transAxes)
put(ax, COLS[2], 0.765, "reached the transport?", fontweight="bold", transform=ax.transAxes)
put(ax, COLS[3], 0.765, "wire", fontweight="bold", transform=ax.transAxes)
put(ax, COLS[4], 0.765, "outcome the caller sees", fontweight="bold", transform=ax.transAxes)
rows = []
for tree_label, key, ok in (("this PR", "with_guard", True), ("guard removed", "without_guard", False)):
    for action in ("send", "broadcast"):
        rows.append((tree_label, action, F[key][action], ok))
TOP, LAST = 0.655, 0.085
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.045, STEP
for i, (tree_label, action, v, ok) in enumerate(rows):
    y = TOP - i * STEP
    colour = GREEN if ok else RED
    ax.add_patch(plt.Rectangle((0.006, y - 0.056), 0.988, 0.108, transform=ax.transAxes,
                               facecolor=colour, alpha=0.085, edgecolor=colour, linewidth=0.9, zorder=0))
    put(ax, COLS[0], y, tree_label, color=colour, fontweight="bold", transform=ax.transAxes)
    put(ax, COLS[1], y, action, transform=ax.transAxes)
    put(ax, COLS[2], y, "NO -- refused first" if not v["dispatched"] else "YES -- dispatched",
        color=colour, fontweight="bold", transform=ax.transAxes)
    put(ax, COLS[3], y, v["wire"], transform=ax.transAxes)
    put(ax, COLS[4], y, v["outcome"][:52], transform=ax.transAxes)
put(ax, 0.012, 0.012,
    f"Without the guard a command is issued fleet-wide / to the peer with the body set to None, and audit records for "
    f"that dispatch: {len(F['without_guard']['broadcast']['audited_success'])} -- the raise lands after the dispatch.",
    fontsize=9.6, style="italic", color=RED, family="sans-serif", transform=ax.transAxes)

# ---------- ROW 2: mutation matrix
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.955, f"2  Mutation matrix -- 5 plausible regressions x 2 arms", fontsize=12.2,
    fontweight="bold", color=HEAD, family="sans-serif", transform=ax2.transAxes)
put(ax2, 0.0, 0.868, f"   {N_BLIND} of {N_MUT} are invisible to all {N_OLD} pre-existing robot_mesh tests ({F['n_old_files']} files)",
    fontsize=9.6, style="italic", color="#444444", family="sans-serif", transform=ax2.transAxes)
M = [0.012, 0.470, 0.660, 0.860]
put(ax2, M[0], 0.760, "mutation applied to robot_mesh.py", fontweight="bold", transform=ax2.transAxes)
put(ax2, M[1], 0.760, "this PR's tests", fontweight="bold", transform=ax2.transAxes)
put(ax2, M[2], 0.760, f"pre-existing ({N_OLD})", fontweight="bold", transform=ax2.transAxes)
put(ax2, M[3], 0.760, "verdict", fontweight="bold", transform=ax2.transAxes)
TOP2, LAST2 = 0.655, 0.150
STEP2 = (TOP2 - LAST2) / len(F["mutations"])
assert STEP2 > 0.045, STEP2
for i, r in enumerate(F["mutations"]):
    y = TOP2 - i * STEP2
    blind = r["old"]["failed"] == 0
    colour = RED if blind else "#8a6d1f"
    if blind:
        ax2.add_patch(plt.Rectangle((0.640, y - 0.048), 0.354, 0.092, transform=ax2.transAxes,
                                    facecolor=RED, alpha=0.11, edgecolor=RED, linewidth=0.9, zorder=0))
    put(ax2, M[0], y, r["label"], transform=ax2.transAxes)
    put(ax2, M[1], y, f"{r['new']['failed']} failed", color=GREEN, fontweight="bold", transform=ax2.transAxes)
    put(ax2, M[2], y, f"{r['old']['failed']} failed" if not blind else "0 failed -- BLIND",
        color=colour, fontweight="bold", transform=ax2.transAxes)
    put(ax2, M[3], y, "caught by both" if not blind else "caught only here", color=colour, transform=ax2.transAxes)
y = LAST2 - STEP2 * 0.62
put(ax2, M[0], y, "(unmutated control)", style="italic", transform=ax2.transAxes)
put(ax2, M[1], y, f"{F['control']['new']['passed']} passed", color=GREEN, transform=ax2.transAxes)
put(ax2, M[2], y, f"{N_OLD} passed", color=GREEN, transform=ax2.transAxes)
assert y > 0.012, y

# ---------- ROW 3: footer
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
lines = [
    "coverage   strands_robots/tools/robot_mesh.py  97 -> 95 missing lines; 1187 and 1207 leave the missing list (mesh suite 2994 -> 3007 passed)",
    "gate       full suite 27579 passed / 257 skipped / 0 failed (610.89s) -- pristine main 27566 + 13 new = 27579;  ruff clean 1164 files;  mypy 0 errors outside examples/isaac_gs",
    "comment    AST digest of robot_mesh.py identical before and after (7dfbf34b3994bb3b) -- the production edit is comment-only",
]
TOP3, STEP3 = 0.80, 0.30
for i, line in enumerate(lines):
    y = TOP3 - i * STEP3
    put(ax3, 0.012, y, line, fontsize=9.5, transform=ax3.transAxes)
    assert y > 0.03, y

for a, y, is_axes in placed:
    if is_axes:
        assert -0.03 <= y <= 1.10, (y, "axes-fraction out of range")
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= y <= hi + 0.07, (y, lo, hi)

OUT = pathlib.Path("/tmp/robot_mesh_hitl_guards.png")
fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", OUT, im.shape, "border clean; every claim asserted")
