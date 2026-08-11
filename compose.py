"""Render the measurement figure. Every number is read from facts JSON."""
import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

RUN = os.environ["GITHUB_RUN_ID"]
F = json.load(open(f"/tmp/facts-{RUN}.json"))
ROOT = pathlib.Path(__file__).resolve().parents[1]
assert F["tree"] == str(ROOT), (F["tree"], ROOT)

M, COV, MUT = F["matrix"], F["coverage"], F["mutations"]
ISAAC = "strands_robots/simulation/isaac/recording.py"
NEWTON = "strands_robots/simulation/newton/recording.py"
holes_before = sum(1 for r in M for b in r["backends"].values() if not b["before"])
holes_after = sum(1 for r in M for b in r["backends"].values() if not b["after"])
cells = sum(len(r["backends"]) for r in M)
assert (cells, holes_before, holes_after) == (8, 5, 0), (cells, holes_before, holes_after)
real = [m for m in MUT if m["label"].startswith("M")]
caught = sum(1 for m in real if m["new"]["failed"] > 0)
blind = sum(1 for m in real if m["old"]["failed"] == 0)
assert (len(real), caught, blind) == (6, 6, 6), (len(real), caught, blind)
assert COV[ISAAC]["before_miss"] == 15 and COV[ISAAC]["after_miss"] == 5
assert COV[NEWTON]["before_miss"] == 2 and COV[NEWTON]["after_miss"] == 1

GREEN, RED, INK, MUTED = "#1b7f4b", "#b3261e", "#14181f", "#5b6472"
placed = []
def put(ax, x, y, s, axes_coords=True, **kw):
    if axes_coords:
        kw.setdefault("transform", ax.transAxes)
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.2, 11.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.06, 0.92, 0.36], hspace=0.20,
                      left=0.035, right=0.972, top=0.925, bottom=0.035)

fig.suptitle("Recording lifecycle guards: which contracts each backend's capture path actually executes",
             fontsize=17.5, fontweight="bold", color=INK, y=0.977)
fig.text(0.5, 0.949,
         "Isaac and Newton each define start_recording and the per-step capture hook in their own backend mixin, so these are "
         "independent copies of one contract.  Tests only \u2014 0 production lines changed.",
         ha="center", fontsize=11.3, color=MUTED)

# ---------------- row 1: the matrix -----------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.005, "Contract  \u00d7  backend      (a cell is green when the suite executes the line that proves it ran)",
    fontsize=12.6, fontweight="bold", color=INK)
COLX = {"before": 0.585, "after": 0.808}
for phase, cx in COLX.items():
    put(ax, cx, 0.905, ("on main" if phase == "before" else "with this PR"),
        fontsize=12.0, fontweight="bold", color=INK, ha="center")
    for i, be in enumerate(("isaac", "newton")):
        put(ax, cx - 0.058 + i * 0.116, 0.845, be, fontsize=10.6, color=MUTED, ha="center")
TOP, LAST = 0.735, 0.115
step = (TOP - LAST) / (len(M) - 1)
assert step > 0.030, step
for k, row in enumerate(M):
    y = TOP - k * step
    put(ax, 0.0, y, row["contract"], fontsize=11.9, color=INK, va="center")
    for phase, cx in COLX.items():
        for i, be in enumerate(("isaac", "newton")):
            c = row["backends"][be]
            ok = c[phase]
            x = cx - 0.058 + i * 0.116
            ax.add_patch(Rectangle((x - 0.049, y - 0.036), 0.098, 0.072,
                                   transform=ax.transAxes, facecolor=GREEN if ok else RED,
                                   alpha=0.16, edgecolor=GREEN if ok else RED, lw=1.4))
            put(ax, x, y, ("driven" if ok else "never run"), fontsize=10.5, ha="center", va="center",
                color=GREEN if ok else RED, fontweight="bold")
            put(ax, x, y - 0.052, f"L{c['line']}", fontsize=8.6, ha="center", va="center", color=MUTED)
assert abs((TOP - (len(M) - 1) * step) - LAST) < 1e-9
put(ax, 0.0, 0.020,
    f"{holes_before} of {cells} cells never executed  \u2192  {holes_after} of {cells}."
    "   The last row was driven on neither backend: Newton's guard module docstring enumerated two of the hook's three no-write states.",
    fontsize=11.2, color=INK, fontweight="bold")

# ---------------- row 2: mutation table -------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.010, "Do the new tests hold the contract?   6 plausible regressions \u00d7 2 arms",
    fontsize=12.6, fontweight="bold", color=INK)
put(ax2, 0.615, 0.918, "these 5 new tests", fontsize=11.0, fontweight="bold", ha="center", color=INK)
put(ax2, 0.845, 0.918, "the 38 pre-existing recording tests", fontsize=11.0, fontweight="bold", ha="center", color=INK)
rows = MUT
T2, L2 = 0.815, 0.075
st2 = (T2 - L2) / (len(rows) - 1)
assert st2 > 0.030, st2
for k, m in enumerate(rows):
    y = T2 - k * st2
    ctrl = not m["label"].startswith("M")
    put(ax2, 0.0, y, m["label"], fontsize=11.2, va="center",
        color=MUTED if ctrl else INK, style="italic" if ctrl else "normal")
    for cx, key in ((0.615, "new"), (0.845, "old")):
        r = m[key]
        if ctrl:
            txt, col = f"{r['passed']} passed", MUTED
        else:
            hit = r["failed"] > 0
            txt = f"{r['failed']} failed / {r['passed']} passed"
            col = GREEN if (hit if key == "new" else False) else RED
            if key == "old":
                txt += "   \u2190 BLIND"
        put(ax2, cx, y, txt, fontsize=10.9, ha="center", va="center", color=col,
            fontweight="normal" if ctrl else "bold")
assert abs((T2 - (len(rows) - 1) * st2) - L2) < 1e-9
put(ax2, 0.0, 0.000,
    f"caught by the new tests: {caught}/6.   Invisible to the pre-existing suite: {blind}/6 \u2014 including M2, which keeps the "
    "guard's condition and discards only its return, the shape a structural \u201cthe guard is called\u201d check cannot see.",
    fontsize=11.2, color=INK, fontweight="bold")

# ---------------- row 3: footer ---------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(Rectangle((0, 0.02), 1, 0.96, transform=ax3.transAxes,
                        facecolor="#f4f6f8", edgecolor="#d8dee6", lw=1.1))
i_, n_ = COV[ISAAC], COV[NEWTON]
lines = [
    f"isaac/recording.py   {i_['before_miss']:>2} missing \u2192 {i_['after_miss']:>2}   "
    f"({i_['before_pct']:.1f}% \u2192 {i_['after_pct']:.1f}%)      closed: {i_['closed']}",
    f"newton/recording.py  {n_['before_miss']:>2} missing \u2192 {n_['after_miss']:>2}   "
    f"({n_['before_pct']:.1f}% \u2192 {n_['after_pct']:.1f}%)      closed: {n_['closed']}",
    f"full suite  {F['suite']['before']} \u2192 {F['suite']['after']} passed / {F['suite']['skipped']} skipped, 0 failed"
    f"      ruff clean \u00b7 mypy 0 errors outside examples/isaac_gs \u00b7 0 lines under strands_robots/ changed",
]
T3, L3 = 0.735, 0.215
st3 = (T3 - L3) / (len(lines) - 1)
assert st3 > 0.030, st3
for k, s in enumerate(lines):
    put(ax3, 0.016, T3 - k * st3, s, fontsize=11.0, family="monospace", color=INK, va="center")

for a, y, axc in placed:
    if axc:
        assert -0.03 <= y <= 1.07, (y, a)
    else:
        lo, hi = a.get_ylim(); assert lo - 0.03 <= y <= hi + 0.07, (y, lo, hi)

out = pathlib.Path("_art/recording_lifecycle_guard_parity.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band - 255).sum(2) > 12).sum())
    assert bad == 0, (name, bad)
print("WROTE", out, Image.open(out).size)
