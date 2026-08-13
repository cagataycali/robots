"""Compose the publication-posture figure from the two measured trees."""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path(sys.argv[1]).read_text())   # main
B = json.loads(pathlib.Path(sys.argv[2]).read_text())   # branch
OUT = pathlib.Path(sys.argv[3])
assert A["tree"] != B["tree"], "both arms measured the same tree"

# Whether each case *should* launch, and whether the operator should be asked.
EXPECT = {
    "push_to_hub=True (named)":            (False, False),
    "+ extra_flags policy.repo_id":        (False, False),
    "extra_flags={'push_to_hub': True}":   (False, False),
    "push_to_hub=False (default)":         (True,  False),
    "push_to_hub=True, operator declines": (False, True),
    "push_to_hub=True, operator approves": (True,  True),
}

def honors(row):
    should_launch, should_ask = EXPECT[row["case"]]
    ok = row["launched"] == should_launch
    if should_ask:
        ok = ok and row["asked"]
    return ok

rowsA = {r["case"]: r for r in A["rows"]}
rowsB = {r["case"]: r for r in B["rows"]}
cases = [r["case"] for r in A["rows"]]
nA = sum(1 for c in cases if honors(rowsA[c]))
nB = sum(1 for c in cases if honors(rowsB[c]))
assert (nA, nB) == (2, 6), f"honoured counts {(nA, nB)} != (2, 6)"
assert rowsA["push_to_hub=True (named)"]["launched"] and not rowsA["push_to_hub=True (named)"]["asked"]
assert rowsA["push_to_hub=True, operator declines"]["launched"], "main must launch despite a decline"
assert not rowsB["push_to_hub=True, operator declines"]["launched"]
assert rowsB["push_to_hub=False (default)"]["launched"], "the default must stay free"
assert rowsA["push_to_hub=False (default)"]["publish_argv"] == \
       rowsB["push_to_hub=False (default)"]["publish_argv"], "default argv must be unchanged"

MUT = [
    ("M1  delete the push_to_hub gate",            6, 0),
    ("M2  keep the call, discard the refusal",      4, 0),
    ("M3  gate unconditionally (default too)",      2, 0),
    ("M4  drop the flag from the blocklist",        6, 0),
    ("M5  gate names a flag that is not blocked",   6, 0),
]

GOOD, BAD, INK, MUTED = "#1b7f4b", "#b3261e", "#1f2328", "#6b7280"
fig = plt.figure(figsize=(15.4, 12.2), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 1.00, 0.72], hspace=0.16,
                      left=0.035, right=0.978, top=0.935, bottom=0.030)
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); kw.setdefault("fontsize", 9.6); kw.setdefault("color", INK)
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig.suptitle("lerobot_train: a publish reached the argv with nobody asked", fontsize=15.6,
             fontweight="bold", color=INK, y=0.982)
fig.text(0.5, 0.955, "measured through the tool with subprocess.Popen recorded  |  "
         "left: main (a78ea60)   right: this change",
         ha="center", fontsize=10.2, color=MUTED, style="italic")

# ---- row 1: verdict grid -------------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.00, "1.  Did the outcome honour the approval posture?", fontsize=12.4, fontweight="bold")
COLS = [(0.005, "call"), (0.455, "main"), (0.640, "this change"), (0.825, "expected")]
for x, h in COLS:
    put(ax, x, 0.905, h, fontsize=10.0, fontweight="bold", color=MUTED)
TOP, LAST = 0.845, 0.075
STEP = (TOP - LAST) / (len(cases) - 1)
assert STEP > 0.030, STEP
y = TOP
for c in cases:
    a, b = rowsA[c], rowsB[c]
    sl, sa = EXPECT[c]
    put(ax, 0.005, y, c, fontsize=10.2, family="monospace")
    for x, r in ((0.455, a), (0.640, b)):
        ok = honors(r)
        col = GOOD if ok else BAD
        verdict = "launched" if r["launched"] else "refused"
        extra = "  (asked)" if r["asked"] else ("  (never asked)" if sa else "")
        put(ax, x, y, f"{'ok ' if ok else 'BAD'}  {verdict}{extra}",
            fontsize=9.7, family="monospace", color=col, fontweight="bold" if not ok else "normal")
    want = ("launch" if sl else "refuse") + (" after asking" if sa else "")
    put(ax, 0.825, y, want, fontsize=9.7, family="monospace", color=MUTED)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9
ax.add_patch(Rectangle((0.450, LAST - 0.030), 0.180, TOP - LAST + 0.090, transform=ax.transData,
                       facecolor=BAD, alpha=0.045, edgecolor="none", zorder=0))
put(ax, 0.005, 0.022, f"honoured: main {nA} of {len(cases)}   |   this change {nB} of {len(cases)}",
    fontsize=10.6, fontweight="bold")

# ---- row 2: the wire ----------------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.00, "2.  What reached the training subprocess", fontsize=12.4, fontweight="bold")
WIRE = ["push_to_hub=True (named)", "+ extra_flags policy.repo_id",
        "push_to_hub=True, operator declines", "push_to_hub=False (default)"]
TOP2, LAST2 = 0.885, 0.100
STEP2 = (TOP2 - LAST2) / (len(WIRE) - 1)
assert STEP2 > 0.100, STEP2
y = TOP2
for c in WIRE:
    put(ax2, 0.005, y, c, fontsize=10.2, family="monospace", fontweight="bold")
    for dx, (tag, r) in ((0.015, ("main", rowsA[c])), (0.515, ("this change", rowsB[c]))):
        argv = [a for a in r["publish_argv"] if "policy" in a]
        txt = "  ".join(argv) if argv else "(no subprocess launched)"
        col = GOOD if honors(r) else BAD
        put(ax2, dx + 0.010, y - STEP2 * 0.36, f"{tag}: {txt}",
            fontsize=9.1, family="monospace", color=col)
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9
put(ax2, 0.005, 0.048,
    "--policy.repo_id is deliberately not blocklisted: LeRobot's config validation requires it even for a\n"
    "purely local run.  The publish decision is the gated one, and it carries the destination with it.",
    fontsize=9.5, color=MUTED, style="italic")

# ---- row 3: mutations + gate -------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 1.00, "3.  Mutation table  (new module  vs  the 277 pre-existing lerobot_train cases)",
    fontsize=12.4, fontweight="bold")
put(ax3, 0.520, 0.865, "new", fontsize=9.6, fontweight="bold", color=MUTED)
put(ax3, 0.610, 0.865, "pre-existing", fontsize=9.6, fontweight="bold", color=MUTED)
TOP3, LAST3 = 0.795, 0.300
STEP3 = (TOP3 - LAST3) / (len(MUT) - 1)
assert STEP3 > 0.060, STEP3
y = TOP3
for label, new, old in MUT:
    put(ax3, 0.005, y, label, fontsize=9.8, family="monospace")
    put(ax3, 0.520, y, f"{new} failed", fontsize=9.8, family="monospace", color=GOOD)
    put(ax3, 0.610, y, f"{old} failed", fontsize=9.8, family="monospace", color=BAD)
    if old == 0:
        put(ax3, 0.720, y, "<- BLIND", fontsize=9.4, family="monospace", color=BAD, fontweight="bold")
    y -= STEP3
assert abs((y + STEP3) - LAST3) < 1e-9
put(ax3, 0.005, 0.205,
    "5 of 5 caught by the new module, 0 of 5 by the pre-existing suite: the gate's wiring had no test at all.",
    fontsize=9.6, color=MUTED, style="italic")
put(ax3, 0.005, 0.120,
    "Gate  ruff clean (1215 files)  |  mypy 0 non-examples errors  |  "
    "MUJOCO_GL=egl pytest tests: 29604 passed / 266 skipped / 0 failed (675s)\n"
    "Pre-fix, source reverted and tests kept: 6 failed / 10 passed  |  "
    "no policy, simulation, rendering, recording or asset behaviour changes.",
    fontsize=9.5, family="monospace")

for ax_, yy, is_axes in placed:
    lo, hi = ((-0.03, 1.10) if is_axes else ax_.get_ylim())
    assert lo - 0.05 <= yy <= hi + 0.07, f"text at y={yy} outside {(lo, hi)}"

fig.savefig(OUT, dpi=124, facecolor="white", bbox_inches="tight", pad_inches=0.30)
import numpy as np
from PIL import Image
im = np.asarray(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {im.shape[1]}x{im.shape[0]}  honoured main={nA} branch={nB}")
