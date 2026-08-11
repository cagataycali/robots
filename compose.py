import json, os, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RUN = os.environ["GITHUB_RUN_ID"]
F = json.loads(pathlib.Path(f"/tmp/art-{RUN}.json").read_text())
frame = np.load(f"/tmp/art-frame-{RUN}.npy")

# --- self-audit every claim before drawing -------------------------------
assert F["present"]["main"] == {"passed": 37, "skipped": 0, "failed": 0}, F["present"]["main"]
assert F["present"]["branch"] == {"passed": 39, "skipped": 0, "failed": 0}, F["present"]["branch"]
assert F["blocked"]["main"] == {"passed": 26, "skipped": 11, "failed": 0}, F["blocked"]["main"]
assert F["blocked"]["branch"] == {"passed": 39, "skipped": 0, "failed": 0}, F["blocked"]["branch"]
assert F["nine_cells_blocked"]["main"]["passed"] == 6 and F["nine_cells_blocked"]["main"]["skipped"] == 3
assert F["nine_cells_blocked"]["branch"]["passed"] == 9 and F["nine_cells_blocked"]["branch"]["skipped"] == 0
assert F["mutation_blocked"]["main"]["failed"] == 0 and F["mutation_blocked"]["main"]["skipped"] == 11
assert F["mutation_blocked"]["branch"]["failed"] == 2 and F["mutation_blocked"]["branch"]["skipped"] == 0
r = F["recording"]
assert r["start"] == r["rollout"] == r["stop"] == "success", r
assert (r["episodes"], r["frames"], r["fps"], r["decoded_frames"]) == (1, 24, 20, 24), r
assert r["saturated_frac"] > 0.5, r["saturated_frac"]

CAUSES = ["lerobot extra absent", "module did not import", "no DatasetRecorder symbol"]
BACKENDS = ["MuJoCo", "Newton", "Isaac"]
# on main the three MuJoCo cells are the skipped ones
RAN_MAIN = {(c, b): (b != "MuJoCo") for c in CAUSES for b in BACKENDS}
assert sum(RAN_MAIN.values()) == 6 == F["nine_cells_blocked"]["main"]["passed"]

GREEN, RED, INK, MUTED = "#1a7f37", "#b3261e", "#14181f", "#5b6470"
placed = []
def put(ax, x, y, s, axes_coords=True, **kw):
    if axes_coords: kw.setdefault("transform", ax.transAxes)
    placed.append((ax, y, axes_coords)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 11.4), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.30, 0.92, 0.30], width_ratios=[1.06, 1.0, 1.0],
                      hspace=0.30, wspace=0.20, left=0.035, right=0.972, top=0.905, bottom=0.035)

fig.suptitle("start_recording's dataset-stack report: the cells that run on an install without a simulator",
             fontsize=16.5, fontweight="bold", y=0.972, color=INK)
fig.text(0.5, 0.938, "Nine cells pin what a caller is told when the LeRobot dataset stack is unusable. "
         "The install most likely to lack the lerobot extra is also the one most likely to lack a simulator.",
         ha="center", fontsize=11.2, color=MUTED)

# --- row 1 col 1: the honored recording path, unchanged ------------------
axf = fig.add_subplot(gs[0, 0]); axf.imshow(frame); axf.set_xticks([]); axf.set_yticks([])
for s in axf.spines.values(): s.set_edgecolor(GREEN); s.set_linewidth(2.4)
axf.set_title("The honored path still records", fontsize=12.4, fontweight="bold", color=GREEN, pad=8)
axf.set_xlabel(f"frame {r['decoded_frames']} of {r['decoded_frames']}, decoded back out of the dataset's own MP4\n"
               f"1 episode / {r['frames']} frames @ {r['fps']}fps  |  {r['mp4_bytes']/1024:.0f} KB  |  "
               f"start/rollout/stop all success\ntests only: no recording behaviour changes",
               fontsize=9.5, color=MUTED, labelpad=8)

# --- row 1 cols 2-3: the nine cells, main vs this branch -----------------
def grid(ax, ran, title, subtitle, colour):
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.axis("off")
    ax.set_title(title, fontsize=12.4, fontweight="bold", color=colour, pad=8)
    for j, b in enumerate(BACKENDS):
        ax.text(j + 0.5, 3.06, b, ha="center", va="bottom", fontsize=10.4, fontweight="bold", color=INK)
    for i, c in enumerate(CAUSES):
        y = 2.5 - i
        ax.text(-0.06, y, c, ha="right", va="center", fontsize=9.3, color=INK)
        for j, b in enumerate(BACKENDS):
            ok = ran[(c, b)]
            ax.add_patch(Rectangle((j + 0.06, y - 0.42), 0.88, 0.84,
                                   facecolor=(GREEN if ok else RED), alpha=0.16,
                                   edgecolor=(GREEN if ok else RED), linewidth=1.5))
            ax.text(j + 0.5, y, "ran" if ok else "SKIPPED", ha="center", va="center",
                    fontsize=9.6, fontweight="bold", color=(GREEN if ok else RED))
    ax.text(1.5, -0.30, subtitle, ha="center", va="top", fontsize=9.7, color=MUTED)

ax1 = fig.add_subplot(gs[0, 1])
grid(ax1, RAN_MAIN, "main  (mujoco absent)",
     f"{F['nine_cells_blocked']['main']['passed']} of 9 cells executed\n"
     f"module: {F['blocked']['main']['passed']} passed / {F['blocked']['main']['skipped']} skipped", RED)
ax2 = fig.add_subplot(gs[0, 2])
grid(ax2, {k: True for k in RAN_MAIN}, "this branch  (mujoco absent)",
     f"{F['nine_cells_blocked']['branch']['passed']} of 9 cells executed\n"
     f"module: {F['blocked']['branch']['passed']} passed / {F['blocked']['branch']['skipped']} skipped", GREEN)

# --- row 2: the measured table -------------------------------------------
axt = fig.add_subplot(gs[1, :]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
rows = [
    ("environment", "measurement", "main", "this branch", None),
    ("mujoco installed", "whole module", f"{F['present']['main']['passed']} passed",
     f"{F['present']['branch']['passed']} passed", "eq"),
    ("mujoco ABSENT", "whole module",
     f"{F['blocked']['main']['passed']} passed / {F['blocked']['main']['skipped']} skipped",
     f"{F['blocked']['branch']['passed']} passed / {F['blocked']['branch']['skipped']} skipped", "fix"),
    ("mujoco ABSENT", "the nine matrix cells",
     f"{F['nine_cells_blocked']['main']['passed']}/9 ran", f"{F['nine_cells_blocked']['branch']['passed']}/9 ran", "fix"),
    ("mujoco ABSENT", "drop MuJoCo's no-recorder-symbol diagnosis",
     f"{F['mutation_blocked']['main']['failed']} failed  (regression invisible)",
     f"{F['mutation_blocked']['branch']['failed']} failed  (regression caught)", "fix"),
]
TOP, LAST = 0.86, 0.16
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
put(axt, 0.5, 1.00, "Measured with the mujoco package blocked at the import system", ha="center",
    fontsize=12.2, fontweight="bold", color=INK)
COLS = (0.015, 0.175, 0.505, 0.735)
y = TOP
for k, (a, b, c, d, kind) in enumerate(rows):
    head = k == 0
    col = INK if head or kind is None else (GREEN if kind == "fix" else MUTED)
    for x, s, mono in zip(COLS, (a, b, c, d), (False, False, True, True), strict=True):
        put(axt, x, y, s, fontsize=(10.3 if head else 9.9), fontweight=("bold" if head else "normal"),
            color=(INK if head else (RED if (mono and x == COLS[2] and kind == "fix") else col)),
            family=("monospace" if mono and not head else "sans-serif"), va="center")
    if head:
        axt.plot([0.012, 0.985], [y - STEP * 0.45] * 2, color="#c8ccd2", lw=1.0, transform=axt.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
put(axt, 0.015, 0.045,
    "The three MuJoCo cells were the skipped ones: their engine was a real Simulation behind "
    "pytest.importorskip(\"mujoco\").\nThe block under test runs before the lock and before any MuJoCo call, "
    "so a __new__ skeleton reaches all three - no compiled model needed.",
    fontsize=9.8, color=MUTED, va="center")

axg = fig.add_subplot(gs[2, :]); axg.axis("off"); axg.set_xlim(0, 1); axg.set_ylim(0, 1)
put(axg, 0.5, 0.55,
    "Gate: MUJOCO_GL=egl pytest tests -> 28176 passed / 257 skipped / 0 failed  |  ruff check + format clean  |  "
    "mypy 0 errors outside examples/  |  tests only, 1 test file, no production line changes",
    ha="center", fontsize=10.2, color=INK, family="monospace")

for ax, yy, axes_coords in placed:
    if axes_coords:
        assert -0.36 <= yy <= 1.08, (ax, yy)
    else:
        lo, hi = ax.get_ylim(); assert lo - 0.05 * abs(hi - lo) <= yy <= hi + 0.10 * abs(hi - lo), (yy, lo, hi)

OUTP = pathlib.Path("_art/dataset_stack_cells_no_simulator.png")
fig.savefig(OUTP, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(OUTP).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print("wrote", OUTP, im.shape)
