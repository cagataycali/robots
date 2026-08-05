"""Compose the verify-dataset gate figure from the two measured trees."""
from __future__ import annotations
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path("/tmp/art_main/facts.json").read_text())    # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch/facts.json").read_text())  # this change
assert A["tree"] != B["tree"], "both dumps came from the same tree"

fa = np.load(A["frame_npy"]); fb = np.load(B["frame_npy"])
sat = ((fb.max(2).astype(int) - fb.min(2).astype(int)) > 45).mean()
assert sat > 0.03, f"decoded frame looks blank ({sat:.4f})"
frame_delta = int(np.abs(fa.astype(int) - fb.astype(int)).max())

# --- self-audit every claim the figure makes -------------------------------
assert A["cli"]["corrupt_min_frames_neg5"] == 0, A["cli"]
assert B["cli"]["corrupt_min_frames_neg5"] == 1, B["cli"]
assert A["cli"]["corrupt_min_frames_0"] == 0 and B["cli"]["corrupt_min_frames_0"] == 0
assert A["cli"]["corrupt_default"] == 1 and B["cli"]["corrupt_default"] == 1
assert A["cli"]["healthy_default"] == 0 and B["cli"]["healthy_default"] == 0
assert A["corruption"]["reader_frames_per_episode"] == [24, 24, 0]
assert B["corruption"]["reader_frames_per_episode"] == [24, 24, 0]
am = {r["value"]: r for r in A["verdicts"]}
bm = {r["value"]: r for r in B["verdicts"]}
CERTIFIED = [v for v, r in am.items() if r["outcome"] == "success" and v != "0"]
RAISED = [v for v, r in am.items() if r["outcome"].startswith("raised")]
assert set(CERTIFIED) == {"-5", "False", "nan"}, CERTIFIED
assert set(RAISED) == {"'2'", "None"}, RAISED
assert all(r["outcome"] == "error" for v, r in bm.items() if v not in ("1", "0"))
assert bm["0"]["outcome"] == "success" and bm["1"]["named_the_short_episode"]

placed: list[tuple] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.0, 9.4), dpi=125)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.32], height_ratios=[1.0, 0.40],
                      hspace=0.20, wspace=0.10, left=0.035, right=0.972, top=0.895, bottom=0.045)

fig.suptitle("A dataset-integrity gate must not be switched off by the threshold that drives it",
             fontsize=17, fontweight="bold", y=0.972)
fig.text(0.5, 0.930,
         "Real MuJoCo rollout recorded to a LeRobot dataset (3 episodes x 24 frames @ 30 fps), then episode 2's "
         "length set to 0 - the on-disk signature of a zero-length episode.\n"
         "verify_dataset's check 2 runs only when min_frames > 0, so a value failing that comparison does not fail "
         "loudly: it skips the check.",
         ha="center", va="top", fontsize=10.6, color="#333333")

# ---- panel 1: the recording under test, decoded back out of its own MP4 ----
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(fb); ax0.set_xticks([]); ax0.set_yticks([])
ax0.set_title("The dataset under test is a real recording", fontsize=11.5, fontweight="bold", pad=7)
ax0.set_xlabel(
    f"frame decoded back out of {pathlib.Path(B['recording']['mp4']).name}\n"
    f"{B['recording']['decoded_frames']} frames, {B['recording']['mp4_kb']} KB, "
    f"stop_recording={B['recording']['stop_status']}\n"
    f"reader sees frames_per_episode = {B['corruption']['reader_frames_per_episode']}",
    fontsize=9.4, labelpad=7)
for sp in ax0.spines.values(): sp.set_edgecolor("#3f7fbf"); sp.set_linewidth(2.0)

# ---- panel 2: the verdict table -------------------------------------------
ax1 = fig.add_subplot(gs[0, 1]); ax1.axis("off"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ORDER = ["1", "0", "-5", "False", "nan", "2.7", "'2'", "None"]
LABEL = {"1": "1  (default)", "0": "0  (documented skip)", "-5": "-5", "False": "False",
         "nan": "nan", "2.7": "2.7", "'2'": "'2'", "None": "None"}
cols = [0.005, 0.175, 0.560]
put(ax1, cols[0], 0.968, "--min-frames", fontsize=11, fontweight="bold")
put(ax1, cols[1], 0.968, "on main", fontsize=11, fontweight="bold")
put(ax1, cols[2], 0.968, "with this change", fontsize=11, fontweight="bold")
ax1.plot([0.0, 1.0], [0.945, 0.945], color="#333333", lw=1.3)

RED, GREEN, GREY = "#c0392b", "#1e8449", "#5d6d7e"
def describe(r):
    if r["outcome"].startswith("raised"):
        return f"{r['outcome']}  (no report)", RED, "escaped"
    if r["outcome"] == "success":
        return "status=success  -> CERTIFIED", RED, "certified"
    if r["named_the_short_episode"]:
        return "status=error  names episode 2 = 0 frame(s)", GREEN, "caught"
    return "status=error  refuses the threshold", GREEN, "refused"

y, step = 0.885, 0.1065
for key in ORDER:
    a_txt, a_col, a_kind = describe(am[key])
    b_txt, b_col, b_kind = describe(bm[key])
    bad = a_kind in ("certified", "escaped")
    if bad:
        ax1.add_patch(Rectangle((0.0, y - 0.036), 1.0, 0.088, transform=ax1.transData,
                                facecolor="#c0392b", alpha=0.085, zorder=0))
    put(ax1, cols[0], y, LABEL[key], fontsize=11.4, family="monospace",
        fontweight="bold" if bad else "normal")
    put(ax1, cols[1], y, a_txt, fontsize=10.4, color=a_col,
        fontweight="bold" if bad else "normal")
    put(ax1, cols[2], y, b_txt, fontsize=10.4, color=b_col)
    if key == "2.7":
        put(ax1, cols[1], y - 0.043, 'reported as "below min_frames=2.7"', fontsize=8.6,
            color=GREY, style="italic")
    y -= step

put(ax1, cols[0], 0.055,
    f"main certifies the corrupt dataset for {len(CERTIFIED)} of 8 thresholds and escapes as a bare "
    f"TypeError for {len(RAISED)} more;\nwith this change every unusable threshold is reported, and 0 keeps "
    "skipping the check exactly as documented.",
    fontsize=10.4, fontweight="bold")

# ---- panel 3: the CLI band ------------------------------------------------
ax2 = fig.add_subplot(gs[1, :]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.005, 0.90, "The same gate from the shell, where it is used as a CI check "
    "(exit 0 = this dataset is certified)", fontsize=11.5, fontweight="bold")
ax2.plot([0.0, 1.0], [0.80, 0.80], color="#333333", lw=1.1)
CLI = [
    ("strands-robots verify-dataset <corrupt>", "corrupt_default", "the zero-length episode is caught"),
    ("strands-robots verify-dataset <corrupt> --min-frames -5", "corrupt_min_frames_neg5",
     "a threshold no episode count can be measured against"),
    ("strands-robots verify-dataset <corrupt> --min-frames 0", "corrupt_min_frames_0",
     "the documented way to skip the length check"),
    ("strands-robots verify-dataset <healthy>", "healthy_default", "an intact recording, unchanged"),
]
yy = 0.66
for cmd, key, note in CLI:
    a, b = A["cli"][key], B["cli"][key]
    put(ax2, 0.005, yy, cmd, fontsize=10.4, family="monospace")
    put(ax2, 0.470, yy, f"main exit {a}", fontsize=10.4, family="monospace",
        color=RED if (a == 0 and key == "corrupt_min_frames_neg5") else GREY,
        fontweight="bold" if a != b else "normal")
    put(ax2, 0.575, yy, f"this change exit {b}", fontsize=10.4, family="monospace",
        color=GREEN if a != b else GREY, fontweight="bold" if a != b else "normal")
    put(ax2, 0.740, yy, note, fontsize=9.6, color=GREY, style="italic")
    yy -= 0.175
put(ax2, 0.005, -0.02,
    f"No recording behaviour changes: the frame decoded out of each tree's own dataset differs by "
    f"max {frame_delta}/255 across the two independent runs.  "
    "Both trees record 3 episodes x 24 frames and both certify the intact dataset.",
    fontsize=9.8, color="#333333")

for ax, yv in placed:
    lo, hi = ax.get_ylim()
    assert lo - 0.06 <= yv <= hi + 0.08, f"text at y={yv} outside {ax.get_ylim()}"

out = pathlib.Path("/tmp/verify_dataset_gate.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.array(plt.imread(out) * 255).astype(int)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 20).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  frame_delta={frame_delta}  "
      f"certified={CERTIFIED} raised={RAISED} sat={sat:.4f}")
