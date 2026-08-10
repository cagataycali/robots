import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path("/tmp/art_facts_main.json").read_text())
B = json.loads(pathlib.Path("/tmp/art_facts_pr.json").read_text())
assert A["tree"] != B["tree"], "both arms resolved the same tree"

fb_main = np.load("/tmp/art_frame_before_main.npy")
fb_pr = np.load("/tmp/art_frame_before_pr.npy")
fa_pr = np.load("/tmp/art_frame_after_pr.npy")
assert not pathlib.Path("/tmp/art_frame_after_main.npy").exists(), "main left a decodable frame"

# the caller's recording is bit-for-bit untouched on the branch
untouched = int(np.abs(fb_pr.astype(int) - fa_pr.astype(int)).max())
assert untouched == 0, untouched
# the two seed recordings agree (same deterministic sim, same encoder)
seed_delta = int(np.abs(fb_main.astype(int) - fb_pr.astype(int)).max())

# measured facts the figure asserts
assert (A["before"]["eps"], A["before"]["frames"], A["before_mp4s"]) == (1, 8, 1)
assert (A["after"]["eps"], A["after"]["frames"], A["after_mp4s"]) == (0, 0, 0)
assert A["after_frame_decodable"] is False
assert (B["before"]["eps"], B["before"]["frames"], B["before_mp4s"]) == (1, 8, 1)
assert (B["after"]["eps"], B["after"]["frames"], B["after_mp4s"]) == (1, 8, 1)
assert B["after_frame_decodable"] is True
assert A["ep0_text"] == "run_policy: control_frequency must be > 0, got 0.0."
assert B["summary"] == "run_policy: control_frequency must be > 0, got 0.0."
assert B["ep0_text"] == ""

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 9.6), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.32, 1.0], hspace=0.30, wspace=0.12)

fig.suptitle(
    "run_policy(control_frequency=0.0, dataset_root=...) - the caller's recorded episode",
    fontsize=15.5, fontweight="bold", y=0.975,
)
fig.text(0.5, 0.936,
         "The tool starts its recording with overwrite=True, so the knob the facade refuses is reached "
         "after the existing dataset is gone.\nAll three panels are real LeRobot MP4 frames decoded back "
         "out of the dataset on disk (or their absence).",
         ha="center", fontsize=10.4, color="#333333")

# ---- panel 1: what the caller had ----
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(fb_pr); ax0.set_xticks([]); ax0.set_yticks([])
ax0.set_title("1. Before either call\nmeta/info.json: 1 episode, 8 frames", fontsize=11.2, fontweight="bold")
ax0.set_xlabel("decoded from the dataset's own MP4\n(identical on both trees to "
               f"{seed_delta}/255)", fontsize=9.6)
for sp in ax0.spines.values():
    sp.set_edgecolor("#3d6fb4"); sp.set_linewidth(3)

# ---- panel 2: main ----
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor("#241a1a"); ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
put(ax1, 0.5, 0.72, "NO EPISODE ON DISK", ha="center", va="center", fontsize=17,
    fontweight="bold", color="#ff6b6b", transform=ax1.transAxes)
put(ax1, 0.5, 0.55, "0 MP4 files under the dataset root", ha="center", va="center",
    fontsize=11, color="#ffb3b3", transform=ax1.transAxes)
put(ax1, 0.5, 0.36,
    "meta/info.json:\ntotal_episodes = 0\ntotal_frames = 0",
    ha="center", va="center", fontsize=11.4, family="monospace", color="#ffdddd",
    transform=ax1.transAxes)
put(ax1, 0.5, 0.15, 'returned "error", reason only inside episodes[0]',
    ha="center", va="center", fontsize=9.6, style="italic", color="#e0a0a0",
    transform=ax1.transAxes)
ax1.set_title("2. main: after the refused call\nthe recording was removed", fontsize=11.2,
              fontweight="bold", color="#b3261e")
ax1.set_xlabel("overwrite=True ran before the loop saw the knob", fontsize=9.6, color="#b3261e")
for sp in ax1.spines.values():
    sp.set_edgecolor("#b3261e"); sp.set_linewidth(3)

# ---- panel 3: branch ----
ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(fa_pr); ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title("3. This change: after the refused call\nmeta/info.json: 1 episode, 8 frames",
              fontsize=11.2, fontweight="bold", color="#1e7d32")
ax2.set_xlabel(f"the same MP4, bit-for-bit unchanged (max|delta| = {untouched}/255)",
               fontsize=9.6, color="#1e7d32")
for sp in ax2.spines.values():
    sp.set_edgecolor("#1e7d32"); sp.set_linewidth(3)

# ---- fact table ----
axt = fig.add_subplot(gs[1, :]); axt.axis("off")
axt.set_xlim(0, 1); axt.set_ylim(0, 1)

rows = [
    ("what the caller asked for", "run_policy(..., control_frequency=0.0, dataset_root=<a dataset with 1 episode>)", "#333333"),
    ("main  - dataset after the call", f"total_episodes={A['after']['eps']}, total_frames={A['after']['frames']}, mp4_files={A['after_mp4s']}   (was 1, 8, 1)", "#b3261e"),
    ("main  - what it reported", A["summary"], "#b3261e"),
    ("main  - where the reason was", f'episodes[0]["text"] = {A["ep0_text"]!r}', "#b3261e"),
    ("this change - dataset after", f"total_episodes={B['after']['eps']}, total_frames={B['after']['frames']}, mp4_files={B['after_mp4s']}   (unchanged)", "#1e7d32"),
    ("this change - what it reported", B["summary"], "#1e7d32"),
    ("recorder / facade calls made", "start_recording: 0, run_policy: 0  (the refusal precedes both)", "#1e7d32"),
    ("unusable values measured", "control_frequency in {0, -5, nan, inf, True, '30', None, [30]} and action_horizon in {0, -5, 2.7, nan, True, '8', None, [8]}", "#333333"),
    ("datasets destroyed", "main: 16 of 16          this change: 0 of 16", "#333333"),
    ("reported reason", "byte-identical for all 16 - only the timing changes", "#333333"),
]
TOP, LAST = 0.90, 0.10
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
put(axt, 0.012, 0.985, "Measured", fontsize=12, fontweight="bold", transform=axt.transAxes)
y = TOP
for label, value, colour in rows:
    put(axt, 0.012, y, label, fontsize=10.4, fontweight="bold", va="center",
        color=colour, transform=axt.transAxes)
    put(axt, 0.245, y, value, fontsize=10.0, family="monospace", va="center",
        color=colour, transform=axt.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
axt.add_patch(Rectangle((0.004, LAST - 0.045), 0.992, (TOP - LAST) + 0.10,
                        transform=axt.transAxes, fill=False, edgecolor="#bbbbbb", lw=1.1))

for ax, yy, is_axes in placed:
    lo, hi = (-0.03, 1.10) if is_axes else ax.get_ylim()
    assert lo <= yy <= hi, (yy, lo, hi)

out = pathlib.Path("/tmp/run_policy_knob_preflight.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, (name, n)
print(f"OK {out}  size={im.shape}  seed_delta={seed_delta}  untouched={untouched}")
