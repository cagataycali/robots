import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = json.loads(Path("/tmp/artv2/main/facts.json").read_text())      # upstream/main
B = json.loads(Path("/tmp/artv2/branch/facts.json").read_text())    # this change
assert A["tree"] != B["tree"], "both dumps came from the same tree"
ra = {r["scenario"]: r for r in A["rows"]}
rb = {r["scenario"]: r for r in B["rows"]}

src = np.load("/tmp/artv2/branch/source_frame.npy")
fa = np.load("/tmp/artv2/main/frame_honored.npy")
fb = np.load("/tmp/artv2/branch/frame_honored.npy")

# --- self-audit ---------------------------------------------------------
assert ra["honored"]["frames"] == rb["honored"]["frames"] == 12
assert ra["honored"]["declared"] == rb["honored"]["declared"] == "(3, 240, 320)"
assert ra["honored"]["decoded_frames"] == rb["honored"]["decoded_frames"] == 12
rt = int(np.abs(fa.astype(int) - fb.astype(int)).max())
assert rt == 0, f"honored round-trip differs across trees: {rt}"
assert ra["typo"]["create"] == "success" and ra["typo"]["declared"] == "(3, 480, 640)"
assert ra["typo"]["frames"] == 0 and ra["typo"]["mp4"] is None
assert rb["typo"]["create"] == "REFUSED" and "is not a declared camera" in rb["typo"]["message"]
assert src.shape == fb.shape == (240, 320, 3)
print(f"audit ok: round-trip max|delta| = {rt}; source {src.shape}")

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes)
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.0, 9.6), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.30, wspace=0.14)
fig.suptitle("DatasetRecorder.create: a camera_dims key the schema does not declare",
             fontsize=15, fontweight="bold", y=0.975)
fig.text(0.5, 0.940,
         "12-step headless MuJoCo rollout, camera streaming 240x320, recorded through the "
         "direct DatasetRecorder API. Defaults are video_height/video_width = 480/640.",
         ha="center", fontsize=9.6, style="italic")

# ---- row 1: the real round trip ---------------------------------------
for col, (img, title, sub, colour) in enumerate([
    (src, "1. Rendered frame (source)",
     f"MuJoCo headless, {src.shape[1]}x{src.shape[0]}", "#333333"),
    (fa, "2. Decoded back out of the MP4 - main",
     f"camera_dims={{'image': ...}} correct: {ra['honored']['frames']} frames", "#1b7f3b"),
    (fb, "3. Decoded back out of the MP4 - this change",
     f"byte-identical to panel 2 (max|delta| = {rt})", "#1b7f3b"),
]):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(colour); sp.set_linewidth(2.4)
    ax.set_title(title, fontsize=10.6, fontweight="bold", pad=6)
    ax.set_xlabel(sub, fontsize=9.0, color=colour)

# ---- row 2: the verdict ledger ----------------------------------------
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965,
    "camera_dims={'imagee': (240, 320)}  against  camera_keys=['image']   "
    "(one mistyped camera name)",
    fontsize=11.4, fontweight="bold")

ROWS = [
    ("DatasetRecorder.create(...)", "success", "ValueError, naming the key"),
    ("declared image feature shape", "(3, 480, 640)  <- the global pair", "nothing declared"),
    ("the camera actually streams", "(3, 240, 320)", "(3, 240, 320)"),
    ("frames written to disk", "0", "0 (nothing was created)"),
    ("MP4 produced", "none", "none"),
    ("where the caller finds out", "add_frame, 0 frames in", "at the call, by parameter name"),
    ("dataset directory", "created, unusable", "untouched"),
]
TOP, LAST = 0.845, 0.300
STEP = (TOP - LAST) / (len(ROWS) - 1)
assert STEP > 0.045, STEP
put(ax, 0.015, TOP + 0.075, "", fontsize=1)
for label, x in (("", 0.015), ("upstream/main", 0.375), ("this change", 0.700)):
    if label:
        put(ax, x, TOP + 0.070, label, fontsize=10.6, fontweight="bold",
            color="#a11" if "main" in label else "#1b7f3b")
y = TOP
for label, main_v, pr_v in ROWS:
    put(ax, 0.015, y, label, fontsize=10.0, color="#222")
    put(ax, 0.375, y, main_v, fontsize=10.0, family="monospace", color="#a11")
    put(ax, 0.700, y, pr_v, fontsize=10.0, family="monospace", color="#1b7f3b")
    y -= STEP
assert abs(y + STEP - LAST) < 1e-9, y
put(ax, 0.015, 0.185,
    "main:   " + ra["typo"]["message"] if ra["typo"].get("message") else
    "main:   create() returned success; the mismatch surfaced against add_frame",
    fontsize=9.0, family="monospace", color="#a11")
msg = rb["typo"]["message"].split(": ", 1)[1]
put(ax, 0.015, 0.110, "branch: " + msg[:104], fontsize=9.0, family="monospace", color="#1b7f3b")
put(ax, 0.015, 0.048, "        " + msg[104:206], fontsize=9.0, family="monospace", color="#1b7f3b")

for _ax, yy, is_axes in placed:
    lo, hi = (-0.03, 1.08) if is_axes else _ax.get_ylim()
    assert lo <= yy <= hi, (yy, lo, hi)

out = Path("/tmp/artv2/frame_shape.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(plt.imread(out) * 255).astype(np.uint8)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("WROTE", out, im.shape)
