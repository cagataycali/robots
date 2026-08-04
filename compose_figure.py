import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

A = json.load(open("/tmp/argv_main.json")); B = json.load(open("/tmp/argv_branch.json"))
assert A["tree"] != B["tree"]

# ---- measured facts, asserted before anything is drawn ----
leaked = [(x, y) for x, y in zip(A["rows"], B["rows"])
          if x["outcome"] == "built" and y["outcome"] == "refused"]
assert len(leaked) == 15, len(leaked)
assert all(A["falsy"][k]["flag_present"] is False for k in A["falsy"])
assert all(B["falsy"][k]["outcome"] == "refused" for k in B["falsy"])
identical = [k for k in A["honored"] if A["honored"][k]["argv"] == B["honored"][k]["argv"]]
assert len(identical) == 5 and len(A["honored"]) == 6, (len(identical), len(A["honored"]))
coerced = [k for k in A["honored"] if k not in identical]
assert coerced == ["record fps=30.0"], coerced
assert A["honored"]["record fps=30.0"]["token"] == "30.0"
assert B["honored"]["record fps=30.0"]["token"] == "30"

RED, GRN, AMB, INK = "#b3261e", "#1b6e3c", "#8a5a00", "#1a1a1a"
fig = plt.figure(figsize=(14.6, 12.4))
gs = fig.add_gridspec(3, 2, height_ratios=[2.55, 0.92, 0.60], hspace=0.30, wspace=0.10)
placed = []
def put(ax, x, y, s, **kw):
    placed.append(y); return ax.text(x, y, s, **kw)

fig.suptitle("build_lerobot_command: the value that reached the lerobot command line",
             fontsize=17, fontweight="bold", y=0.972)
fig.text(0.5, 0.945,
         "Every numeric option is interpolated with str() into the argv of a subprocess launched with "
         "start_new_session=True.\nThat detached process cannot report back to the caller: the session starts, "
         "status=\"success\" is returned with a pid.",
         ha="center", fontsize=10.6, color="#40464d")

# ---------------- Panel A: the leak table ----------------
ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.5, 1.055, "main (5221459a) --  what the CLI received      |      this change", ha="center",
    fontsize=12.5, fontweight="bold", color=INK)
cols = (0.035, 0.235, 0.325, 0.595)
hdr = 0.985
for x, lbl in zip(cols, ("mode", "option", "caller passed", "argv token handed to lerobot")):
    put(ax, x, hdr, lbl, fontsize=10.3, fontweight="bold", color="#5a6169")
put(ax, 0.815, hdr, "this change", fontsize=10.3, fontweight="bold", color="#5a6169")
ax.plot([0.02, 0.98], [hdr - 0.022, hdr - 0.022], color="#c9ced4", lw=1.0)

step = 0.0565
for i, (x, y) in enumerate(leaked):
    yy = hdr - 0.055 - i * step
    ax.add_patch(Rectangle((0.02, yy - 0.021), 0.96, step * 0.86,
                           facecolor="#fdecea" if i % 2 == 0 else "#fbe3e0",
                           edgecolor="none", zorder=0))
    put(ax, cols[0], yy, x["mode"], fontsize=10.0, color="#5a6169", family="monospace")
    put(ax, cols[1], yy, x["knob"], fontsize=10.0, color=INK, family="monospace")
    put(ax, cols[3] - 0.27, yy, x["value"], fontsize=10.2, color=INK, family="monospace",
        fontweight="bold", ha="right")
    flag = {"dataset_fps": "--dataset.fps", "dataset_num_episodes": "--dataset.num_episodes",
            "dataset_episode_time_s": "--dataset.episode_time_s",
            "replay_episode": "--dataset.episode"}[x["knob"]]
    put(ax, cols[3], yy, f"{flag} {x['token']}", fontsize=10.1, color=RED,
        family="monospace", fontweight="bold")
    put(ax, 0.815, yy, "refused, no argv built", fontsize=10.0, color=GRN, family="monospace")
put(ax, 0.5, hdr - 0.055 - len(leaked) * step - 0.012,
    f"{len(leaked)} of {len(leaked)} values no run can be given were placed on the argv and reported success.",
    ha="center", fontsize=11.0, color=RED, fontweight="bold")

# ---------------- Panel B: the two truthiness reads ----------------
ax2 = fig.add_subplot(gs[1, :]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.5, 1.03, "0 read for truthiness meant the OPPOSITE of the request", ha="center",
    fontsize=12.5, fontweight="bold", color=INK)
notes = [
    ("teleop_time_s=0", "\"stop at once\"",
     "no --teleop_time_s emitted  ->  unbounded session", "refused"),
    ("replay dataset_fps=0", "an unusable rate",
     "no --dataset.fps emitted  ->  lerobot's default, not the caller's", "refused"),
]
for i, (call, meant, was, now) in enumerate(notes):
    yy = 0.80 - i * 0.40
    ax2.add_patch(Rectangle((0.02, yy - 0.15), 0.96, 0.30, facecolor="#fff6e5",
                            edgecolor="#e6c47a", lw=0.9, zorder=0))
    put(ax2, 0.045, yy + 0.055, call, fontsize=11.2, family="monospace", fontweight="bold", color=INK)
    put(ax2, 0.045, yy - 0.075, f"caller meant: {meant}", fontsize=9.9, color="#5a6169")
    put(ax2, 0.30, yy + 0.055, f"main:  {was}", fontsize=10.5, family="monospace", color=AMB, fontweight="bold")
    put(ax2, 0.30, yy - 0.075, f"this change:  {now}", fontsize=10.5, family="monospace", color=GRN)

# ---------------- Panel C: no-regression ledger ----------------
ax3 = fig.add_subplot(gs[2, :]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.5, 1.02, "A call that can be honored is unchanged", ha="center",
    fontsize=12.5, fontweight="bold", color=INK)
put(ax3, 0.045, 0.70,
    f"{len(identical)} of {len(A['honored'])} honored calls build a BYTE-IDENTICAL argv on both trees:",
    fontsize=10.8, color=INK)
put(ax3, 0.075, 0.47, "   ".join(f"{k}" for k in identical), fontsize=9.9,
    family="monospace", color=GRN)
put(ax3, 0.045, 0.21,
    "The 1 deliberate difference:  record fps=30.0   ->   main '30.0'   |   this change '30'"
    "     (lerobot declares DatasetRecordConfig.fps an int, so the token must be an int literal)",
    fontsize=10.2, family="monospace", color="#2b5fa8")

assert all(0.0 <= y <= 1.07 for y in placed), [y for y in placed if not 0.0 <= y <= 1.07]
out = "/tmp/artifact_teleop_argv.png"
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.34, facecolor="white")
plt.close(fig)
im = np.asarray(Image.open(out).convert("RGB")).astype(int)
b = 8
border = np.concatenate([im[:b].ravel(), im[-b:].ravel(), im[:, :b].ravel(), im[:, -b:].ravel()])
assert int((border != 255).sum()) == 0, int((border != 255).sum())
red = int((np.abs(im - np.array([179, 38, 30])).sum(2) < 60).sum())
grn = int((np.abs(im - np.array([27, 110, 60])).sum(2) < 60).sum())
assert red > 2000 and grn > 2000, (red, grn)
print(f"OK {out}  size={Image.open(out).size}  red_px={red}  green_px={grn}  border_clean")
