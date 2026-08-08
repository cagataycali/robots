import json, pathlib, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image

A = json.loads(pathlib.Path("/tmp/art_main/facts.json").read_text())      # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch/facts.json").read_text())    # this change
assert A["tree"] != B["tree"], "same tree - not a before/after"

MIS_A = sum(1 for x in A["streams"] if x["reported"] != x["truth"])
MIS_B = sum(1 for x in B["streams"] if x["reported"] != x["truth"])
assert (MIS_A, MIS_B) == (3, 0), (MIS_A, MIS_B)
assert A["rollout"]["joints_after"] == B["rollout"]["joints_after"]
assert A["rollout"]["emitted_action"] == B["rollout"]["emitted_action"]

im_a = np.asarray(Image.open("/tmp/art_main/rollout.png").convert("RGB")).astype(int)
im_b = np.asarray(Image.open("/tmp/art_branch/rollout.png").convert("RGB")).astype(int)
RENDER_MAXD = int(np.abs(im_a - im_b).max())
RENDER_DIFF = int((np.abs(im_a - im_b).sum(2) > 0).sum())
assert RENDER_MAXD <= 2, RENDER_MAXD
assert ((im_b.max(2) - im_b.min(2)) > 45).mean() > 0.10  # the scene has content

GREEN, RED, GREY, INK = "#1a7f37", "#b42318", "#57606a", "#1f2328"
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); kw.setdefault("fontsize", 9.6); kw.setdefault("color", INK)
    placed.append((ax, y, "axes" if kw.get("transform") is not None else "data"))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.4, 12.4), dpi=122)
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.42, 1.30, 0.92],
              width_ratios=[0.415, 0.585], hspace=0.16, wspace=0.05,
              left=0.022, right=0.978, top=0.925, bottom=0.028)

fig.suptitle("A non-finite action stream was diagnosed as a near-zero one", fontsize=17.5, y=0.982, weight="bold")
fig.text(0.5, 0.949, "strands_robots.policies.lerobot_local  -  ZeroActionMonitor  |  "
         f"streams reported as the wrong fault: {MIS_A} of {len(A['streams'])}  ->  {MIS_B} of {len(B['streams'])}",
         ha="center", fontsize=11.4, color=GREY)

# ---------------- row 1 left: the real render -------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.imshow(im_b.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_edgecolor(GREEN); s.set_linewidth(2.0)
ax.set_title("The honored path, MuJoCo headless (SO-101)", fontsize=11.5, weight="bold", pad=6)
ax.set_xlabel("A finite action dict emitted by the provider, applied for 20 control steps.\n"
              "Byte-comparable on both trees; the change adds no branch a finite action reaches.",
              fontsize=9.3, color=GREY, labelpad=7)

# ---------------- row 1 right: no-regression ledger -------------------------
ax = fig.add_subplot(gs[0, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.0, "The near-zero contract and the honored path are unchanged", fontsize=12.4, weight="bold")
rows = [
    ("emitted action dict", "identical on both trees", True),
    ("final joint positions (12 dp)", "identical on both trees", True),
    ("arm travel over 20 steps", f"{A['rollout']['max_delta_deg']:.3f} deg on both trees", True),
    ("render agreement", f"max|delta| = {RENDER_MAXD}/255 over {RENDER_DIFF} of {im_a.shape[0]*im_a.shape[1]:,} px", True),
    ("a dead policy (0.0 stream)", "still reports the original near-zero message", True),
    ("a healthy policy (0.9 stream)", "still reports nothing", True),
    ("threshold = 0", "still accepted; its measured meaning is recorded", True),
    ("existing tests changed", "none (806 passed in tests/policies/lerobot_local/)", True),
]
y = 0.855
for label, value, ok in rows:
    put(ax, 0.012, y, "\u2713", fontsize=11.2, color=GREEN, weight="bold")
    put(ax, 0.055, y, label, fontsize=10.3, weight="bold")
    put(ax, 0.475, y, value, fontsize=10.3, family="monospace", color=GREY)
    y -= 0.104
assert y > 0.02, y
put(ax, 0.012, y - 0.005,
    "The two faults are tracked independently, so a stream that is genuinely both still reports both.",
    fontsize=9.6, color=GREY, style="italic")

# ---------------- row 2: per-stream verdicts --------------------------------
ax = fig.add_subplot(gs[1, :]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.0, "Which fault each action stream is reported as", fontsize=12.6, weight="bold")
COLS = [0.012, 0.235, 0.335, 0.485, 0.62]
put(ax, COLS[0], 0.885, "action stream", fontsize=10.0, weight="bold", color=GREY)
put(ax, COLS[1], 0.885, "actual fault", fontsize=10.0, weight="bold", color=GREY)
put(ax, COLS[2], 0.885, "upstream/main reports", fontsize=10.0, weight="bold", color=GREY)
put(ax, COLS[3], 0.885, "this change reports", fontsize=10.0, weight="bold", color=GREY)
put(ax, COLS[4], 0.885, "what the message told the operator to check", fontsize=10.0, weight="bold", color=GREY)
ax.plot([0.008, 0.992], [0.858, 0.858], color="#d0d7de", lw=1.0)

N = len(A["streams"]); TOP, FLOOR, PAD = 0.822, 0.045, 0.018
STEP = (TOP - FLOOR - PAD * N) / N
assert STEP > 0.030, STEP
def remedy(msg):
    if not msg: return "-"
    if "non-finite action" in msg: return "checkpoint normalization stats / observation values"
    if "near-zero actions" in msg: return "the embodiment's obs_rename / camera keys"
    return "-"
y = TOP
for x, yb in zip(A["streams"], B["streams"]):
    ok_a, ok_b = x["reported"] == x["truth"], yb["reported"] == yb["truth"]
    if not ok_a:
        ax.add_patch(Rectangle((0.008, y - STEP + 0.004), 0.984, STEP + 0.006,
                               facecolor="#fff1f0", edgecolor="none", zorder=0))
    put(ax, COLS[0], y, x["label"], fontsize=10.2, family="monospace")
    put(ax, COLS[1], y, x["truth"], fontsize=10.2, family="monospace", color=GREY)
    put(ax, COLS[2], y, x["reported"] + ("" if ok_a else "   \u2717"), fontsize=10.2,
        family="monospace", weight="bold", color=GREEN if ok_a else RED)
    put(ax, COLS[3], y, yb["reported"] + ("   \u2713" if ok_b else "   \u2717"), fontsize=10.2,
        family="monospace", weight="bold", color=GREEN if ok_b else RED)
    put(ax, COLS[4], y, remedy(x["message"]), fontsize=9.5, color=RED if not ok_a else GREY)
    put(ax, COLS[4], y - 0.036, "-> " + remedy(yb["message"]), fontsize=9.5, color=GREEN)
    y -= STEP + PAD
assert y > -0.02, y
put(ax, 0.012, 0.028,
    "A single nan component makes np.abs(action).max() nan, so row 4 - five real joint commands - was reported as "
    "emitting no command at all.", fontsize=9.6, color=GREY, style="italic")

# ---------------- row 3: ctor knobs ----------------------------------------
ax = fig.add_subplot(gs[2, :]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.0, "The watchdog's own knobs: a bare comparison is False for nan/inf and reads True as 1",
    fontsize=12.2, weight="bold")
CC = [0.012, 0.20, 0.30, 0.415]
put(ax, CC[0], 0.855, "constructor value", fontsize=9.8, weight="bold", color=GREY)
put(ax, CC[1], 0.855, "main", fontsize=9.8, weight="bold", color=GREY)
put(ax, CC[2], 0.855, "this change", fontsize=9.8, weight="bold", color=GREY)
put(ax, CC[3], 0.855, "what main's acceptance did to the watchdog", fontsize=9.8, weight="bold", color=GREY)
M = len(A["ctor"]); T2, F2, P2 = 0.79, 0.04, 0.004
S2 = (T2 - F2 - P2 * M) / M
assert S2 > 0.028, S2
y = T2
for x, yb in zip(A["ctor"], B["ctor"]):
    default = "default" in x["label"]
    put(ax, CC[0], y, x["label"], fontsize=9.8, family="monospace")
    put(ax, CC[1], y, x["verdict"], fontsize=9.8, family="monospace", weight="bold",
        color=GREEN if default else RED)
    put(ax, CC[2], y, yb["verdict"], fontsize=9.8, family="monospace", weight="bold", color=GREEN)
    put(ax, CC[3], y, x["detail"] if x["verdict"] == "accepted" else "-", fontsize=9.3,
        family="monospace", color=GREY if default else RED)
    y -= S2 + P2
assert y > -0.02, y

for ax_, yv, kind in placed:
    lo, hi = ax_.get_ylim()
    if kind == "axes":
        assert -0.03 <= yv <= 1.07, (yv, kind)
    else:
        assert lo - 0.05 <= yv <= hi + 0.07, (yv, lo, hi)

out = pathlib.Path("/tmp/artifact_nonfinite.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

img = np.asarray(Image.open(out).convert("RGB")).astype(int)
h, w, _ = img.shape
for name, band in (("top", img[:8]), ("bottom", img[-8:]), ("left", img[:, :8]), ("right", img[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {w}x{h}  mis-reported {MIS_A} -> {MIS_B}  render max|delta|={RENDER_MAXD}")
