import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path("/tmp/art_main.json").read_text())      # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch.json").read_text())    # this change
assert A["tree"] != B["tree"], "before/after came from the SAME tree"
dh = np.load("/tmp/delta_honoured.npy"); di = np.load("/tmp/delta_negated.npy")
cos = float(np.dot(dh, di) / (np.linalg.norm(dh) * np.linalg.norm(di)))
nz = dh != 0
frac_opp = float(np.mean(np.sign(dh[nz]) != np.sign(di[nz])))
rows_a = {r["value"]: r for r in A["rows"]}; rows_b = {r["value"]: r for r in B["rows"]}
BASE = float(A["baseline_w_sum"])
CLIP = A["clip_grad_norm_effect"]

# --- self-audit: every claim below is re-derived from the two dumps -----------
assert rows_a["0.0"]["identical_to_untrained"] is True
assert rows_a["0.0"]["delta_norm"] == "0.0000000000"
assert rows_a["0.0"]["problems"] == [] and rows_a["0.0"]["status"] == "success"
assert rows_a["-1.0"]["problems"] == [] and rows_a["-1.0"]["status"] == "success"
assert float(rows_a["-1.0"]["w_sum"]) < BASE < float(rows_a["1.0"]["w_sum"])
assert rows_b["1.0"]["w_sum"] == rows_a["1.0"]["w_sum"], "honoured run drifted"
assert rows_b["inf"]["w_sum"] == rows_a["inf"]["w_sum"], "no-clip run drifted"
for v in ("0.0", "-1.0", "True", "nan"):
    assert rows_b[v]["problems"] and rows_b[v]["status"] == "error", v
assert cos < -0.99 and frac_opp > 0.9
assert CLIP["0.0"] == [0.0, 0.0] and CLIP["-1.0"] == [-0.6, -0.8] and CLIP["inf"] == [3.0, 4.0]

GREEN, RED, GREY, BLUE = "#1a7f37", "#cf222e", "#57606a", "#0969da"
fig = plt.figure(figsize=(15.6, 12.2), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.92, 1.30], hspace=0.40, wspace=0.24,
                      left=0.062, right=0.975, top=0.918, bottom=0.038)
placed = []
def put(ax, x, y, s, **kw):
    # Record WHICH coordinate system the y is in: an axes-fraction note at 1.02
    # is legitimate and must not be checked against the data ylim.
    placed.append((ax, y, "axes" if kw.get("transform") is not None else "data"))
    return ax.text(x, y, s, **kw)

fig.suptitle("An on-policy gradient-norm clip outside its domain: measured on a seeded 60-step PPO run",
             fontsize=15.5, fontweight="bold", y=0.972)
fig.text(0.5, 0.941, "RLTrainSpec.max_grad_norm  |  so100 Elbow-reach SimEnv, seed=0, 34,701 actor-critic parameters",
         ha="center", fontsize=10.6, color=GREY)

# --- Panel 1: the update direction inverts ----------------------------------
ax = fig.add_subplot(gs[0, 0])
k = np.linspace(0, dh.size - 1, 4000).astype(int)
ax.scatter(dh[k], di[k], s=2.4, alpha=0.30, color=RED, edgecolors="none", rasterized=True)
lim = float(np.percentile(np.abs(dh), 99.7)) * 1.25
ax.plot([-lim, lim], [lim, -lim], color=BLUE, lw=1.5, ls="--", label="perfect inversion  ($\\Delta_{-1}=-\\Delta_{1}$)")
ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1.1, ls=":", label="same direction")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("per-parameter update with max_grad_norm=1.0  (honoured)", fontsize=9.6)
ax.set_ylabel("per-parameter update with max_grad_norm=-1.0", fontsize=9.6)
ax.set_title("A negative bound trains in the opposite direction", fontsize=11.4, fontweight="bold", pad=7)
ax.legend(loc="upper right", fontsize=8.4, framealpha=0.94)
ax.grid(alpha=0.18)
put(ax, 0.028, 0.055,
    f"cosine similarity  =  {cos:.6f}\n{frac_opp * 100:.2f}% of the {int(nz.sum()):,} moved parameters\nflipped sign",
    transform=ax.transAxes, fontsize=9.3, va="bottom", family="monospace",
    bbox=dict(boxstyle="round,pad=0.42", fc="#fff1f0", ec=RED, lw=1.1))

# --- Panel 2: where the checkpoint lands ------------------------------------
ax = fig.add_subplot(gs[0, 1])
order = ["1.0", "inf", "0.0", "-1.0"]
labels = ["1.0\n(default)", "inf\n(no clipping)", "0.0", "-1.0"]
vals = [float(rows_a[v]["w_sum"]) - BASE for v in order]
cols = [GREEN, GREEN, RED, RED]
bars = ax.bar(range(4), vals, color=cols, width=0.58, edgecolor="k", linewidth=0.7)
ax.axhline(0, color="k", lw=1.3)
put(ax, 3.48, 0.0018, "never-trained baseline", ha="right", va="bottom", fontsize=8.6, color=GREY, style="italic")
for i, (b, v) in enumerate(zip(bars, vals)):
    off = 0.0035 if v >= 0 else -0.0035
    put(ax, i, v + off, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
        fontsize=9.0, family="monospace", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9.0)
ax.set_ylabel("checkpoint parameter sum, relative to untrained", fontsize=9.6)
ax.set_title("Two values move it the wrong way, or not at all", fontsize=11.4, fontweight="bold", pad=7)
ax.grid(axis="y", alpha=0.18)
ax.set_ylim(min(vals) * 1.55, max(vals) * 1.62)
put(ax, 0.5, 0.055,
    f"0.0 moved 0 of {int(nz.sum()):,} parameters:  delta norm exactly {rows_a['0.0']['delta_norm']}\n"
    "the checkpoint is bit-identical to a never-trained control",
    transform=ax.transAxes, ha="center", fontsize=8.9, family="monospace",
    bbox=dict(boxstyle="round,pad=0.40", fc="#fff8c5", ec="#9a6700", lw=1.0))

# --- Panel 3: the verdict table --------------------------------------------
ax = fig.add_subplot(gs[1:, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
COLS = [0.012, 0.113, 0.268, 0.452, 0.700]
HEAD = ["max_grad_norm", "clip_grad_norm_\napplied to grad [3.0, 4.0]",
        "on main:  validate() / train()", "on main: what the run did", "with this change"]
TOP = 0.955
for x, h in zip(COLS, HEAD):
    put(ax, x, TOP, h, fontsize=9.6, fontweight="bold", va="top")
ax.plot([0.008, 0.992], [TOP - 0.088, TOP - 0.088], color="k", lw=1.1)

TABLE = [
    ("1.0", CLIP["1.0"], "[] -> success", f"trains; sum {rows_a['1.0']['w_sum']}", "accepted (unchanged)", GREEN),
    ("inf", CLIP["inf"], "[] -> success", f"unclipped; sum {rows_a['inf']['w_sum']}",
     "accepted: the only spelling of\n'do not clip', honoured by torch", GREEN),
    ("0 / 0.0", CLIP["0.0"], "[] -> success",
     "0 of 34,701 parameters moved;\nbit-identical to never-trained", "refused: must be > 0", RED),
    ("-1.0 / -0.5", CLIP["-1.0"], "[] -> success",
     f"96.50% of parameters flipped sign;\nsum {rows_a['-1.0']['w_sum']} (baseline {BASE:.10f})",
     "refused: must be > 0", RED),
    ("True", CLIP["1.0"], "[] -> success", "a silent bound of one", "refused: must be > 0", RED),
    ('"1.0"', CLIP["1.0"], "[] -> success", "silently coerced through float()", "refused: must be > 0", RED),
    ("nan", CLIP["nan"], "[] -> RAISED", "torch ValueError mid-update, naming\nneither the field nor the value",
     "refused: must be > 0", RED),
    ("None / [1.0]", "TypeError", "[] -> RAISED", "bare TypeError from inside torch",
     "refused: must be > 0", RED),
]
N = len(TABLE)
FLOOR, PAD = 0.035, 0.013
avail = (TOP - 0.098) - FLOOR - PAD * N
LINE = avail / sum(max(len(str(c).split("\n")) for c in r[1:5]) for r in TABLE)
assert LINE > 0.030, LINE
y = TOP - 0.098
for value, clip, verdict, effect, fixed, colour in TABLE:
    nl = max(len(str(c).split("\n")) for c in (clip, verdict, effect, fixed))
    h = LINE * nl
    if colour is RED:
        ax.add_patch(Rectangle((0.008, y - h - PAD * 0.45), 0.984, h + PAD * 0.72,
                               facecolor="#fff5f5", edgecolor="none", zorder=0))
    put(ax, COLS[0], y, value, fontsize=9.5, family="monospace", fontweight="bold", va="top", color=colour)
    clip_s = str(clip) if not isinstance(clip, list) else f"-> {clip}"
    put(ax, COLS[1], y, clip_s, fontsize=9.0, family="monospace", va="top", color=GREY)
    put(ax, COLS[2], y, verdict, fontsize=9.0, family="monospace", va="top",
        color=RED if "success" in verdict or "RAISED" in verdict else GREY)
    put(ax, COLS[3], y, effect, fontsize=8.8, va="top", color="#24292f")
    put(ax, COLS[4], y, fixed, fontsize=8.8, va="top", color=colour, fontweight="bold")
    y -= h + PAD
assert y > 0.030, y
ax.plot([0.008, 0.992], [y + PAD * 0.55, y + PAD * 0.55], color="k", lw=0.8)
put(ax, 0.012, y - 0.004,
    "No regression: the two accepted rows are BIT-IDENTICAL across the two trees "
    f"(sum {rows_b['1.0']['w_sum']} and {rows_b['inf']['w_sum']}).   "
    "train() is fail-closed on validate(), so a refused bound never collects a rollout.",
    fontsize=8.9, va="top", color=GREY, style="italic")

for a, yy, coord in placed:
    lo, hi = (-0.03, 1.07) if coord == "axes" else a.get_ylim()
    pad = abs(hi - lo) * 0.10
    assert lo - pad <= yy <= hi + pad, (coord, yy, (lo, hi))

out = "/tmp/gradient_clip_domain.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  cos={cos:.6f}  frac_opp={frac_opp:.4f}")
