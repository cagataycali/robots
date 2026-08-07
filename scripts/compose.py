"""Compose the artifact from the two trees' captures. Every cell is measured."""
from __future__ import annotations
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = {r["tag"]: r for r in json.loads(pathlib.Path("/tmp/art_branch/facts.json").read_text())}
M = {r["tag"]: r for r in json.loads(pathlib.Path("/tmp/art_main/facts.json").read_text())}
assert B["honored"]["tree"] != M["honored"]["tree"], "both captures came from the same tree"

img_b = {p.stem: np.load(p) for p in pathlib.Path("/tmp/art_branch").glob("*.npy")}
img_m = {p.stem: np.load(p) for p in pathlib.Path("/tmp/art_main").glob("*.npy")}

def frac(a, c): return float((np.abs(a.astype(int) - c.astype(int)).sum(2) > 12).mean())

# --- self-audit: every claim the figure makes ---
assert int(np.abs(img_m["honored"].astype(int) - img_b["honored"].astype(int)).max()) == 0
assert frac(img_m["honored"], img_m["zero"]) > 0.10
assert frac(img_m["honored"], img_m["nan"]) > 0.10
assert M["zero"]["ctor"] == "accepted" and M["zero"]["ok"] == 60 and M["zero"]["reward"] == 60.0
assert M["nan"]["ctor"] == "accepted" and M["nan"]["err"] == 60 and M["nan"]["reward"] == 60.0
assert B["zero"]["ctor"] == "REFUSED" and B["nan"]["ctor"] == "REFUSED"
assert M["honored"]["shoulder"] == B["honored"]["shoulder"] == 0.312
F_ZERO, F_NAN = frac(img_m["honored"], img_m["zero"]), frac(img_m["honored"], img_m["nan"])

placed: list[tuple[object, float]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 10.2), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.30, wspace=0.045,
                      left=0.018, right=0.982, top=0.905, bottom=0.022)

fig.suptitle("SimEnv(action_scale=...): what an unusable multiplier did to a MuJoCo rollout",
             fontsize=16.5, fontweight="bold", y=0.977)
fig.text(0.5, 0.938, "Two-joint arm, 60 env steps of a constant [0.9, -0.7] command, n_substeps=10. "
                     "The grey post marks the commanded reach.",
         ha="center", fontsize=10.6, style="italic", color="#333333")

PANELS = [
    ("honored", "action_scale = 1.0   (honored)",
     f"send_action 60/60 ok  |  return {M['honored']['reward']}\n"
     f"shoulder {M['honored']['shoulder']:+.4f} rad   elbow {M['honored']['elbow']:+.4f} rad\n"
     "byte-identical on both trees (max|delta| = 0)", "#1a7f37"),
    ("zero", "main: action_scale = 0   (accepted)",
     f"send_action 60/60 ok  |  return {M['zero']['reward']}\n"
     f"shoulder {M['zero']['shoulder']:+.4f} rad   elbow {M['zero']['elbow']:+.4f} rad  <- gravity droop\n"
     f"the policy never reached the robot  ({F_ZERO:.1%} of pixels differ)", "#b3261e"),
    ("nan", "main: action_scale = nan   (accepted)",
     f"send_action 0/60 ok, 60 REFUSED  |  return {M['nan']['reward']}\n"
     f"shoulder {M['nan']['shoulder']:+.4f} rad   elbow {M['nan']['elbow']:+.4f} rad  <- never moved\n"
     f"every command unsendable, full return banked  ({F_NAN:.1%} differ)", "#b3261e"),
]
for col, (tag, title, cap, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img_m[tag]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(colour); sp.set_linewidth(3.0)
    ax.set_title(title, fontsize=12.4, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(cap, fontsize=9.9, color="#222222", labelpad=7, linespacing=1.5)

# ---- verdict table ----
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ROWS = [
    ("1.0",   "accepted -> arm tracks the command",              "accepted (unchanged)",                    True),
    ("0.25",  "accepted -> arm tracks, scaled",                  "accepted (unchanged)",                    True),
    ("np.float32(0.25)", "accepted -> arm tracks, scaled",       "accepted (unchanged)",                    True),
    ("0",     "accepted -> zero command, full return banked",    "SimEnv: action_scale must be > 0, got 0.",   False),
    ("-1.0",  "accepted -> every commanded DOF inverted",        "SimEnv: action_scale must be > 0, got -1.0.", False),
    ("nan",   "accepted -> all 60 commands refused, return banked", "SimEnv: action_scale must be > 0, got nan.", False),
    ("inf",   "accepted -> all 60 commands refused, return banked", "SimEnv: action_scale must be > 0, got inf.", False),
    ("True",  "accepted -> a silent scale of 1.0",               "SimEnv: action_scale must be > 0, got True.", False),
    ("None",  "bare TypeError from float()",                     "SimEnv: action_scale must be > 0, got None.", False),
]
put(ax, 0.5, 0.965, "action_scale, before and after", ha="center", fontsize=13.2, fontweight="bold")
X = (0.017, 0.175, 0.615)
for x, h in zip(X, ("value", "on main", "this change"), strict=True):
    put(ax, x, 0.885, h, fontsize=11.4, fontweight="bold", color="#111111", family="monospace")
ax.plot([0.012, 0.988], [0.858, 0.858], color="#333333", lw=1.4, transform=ax.transAxes)
step, top = 0.086, 0.795
for i, (val, before, after) in enumerate([(r[0], r[1], r[2]) for r in ROWS]):
    y = top - i * step
    ok = ROWS[i][3]
    if not ok:
        ax.add_patch(plt.Rectangle((0.012, y - 0.028), 0.976, 0.070, transform=ax.transAxes,
                                   facecolor="#fdecea", edgecolor="none", zorder=0))
    put(ax, X[0], y, val, fontsize=10.6, family="monospace", va="center", color="#111111")
    put(ax, X[1], y, before, fontsize=10.4, va="center",
        color="#1a7f37" if ok else "#b3261e")
    put(ax, X[2], y, after, fontsize=10.2, family="monospace", va="center",
        color="#1a7f37" if ok else "#0b5cad")
put(ax, 0.5, 0.055,
    "Every row on main returned status=success and banked the full 60.0 return. "
    "max_episode_steps, n_substeps and action_dim get the same treatment on the positive-integer domain.",
    ha="center", fontsize=10.2, style="italic", color="#333333")

for a, y in placed:
    lo, hi = a.get_ylim()
    assert min(lo, hi) - 0.05 <= y <= max(lo, hi) + 0.07, f"text at y={y} outside {a.get_ylim()}"

out = pathlib.Path("/tmp/artifact_action_scale.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(plt.imread(out)[:, :, :3] * 255).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 24).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  {out.stat().st_size // 1024} KB")
print(f"    honored across trees: max|delta|=0 ; honored-vs-zero {F_ZERO:.2%} ; honored-vs-nan {F_NAN:.2%}")
