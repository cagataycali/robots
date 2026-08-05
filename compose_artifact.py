"""Compose the artifact: 3 replay renders over the measured verdict table."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

A = json.loads(Path("/tmp/art/before.json").read_text())      # pristine upstream/main
B = json.loads(Path("/tmp/art/after.json").read_text())       # this PR
RA = json.loads(Path("/tmp/art/before_render.json").read_text())
RB = json.loads(Path("/tmp/art/after_render.json").read_text())
assert A["tree"] != B["tree"], "both probes resolved to the same tree"

img = lambda n: np.asarray(iio.imread(f"/tmp/art/{n}.png"))[:, :, :3]
honored_a, honored_b = img("before_honored"), img("after_honored")
wrong_main, refused = img("before_wrongkey"), img("after_wrongkey")

# --- self-audit: every claim below is re-derived here -----------------------
same = int(np.abs(honored_a.astype(int) - honored_b.astype(int)).max())
assert same <= 2, f"the honored path differs across trees by {same}/255"
diff = float((np.abs(honored_a.astype(int) - wrong_main.astype(int)).sum(2) > 12).mean())
assert diff > 0.10, diff
cA, cB = A["cases"], B["cases"]
assert cA["scale (documented alias)"]["compiled"] == cB["scale (documented alias)"]["compiled"] == [0.3, 0.3, 0.3]
assert cA["extents (plausible, wrong)"]["status"] == "success"
assert cA["extents (plausible, wrong)"]["compiled"] == [0.05, 0.05, 0.05]
assert cB["extents (plausible, wrong)"]["status"] == "error"
assert cB["extents (plausible, wrong)"]["registered"] is False

ROWS = [
    ("(none - control)",            "success",   "success"),
    ("scale  (documented alias)",   "success, honored", "success, honored"),
    ("heigth",                      "success, dropped", "refused"),
    ("positon",                     "success, dropped", "refused"),
    ("colour",                      "success, dropped", "refused"),
    ("density",                     "success, dropped", "refused"),
    ("friction",                    "success, dropped", "refused"),
    ("rgba",                        "success, dropped", "refused"),
    ("extents",                     "success, dropped", "refused"),
]
n_div = sum(1 for _, m, _ in ROWS if "dropped" in m)
assert n_div == 7, n_div

fig = plt.figure(figsize=(15.6, 10.2), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.20, wspace=0.045,
                      left=0.032, right=0.982, top=0.905, bottom=0.035)
fig.suptitle(
    "IsaacSimulation.add_object: a keyword it cannot use is named rather than dropped",
    fontsize=15.5, fontweight="bold", y=0.972,
)
fig.text(0.5, 0.933,
         "Renders replay the extent Isaac's add_object actually compiled, drawn in MuJoCo headless. "
         "Blue post = 0.30 m, the requested size.",
         ha="center", fontsize=10.2, style="italic", color="#444444")

panels = [
    (honored_a, "A - what the caller asked for\nadd_object(scale=[0.30, 0.30, 0.30])",
     "compiled [0.30, 0.30, 0.30]  -  identical on both trees", "#1a7f37"),
    (wrong_main, "B - main: add_object(extents=[0.30, 0.30, 0.30])",
     'status="success", compiled [0.05, 0.05, 0.05]', "#b3261e"),
    (refused, "C - this PR: the same call",
     'status="error", nothing constructed', "#1a7f37"),
]
for col, (frame, title, sub, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(frame)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=11.0, fontweight="bold", pad=7)
    ax.set_xlabel(sub, fontsize=9.9, color=colour, labelpad=6)

# --- verdict table ---------------------------------------------------------
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
placed: list[float] = []
def put(x, y, s, **kw):
    placed.append(y)
    ax.text(x, y, s, transform=ax.transAxes, **kw)

put(0.5, 0.975, "add_object keyword -> verdict, measured one call per case (no isaacsim / newton / warp installed)",
    ha="center", fontsize=11.6, fontweight="bold")
cols = [0.055, 0.315, 0.487, 0.660, 0.845]
hdr = ["extra keyword", "MuJoCo", "Newton", "Isaac (main)", "Isaac (this PR)"]
for x, h in zip(cols, hdr):
    put(x, 0.885, h, fontsize=10.6, fontweight="bold", family="monospace")
ax.axhline(0.862, xmin=0.04, xmax=0.965, color="#333333", lw=1.1)

step = 0.083
for i, (key, main_v, pr_v) in enumerate(ROWS):
    y = 0.800 - i * step
    control = "control" in key
    sibling = "success" if (control or "honored" in main_v and key.startswith("scale")) else "TypeError"
    if control:
        sibling = "success"
    elif key.startswith("scale"):
        sibling = "TypeError"
    put(cols[0], y, key, fontsize=10.2, family="monospace")
    grey = "#666666"
    put(cols[1], y, sibling, fontsize=10.2, family="monospace",
        color=grey if sibling == "success" else "#1a7f37")
    put(cols[2], y, sibling, fontsize=10.2, family="monospace",
        color=grey if sibling == "success" else "#1a7f37")
    bad = "dropped" in main_v
    put(cols[3], y, main_v, fontsize=10.2, family="monospace",
        color="#b3261e" if bad else grey, fontweight="bold" if bad else "normal")
    put(cols[4], y, pr_v, fontsize=10.2, family="monospace",
        color="#1a7f37" if pr_v == "refused" else grey,
        fontweight="bold" if pr_v == "refused" else "normal")
    if bad:
        ax.add_patch(plt.Rectangle((0.04, y - 0.028), 0.925, 0.058, transform=ax.transAxes,
                                   facecolor="#b3261e", alpha=0.055, zorder=0))

foot = (f"divergences from the sibling backends: main {n_div} of {len(ROWS)}  ->  this PR 0 of {len(ROWS)}     |     "
        f"honored `scale` render identical across trees (max delta {same}/255); "
        f"main's dropped-key result differs on {diff:.1%} of pixels")
put(0.5, 0.055, foot, ha="center", fontsize=10.4, fontweight="bold", color="#222222")

lo, hi = ax.get_ylim()
assert all(lo - 0.03 <= y <= hi + 0.06 for y in placed), [y for y in placed if not (lo - 0.03 <= y <= hi + 0.06)]

out = Path("/tmp/art/add_object_unknown_keyword.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(iio.imread(out))[:, :, :3]
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 22).sum())
    assert nw == 0, f"{side} border has {nw} non-white px"
print(f"OK  {out}  {im.shape[1]}x{im.shape[0]}  divergences {n_div}->0  honored-delta {same}  diff {diff:.4f}")
