import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.load(open("/tmp/art_main/facts.json"))   # before
B = json.load(open("/tmp/art_pr/facts.json"))     # after
assert A["tree"] != B["tree"], "both dumps came from the same tree"
a = {r["tag"]: r for r in A["rows"]}
b = {r["tag"]: r for r in B["rows"]}

# ---- measured facts, asserted before anything is drawn ---------------------
assert a["honored"]["inferences"] == b["honored"]["inferences"] == 12
assert a["honored"]["joints"] == b["honored"]["joints"], "honored rollout diverged across trees"
assert a["clamped"]["outcome"] == "success" and a["clamped"]["inferences"] == 96
assert b["clamped"]["outcome"] == "raised ValueError" and b["clamped"]["inferences"] == 0
assert a["nonfinite"]["inferences"] == 1 and "convert float NaN" in a["nonfinite"]["text"]
assert b["nonfinite"]["inferences"] == 0 and "action_horizon" in b["nonfinite"]["text"]

hon_a = np.asarray(Image.open("/tmp/art_main/honored.png").convert("RGB"), dtype=np.int16)
hon_b = np.asarray(Image.open("/tmp/art_pr/honored.png").convert("RGB"), dtype=np.int16)
clamped = np.asarray(Image.open("/tmp/art_main/clamped.png").convert("RGB"), dtype=np.int16)
same = int(np.abs(hon_a - hon_b).max())
npix = int((np.abs(hon_a - hon_b).sum(2) > 0).sum())
diff_frac = float((np.abs(hon_a - clamped).sum(2) > 12).mean())
assert same <= 2, f"honored renders differ across trees by {same}"
assert diff_frac > 0.10, f"clamped pose only {diff_frac:.2%} different - reframe"
print(f"honored across trees: max|delta|={same} over {npix} of {hon_a[:, :, 0].size} px")
print(f"clamped vs honored:   {diff_frac:.2%} of pixels differ")

placed: list[tuple] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

MONO = {"family": "monospace"}
fig = plt.figure(figsize=(15.6, 11.0), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.14, wspace=0.045)

fig.suptitle(
    "action_horizon on the directly-drivable runner surfaces: same 96-step rollout, three requests",
    fontsize=14.5, fontweight="bold", y=0.973,
)
fig.text(0.5, 0.941,
    "PolicyRunner.run / .evaluate consume the horizon and are documented as drivable directly. "
    "Every public entry point validates it; these two did not.",
    ha="center", fontsize=10.4, style="italic", color="#333333")

panels = [
    ("A  action_horizon=8  (honored)", hon_a.astype(np.uint8), "#1a7f37",
     ["main   : success, 12 inferences", "branch : success, 12 inferences",
      f"renders agree to {same}/255 -- unchanged"]),
    ("B  action_horizon=0  on main", clamped.astype(np.uint8), "#b42318",
     ["success -- and 96 inferences,", "8x the model calls for a horizon", "the caller set to 0"]),
    ("C  action_horizon=0  on this branch", None, "#1a7f37",
     ["ValueError:", "PolicyRunner.run: action_horizon", "must be a positive integer, got 0.",
      "", "0 inferences, 0 actions applied,", "nothing rendered -- the rollout", "never started"]),
]
for col, (title, img, colour, notes) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.4)
    if img is not None:
        ax.imshow(img)
        ax.set_xlabel("\n".join(notes), fontsize=9.3, labelpad=7, color="#222222", **MONO)
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_facecolor("#f5f2ef")
        y = 0.80
        for line in notes:
            put(ax, 0.06, y, line, fontsize=10.2, color="#7a1f16" if "ValueError" in line else "#222222",
                fontweight="bold" if "ValueError" in line else "normal", transform=ax.transAxes, **MONO)
            y -= 0.088
    ax.set_title(title, fontsize=11.4, fontweight="bold", color=colour, pad=8)

# ---- verdict table --------------------------------------------------------
ax = fig.add_subplot(gs[1, :])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.035, "Measured on both trees -- PolicyRunner.run, so100, 96 control steps, MockPolicy",
    fontsize=11.6, fontweight="bold", transform=ax.transAxes)

COLS = [0.0, 0.135, 0.315, 0.44, 0.615, 0.735]
HEAD = ["action_horizon", "main: outcome", "infers", "this branch: outcome", "infers", "final Elbow (rad)"]
TOP, FLOOR = 0.945, 0.115
rows = [
    ("8", "success", "12", "success", "12", f"{a['honored']['joints']['Elbow']:.6f}  (both trees)", None),
    ("0", "success  <- clamped to 1", "96", "refused, names the parameter", "0",
     f"{a['clamped']['joints']['Elbow']:.6f}  (main)", "bad"),
    ("nan", 'error: "cannot convert float NaN"', "1", "refused, names the parameter", "0", "-", "bad"),
    ("-5 / True / 2.7 / \"8\"", "success  <- clamped or truncated", "-", "refused, names the parameter", "0",
     "-", "bad"),
]
STEP = (TOP - FLOOR) / (len(rows) + 1)
assert STEP > 0.030, STEP
y = TOP
for i, h in enumerate(HEAD):
    put(ax, COLS[i], y, h, fontsize=9.9, fontweight="bold", transform=ax.transAxes)
y -= STEP * 0.55
ax.plot([0, 0.88], [y, y], transform=ax.transAxes, color="#999999", lw=0.9)
y -= STEP * 0.45
for cells in rows:
    flag = cells[-1]
    if flag == "bad":
        ax.add_patch(plt.Rectangle((-0.008, y - 0.021), 0.895, STEP * 0.86, transform=ax.transAxes,
                                   facecolor="#fdecea", edgecolor="none", zorder=0))
    for i, cell in enumerate(cells[:-1]):
        colour = "#b42318" if (flag == "bad" and i in (1, 2)) else ("#1a7f37" if (flag == "bad" and i in (3, 4)) else "#222222")
        put(ax, COLS[i], y, cell, fontsize=9.4, color=colour, transform=ax.transAxes, **MONO)
    y -= STEP
assert y > 0.02, y

put(ax, 0.0, y - 0.012,
    "Both sibling knobs of the same signature already raised here: control_substeps "
    "(its docstring calls the raise \"the guarantee for callers driving PolicyRunner directly\")\n"
    "and control_frequency. The domain is the entry point's verbatim -- "
    "strands_robots.utils.positive_count_error -- with this surface as the context.",
    fontsize=9.5, color="#333333", transform=ax.transAxes, va="top")

for ax_, yy, is_axes in placed:
    if is_axes:
        assert -0.05 <= yy <= 1.07, (yy, "axes-fraction text outside panel")
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.03 <= yy <= hi + 0.07, (yy, lo, hi)

out = pathlib.Path("/tmp/artifact_action_horizon.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white px"
print("SAVED", out, Image.open(out).size)
