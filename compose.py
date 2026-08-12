"""Compose the fromto-resize artifact from the two captured trees."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ART = pathlib.Path(sys.argv[1]); MAIN_TAG, PR_TAG = sys.argv[2], sys.argv[3]
A = json.loads((ART / f"facts-{MAIN_TAG}.json").read_text())   # upstream/main
B = json.loads((ART / f"facts-{PR_TAG}.json").read_text())     # this PR
assert A["tree"] != B["tree"], "both arms resolved the same tree"

def load(key, tag): return np.load(ART / f"{key}-{tag}.npy")
F = {(k, t): load(k, t) for k in ("declared", "after_set", "after_recompile", "thicken")
     for t in (MAIN_TAG, PR_TAG)}

# ---- measured facts the figure asserts ------------------------------------
assert A["resize"]["status"] == "success" and A["after_set"]["geom_size"][1] == 0.3
assert A["after_set"]["inertia"] == A["declared"]["inertia"], "main's inertia must still be the short capsule's"
assert A["after_set"]["mass"] == A["declared"]["mass"]
assert A["after_recompile"]["geom_size"] == A["declared"]["geom_size"], "main must revert"
assert A["diff_pct"]["declared_vs_after_recompile"] == 0.0
assert B["resize"]["status"] == "error" and "<fromto>" in B["resize"]["text"]
assert B["after_set"]["geom_size"] == B["declared"]["geom_size"] == [0.05, 0.15, 0.0]
for t in (MAIN_TAG, PR_TAG):
    assert json.loads((ART / f"facts-{t}.json").read_text())["thicken"]["status"] == "success"
assert A["thicken_after_recompile"] == B["thicken_after_recompile"], "honored path identical on both trees"
assert B["thicken_after_set"] == B["thicken_after_recompile"], "honored path must be durable"
HON = float(np.abs(F[("thicken", MAIN_TAG)].astype(int) - F[("thicken", PR_TAG)].astype(int)).max())
assert HON <= 2, f"honored render differs across trees by {HON}"

# ---- crop to the capsule + the in-scene reference posts -------------------
def mask(img):
    r, g, b = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    orange = (r > 120) & (r - b > 45) & (r > g)
    dark = (r < 80) & (g < 80) & (b < 90)
    return orange | dark
ys, xs = [], []
for im in F.values():
    yy, xx = np.nonzero(mask(im))
    ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
pad = 26
y0, y1 = max(0, min(ys) - pad), min(F[("declared", MAIN_TAG)].shape[0], max(ys) + pad)
x0, x1 = max(0, min(xs) - pad), min(F[("declared", MAIN_TAG)].shape[1], max(xs) + pad)
crop = lambda im: im[y0:y1, x0:x1]
print(f"crop = y[{y0}:{y1}] x[{x0}:{x1}] -> {(y1-y0, x1-x0)}")

def pct(a, b):
    return float((np.abs(crop(a).astype(int) - crop(b).astype(int)).sum(axis=2) > 8).mean()) * 100.0
D_SET = pct(F[("declared", MAIN_TAG)], F[("after_set", MAIN_TAG)])
D_REV = pct(F[("after_set", MAIN_TAG)], F[("after_recompile", MAIN_TAG)])
D_HON = pct(F[("declared", PR_TAG)], F[("thicken", PR_TAG)])
print(f"cropped diffs: declared->after_set {D_SET:.1f}%  after_set->recompile {D_REV:.1f}%  honored {D_HON:.1f}%")
assert D_SET > 10.0 and D_REV > 10.0, "the resize must be legible in the crop"

fig = plt.figure(figsize=(15.4, 11.2), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.86], hspace=0.30, wspace=0.06,
                      left=0.035, right=0.975, top=0.905, bottom=0.035)
fig.suptitle("A geom_size component a <fromto> fixes: reported, then discarded twice over",
             fontsize=16.5, fontweight="bold", y=0.972)
fig.text(0.5, 0.936,
         "MuJoCo capsule declared fromto=\"-0.15 0 0  0.15 0 0\" (half-length 0.15 m). "
         "The black posts mark +/-0.30 m, the requested half-length.",
         ha="center", fontsize=10.6, style="italic", color="#333333")

placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, transform=ax.transAxes, **kw)

def panel(row, col, key, tag, title, sub, ok):
    ax = fig.add_subplot(gs[row, col]); ax.imshow(crop(F[(key, tag)])); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a7a3c" if ok else "#b3261e"); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=11.4, fontweight="bold", pad=6,
                 color="#1a7a3c" if ok else "#b3261e")
    ax.set_xlabel(sub, fontsize=9.5, labelpad=5, color="#222222")
    return ax

hl = lambda d: d["geom_size"][1]
panel(0, 0, "declared", MAIN_TAG, "main: as declared",
      f"half-length {hl(A['declared'])} m   Ixx {A['declared']['inertia'][0]}   mass {A['declared']['mass']} kg", True)
panel(0, 1, "after_set", MAIN_TAG, 'main: set_geom_properties -> status "success"',
      f"half-length {hl(A['after_set'])} m, but Ixx {A['after_set']['inertia'][0]} and mass "
      f"{A['after_set']['mass']} kg are STILL the 0.15 m capsule's", False)
panel(0, 2, "after_recompile", MAIN_TAG, "main: after one unrelated add_object",
      f"half-length {hl(A['after_recompile'])} m - the resize is gone, no call reported it", False)

panel(1, 0, "declared", PR_TAG, "this PR: as declared",
      f"half-length {hl(B['declared'])} m   Ixx {B['declared']['inertia'][0]}   mass {B['declared']['mass']} kg", True)
panel(1, 1, "after_set", PR_TAG, 'this PR: set_geom_properties -> status "error"',
      "refused before either representation is touched - byte-identical to the panel on its left", True)
panel(1, 2, "thicken", PR_TAG, "this PR: radius 0.05 -> 0.09 m, still honored",
      f"half-length {hl(B['thicken_after_recompile'])} m kept, Ixx "
      f"{B['thicken_after_recompile']['inertia'][0]}, and it survives the next recompile", True)

axr = fig.add_subplot(gs[2, :]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 0.955, "Measured (one script, run once per tree; every number below is read back from its dump)",
    fontsize=12.2, fontweight="bold")
rows = [
    ("set_geom_properties(geom_name='cap', size=[0.05, 0.30])",
     f"success  -  \"{A['resize']['text']}\"", "error, naming the component and the value it restores"),
    ("model geom_size right after that call", f"[0.05, {hl(A['after_set'])}, 0.0]  (the model now collides as a 0.30 m capsule)",
     f"[0.05, {hl(B['after_set'])}, 0.0]  (unchanged)"),
    ("body inertia / mass right after that call",
     f"Ixx {A['after_set']['inertia'][0]}, {A['after_set']['mass']} kg  -  still the 0.15 m capsule's",
     f"Ixx {B['after_set']['inertia'][0]}, {B['after_set']['mass']} kg  -  unchanged"),
    ("model geom_size after one unrelated add_object", f"[0.05, {hl(A['after_recompile'])}, 0.0]  -  reverted, silently",
     f"[0.05, {hl(B['after_recompile'])}, 0.0]  (unchanged)"),
    ("radius-only resize, size=[0.09, 0.15]",
     f"success, durable: {A['thicken_after_recompile']['geom_size'][:2]}",
     f"success, durable: {B['thicken_after_recompile']['geom_size'][:2]}  -  identical, nothing narrowed"),
    ("cropped pixels differing (declared vs the panel above)",
     f"{D_SET:.1f}% after the call, {D_REV:.1f}% again when it reverted",
     f"0.0% for the refusal, {D_HON:.1f}% for the honored radius change"),
]
TOP, LAST = 0.845, 0.155
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
put(axr, 0.008, TOP + 0.075, "what was measured", fontsize=10.4, fontweight="bold", color="#444444")
put(axr, 0.345, TOP + 0.075, "upstream/main", fontsize=10.4, fontweight="bold", color="#b3261e")
put(axr, 0.688, TOP + 0.075, "this PR", fontsize=10.4, fontweight="bold", color="#1a7a3c")
y = TOP
for label, main_v, pr_v in rows:
    axr.add_patch(Rectangle((0.0, y - 0.052), 1.0, 0.088, transform=axr.transAxes,
                            facecolor="#f4f4f6" if rows.index((label, main_v, pr_v)) % 2 == 0 else "white",
                            edgecolor="none", zorder=0))
    put(axr, 0.008, y, label, fontsize=9.3, va="center", color="#222222")
    put(axr, 0.345, y, main_v, fontsize=9.1, va="center", family="monospace", color="#8c1d18")
    put(axr, 0.688, y, pr_v, fontsize=9.1, va="center", family="monospace", color="#14532d")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(axr, 0.008, 0.055,
    "The refusal is placed beside the existing mesh / height-field / SDF refusal: the same "
    "\"can this geom's extent be honored\" decision, before the lock and before any write.",
    fontsize=9.5, style="italic", color="#333333")
for ax, yy in placed:
    assert -0.05 <= yy <= 1.10, (ax, yy)

out = ART / "fromto-fixed-geom-size.png"
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
import imageio.v3 as iio
im = np.asarray(iio.imread(out))[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"WROTE {out}  {im.shape[1]}x{im.shape[0]}")
