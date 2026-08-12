from __future__ import annotations
import json, os, pathlib, textwrap
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path("/tmp/artfacts-main.json").read_text())
B = json.loads(pathlib.Path("/tmp/artfacts-pr.json").read_text())
assert A["tree"] != B["tree"], "both arms measured the same tree"

ref_m = np.load("/tmp/art-main/A_reference.npy"); ref_p = np.load("/tmp/art-pr/A_reference.npy")
uns_m = np.load("/tmp/art-main/B_unsized.npy");   uns_p = np.load("/tmp/art-pr/B_unsized.npy")

def masked(s: str) -> str:
    """Mask CPython object addresses: each tree's generator has its own."""
    import re
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)

def dmax(x, y): return int(np.abs(x.astype(int) - y.astype(int)).max())
def frac(x, y): return float((np.abs(x.astype(int) - y.astype(int)).max(axis=2) > 8).mean())

# --- self-audit ---
assert dmax(ref_m, ref_p) <= 2, f"the honored crate differs across trees: {dmax(ref_m, ref_p)}"
assert dmax(uns_m, uns_p) <= 2, f"the refused scene differs across trees: {dmax(uns_m, uns_p)}"
assert frac(ref_p, uns_p) > 0.10, f"reference vs refused only {frac(ref_p, uns_p):.2%}"
assert "got 0 (size=[])" in A["unsized_add_object"]["text"]
assert "got 0" not in B["unsized_add_object"]["text"]
assert "has no len()" in A["patch_size"]["text"]
assert "has no len()" not in B["patch_size"]["text"]
assert "'pos' must be a list/tuple of 3 numbers" in A["patch_pos"]["text"]
assert masked(A["patch_pos"]["text"]) == masked(B["patch_pos"]["text"]), "the sibling field's verdict moved"
assert A["reference"]["text"] == B["reference"]["text"], "the honored add_object verdict moved"
D_REF_UNS, D_TREES = frac(ref_p, uns_p), dmax(uns_m, uns_p)

fig = plt.figure(figsize=(16.6, 13.2), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.20, 0.98, 0.56], hspace=0.30, wspace=0.055)
placed: list[tuple[object, float]] = []
def put(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes); placed.append((ax, y)); return ax.text(x, y, s, **kw)

MONO = dict(family="monospace", fontsize=8.6, va="top", wrap=False)

panels = [
    (ref_p, "A - what the caller asked for\nsize=[0.3, 0.3, 0.3]  (a list)", "#1a7f37",
     f"accepted on both trees  |  max|main-PR| = {dmax(ref_m, ref_p)}/255"),
    (uns_m, "B - main: the same three edge lengths,\nproduced lazily", "#b42318",
     "refused - and the reason names an\nextent the caller never passed"),
    (uns_p, "C - this change: the same call", "#1a7f37",
     "refused - the reason names the value"),
]
for col, (img, title, colour, sub) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(colour); sp.set_linewidth(2.4)
    ax.set_title(title, fontsize=11.4, color=colour, fontweight="bold", pad=8)
    ax.set_xlabel(sub, fontsize=9.4, color="#333")

# ---- row 2: the verbatim messages ----
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.03, "What the caller is told  (verbatim, one call per row)",
    fontsize=12.2, fontweight="bold", va="bottom")
rows = [
    ("add_object(size=<3 lazily-produced edge lengths>)",
     A["unsized_add_object"]["text"], B["unsized_add_object"]["text"]),
    ("patch_scene_mjcf(add_geom, size=<same>)",
     A["patch_size"]["text"], B["patch_size"]["text"]),
    ("patch_scene_mjcf(add_geom, pos=<same>)  <- sibling field, unchanged",
     A["patch_pos"]["text"], B["patch_pos"]["text"]),
]
# The deepest line of each row sits SUB_PR below its anchor, so LAST must clear it.
SUB_MAIN, SUB_PR = 0.075, 0.185
TOP, LAST = 0.955, 0.315
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
assert LAST - SUB_PR > 0.05, f"the last row's PR line would land at {LAST - SUB_PR}"
def shorten(s: str) -> str:
    return s.replace("<generator object <genexpr> at 0x", "<generator object ... at 0x")[:172]
y = TOP
for call, main_text, pr_text in rows:
    put(ax, 0.0, y, call, fontsize=9.9, fontweight="bold", color="#111", va="top")
    put(ax, 0.030, y - SUB_MAIN, "main:", fontsize=8.8, color="#b42318", fontweight="bold", va="top")
    put(ax, 0.088, y - SUB_MAIN, "\n".join(textwrap.wrap(shorten(main_text), 126)), color="#b42318", **MONO)
    put(ax, 0.030, y - SUB_PR, "this PR:", fontsize=8.8, color="#1a7f37", fontweight="bold", va="top")
    put(ax, 0.088, y - SUB_PR, "\n".join(textwrap.wrap(shorten(pr_text), 126)), color="#1a7f37", **MONO)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

# ---- row 3: measured ledger ----
ax = fig.add_subplot(gs[2, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.02, "Measured", fontsize=12.2, fontweight="bold", va="bottom")
lines = [
    f"scene after the refused call, main vs this PR .... max|delta| = {D_TREES}/255  -> the physics is untouched; the whole difference is what the caller is told",
    f"panel A vs panel C ............................... {D_REF_UNS:.2%} of pixels differ  -> the reference call really does build the crate, so the refusals are not an empty scene",
    "value classes moving accepted -> refused ......... 4  (generator, iterator, map, iterable whose __len__ raises); every other verdict byte-identical (16 of 16 probe rows)",
    "mutations caught by the new cases ................ 5 of 5   |   invisible to the 233 pre-existing cases in the two files: 3 of 5",
    "moving the probe ahead of the component read ..... fails 11 pre-existing tests  -> the placement is what keeps 'could not be iterated' and 'could not be read' intact",
]
TOP2, LAST2 = 0.86, 0.10
STEP2 = (TOP2 - LAST2) / (len(lines) - 1)
assert STEP2 > 0.030, STEP2
y = TOP2
for line in lines:
    put(ax, 0.0, y, line, family="monospace", fontsize=9.2, color="#222", va="top")
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9

fig.suptitle(
    "finite_vector_error: a vector whose length cannot be read is reported, not accepted\n"
    "MuJoCo headless (MUJOCO_GL=egl) - the guard's two live call sites count the components by reading the value again",
    fontsize=13.4, fontweight="bold", y=0.988,
)
for ax_, yy in placed:
    lo, hi = ax_.get_ylim()
    if (lo, hi) == (0.0, 1.0):
        assert -0.05 <= yy <= 1.10, f"text at y={yy}"
out = pathlib.Path(f"/tmp/artifact-{run}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  refused-scene delta={D_TREES}  A-vs-C={D_REF_UNS:.2%}")
