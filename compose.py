import io, json, os, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUN = os.environ["GITHUB_RUN_ID"]; D = pathlib.Path(f"/tmp/art-{RUN}")
A = json.loads((D / f"facts-wt-main-{RUN}.json").read_text())          # main
B = json.loads((D / f"facts-robots-mine-{RUN}.json").read_text())      # branch
assert A["tree"] != B["tree"], "both arms measured the same tree"

def img(p): return np.asarray(Image.open(io.BytesIO((D / p).read_bytes())).convert("RGB"), np.uint8)
ref_a, ref_b = img(f"intended-wt-main-{RUN}.png"), img(f"intended-robots-mine-{RUN}.png")
mal_a, mal_b = img(f"malformed-wt-main-{RUN}.png"), img(f"malformed-robots-mine-{RUN}.png")

def dmax(x, y): return int(np.abs(x.astype(int) - y.astype(int)).max())
def dfrac(x, y): return float((np.abs(x.astype(int) - y.astype(int)).max(2) > 8).mean())

# --- audited claims -------------------------------------------------------
assert dmax(ref_a, ref_b) <= 2, dmax(ref_a, ref_b)          # same rig, both trees
assert dmax(ref_b, mal_b) <= 2, dmax(ref_b, mal_b)          # branch reaches the intended pose
assert dfrac(ref_a, mal_a) > 0.10, dfrac(ref_a, mal_a)      # main's loss is legible
assert A["with_malformed"]["applied"] == 1 and B["with_malformed"]["applied"] == 2
assert A["intended"]["applied"] == 2 and B["intended"]["applied"] == 2
assert B["joints_match_intended"] is True and A["joints_match_intended"] is False
assert A["intended"]["joints"] == B["intended"]["joints"]
ERR = max(abs(A["with_malformed"]["joints"][k] - A["intended"]["joints"][k]) for k in A["intended"]["joints"])
print(f"audited: ref delta={dmax(ref_a, ref_b)}  branch-vs-intended={dmax(ref_b, mal_b)}"
      f"  main-vs-intended={dfrac(ref_a, mal_a)*100:.2f}%  worst joint error={ERR:.4f} rad")

MUT = [
    ("M1  revert the fix (raise again)",              "15", "0"),
    ("M2  partially apply instead of rejecting whole", "9", "0"),
    ("M3  refuse but say nothing",                     "6", "0"),
    ("M4  narrow the except to ValueError only",       "8", "0"),
    ("M5  reword the refusal locally",                 "6", "0"),
    ("M6  poll loop: no reader tolerance",             "1", "0"),
    ("M7  _start_poll not idempotent",                 "1", "0"),
]

fig = plt.figure(figsize=(15.4, 12.4), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.34, 0.62, 0.60], hspace=0.30, wspace=0.05)
placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords)); ax.text(x, y, s, **kw)

fig.suptitle(
    "One malformed joint_command must not discard the commands queued behind it\n"
    "cyclonedds take(N=10) batches, so the parser's raise aborted the dispatch loop mid-batch",
    fontsize=15, fontweight="bold", y=0.982)

CAP = [
    (ref_b, "1. what the operator commanded\n(valid pose, then the final pose)", "#1a7f37"),
    (mal_a, "2. main: a malformed sample between them\nthe final command never arrived", "#b3261e"),
    (mal_b, "3. this PR: same batch\nthe final command still arrives", "#1a7f37"),
]
for col, (im, cap, colour) in enumerate(CAP):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_xlabel(cap, fontsize=10.5, color=colour, fontweight="bold", labelpad=7)

# --- row 2: measured ledger ----------------------------------------------
axl = fig.add_subplot(gs[1, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.955, "Measured on a real MuJoCo so101 driven through the bridge's own poll loop",
    fontsize=12, fontweight="bold", transform=axl.transAxes)
rows = [
    ("batch fed to _poll_loop", "3 samples: valid pose, malformed position, final pose", ""),
    ("commands applied  (main)", f"{A['with_malformed']['applied']} of 2 valid", "#b3261e"),
    ("commands applied  (this PR)", f"{B['with_malformed']['applied']} of 2 valid", "#1a7f37"),
    ("final pose  (main)", "stranded at the intermediate pose", "#b3261e"),
    ("final pose  (this PR)", "the commanded final pose, joint for joint", "#1a7f37"),
    ("worst joint error from the lost command", f"{ERR:.4f} rad  ({np.degrees(ERR):.0f} deg)", "#b3261e"),
    ("renders: intended pose, main vs this PR", f"max delta {dmax(ref_a, ref_b)}/255  (renderer noise)", "#555555"),
    ("renders: panel 2 vs panel 1", f"{dfrac(ref_a, mal_a)*100:.2f}% of pixels differ", "#555555"),
]
TOP, LAST = 0.80, 0.06
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
y = TOP
for label, value, colour in rows:
    put(axl, 0.008, y, label, fontsize=10.4, transform=axl.transAxes)
    put(axl, 0.44, y, value, fontsize=10.4, family="monospace",
        color=colour or "#111111", fontweight="bold" if colour else "normal", transform=axl.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

# --- row 3: mutation matrix ----------------------------------------------
axm = fig.add_subplot(gs[2, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 0.955, "Mutation table  -  7 plausible regressions x 2 arms",
    fontsize=12, fontweight="bold", transform=axm.transAxes)
put(axm, 0.008, 0.845, "regression introduced", fontsize=9.6, style="italic", transform=axm.transAxes)
put(axm, 0.575, 0.845, "new cases (24)", fontsize=9.6, style="italic", transform=axm.transAxes)
put(axm, 0.775, 0.845, "pre-existing (202)", fontsize=9.6, style="italic", transform=axm.transAxes)
TOP2, LAST2 = 0.735, 0.075
step2 = (TOP2 - LAST2) / (len(MUT) - 1)
assert step2 > 0.030, step2
y = TOP2
for label, new, old in MUT:
    put(axm, 0.008, y, label, fontsize=9.9, family="monospace", transform=axm.transAxes)
    put(axm, 0.575, y, f"{new} failed", fontsize=9.9, family="monospace",
        color="#1a7f37", fontweight="bold", transform=axm.transAxes)
    put(axm, 0.775, y, f"{old} failed  <- BLIND", fontsize=9.9, family="monospace",
        color="#b3261e", fontweight="bold", transform=axm.transAxes)
    y -= step2
assert abs((y + step2) - LAST2) < 1e-9, (y, LAST2)
put(axm, 0.008, 0.005,
    "Every regression, including reverting the fix itself, is invisible to all 202 pre-existing cases.",
    fontsize=9.6, style="italic", color="#444444", transform=axm.transAxes)

for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.07, (y, ax)
    else:
        lo, hi = ax.get_ylim(); assert lo - 0.05 <= y <= hi + 0.07, (y, lo, hi)

out = D / "artifact.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("wrote", out, im.shape)
