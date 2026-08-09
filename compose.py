"""Compose the wire-trace figure from the two measured trees."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path("/tmp/art_main.json").read_text())   # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch.json").read_text()) # this change
assert A["tree"] != B["tree"], "both dumps came from the same tree"
print("main tree  :", A["tree"])
print("branch tree:", B["tree"])

COLS = [("sync:create", "sync_dataset_to_bucket\ncreate"),
        ("sync:private", "sync_dataset_to_bucket\nprivate"),
        ("sync:delete", "sync_dataset_to_bucket\ndelete"),
        ("push:private", "push_to_hub\nprivate")]
VALS = ["'false'", "'no'", "'off'", "'0'", "1", "nan", "0", "''", "None", "[]"]

def reached_remote(row):
    """Did a posture the caller never expressed reach the remote store?"""
    if row["surface"] == "push_to_hub":
        return bool(row.get("published"))
    return bool(row["created"] or row["mirror_deleted"] or row["argv"])

def cell(dump, col, val):
    row = dump["rows"][f"{col.split(':')[0].replace('sync','sync').replace('push','push')}:{col.split(':')[1]}:{val}"]
    return row

bad_main = bad_branch = 0
grid = {}
for key, _ in COLS:
    for v in VALS:
        rm = A["rows"][f"{key}:{v}"]; rb = B["rows"][f"{key}:{v}"]
        m_bad, b_bad = reached_remote(rm), reached_remote(rb)
        bad_main += m_bad; bad_branch += b_bad
        grid[(key, v)] = (m_bad, b_bad, rm, rb)

CELLS = len(COLS) * len(VALS)
print(f"cells={CELLS}  reached the remote: main {bad_main}, this change {bad_branch}")
assert CELLS == 40 and bad_branch == 0 and bad_main > 0, (CELLS, bad_main, bad_branch)

# no-regression: every honoured posture builds a byte-identical command line
same = [n for n in A["controls"] if A["controls"][n] == B["controls"][n]]
print(f"honoured controls byte-identical across trees: {len(same)} of {len(A['controls'])}")
assert len(same) == len(A["controls"]) == 4, (same,)

GREEN, RED, INK = "#1b7f3b", "#b3261e", "#1a1a1a"
fig = plt.figure(figsize=(15.4, 12.6), dpi=124)
gs = fig.gridspec = fig.add_gridspec(3, 1, height_ratios=[1.62, 1.00, 0.62], hspace=0.20)

placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

# ---- row 1: verdict matrix -------------------------------------------------
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.5, 1.045, "What posture actually reaches the remote store, per flag value",
    ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK, transform=ax.transAxes)
put(ax, 0.5, 1.005,
    "each value is one an operator reaches for when opting out, or a falsy non-boolean;  "
    "green = refused naming the flag,  red = a posture the caller never expressed reached the remote",
    ha="center", va="bottom", fontsize=9.6, color="#444", transform=ax.transAxes)

X0, XW, PAD = 0.175, 0.196, 0.008
TOP, LAST = 0.845, 0.075
STEP = (TOP - LAST) / (len(VALS) - 1)
assert STEP > 0.030, STEP

for j, (key, title) in enumerate(COLS):
    cx = X0 + j * XW
    put(ax, cx + XW / 2 - PAD, 0.955, title, ha="center", va="center",
        fontsize=10.2, fontweight="bold", color=INK, family="monospace")
    put(ax, cx + XW * 0.245, 0.898, "main", ha="center", va="center", fontsize=9, color="#666")
    put(ax, cx + XW * 0.745, 0.898, "this change", ha="center", va="center", fontsize=9, color="#666")

put(ax, 0.163, 0.955, "flag value", ha="right", va="center", fontsize=10.2,
    fontweight="bold", color=INK)

for i, v in enumerate(VALS):
    y = TOP - i * STEP
    put(ax, 0.163, y, v, ha="right", va="center", fontsize=10.4, family="monospace", color=INK)
    for j, (key, _t) in enumerate(COLS):
        m_bad, b_bad, rm, rb = grid[(key, v)]
        for k, bad in enumerate((m_bad, b_bad)):
            bx = X0 + j * XW + k * (XW / 2 - PAD / 2)
            ax.add_patch(Rectangle((bx, y - STEP * 0.36), XW / 2 - PAD, STEP * 0.72,
                                   facecolor=RED if bad else GREEN, alpha=0.90,
                                   edgecolor="white", lw=1.0, transform=ax.transAxes))
            lab = ("reached" if bad else "refused")
            put(ax, bx + (XW / 2 - PAD) / 2, y, lab, ha="center", va="center",
                fontsize=8.4, color="white", fontweight="bold", transform=ax.transAxes)

put(ax, 0.5, 0.012,
    f"a posture the caller never expressed reached the remote in {bad_main} of {CELLS} cells on main,  "
    f"{bad_branch} of {CELLS} with this change",
    ha="center", va="center", fontsize=11.4, fontweight="bold", color=INK, transform=ax.transAxes)

# ---- row 2: the two decisive command lines --------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
put(ax2, 0.5, 1.03, "The command line the two decisive values built", ha="center", va="bottom",
    fontsize=14, fontweight="bold", color=INK, transform=ax2.transAxes)

def argv_block(x, w, title, sub, rm, rb):
    put(ax2, x, 0.90, title, ha="left", va="center", fontsize=11.2, fontweight="bold",
        color=INK, family="monospace")
    put(ax2, x, 0.815, sub, ha="left", va="center", fontsize=9.4, color="#444")
    y = 0.700
    for tree_label, row, colour in (("main", rm, RED), ("this change", rb, GREEN)):
        put(ax2, x, y, tree_label, ha="left", va="center", fontsize=9.8,
            fontweight="bold", color=colour)
        y -= 0.105
        if row["argv"]:
            for c in row["argv"]:
                put(ax2, x + 0.012, y, "$ " + c, ha="left", va="center", fontsize=9.0,
                    family="monospace", color=INK)
                y -= 0.098
        else:
            reason = row["msg"].split(".")[0][:96] + "."
            put(ax2, x + 0.012, y, "(no command run)", ha="left", va="center", fontsize=9.0,
                family="monospace", color="#555")
            y -= 0.098
            put(ax2, x + 0.012, y, reason, ha="left", va="center", fontsize=8.8, color="#333")
            y -= 0.098
        y -= 0.030
    return y

m1, b1 = A["rows"]["sync:delete:'false'"], B["rows"]["sync:delete:'false'"]
argv_block(0.020, 0.47, 'delete="false"',
           "reads as off; every non-empty string is truthy",
           m1, b1)
m2, b2 = A["rows"]["sync:private:0"], B["rows"]["sync:private:0"]
argv_block(0.525, 0.47, "private=0",
           "reads as not-private; a falsy non-boolean takes the other branch",
           m2, b2)
assert m1["mirror_deleted"] and not b1["argv"], "delete row not as measured"
assert m2["created"] and not m2["private_flag"] and not b2["argv"], "private row not as measured"

# ---- row 3: no-regression ledger ------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
put(ax3, 0.5, 1.02, "Every posture that could already be expressed is unchanged",
    ha="center", va="bottom", fontsize=13.4, fontweight="bold", color=INK, transform=ax3.transAxes)
names = list(A["controls"])
TOP3, LAST3 = 0.76, 0.16
STEP3 = (TOP3 - LAST3) / (len(names) - 1)
assert STEP3 > 0.030, STEP3
for i, n in enumerate(names):
    y = TOP3 - i * STEP3
    ca, cb = A["controls"][n], B["controls"][n]
    put(ax3, 0.055, y, n, ha="left", va="center", fontsize=10.2, family="monospace",
        fontweight="bold", color=INK)
    put(ax3, 0.235, y, " ; ".join(ca["argv"]), ha="left", va="center", fontsize=8.9,
        family="monospace", color=INK)
    put(ax3, 0.905, y, "identical" if ca == cb else "DIFFERS", ha="left", va="center",
        fontsize=9.6, fontweight="bold", color=GREEN if ca == cb else RED)
put(ax3, 0.5, 0.045,
    f"{len(same)} of {len(names)} honoured command lines are byte-identical across the two trees",
    ha="center", va="center", fontsize=10.8, color=INK, transform=ax3.transAxes)

for a, y, is_axes in placed:
    lo, hi = (-0.05, 1.10) if is_axes else a.get_ylim()
    assert lo - 0.03 <= y <= hi + 0.08, (y, lo, hi, is_axes)

out = pathlib.Path("/tmp/art_posture_flags.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

from PIL import Image
import numpy as np
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print(f"OK {out} {im.shape[1]}x{im.shape[0]}  border clean")
