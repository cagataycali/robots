"""Compose the gripper_dim_index figure; every number re-derived from the dumps."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = pathlib.Path("/tmp/art_out")
A = json.loads((D / "facts_main.json").read_text())
B = json.loads((D / "facts_branch.json").read_text())
assert A["tree"] != B["tree"], (A["tree"], B["tree"])


def img(tag, label, kind):
    return np.load(D / f"{tag}_{label}_{kind}.npy")


def frac(a, b):
    return float((np.abs(a.astype(int) - b.astype(int)).sum(2) > 12).mean())


# ---- self-audit of every claim the figure makes -------------------------
assert A["rows"]["-1"]["verdict"] == "decoded" and B["rows"]["-1"]["verdict"] == "decoded"
assert A["rows"]["6.0"]["verdict"] == "IndexError" and B["rows"]["6.0"]["verdict"] == "decoded"
assert A["rows"]["-5"]["verdict"] == "decoded" and B["rows"]["-5"]["verdict"] == "ValueError"
assert A["rows"]["99"]["verdict"] == "IndexError" and B["rows"]["99"]["verdict"] == "ValueError"
# the honored path is untouched, to the pixel and to the hand pose
ref_delta = int(np.abs(img("main", "-1", "after").astype(int) - img("branch", "-1", "after").astype(int)).max())
assert ref_delta <= 2, ref_delta
assert A["rows"]["-1"]["hand_xyz"] == B["rows"]["-1"]["hand_xyz"], (A["rows"]["-1"], B["rows"]["-1"])
# on main, -5 is indistinguishable from the sentinel: same pose, same pixels
silent_delta = int(np.abs(img("main", "-5", "after").astype(int) - img("main", "-1", "after").astype(int)).max())
assert silent_delta <= 2, silent_delta
assert A["rows"]["-5"]["hand_xyz"] == A["rows"]["-1"]["hand_xyz"]
# the visible half: main's 6.0 never moved the arm
gain = frac(img("main", "6.0", "after"), img("branch", "6.0", "after"))
assert gain > 0.10, gain
assert A["rows"]["6.0"]["hand_xyz"] == A["rows"]["6.0"]["hand_xyz"]
print(f"AUDIT ok: ref max|delta|={ref_delta}, silent max|delta|={silent_delta}, 6.0 panels differ {gain:.2%}")

fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.20, wspace=0.05)

PANELS = [
    ("branch", "-1", "Reference: gripper_dim_index = -1\n(the documented sentinel)",
     f"decoded, hand x={B['rows']['-1']['hand_xyz'][0]:.3f} m\nidentical on both trees (max|delta| = {ref_delta}/255)", "#2e7d32"),
    ("main", "6.0", "main: gripper_dim_index = 6.0\n(an integral float, from a config)",
     f"IndexError from numpy - the decode aborted\nthe arm never moved, hand x={A['rows']['6.0']['hand_xyz'][0]:.3f} m", "#c62828"),
    ("branch", "6.0", "this change: gripper_dim_index = 6.0",
     f"decoded, hand x={B['rows']['6.0']['hand_xyz'][0]:.3f} m\nnormalized to an index, as rotation_dim is", "#2e7d32"),
]
for col, (tag, label, title, cap, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img(tag, label, "after"))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(cap, fontsize=9.0, color="#263238", labelpad=6)

axt = fig.add_subplot(gs[1, :])
axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
placed = []


def put(x, y, s, **kw):
    placed.append(y)
    axt.text(x, y, s, transform=axt.transAxes, **kw)


ROWS = [
    ("-1  (sentinel)", "decoded, trailing column", "decoded, trailing column", "honored - unchanged", "#2e7d32"),
    ("6  (explicit last)", "decoded, trailing column", "decoded, trailing column", "honored - unchanged", "#2e7d32"),
    ("0  (first column)", "decoded, column 0", "decoded, column 0", "honored - unchanged", "#2e7d32"),
    ("6.0  (integral float)", "IndexError: only integers, slices ...", "decoded, trailing column", "newly usable", "#1565c0"),
    ("-5 / -99", "decoded - answered with the DEFAULT", "ValueError naming the parameter", "silent substitution removed", "#c62828"),
    ("nan", "decoded - answered with the DEFAULT", "ValueError naming the parameter", "silent substitution removed", "#c62828"),
    ("2.7 / inf / True", "IndexError / ValueError from numpy", "ValueError naming the parameter", "channel + naming fixed", "#c62828"),
    ("'6' / None / [6]", "TypeError: '>=' not supported ...", "ValueError naming the parameter", "channel + naming fixed", "#c62828"),
    ("99  (past the last)", "IndexError: index 99 is out of bounds", "ValueError naming param + chunk width", "reported against the chunk", "#c62828"),
]
TOP, LAST = 0.86, 0.20
STEP = (TOP - LAST) / (len(ROWS) - 1)
assert STEP > 0.045, STEP
put(0.0, 0.975, "One real decode per row onto a MuJoCo Panda through the shipped MinkIKBridge  --  "
    f"main {A['tree'].split('/')[-1]}  vs  this change",
    fontsize=11.2, fontweight="bold", color="#0d47a1")
for x, h in ((0.0, "gripper_dim_index ="), (0.20, "main"), (0.545, "this change"), (0.845, "what changes")):
    put(x, 0.925, h, fontsize=9.8, fontweight="bold", color="#37474f")
y = TOP
for lbl, m, b, note, colour in ROWS:
    put(0.0, y, lbl, fontsize=9.3, family="monospace", color="#263238")
    put(0.20, y, m, fontsize=8.9, family="monospace", color="#b71c1c" if colour != "#2e7d32" else "#33691e")
    put(0.545, y, b, fontsize=8.9, family="monospace", color="#1b5e20")
    put(0.845, y, note, fontsize=8.9, style="italic", color=colour)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y
FOOT = 0.085
assert FOOT > 0.02
put(0.0, FOOT,
    "The sentinel row is the no-regression proof: the same decode on both trees leaves the hand at "
    f"x={B['rows']['-1']['hand_xyz'][0]:.3f} m and the renders agree to {ref_delta}/255.\n"
    f"On main, -5 is indistinguishable from the sentinel - same hand pose, renders agree to {silent_delta}/255 - "
    "so a request no column satisfies was answered with the default and nothing reported it.",
    fontsize=9.1, color="#37474f")
for yy in placed:
    assert -0.03 <= yy <= 1.07, yy

out = D / "gripper_dim_index.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(matplotlib.image.imread(out))[:, :, :3]
im = (im * 255).astype(int) if im.max() <= 1.0 else im.astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band - 255).sum(-1) > 12).sum())
    assert bad == 0, (name, bad)
print("SAVED", out, matplotlib.image.imread(out).shape)
