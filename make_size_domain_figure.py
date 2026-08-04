"""Compose the measured verdict figure from the two trees' JSON dumps."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(Path("/tmp/art/before.json").read_text())   # upstream/main
B = json.loads(Path("/tmp/art/after.json").read_text())    # this change
assert A["tree"] != B["tree"], "both halves came from the same tree"

rows_a, rows_b = A["rows"], B["rows"]
assert [ (r["param"], r["label"]) for r in rows_a ] == [ (r["param"], r["label"]) for r in rows_b ]

UNUSABLE = {("steps","0"),("steps","-5"),("steps","True"),("steps","2.7"),("steps","nan"),("steps","inf"),
            ("batch_size","0"),("batch_size","-8"),("batch_size","True"),("batch_size","2.7")}
def key(r): return (r["param"], r["label"])
acc_before = [r for r in rows_a if key(r) in UNUSABLE and r["v"] == "accepted"]
acc_after  = [r for r in rows_b if key(r) in UNUSABLE and r["v"] == "accepted"]
assert len(acc_before) == 10, f"expected 10 accepted-unusable on main, got {len(acc_before)}"
assert len(acc_after) == 0, f"expected 0 after the fix, got {len(acc_after)}"
assert A["control_argv"] == B["control_argv"], "a usable call's argv changed"
for lbl in (("steps","None"), ("save_freq","0")):
    for rows in (rows_a, rows_b):
        r = next(x for x in rows if key(x) == lbl)
        assert r["v"] == "accepted", f"{lbl} must stay accepted on both trees"

GOOD, BAD, NEUTRAL = "#1b7f4b", "#b3261e", "#4a5568"
placed: list[tuple[object, float]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 10.4), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[5.0, 1.05, 1.35], hspace=0.30,
                      left=0.035, right=0.975, top=0.925, bottom=0.035)

# --- row 1: the verdict matrix -------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.055, "lerobot_train: a run size the trainer cannot honor", fontsize=17,
    fontweight="bold", va="bottom")
put(ax, 0.0, 1.008,
    "Each value below is supplied to build_train_command, whose argv the tool then launches as a DETACHED process. "
    "Measured on two trees, one script.",
    fontsize=10.4, color=NEUTRAL, va="bottom")

COLS = [(0.000, "parameter"), (0.115, "value"), (0.200, "main"), (0.283, "what lerobot then does with the token"),
        (0.700, "this change")]
for x, name in COLS:
    put(ax, x, 0.955, name, fontsize=10.6, fontweight="bold", color="#111")
ax.plot([0, 1], [0.938, 0.938], color="#999", lw=1.0)

step = 0.0605
y = 0.938 - 0.040
for ra, rb in zip(rows_a, rows_b):
    unusable = key(ra) in UNUSABLE
    if unusable:
        ax.add_patch(Rectangle((-0.004, y - 0.019), 1.008, step * 0.86,
                               facecolor="#fdecea", edgecolor="none", zorder=0))
    put(ax, 0.000, y, f"{ra['param']}", fontsize=10.2, family="monospace", color="#111")
    put(ax, 0.115, y, ra["label"], fontsize=10.2, family="monospace", fontweight="bold", color="#111")
    ca = BAD if (unusable and ra["v"] == "accepted") else GOOD
    put(ax, 0.200, y, ra["v"], fontsize=10.2, fontweight="bold", color=ca)
    note = ra["consequence"] or ("kept: lerobot's own default applies" if ra["label"] == "None"
                                 else "kept: lerobot disables periodic saving")
    put(ax, 0.283, y, note, fontsize=9.6, family="monospace",
        color=BAD if unusable else NEUTRAL)
    cb = GOOD if (rb["v"] == "refused" or not unusable) else BAD
    txt = rb["v"] if not unusable else "refused before launch"
    put(ax, 0.700, y, txt, fontsize=10.2, fontweight="bold", color=cb)
    y -= step

put(ax, 0.283, y + step - 0.038,
    "The last two rows are the documented capabilities the guard must not take away.",
    fontsize=9.4, style="italic", color=NEUTRAL)

# --- row 2: the divergence this closes -----------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f4f6f8", edgecolor="#ccd2d8"))
put(ax2, 0.012, 0.80, "One parameter, two surfaces onto the same lerobot run", fontsize=12.2, fontweight="bold")
put(ax2, 0.012, 0.50,
    "LerobotTrainer.validate(spec)      steps=0  ->  refused    \"steps must be > 0, got 0\"",
    fontsize=10.4, family="monospace", color=GOOD)
put(ax2, 0.012, 0.22,
    "build_train_command(...)           steps=0  ->  accepted   \"--steps=0\"      (before this change)",
    fontsize=10.4, family="monospace", color=BAD)

# --- row 3: no regression ------------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f2f8f4", edgecolor="#bcd8c6"))
put(ax3, 0.012, 0.855, "A usable call is untouched: the argv is byte-identical on both trees", fontsize=12.2,
    fontweight="bold", color=GOOD)
argv = A["control_argv"]
tail = " ".join(argv[-7:])
put(ax3, 0.012, 0.60, f"steps=20000 batch_size=8 save_freq=5000  ->  {len(argv)} tokens, ending", fontsize=10.2,
    family="monospace", color="#111")
put(ax3, 0.012, 0.375, f"    {tail}", fontsize=9.9, family="monospace", color="#111")
put(ax3, 0.012, 0.13,
    f"unusable values accepted:  main {len(acc_before)} of {len(UNUSABLE)}   ->   this change {len(acc_after)} of {len(UNUSABLE)}",
    fontsize=11.0, fontweight="bold", color=GOOD)

for a, yv in placed:
    lo, hi = a.get_ylim()
    span = hi - lo
    assert lo - 0.06 * span <= yv <= hi + 0.06 * span, f"text at y={yv} outside {a.get_ylim()}"

out = Path("/tmp/art/lerobot_train_size_domain.png")
fig.savefig(out, dpi=125, facecolor="white", bbox_inches="tight", pad_inches=0.28)
plt.close(fig)

import numpy as np
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band.astype(int) - 255).sum(axis=2) > 20).sum())
    assert bad == 0, f"{name} border has {bad} non-white px"
print(f"OK {out} {im.shape[1]}x{im.shape[0]}  accepted-unusable {len(acc_before)} -> {len(acc_after)}")
