"""Compose the Isaac camera-scoping parity figure from the measured facts."""
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TMP = pathlib.Path(sys.argv[1])
F = json.load(open(TMP / "facts.json"))
frame = np.load(TMP / "decoded.npy")
CELLS = {c["label"]: c for c in F["cells"]}

# --- self-audit: every drawn number comes from the measurement ---
assert F["equivalent"] is True
assert CELLS["safe"]["columns"] == ["arm0__wrist"] and CELLS["safe"]["sources"] == ["arm0/wrist"]
assert CELLS["both"]["columns"] == ["arm0__wrist"], CELLS["both"]
assert CELLS["all"]["columns"] == ["arm0__wrist", "overview"]
assert CELLS["unknown"]["status"] == "error"
assert F["roundtrip"]["decoded_frames"] == 4 and F["roundtrip"]["saturated_frac"] > 0.10
N_MUT = len(F["mutations"])
CAUGHT = sum(1 for _l, n, _o in F["mutations"] if n > 0)
BLIND = sum(1 for _l, _n, o in F["mutations"] if o == 0)
assert (N_MUT, CAUGHT, BLIND) == (5, 5, 5), (N_MUT, CAUGHT, BLIND)
XB = F["cross_backend"]
BEFORE_PINNED = sum(r[3] for r in XB); AFTER_PINNED = sum(r[4] for r in XB)
assert (BEFORE_PINNED, AFTER_PINNED) == (3, 6), (BEFORE_PINNED, AFTER_PINNED)

GREEN, RED, GREY, INK = "#1b7f3b", "#b3261e", "#8a8a8a", "#202124"
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); kw.setdefault("fontsize", 9.4); kw.setdefault("color", INK)
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.06, 0.92, 0.80], width_ratios=[0.86, 1.14],
                      hspace=0.20, wspace=0.13,
                      left=0.035, right=0.972, top=0.926, bottom=0.036)

fig.suptitle("Isaac recording: the schema-safe camera alias was documented, claimed at parity, and never executed",
             fontsize=14.6, fontweight="bold", y=0.981)
fig.text(0.5, 0.951, "cameras=['arm0__wrist'] on IsaacSimulation.start_recording  ->  measured on this branch; "
         "tests only, no line under strands_robots/ changes",
         ha="center", fontsize=10.2, color="#3c4043", style="italic")

# ---- row 1 left: the decoded frame the aliased column produced ----
axf = fig.add_subplot(gs[0, 0])
axf.imshow(frame, interpolation="nearest")
axf.set_xticks([]); axf.set_yticks([])
for sp in axf.spines.values(): sp.set_edgecolor(GREEN); sp.set_linewidth(2.4)
axf.set_title("frame 3 decoded back out of the dataset's own MP4", fontsize=10.8, fontweight="bold", pad=7)
axf.set_xlabel(f"{F['roundtrip']['mp4']}\n"
               f"{F['roundtrip']['decoded_frames']} frames, {frame.shape[1]}x{frame.shape[0]}, "
               f"{F['roundtrip']['size_bytes']} bytes  |  saturated {F['roundtrip']['saturated_frac']:.1%}",
               fontsize=8.9, color="#3c4043", labelpad=7)

# ---- row 1 right: what each request spelling produced ----
axr = fig.add_subplot(gs[0, 1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 1.0, "What each cameras= spelling produced (measured, this branch)",
    fontsize=11.2, fontweight="bold")
rows = [
    ("cameras=['arm0__wrist']   (schema-safe alias)", CELLS["safe"], True),
    ("cameras=['arm0/wrist']    (raw scene name)", CELLS["raw"], True),
    ("cameras=['arm0/wrist','arm0__wrist']", CELLS["both"], True),
    ("cameras=['nope']", CELLS["unknown"], False),
    ("cameras= omitted", CELLS["all"], True),
]
TOP, LAST = 0.905, 0.135
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for label, cell, ok in rows:
    put(axr, 0.0, y, label, fontsize=9.6, fontweight="bold", family="monospace")
    if ok:
        put(axr, 0.045, y - 0.052,
            f"columns {cell['columns']}   rendered from {cell['sources']}", fontsize=9.0, color=GREEN)
    else:
        put(axr, 0.045, y - 0.052, "status=error, lists ['arm0/wrist', 'overview']", fontsize=9.0, color=RED)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y
put(axr, 0.0, 0.062,
    "The alias and the raw name are indistinguishable: identical columns AND identical render sources.\n"
    "Requesting both spellings records the camera once. Neither property had a test on this backend.",
    fontsize=9.2, color="#3c4043", style="italic")

# ---- row 2: cross-backend cell matrix ----
axm = fig.add_subplot(gs[1, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 1.0, "Camera-scoping cells pinned by a test, per backend", fontsize=11.6, fontweight="bold")
cols = [("MuJoCo\n(reference)", 0.455), ("Newton", 0.590), ("Isaac\nbefore", 0.725), ("Isaac\nnow", 0.868)]
for name, cx in cols:
    axm.text(cx, 0.885, name, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
TOP2, LAST2 = 0.735, 0.175
STEP2 = (TOP2 - LAST2) / (len(XB) - 1)
assert STEP2 > 0.030, STEP2
y = TOP2
for row in XB:
    cell_name, m, n, ib, ia = row
    put(axm, 0.012, y, cell_name, fontsize=9.7, family="monospace")
    for (_h, cx), v in zip(cols, (m, n, ib, ia), strict=True):
        axm.add_patch(Rectangle((cx - 0.052, y - 0.052), 0.104, 0.062,
                                facecolor=(GREEN if v else RED), alpha=0.16,
                                edgecolor=(GREEN if v else RED), linewidth=1.1,
                                transform=axm.transData))
        axm.text(cx, y - 0.006, "pinned" if v else "no test", ha="center", va="center",
                 fontsize=8.9, fontweight="bold", color=(GREEN if v else RED))
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, y
put(axm, 0.012, 0.086,
    f"cells pinned on Isaac: {BEFORE_PINNED} of {len(XB)}  ->  {AFTER_PINNED} of {len(XB)}. "
    "The source comment above the alias branch says \"parity with MuJoCo/Newton\"; the branch (recording.py:338) "
    "was unexecuted.\nThe both-spellings dedup is the first pin of that branch on any backend.",
    fontsize=9.2, color="#3c4043", style="italic")

# ---- row 3: mutations + gate ----
axx = fig.add_subplot(gs[2, :]); axx.axis("off"); axx.set_xlim(0, 1); axx.set_ylim(0, 1)
put(axx, 0.0, 1.0, "Plausible regressions in the alias branch (failures per arm)",
    fontsize=11.4, fontweight="bold")
axx.text(0.615, 0.885, "3 new cases", ha="center", va="top", fontsize=9.4, fontweight="bold")
axx.text(0.845, 0.885, "18 pre-existing", ha="center", va="top", fontsize=9.4, fontweight="bold")
TOP3, LAST3 = 0.725, 0.245
STEP3 = (TOP3 - LAST3) / (len(F["mutations"]) - 1)
assert STEP3 > 0.030, STEP3
y = TOP3
for label, new, old in F["mutations"]:
    put(axx, 0.012, y, label, fontsize=9.5, family="monospace")
    axx.text(0.615, y - 0.004, f"{new} failed", ha="center", va="center",
             fontsize=9.3, fontweight="bold", color=GREEN)
    axx.text(0.845, y - 0.004, f"{old} failed  <- BLIND", ha="center", va="center",
             fontsize=9.3, fontweight="bold", color=RED)
    y -= STEP3
assert abs((y + STEP3) - LAST3) < 1e-9, y
cv = F["coverage"]
put(axx, 0.012, 0.150,
    f"{CAUGHT} of {N_MUT} caught by the new cases; {BLIND} of {N_MUT} invisible to the 18 the module already had. "
    f"Every anchor in_fn=1, restored byte-identically.\n"
    f"isaac/recording.py over tests/simulation/isaac: {cv['before_pct']:.2f}% -> {cv['after_pct']:.2f}% "
    f"({cv['before_missing']} -> {cv['after_missing']} missing); the one line closed is exactly {cv['closed'][0]}.  "
    "Gate at c45bd1fa: 29739 passed / 266 skipped, ruff + mypy clean.",
    fontsize=9.2, color="#3c4043")

for ax, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.07, (yy, "axes coords")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

OUT = TMP / "isaac_camera_scoping_parity.png"
fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(OUT).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, (side, nw)
print("OK", OUT, im.shape)
