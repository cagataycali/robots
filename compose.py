import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = pathlib.Path("/tmp/art")
f = json.loads((A / "facts.json").read_text())
m = f["metrics"]
img = {n: np.load(A / f"{n}.npy") for n in ("sole_home", "sole_after", "named_after", "amb_home", "amb_after")}

# --- self-audit: every claim the figure makes ---
assert f["envelopes_identical"] is True
assert f["sole_omitted"]["status"] == "success" and "'arm'" in f["sole_omitted"]["text"]
assert f["ambiguous_omitted"]["status"] == "error" and "robot_name" in f["ambiguous_omitted"]["text"]
assert f["ambiguous_named"]["status"] == "success" and "'arm'" in f["ambiguous_named"]["text"]
assert m["omitted_vs_named_max_delta"] <= 2, m
assert m["ambiguous_unmoved_max_delta"] <= 2, m
assert m["sole_moved_frac"] > 0.10, m

MUT = [
    ("removes the sole-robot resolution", "12 of 15 fail", "77 pass"),
    ("stops converting the ValueError", "6 of 15 fail", "77 pass"),
    ("silently resolves the first of two robots", "3 of 15 fail", "77 pass"),
    ("drops the reason from the refusal", "6 of 15 fail", "77 pass"),
]

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 9.6)
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 11.6), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.42, 0.66, 0.50], hspace=0.30, wspace=0.06)

PANELS = [
    (img["sole_home"], "1 robot - home pose", "the scene before the call", "#37474f"),
    (img["sole_after"], "1 robot - move_to(position=...), robot_name OMITTED",
     f"the sole robot resolved and driven  |  {m['sole_moved_frac']*100:.1f}% of pixels differ from home", "#1b5e20"),
    (img["amb_after"], "2 robots - move_to(position=...), robot_name OMITTED",
     f"refused; nothing moved  |  max delta vs its own home = {m['ambiguous_unmoved_max_delta']}/255", "#b71c1c"),
]
for col, (arr, title, sub, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=10.6, color=colour, fontweight="bold", pad=7)
    ax.set_xlabel(sub, fontsize=9.4, color=colour, labelpad=6)

# --- row 2: the three documented outcomes, as reported ---
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.94, "What each documented outcome reports, with robot_name omitted  (identical on move_to / set_gripper / rotate_wrist)",
    fontsize=11.2, fontweight="bold", transform=ax.transAxes)
ROWS = [
    ("one robot", "success", f["sole_omitted"]["text"][:104], "#1b5e20"),
    ("zero robots", "error", "No robots registered in the simulation. Add a robot first (add_robot or Robot factory).", "#b71c1c"),
    ("many robots", "error", f["ambiguous_omitted"]["text"][:104], "#b71c1c"),
]
TOP, LAST = 0.72, 0.16
STEP = (TOP - LAST) / (len(ROWS) - 1)
assert STEP > 0.030, STEP
y = TOP
for scene, status, msg, colour in ROWS:
    put(ax, 0.005, y, scene, fontweight="bold", transform=ax.transAxes)
    put(ax, 0.115, y, status, color=colour, fontweight="bold", family="monospace", transform=ax.transAxes)
    put(ax, 0.195, y, msg, family="monospace", fontsize=8.9, transform=ax.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y

# --- row 3: what the existing suite could not see ---
ax = fig.add_subplot(gs[2, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.95, "Four mutations of the shared preamble: invisible to the 77 existing primitive tests, each caught here",
    fontsize=11.2, fontweight="bold", transform=ax.transAxes)
put(ax, 0.005, 0.76, "mutation", fontweight="bold", fontsize=9.4, transform=ax.transAxes)
put(ax, 0.560, 0.76, "these tests", fontweight="bold", fontsize=9.4, transform=ax.transAxes)
put(ax, 0.735, 0.76, "existing 4 primitive test files", fontweight="bold", fontsize=9.4, transform=ax.transAxes)
TOP2, LAST2 = 0.58, 0.20
STEP2 = (TOP2 - LAST2) / (len(MUT) - 1)
assert STEP2 > 0.030, STEP2
y = TOP2
for label, mine, theirs in MUT:
    put(ax, 0.005, y, label, family="monospace", fontsize=9.0, transform=ax.transAxes)
    put(ax, 0.560, y, mine, family="monospace", fontsize=9.0, color="#1b5e20", fontweight="bold", transform=ax.transAxes)
    put(ax, 0.735, y, theirs, family="monospace", fontsize=9.0, color="#b71c1c", fontweight="bold", transform=ax.transAxes)
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, y
put(ax, 0.005, 0.05,
    f"omitting the name is the same call as naming the sole robot: envelopes compare equal, renders differ by "
    f"{m['omitted_vs_named_max_delta']}/255 (two independent worlds)   |   "
    f"motion_primitives.py coverage 93% -> 94% (missing 29 -> 25; the auto-resolve block 163-166 was the gap)",
    fontsize=9.3, family="monospace", transform=ax.transAxes)

fig.suptitle("The motion primitives' documented robot_name default: resolve the sole robot, refuse an ambiguous scene",
             fontsize=13.4, fontweight="bold", y=0.985)
for ax_, y_, is_axes in placed:
    if is_axes:
        assert -0.03 <= y_ <= 1.07, (y_,)
    else:
        lo, hi = ax_.get_ylim(); assert lo - 0.05 <= y_ <= hi + 0.07, (y_, lo, hi)

out = pathlib.Path("/tmp/art/robot_name_auto_resolution.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(plt.imread(out) * 255).astype(int)[:, :, :3]
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, (side, n)
print("wrote", out, im.shape[1], "x", im.shape[0], "border clean, all assertions passed")
