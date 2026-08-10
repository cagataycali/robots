import json, pathlib
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path("/tmp/art_main.json").read_text())     # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch.json").read_text())    # this change
assert A["tree"] != B["tree"], "both dumps came from one tree"
assert A["tag"] == "main" and B["tag"] == "branch"

im_a = np.asarray(Image.open("/tmp/art_render_main.png").convert("RGB")).astype(int)
im_b = np.asarray(Image.open("/tmp/art_render_branch.png").convert("RGB")).astype(int)
assert im_a.shape == im_b.shape
maxdelta = int(np.abs(im_a - im_b).max())
sat = float(((im_b.max(2) - im_b.min(2)) > 45).mean())
assert sat > 0.05, sat                                   # the render has real content
assert maxdelta <= 2, maxdelta                           # honored path untouched

KINDS = ["Robot (facade)", "Robot (MuJoCo)", "Object", "Camera", "Body", "Joint", "Sensor"]
FALSE = ("No robots in the scene", "No objects in the scene")

def verdict(rec):
    t = rec["text"]
    if not rec["ok"]:
        return "raised", "escaped the tool contract"
    if any(c in t for c in FALSE):
        return "false", "claims the scene is empty"
    if "Available" not in t:
        return "deadend", "no listing at all"
    return "good", "lists what is registered"

va = {k: verdict(A["messages"][k]) for k in KINDS}
vb = {k: verdict(B["messages"][k]) for k in KINDS}
n_bad_a = sum(1 for k in KINDS if va[k][0] != "good")
n_bad_b = sum(1 for k in KINDS if vb[k][0] != "good")
assert (n_bad_a, n_bad_b) == (7, 0), (n_bad_a, n_bad_b)
assert sum(1 for k in KINDS if va[k][0] == "false") == 3
assert sum(1 for k in KINDS if va[k][0] == "deadend") == 4
for k in KINDS:                                          # str typo unchanged
    pass
for k, v in A["str_typo"].items():
    assert B["str_typo"][k] == v, k
n_reach_a = sum(1 for r in A["reached"].values() if not r["ok"])
n_reach_b = sum(1 for r in B["reached"].values() if not r["ok"])
assert (n_reach_a, n_reach_b) == (2, 0), (n_reach_a, n_reach_b)

COL = {"good": "#1b7f4b", "false": "#b3261e", "deadend": "#8a5a00", "raised": "#b3261e"}
placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.2, 11.6), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.24, 0.60, 0.92], width_ratios=[1.0, 1.06],
                      hspace=0.20, wspace=0.10)
fig.suptitle("A non-str entity name is reported, so the report has to be usable",
             fontsize=17, fontweight="bold", y=0.975)
fig.text(0.5, 0.947,
         "One call: sim.move_to([0.2, 0.0, 0.1]) - robot_name is move_to's first positional, so a position list lands in it. "
         "Scene holds robot 'arm', object 'crate', camera 'look'.",
         ha="center", fontsize=10.4, color="#333")

# --- row 1: the verdict grid -------------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.02, "What the caller is told, per unknown-entity message  (requested = ['front'])",
    transform=ax.transAxes, fontsize=12.6, fontweight="bold")
TOP, LAST = 0.90, 0.075
STEP = (TOP - LAST) / (len(KINDS) - 1)
assert STEP > 0.045, STEP
put(ax, 0.010, TOP + 0.062, "entity kind", transform=ax.transAxes, fontsize=10.2, fontweight="bold", color="#555")
put(ax, 0.150, TOP + 0.062, "main", transform=ax.transAxes, fontsize=10.2, fontweight="bold", color="#555")
put(ax, 0.575, TOP + 0.062, "this change", transform=ax.transAxes, fontsize=10.2, fontweight="bold", color="#555")
for i, k in enumerate(KINDS):
    y = TOP - i * STEP
    put(ax, 0.010, y, k, transform=ax.transAxes, fontsize=10.4, fontweight="bold")
    for x0, w, (kind, why) in ((0.150, 0.415, va[k]), (0.575, 0.415, vb[k])):
        ax.add_patch(Rectangle((x0, y - 0.021), w, 0.050, transform=ax.transAxes,
                               facecolor=COL[kind], alpha=0.16, edgecolor=COL[kind], lw=1.1))
        put(ax, x0 + 0.010, y, why, transform=ax.transAxes, fontsize=9.9, color=COL[kind], fontweight="bold")
assert abs((TOP - (len(KINDS) - 1) * STEP) - LAST) < 1e-9
put(ax, 0.010, 0.005,
    f"messages that do not list what is registered:  main {n_bad_a} of {len(KINDS)}   ->   "
    f"this change {n_bad_b} of {len(KINDS)}     (3 asserted the scene was empty, 4 gave a bare dead end)",
    transform=ax.transAxes, fontsize=10.6, fontweight="bold", color="#111")

# --- row 2: the verbatim messages -------------------------------------------
for col, (label, dump, tint) in enumerate((("main", A, "#b3261e"), ("this change", B, "#1b7f4b"))):
    axm = fig.add_subplot(gs[1, col]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
    put(axm, 0.0, 1.05, f"{label} - Robot (MuJoCo), verbatim", transform=axm.transAxes,
        fontsize=11.4, fontweight="bold", color=tint)
    msg = dump["messages"]["Robot (MuJoCo)"]["text"]
    wrapped, line = [], ""
    for word in msg.split():
        if len(line) + len(word) + 1 > 56:
            wrapped.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    wrapped.append(line)
    yy = 0.80
    for w in wrapped:
        put(axm, 0.015, yy, w, transform=axm.transAxes, fontsize=10.0, family="monospace", color="#111")
        yy -= 0.155
    assert yy > -0.12, yy
    reach = dump["reached"]
    put(axm, 0.015, 0.115, "lookups in front of their own report:", transform=axm.transAxes,
        fontsize=9.6, color="#555", fontweight="bold")
    put(axm, 0.015, 0.020,
        "   " + "   ".join(f"{k}: {'RAISED TypeError' if not v['ok'] else 'reports'}" for k, v in reach.items()),
        transform=axm.transAxes, fontsize=9.6, family="monospace", color=tint, fontweight="bold")

# --- row 3: the render + the unchanged facts --------------------------------
axr = fig.add_subplot(gs[2, 0])
axr.imshow(np.asarray(Image.open("/tmp/art_render_branch.png").convert("RGB")))
axr.set_xticks([]); axr.set_yticks([])
axr.set_title("the honored path is untouched (headless MuJoCo, MUJOCO_GL=egl)", fontsize=11.2, fontweight="bold")
axr.set_xlabel(f"same scene rendered independently in both trees: max|main - branch| = {maxdelta}/255\n"
               f"60 send_action ticks, arm/link at {B['joint']} in both", fontsize=9.6)

axf = fig.add_subplot(gs[2, 1]); axf.axis("off"); axf.set_xlim(0, 1); axf.set_ylim(0, 1)
put(axf, 0.0, 1.03, "unchanged by this change", transform=axf.transAxes, fontsize=11.8, fontweight="bold")
rows = [
    ("a str typo keeps its close match + listing", f"{len(A['str_typo'])} of {len(A['str_typo'])} byte-identical"),
    ("Robot 'arm0'", "Did you mean: arm? Available robots: ['arm']"),
    ("Object 'crat'", "Available objects: ['crate']"),
    ("an empty scene is still reported empty", "No robots in the scene (pinned)"),
    ("difflib suggestion for a non-str name", "still omitted - nothing to match"),
    ("full suite", "27187 passed / 257 skipped / 0 failed"),
    ("pre-fix", "25 failed / 37 passed"),
]
TOPF, LASTF = 0.86, 0.10
STEPF = (TOPF - LASTF) / (len(rows) - 1)
assert STEPF > 0.045, STEPF
for i, (k, v) in enumerate(rows):
    y = TOPF - i * STEPF
    put(axf, 0.015, y, k, transform=axf.transAxes, fontsize=10.0, color="#222")
    put(axf, 0.520, y, v, transform=axf.transAxes, fontsize=9.7, family="monospace",
        color="#1b7f4b", fontweight="bold")
assert abs((TOPF - (len(rows) - 1) * STEPF) - LASTF) < 1e-9

for ax_, y, axes_coords in placed:
    if axes_coords:
        assert -0.04 <= y <= 1.10, (y, ax_.get_title())
out = pathlib.Path("/tmp/artifact_unknown_entity.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, (name, nw)
print(f"OK {out}  size={im.shape}  bad main={n_bad_a} branch={n_bad_b}  raised {n_reach_a}->{n_reach_b}  maxdelta={maxdelta}  sat={sat:.3f}")
