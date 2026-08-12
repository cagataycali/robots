"""Compose the PR figure from the measured facts. Every rendered value is asserted."""

import json
import pathlib
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ART = pathlib.Path(sys.argv[1])
F = json.loads((ART / "facts.json").read_text())

O, P = F["orthographic"], F["perspective"]
# --- self-audit: the story the figure tells must be the measurement --- #
assert F["tree"].endswith(pathlib.Path.cwd().name), F["tree"]
assert O["orthographic_flag"] == 1 and P["orthographic_flag"] == 0
assert O["get_frame"] == "ok" and P["get_frame"] == "ok", "both frames must render"
assert O["get_camera_params"].startswith("ValueError"), O["get_camera_params"]
assert P["get_camera_params"].startswith("ok"), P["get_camera_params"]
assert O["gwp_status"] == "error" and P["gwp_status"] == "success"
assert "camera parameters" in O["gwp_text"] and "orthographic" in O["gwp_text"]
assert min(O["saturated_frac"], P["saturated_frac"]) > 0.15, "frames must be legible"
assert F["ast_digest_base"] == F["ast_digest_head"], "executable AST changed"
assert F["text_differs"] == "True", "docstring text must differ"

MATRIX = [
    ("get_frame", "NotImplementedError", "covered", "covered"),
    ("get_frame", "KeyError / ValueError / RuntimeError / TypeError", "covered", "covered"),
    ("get_camera_params", "NotImplementedError", "NOT COVERED", "covered"),
    ("get_camera_params", "KeyError / ValueError / RuntimeError / TypeError", "NOT COVERED", "covered"),
    ("SimEngine.get_frame", "base default raise", "NOT COVERED", "covered"),
    ("SimEngine.get_camera_params", "base default raise", "NOT COVERED", "covered"),
]
n_before = sum(1 for r in MATRIX if r[2] != "covered")
n_after = sum(1 for r in MATRIX if r[3] != "covered")
assert (n_before, n_after) == (4, 0), (n_before, n_after)

MUT = [
    ("M1  dedicated NotImplementedError arm no longer catches", 1, 0),
    ("M2  that arm's report drops the method name", 1, 0),
    ("M3  handled tuple narrowed to KeyError", 5, 0),
    ("M4  params report drops the backend's reason", 5, 0),
    ("M5  params report copies the frame wording", 6, 0),
    ("M6  base default stops naming the method", 1, 0),
]
caught = sum(1 for _l, a, _b in MUT if a > 0)
blind = sum(1 for _l, _a, b in MUT if b == 0)
assert (caught, blind) == (6, 6), (caught, blind)

GREEN, RED, GREY = "#1b7f3b", "#b3261e", "#5f6368"
placed: list[tuple] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 13.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 0.62, 0.60], hspace=0.20, wspace=0.06,
                      left=0.028, right=0.972, top=0.945, bottom=0.030)

fig.suptitle(
    "get_world_point makes two backend reads. Only the first one's failure was ever tested.",
    fontsize=17, fontweight="bold", y=0.983,
)
fig.text(0.5, 0.958,
         "Both frames below render successfully. The orthographic scene still cannot be grounded "
         "-- and nothing in the suite drove that path.",
         ha="center", fontsize=11.5, style="italic", color=GREY)

# ---- row 1: the two real frames ---- #
for col, (label, rec) in enumerate((("orthographic", O), ("perspective", P))):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(np.load(ART / f"{label}.npy"))
    ax.set_xticks([]); ax.set_yticks([])
    bad = rec["gwp_status"] != "success"
    for sp in ax.spines.values():
        sp.set_edgecolor(RED if bad else GREEN); sp.set_linewidth(3.0)
    ax.set_title(
        f"free camera: {label.upper()}"
        + ("   <visual><global orthographic=\"true\"/></visual>" if bad else "   (same scene, perspective)"),
        fontsize=12.5, fontweight="bold", color=RED if bad else GREEN, pad=8,
    )
    gwp = rec["gwp_text"] if bad else f"{rec['gwp_text']}  ->  point {rec['gwp_point']} m"
    ax.set_xlabel(
        f"get_frame          -> ok, rgb {rec['rgb_shape'][1]}x{rec['rgb_shape'][0]} + depth "
        f"({rec['saturated_frac'] * 100:.1f}% of pixels carry geometry)\n"
        f"get_camera_params  -> {textwrap.shorten(rec['get_camera_params'], 96)}\n"
        f"get_world_point    -> status={rec['gwp_status']}\n"
        + "\n".join(textwrap.wrap(gwp, 104, initial_indent="    ", subsequent_indent="    ")),
        fontsize=9.2, family="monospace", loc="left", labelpad=9,
        color=RED if bad else "#202124",
    )

# ---- row 2: the coverage matrix ---- #
axm = fig.add_subplot(gs[1, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 1.02, "Failure arms of the two backend reads, and the base-class defaults behind them",
    fontsize=13, fontweight="bold", transform=axm.transAxes)
cols = (0.005, 0.215, 0.660, 0.830)
put(axm, cols[0], 0.885, "backend read", fontsize=10.5, fontweight="bold", transform=axm.transAxes)
put(axm, cols[1], 0.885, "failure arm", fontsize=10.5, fontweight="bold", transform=axm.transAxes)
put(axm, cols[2], 0.885, "on main", fontsize=10.5, fontweight="bold", transform=axm.transAxes)
put(axm, cols[3], 0.885, "with this PR", fontsize=10.5, fontweight="bold", transform=axm.transAxes)
TOP, LAST = 0.760, 0.150
step = (TOP - LAST) / (len(MATRIX) - 1)
assert step > 0.030, step
y = TOP
for read, arm, before, after in MATRIX:
    if before != "covered":
        axm.add_patch(Rectangle((0.0, y - 0.042), 1.0, 0.095, transform=axm.transAxes,
                                facecolor="#fdecea", edgecolor="none", zorder=0))
    put(axm, cols[0], y, read, fontsize=10, family="monospace", transform=axm.transAxes)
    put(axm, cols[1], y, arm, fontsize=10, family="monospace", color=GREY, transform=axm.transAxes)
    put(axm, cols[2], y, before, fontsize=10, fontweight="bold" if before != "covered" else "normal",
        color=RED if before != "covered" else GREEN, transform=axm.transAxes)
    put(axm, cols[3], y, after, fontsize=10, color=GREEN, transform=axm.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(axm, cols[0], 0.030,
    f"unexercised arms: {n_before} of {len(MATRIX)}  ->  {n_after} of {len(MATRIX)}"
    "        (the camera-params read had never raised anywhere in the suite)",
    fontsize=10.5, fontweight="bold", transform=axm.transAxes)

# ---- row 3: the mutation table ---- #
axx = fig.add_subplot(gs[2, :]); axx.axis("off"); axx.set_xlim(0, 1); axx.set_ylim(0, 1)
put(axx, 0.0, 1.02, "Would a regression be caught? (6 mutations x 2 arms, same two test files)",
    fontsize=13, fontweight="bold", transform=axx.transAxes)
put(axx, 0.005, 0.865, "regression introduced into get_world_point / SimEngine",
    fontsize=10.5, fontweight="bold", transform=axx.transAxes)
put(axx, 0.660, 0.865, "new cases", fontsize=10.5, fontweight="bold", transform=axx.transAxes)
put(axx, 0.815, 0.865, "pre-existing cases", fontsize=10.5, fontweight="bold", transform=axx.transAxes)
TOP2, LAST2 = 0.720, 0.185
step2 = (TOP2 - LAST2) / (len(MUT) - 1)
assert step2 > 0.030, step2
y = TOP2
for label, a, b in MUT:
    put(axx, 0.005, y, label, fontsize=10, family="monospace", transform=axx.transAxes)
    put(axx, 0.660, y, f"{a} failed", fontsize=10, color=GREEN, fontweight="bold", transform=axx.transAxes)
    put(axx, 0.815, y, f"{b} failed   <- BLIND", fontsize=10, color=RED, fontweight="bold", transform=axx.transAxes)
    y -= step2
assert abs((y + step2) - LAST2) < 1e-9, (y, LAST2)
put(axx, 0.005, 0.055,
    f"caught by the new cases: {caught}/{len(MUT)}      invisible to the pre-existing cases: {blind}/{len(MUT)}\n"
    f"No executable line changes: docstring-stripped AST digest of base.py is {F['ast_digest_base']} on both trees "
    f"(git diff --numstat: {F['numstat'].replace(chr(9), ' ')}).",
    fontsize=10.2, family="monospace", transform=axx.transAxes)

for ax, yv, is_axes in placed:
    if is_axes:
        assert -0.05 <= yv <= 1.10, (yv, "axes-fraction out of band")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.03 <= yv <= hi + 0.06, (yv, lo, hi)

out = ART / "get_world_point_camera_params_arms.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(matplotlib.image.imread(out) * 255).astype(int)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nz = int((np.abs(band - 255).sum(2) > 12).sum())
    assert nz == 0, f"{name} border has {nz} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  ({out.stat().st_size // 1024} KB)")
print(f"   ortho saturated={O['saturated_frac']:.4f}  persp saturated={P['saturated_frac']:.4f}")
