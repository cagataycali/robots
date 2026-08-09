import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.load(open("/tmp/art_ctrl-main-2241776.json"))     # main
B = json.load(open("/tmp/art_robots-mine-2238590.json"))   # branch
assert A["tree"] != B["tree"], "the two dumps came from one tree"
ga = np.load("/tmp/ground_ctrl-main-2241776.npy"); gb = np.load("/tmp/ground_robots-mine-2238590.npy")

def divergences(d):
    return sum(1 for r in d["rows"] if r["render"] != r["viewer"])

DIV_A, DIV_B = divergences(A), divergences(B)
N = len(A["rows"])
assert (N, DIV_A, DIV_B) == (13, 12, 0), (N, DIV_A, DIV_B)
assert A["ledger"]["first_status"] == "success" and A["ledger"]["first_built"] == "0x0"
assert A["ledger"]["retry_built"] == "0x0" and "already open" in A["ledger"]["retry_text"]
assert B["ledger"]["first_status"] == "error" and B["ledger"]["first_built"] is None
assert B["ledger"]["retry_built"] == "1280x720"
delta = int(np.abs(ga.astype(int) - gb.astype(int)).max())
assert delta <= 2, delta
assert ga.shape == (720, 1280, 3), ga.shape

RED, GREEN, GREY = "#c62828", "#1b5e20", "#37474f"
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 9.4)
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 13.6), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 0.50, 1.05], hspace=0.20, wspace=0.09)

fig.suptitle(
    "Newton: one pixel quantity, two contracts - the viewer window vs the frame",
    fontsize=15.2, fontweight="bold", y=0.983,
)
fig.text(0.5, 0.960,
         "Every verdict measured against a recording stand-in for newton.viewer; the render column is _resolve_camera_view,\n"
         "the funnel render / get_frame / get_camera_params apply the domain in. Same engine, same session.",
         ha="center", fontsize=9.6, color=GREY)

# ---------------------------------------------------------------- row 1: matrix
for col, (label, d, ndiv) in enumerate(
    (("main - open_viewer applies no domain", A, DIV_A), ("this change - one floor for both", B, DIV_B))
):
    ax = fig.add_subplot(gs[0, col]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    put(ax, 0.0, 1.045, label, fontsize=11.4, fontweight="bold",
        color=RED if ndiv else GREEN, transform=ax.transAxes)
    put(ax, 0.0, 0.985, f"disagreements: {ndiv} of {N}", fontsize=10.2,
        color=RED if ndiv else GREEN, transform=ax.transAxes)
    put(ax, 0.005, 0.930, f"{'width=':>12}  {'render(...)':<10} {'open_viewer':<11} window built",
        family="monospace", fontsize=9.0, fontweight="bold")
    TOP, LAST = 0.885, 0.075
    step = (TOP - LAST) / (N - 1)
    assert step > 0.030, step
    for i, r in enumerate(d["rows"]):
        y = TOP - i * step
        bad = r["render"] != r["viewer"]
        if bad:
            ax.add_patch(Rectangle((0.0, y - 0.021), 1.0, 0.046, color=RED, alpha=0.10, lw=0))
        built = r["built"] or "-- nothing --"
        put(ax, 0.005, y, f"{r['value']:>12}  {r['render']:<10} {r['viewer']:<11} {built}",
            family="monospace", fontsize=9.0, color=RED if bad else GREY)
    assert TOP - (N - 1) * step > 0.04

# ---------------------------------------------------------------- row 2: ledger
for col, (label, d, ok) in enumerate((("main", A, False), ("this change", B, True))):
    ax = fig.add_subplot(gs[1, col]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    L = d["ledger"]
    put(ax, 0.0, 1.05, f"the single viewer slot, {label}", fontsize=11.0, fontweight="bold",
        color=GREEN if ok else RED, transform=ax.transAxes)
    lines = [
        f'open_viewer("gl", width=0, height=0)',
        f'   -> {L["first_status"]:<8} window built: {L["first_built"] or "-- nothing --"}',
        f'open_viewer("gl", width=1280, height=720)   # the retry',
        f'   -> {L["retry_status"]:<8} window built: {L["retry_built"] or "-- nothing --"}',
        f'   -> "{L["retry_text"]}"',
    ]
    TOP, FLOOR = 0.92, 0.10
    st = (TOP - FLOOR) / (len(lines) - 1)
    assert st > 0.030, st
    for i, ln in enumerate(lines):
        y = TOP - i * st
        put(ax, 0.005, y, ln, family="monospace", fontsize=9.2,
            color=(RED if (not ok and i in (1, 3, 4)) else (GREEN if (ok and i in (1, 3, 4)) else GREY)))
    assert TOP - (len(lines) - 1) * st > 0.05

# ---------------------------------------------------------------- row 3: ground
axg = fig.add_subplot(gs[2, :]); axg.imshow(gb); axg.set_xticks([]); axg.set_yticks([])
axg.set_title(
    f"what a usable pixel count means: the same scene rendered offscreen at the viewer's default 1280x720",
    fontsize=11.0, fontweight="bold", pad=8, color=GREEN,
)
axg.set_xlabel(
    f"Rendered independently on both trees: max|main - this change| = {delta}/255 over {ga.shape[1]}x{ga.shape[0]} px "
    "(renderer noise) - the honored path is untouched, and 1280x720 is exactly what a refused size now leaves the slot free for.",
    fontsize=9.3, color=GREY, labelpad=8,
)

for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.10, (y, "axes coords out of band")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.03 <= y <= hi + 0.03, (y, lo, hi)

out = Path("/tmp/newton_viewer_dims.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

from PIL import Image
im = np.array(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print(f"OK {out} {im.shape[1]}x{im.shape[0]}  divergences {DIV_A} -> {DIV_B}  ground delta {delta}")
