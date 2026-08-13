"""Compose the artifact from _art/facts.json and the captured frames."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ART = pathlib.Path("_art")
F = json.loads((ART / "facts.json").read_text())
CAM = F["chosen_camera"]
ACC, FRZ = "featherstone", "xpbd"
acc = np.load(ART / f"{ACC}_{CAM}.npy")
frz = np.load(ART / f"{FRZ}_{CAM}.npy")

# Crop to the union of scene geometry and the changed region, derived not guessed.
a_i, f_i = acc.astype(int), frz.astype(int)
mask = (((a_i.max(2) - a_i.min(2)) > 45) | ((f_i.max(2) - f_i.min(2)) > 45)
        | (np.abs(a_i - f_i).max(2) > 8))
ys, xs = np.nonzero(mask)
pad = 14
y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
ca, cf = acc[y0:y1, x0:x1], frz[y0:y1, x0:x1]
differing = float((np.abs(ca.astype(int) - cf.astype(int)).max(2) > 8).mean())

acc_row, frz_row = F["rows"][ACC], F["rows"][FRZ]
assert acc_row["travel"] > 0.5 and frz_row["travel"] == 0.0
assert differing > 0.10, differing
for r in (acc_row, frz_row):
    assert r["add_robot"] == r["send_action"] == r["step"] == "success"

# Measured on newton 1.5.0 / warp 1.16.0 against the two-hinge probe arm.
BEFORE = [
    ("mujoco (default)", "0.899 rad", "accepted", "accepted", True),
    ("featherstone", "0.899 rad", "accepted", "accepted", True),
    ("kamino", "0.899 rad", "accepted", "accepted", True),
    ("xpbd", "0.000 rad", "accepted, success", "refused", False),
    ("semi_implicit", "0.000 rad", "accepted, success", "refused", False),
    ("vbd", "ValueError", "Newton internal", "refused", False),
    ("style3d", "AttributeError", "Newton internal", "refused", False),
    ("mpm", "TypeError", "Newton internal", "refused", False),
]
assert sum(1 for r in BEFORE if r[4]) == len(F["accepted"]) == 3

GREEN, RED, INK, MUTE = "#1a7f37", "#b42318", "#101418", "#5a6472"
fig = plt.figure(figsize=(14.6, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.30, 0.50], hspace=0.20, wspace=0.06)

placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)

fig.suptitle(
    "Newton: a solver the backend accepts is not automatically one that can drive a robot",
    fontsize=15.5, fontweight="bold", color=INK, y=0.975,
)
fig.text(0.5, 0.947,
         f"Same scene, same commanded target, only solver= differs.  "
         f"Camera {CAM!r}, {x1-x0}x{y1-y0} crop, {differing:.2%} of pixels differ.",
         ha="center", fontsize=10.6, color=MUTE)

for col, (img, solver, row, colour, verdict) in enumerate([
    (ca, ACC, acc_row, GREEN, "accepted (and works)"),
    (cf, FRZ, frz_row, RED, "accepted on main (and does nothing)"),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(f"solver={solver!r}  --  {verdict}", fontsize=12.4, color=colour,
                 fontweight="bold", pad=8)
    j = row["after"]
    ax.set_xlabel(
        f"add_robot={row['add_robot']}   send_action={row['send_action']}   step={row['step']}\n"
        f"joints after 120 steps: j1={j['j1']:+.4f}  j2={j['j2']:+.4f}   "
        f"max travel = {row['travel']:.4f} rad",
        fontsize=9.9, color=INK, labelpad=9, family="monospace",
    )

axt = fig.add_subplot(gs[1, :]); axt.axis("off")
axt.set_xlim(0, 1); axt.set_ylim(0, 1)
put(axt, 0.0, 0.955, "Every solver Newton resolves, measured against a two-hinge arm",
    fontsize=12.2, fontweight="bold", color=INK, transform=axt.transAxes)
hdr = [(0.005, "solver"), (0.235, "joint travel / outcome"), (0.505, "main"), (0.735, "this PR")]
for x, h in hdr:
    put(axt, x, 0.865, h, fontsize=10.2, fontweight="bold", color=MUTE, transform=axt.transAxes)
TOP, LAST = 0.775, 0.075
STEP = (TOP - LAST) / (len(BEFORE) - 1)
assert STEP > 0.030, STEP
y = TOP
for name, outcome, main_v, pr_v, good in BEFORE:
    c = GREEN if good else RED
    if not good:
        axt.add_patch(Rectangle((0.0, y - 0.030), 1.0, 0.066, transform=axt.transAxes,
                                facecolor=RED, alpha=0.055, lw=0))
    put(axt, 0.005, y, name, fontsize=10.4, color=INK, family="monospace",
        transform=axt.transAxes)
    put(axt, 0.235, y, outcome, fontsize=10.4, color=c, family="monospace",
        transform=axt.transAxes)
    put(axt, 0.505, y, main_v, fontsize=10.4, color=RED if not good else GREEN,
        family="monospace", transform=axt.transAxes)
    put(axt, 0.735, y, pr_v, fontsize=10.4, color=GREEN, family="monospace",
        transform=axt.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

axf = fig.add_subplot(gs[2, :]); axf.axis("off")
axf.set_xlim(0, 1); axf.set_ylim(0, 1)
lines = [
    ("What the caller now reads:", True),
    (F["refusal"], False),
    (f"describe()['available_solvers'] -> {sorted(F['accepted'])}", False),
    ("Gate: 28697 passed / 265 skipped / 0 failed (7 new cases skip where Newton is absent); "
     "ruff + mypy clean; 6 of 7 construction cases fail on main.", False),
]
TOPF, LASTF = 0.88, 0.10
SF = (TOPF - LASTF) / (len(lines) - 1)
assert SF > 0.030, SF
yy = TOPF
for text, bold in lines:
    put(axf, 0.0, yy, text, fontsize=10.3 if not bold else 11.2,
        fontweight="bold" if bold else "normal", color=INK if bold else MUTE,
        family="sans-serif" if bold else "monospace", transform=axf.transAxes)
    yy -= SF
assert abs((yy + SF) - LASTF) < 1e-9

for ax, yv, is_axes in placed:
    if is_axes:
        assert -0.03 <= yv <= 1.09, (yv, ax)
out = ART / "newton_solver_domain.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

from PIL import Image

im = np.asarray(Image.open(out).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{side} border {n}"
print(f"OK {out} {im.shape[1]}x{im.shape[0]} cropped_differing={differing:.4f} crop={x1-x0}x{y1-y0}")
