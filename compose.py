"""Compose the artifact. Every rendered number is read from the two capture
dumps; nothing is hand-typed."""

from __future__ import annotations

import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RID = os.environ["GITHUB_RUN_ID"]
A_DIR = pathlib.Path(f"/tmp/art-main-{RID}")      # upstream/main
B_DIR = pathlib.Path(f"/tmp/art-branch-{RID}")    # this PR
A = json.loads(next(A_DIR.glob("facts_*.json")).read_text())
B = json.loads(next(B_DIR.glob("facts_*.json")).read_text())
assert A["tree"] != B["tree"], "both captures ran in the same tree"
a = {r["label"]: r for r in A["runs"]}
b = {r["label"]: r for r in B["runs"]}

def img(d, name): return np.asarray(Image.open(d / name).convert("RGB")).astype(int)
def dmax(x, y): return int(np.abs(x - y).max())
def dfrac(x, y): return float((np.abs(x - y).sum(2) > 8).mean())

healthy_a, healthy_b = img(A_DIR, a["healthy"]["png_end"]), img(B_DIR, b["healthy"]["png_end"])
broken_a, broken_b = img(A_DIR, a["broken_obs"]["png_end"]), img(B_DIR, b["broken_obs"]["png_end"])
start_b = img(B_DIR, b["healthy"]["png_start"])

d_healthy = dmax(healthy_a, healthy_b)
d_broken = dmax(broken_a, broken_b)
f_move = dfrac(start_b, healthy_b)

# --- audit every claim the figure makes -------------------------------------
assert a["healthy"]["install"] == b["healthy"]["install"] == "accepted"
assert a["healthy"]["applied"] == b["healthy"]["applied"] == 26
assert a["broken_obs"]["install"] == "accepted" and a["broken_obs"]["adapter_error"] is None
assert a["broken_obs"]["applied"] == 0 and a["broken_obs"]["action_errors"] == 26
assert b["broken_obs"]["install"] == "refused" and b["broken_obs"]["adapter_error"] is not None
assert b["broken_obs"]["applied"] == 0 and b["broken_obs"]["action_errors"] == 0
assert not b["broken_obs"]["controller_installed"] and a["broken_obs"]["controller_installed"]
assert d_healthy <= 2, d_healthy      # honored path untouched across trees
assert d_broken <= 2, d_broken        # physics untouched across trees
assert f_move > 0.10, f_move          # the controller genuinely moves the arm

placed: list[tuple[object, float]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); ax.text(x, y, s, transform=ax.transAxes, **kw)

fig = plt.figure(figsize=(16.2, 12.4), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.20, wspace=0.06)
MONO = {"family": "monospace", "fontsize": 9.6}

panels = [
    (healthy_b, "1. healthy engine  (identical on both trees)",
     f"install accepted -> 26/26 actions applied\nhand z {b['healthy']['z_start']:.4f} -> {b['healthy']['z_end']:.4f} m  (descends)",
     "#1b7f3b"),
    (broken_a, "2. broken joint-state read  -  upstream/main",
     f"install ACCEPTED, _action_controller_error = None\n0/26 applied, {a['broken_obs']['action_errors']}/26 conversion errors\n"
     f"hand z {a['broken_obs']['z_start']:.4f} -> {a['broken_obs']['z_end']:.4f} m  (still)",
     "#b3261e"),
    (broken_b, "3. broken joint-state read  -  this PR",
     f"install REFUSED at episode start\n0/26 applied, 0 conversion errors\n"
     f"hand z {b['broken_obs']['z_start']:.4f} -> {b['broken_obs']['z_end']:.4f} m  (still)",
     "#1b7f3b"),
]
for col, (im, title, cap, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(im.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=11.4, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(cap, fontsize=9.5, color="#111", linespacing=1.5)

# --- ledger -----------------------------------------------------------------
axl = fig.add_subplot(gs[1, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.965, "A broken kinematics read the install probe could not see", fontsize=13.2, fontweight="bold")
put(axl, 0.0, 0.905,
    "The engine's articulation reports every Franka joint (robot_joint_names); its observation omits panda_joint4.\n"
    "The install validated the first surface; the per-action solve reads the second.",
    fontsize=10.3, color="#333", linespacing=1.5)

rows = [
    ("", "upstream/main", "this PR"),
    ("install verdict", "accepted", "refused"),
    ("_action_controller_error", "None  (the eval reads green)", "set, and it names the read"),
    ("controller left installed", "yes", "no"),
    ("actions applied", "0 of 26", "0 of 26"),
    ("conversion errors mid-eval", f"{a['broken_obs']['action_errors']} of 26", "0  (nothing was installed)"),
    ("hand travel", f"{abs(a['broken_obs']['z_end'] - a['broken_obs']['z_start']) * 1000:.1f} mm  (still)",
     f"{abs(b['broken_obs']['z_end'] - b['broken_obs']['z_start']) * 1000:.1f} mm  (still)"),
]
TOP, LAST = 0.795, 0.345
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
y = TOP
for i, (k, m, p) in enumerate(rows):
    bold = "bold" if i == 0 else "normal"
    put(axl, 0.005, y, k, fontweight="bold" if i else "bold", **MONO)
    put(axl, 0.315, y, m, color="#111" if i == 0 else "#b3261e", fontweight=bold, **MONO)
    put(axl, 0.655, y, p, color="#111" if i == 0 else "#1b7f3b", fontweight=bold, **MONO)
    if i == 0:
        axl.plot([0.0, 1.0], [y - step * 0.34] * 2, color="#bbb", lw=0.9, transform=axl.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9

put(axl, 0.005, 0.255, "Refusal text (this PR):", fontweight="bold", **MONO)
msg = b["broken_obs"]["install_error"]
wrap, line = [], ""
for word in msg.split():
    if len(line) + len(word) + 1 > 128:
        wrap.append(line); line = word
    else:
        line = f"{line} {word}".strip()
wrap.append(line)
FTOP, FSTEP = 0.198, 0.049
for i, ln in enumerate(wrap):
    put(axl, 0.005, FTOP - i * FSTEP, ln, color="#1b7f3b", family="monospace", fontsize=9.0)
foot = FTOP - len(wrap) * FSTEP
put(axl, 0.005, foot,
    f"Renders: panel 1 identical across trees (max |delta| = {d_healthy}/255) - the honored path is untouched.  "
    f"Panels 2 and 3 identical (max |delta| = {d_broken}/255) - the physics is\nunchanged and the whole difference is what the caller is "
    f"told, and when.  Panel 1 vs its own start frame differs on {f_move * 100:.1f}% of pixels, so the controller\ndoes move the arm.  "
    "Production IsaacDeltaEEFController driven over MuJoCo seams (Isaac Sim is not required to reach the install decision).",
    fontsize=9.2, color="#444", linespacing=1.55)
assert foot > 0.02, foot

for ax, yy in placed:
    lo, hi = ax.get_ylim() if ax is not axl else (0.0, 1.0)
    assert -0.03 <= yy <= 1.10, (yy, lo, hi)

out = pathlib.Path(f"/tmp/artifact-{RID}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.32, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((band < 250).any(axis=2).sum())
    assert n == 0, f"{side} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}")
print(f"  panel1 across trees max|d|={d_healthy}  panels2/3 max|d|={d_broken}  move_frac={f_move:.4f}")
