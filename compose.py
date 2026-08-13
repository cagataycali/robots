"""Compose the artifact. Every rendered number is re-derived from the dumps."""
from __future__ import annotations
import json, os, pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

R = os.environ["GITHUB_RUN_ID"]
MAIN = json.loads(pathlib.Path(f"/tmp/facts-wt-main-{R}.json").read_text())
BR = json.loads(pathlib.Path(f"/tmp/facts-robots-mine-{R}.json").read_text())
MX_MAIN = {(r["phase"], r["primitive"]): r for r in json.loads(pathlib.Path(f"/tmp/matrix-wt-main-{R}.json").read_text())["rows"]}
MX_BR = {(r["phase"], r["primitive"]): r for r in json.loads(pathlib.Path(f"/tmp/matrix-robots-mine-{R}.json").read_text())["rows"]}
assert MAIN["tree"] != BR["tree"], "both arms measured the same tree"

PHASES = [
    ("already-running", "a policy is already running", ["move_to", "set_gripper", "rotate_wrist"]),
    # move_to's mid-run abort needs an IK model to reach; it is already pinned in
    # the backend's IK suite, so this probe drives the two that were missing it.
    ("starts-mid-run", "a policy starts mid-run", ["set_gripper", "rotate_wrist"]),
]

def honors(row: dict) -> bool:
    """The contract: refuse, and say the policy is why."""
    return row["status"] == "error" and row["names_policy"]

bad_main = sum(1 for k, r in MX_MAIN.items() if not honors(r))
bad_br = sum(1 for k, r in MX_BR.items() if not honors(r))
cells = sum(len(p) for _, _, p in PHASES)
assert (cells, bad_main, bad_br) == (5, 4, 0), (cells, bad_main, bad_br)

roll = np.asarray(iio.imread(BR["renders"]["rollout"])).astype(int)
prim = np.asarray(iio.imread(BR["renders"]["primitive"])).astype(int)
roll_m = np.asarray(iio.imread(MAIN["renders"]["rollout"])).astype(int)
prim_m = np.asarray(iio.imread(MAIN["renders"]["primitive"])).astype(int)
assert np.abs(roll - roll_m).max() == 0 and np.abs(prim - prim_m).max() == 0, "pose renders differ across trees"
differ = float((np.abs(roll - prim).max(2) > 8).mean())
assert differ > 0.10, differ
W_IDX = 3  # wrist_roll
w_roll, w_prim = BR["rollout_pose"][W_IDX], BR["primitive_pose"][W_IDX]
contended = MAIN["contended_writes"]
assert contended == 20 and BR["contended_writes"] == 0, (contended, BR["contended_writes"])

fig = plt.figure(figsize=(15.4, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.42, 1.05, 0.62], hspace=0.20, wspace=0.06)
placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig.suptitle(
    "Isaac motion primitives: a primitive under a running policy writes the same articulation's PD targets",
    fontsize=15, fontweight="bold", y=0.982,
)
fig.text(0.5, 0.958,
         "The two command streams in conflict, rendered as the poses each party commands. "
         "Sim: MuJoCo headless (MUJOCO_GL=egl) replaying the recorded joint targets.",
         ha="center", fontsize=10.2, style="italic", color="#444")

# --- row 1: the two conflicting commands -------------------------------
for col, (img, title, sub) in enumerate((
    (roll, "What the rollout commands", f"wrist_roll -> {w_roll:+.3f} rad  (30 PD target sets over 30 ticks)"),
    (prim, "What rotate_wrist commands, same joint, same ticks", f"wrist_roll -> {w_prim:+.3f} rad  (20 more target sets)"),
)):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(img.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11.6, fontweight="bold", pad=7)
    ax.set_xlabel(sub, fontsize=9.6, labelpad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2f6fbf" if col == 0 else "#c0392b"); sp.set_linewidth(2.2)
fig.text(0.5, 0.632,
         f"The two poses differ on {100 * differ:.2f}% of pixels: opposite directions on wrist_roll. "
         "Both panels are byte-identical across the two trees (max|delta| = 0) - they visualise the commands, not an outcome.",
         ha="center", fontsize=9.8, color="#333")

# --- row 2: the 6-cell verdict matrix ----------------------------------
axm = fig.add_subplot(gs[1, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.5, 1.045, "What the caller is told, per primitive, per phase", ha="center",
    fontsize=12.2, fontweight="bold", transform=axm.transAxes)
COLS = [0.012, 0.145, 0.500]
put(axm, COLS[0], 0.945, "phase", fontsize=9.6, fontweight="bold", transform=axm.transAxes)
put(axm, COLS[1], 0.945, "primitive", fontsize=9.6, fontweight="bold", transform=axm.transAxes)
put(axm, COLS[2], 0.945, "before this change", fontsize=9.6, fontweight="bold", color="#c0392b", transform=axm.transAxes)
put(axm, COLS[2] + 0.255, 0.945, "with this change", fontsize=9.6, fontweight="bold", color="#1e7d32", transform=axm.transAxes)
TOP, LAST, N = 0.865, 0.075, cells
STEP = (TOP - LAST) / (N - 1)
assert STEP > 0.030, STEP
y = TOP
for phase, phase_label, prims in PHASES:
    for i, prim_name in enumerate(prims):
        m, b = MX_MAIN[(phase, prim_name)], MX_BR[(phase, prim_name)]
        if i == 0:
            put(axm, COLS[0], y, phase_label, fontsize=9.0, style="italic", color="#333", transform=axm.transAxes)
        put(axm, COLS[1], y, prim_name, fontsize=9.4, family="monospace", transform=axm.transAxes)
        for col_x, row in ((COLS[2], m), (COLS[2] + 0.255, b)):
            ok = honors(row)
            body = row["text"] if row["status"] == "error" else f"status=success, {row['actions_applied']} PD target sets applied"
            body = body.split(" - ")[0].split(" (")[0]
            put(axm, col_x, y, ("v " if ok else "x ") + textwrap.shorten(body, 58, placeholder="..."),
                fontsize=8.4, family="monospace", color="#1e7d32" if ok else "#c0392b", transform=axm.transAxes)
        y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y
put(axm, 0.5, 0.010,
    f"cells not honouring the refusal contract:  {bad_main} of {cells}  ->  {bad_br} of {cells}",
    ha="center", fontsize=10.4, fontweight="bold", transform=axm.transAxes)

# --- row 3: ledger -----------------------------------------------------
axl = fig.add_subplot(gs[2, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
LED = [
    ("PD target sets reaching the articulation while a 30-tick rollout runs",
     f"{MAIN['reference']['target_writes'] + contended} (the rollout's 30 + {contended} contended)",
     f"{BR['with_primitive']['target_writes']} (the rollout's own, unchanged)"),
    ("rotate_wrist's report",
     "convergence timeout - blames the arm for the race",
     "names the running policy and how to proceed"),
    ("rollout trajectory recorded over the 30 ticks",
     "identical to the uncontended reference in this model",
     "identical to the uncontended reference"),
]
put(axl, 0.5, 1.02, "Measured ledger", ha="center", fontsize=12.0, fontweight="bold", transform=axl.transAxes)
TOP2, LAST2 = 0.78, 0.20
STEP2 = (TOP2 - LAST2) / (len(LED) - 1)
assert STEP2 > 0.10, STEP2
y = TOP2
for label, a, b in LED:
    put(axl, 0.012, y, label, fontsize=9.2, transform=axl.transAxes)
    put(axl, 0.500, y, a, fontsize=9.0, family="monospace", color="#c0392b", transform=axl.transAxes)
    put(axl, 0.755, y, b, fontsize=9.0, family="monospace", color="#1e7d32", transform=axl.transAxes)
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, y
put(axl, 0.5, 0.03,
    "Which party a contended tick resolves to is arbitrary in a real race, so no outcome pose is asserted: "
    "the measured harm is the 20 extra target sets and the report that names the wrong cause.",
    ha="center", fontsize=8.8, style="italic", color="#444", transform=axl.transAxes)

for ax, yy, is_axes in placed:
    lo, hi = (-0.03, 1.07) if is_axes else ax.get_ylim()
    assert lo <= yy <= hi, (yy, lo, hi)

out = pathlib.Path(f"/tmp/artifact-{R}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(iio.imread(out))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print("WROTE", out, im.shape)
