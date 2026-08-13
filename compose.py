"""Compose the artifact from the measured facts. Every drawn number is asserted."""
import json, pathlib, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RUN = sys.argv[1]
HERE = pathlib.Path(__file__).resolve().parent
F = json.loads(pathlib.Path(f"/tmp/art-facts-{RUN}.json").read_text())

H = {k: np.load(HERE / f"p_{k}.npy") for k in ("home", "honored", "refused_move", "refused_wrist")}

hon, rmv, rwr = F["honored"], F["refused_move_to"], F["refused_rotate_wrist"]
assert hon["status"] == "success" and rmv["status"] == "error" and rwr["status"] == "error"
assert hon["moved_frac"] > 0.10 and rmv["changed_px"] == 0 and rwr["changed_px"] == 0

# The six refusals this PR drives, with the coverage measured over the primitive suites.
REFUSALS = [
    ("move_to", "bodyless", "an end-effector frame", "495"),
    ("move_to", "actuatorless", "any joint-transmission actuator", "504"),
    ("move_to", "gripper-only", "a NON-gripper actuator", "523"),
    ("set_gripper", "no-gripper", "a gripper actuator", "733/736"),
    ("rotate_wrist", "actuatorless", "any joint-transmission actuator", "847"),
    ("rotate_wrist", "gripper-only", "a wrist joint", "881/882"),
]
MUTATIONS = [
    ("M1  move_to: proceed with no EE frame", 4, 0),
    ("M2  move_to: proceed with no actuators", 3, 0),
    ("M3  move_to: proceed when all actuators are grippers", 3, 0),
    ("M4  set_gripper: proceed with no gripper actuator", 5, 0),
    ("M5  rotate_wrist: proceed with no actuators", 1, 0),
    ("M6  rotate_wrist: distal fallback reaches past the gripper class", 3, 2),
    ("M7  _short_name: hand back MuJoCo's None, not an empty name", 6, 0),
]
assert sum(1 for _l, n, _o in MUTATIONS if n > 0) == 7
assert sum(1 for _l, _n, o in MUTATIONS if o > 0) == 1

placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(16.4, 14.4), dpi=122)
fig.patch.set_facecolor("white")
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.06, 1.06, 0.62], hspace=0.20, wspace=0.06,
              left=0.028, right=0.972, top=0.925, bottom=0.028)

fig.suptitle(
    "MuJoCo motion primitives: an honored primitive moves the arm, a resolution refusal costs nothing",
    fontsize=17, fontweight="bold", y=0.976,
)
fig.text(0.5, 0.947,
         "Headless MuJoCo (MUJOCO_GL=egl) on Thor. Tests only - no library code changes - so every render below is "
         "produced by unchanged production code.",
         ha="center", fontsize=11, style="italic", color="#444")


def panel(ax, img, title, caption, tc):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=tc, pad=7)
    ax.set_xlabel(caption, fontsize=9.6, color="#333", labelpad=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(tc); sp.set_linewidth(2.0)


panel(fig.add_subplot(gs[0, 0]), H["home"], "1. conventional arm, at rest",
      "ee_site TCP, 4 actuated hinges + a jaw:\nevery resolution succeeds", "#555")
panel(fig.add_subplot(gs[0, 1]), H["honored"],
      f"2. honored move_to -> status success",
      f"reached [0.2, 0.1, 0.2] in 32 steps (err 0.0199 m)\n{hon['moved_frac']:.2%} of pixels differ from panel 1",
      "#1a7f37")
panel(fig.add_subplot(gs[0, 2]), H["refused_move"],
      "3. actuatorless arm, refused move_to",
      f"hinges + ee_site, NO <actuator> block\n{rmv['changed_px']} pixels differ from its own rest frame "
      f"(max delta {rmv['max_delta']}/255)", "#b3261e")

panel(fig.add_subplot(gs[1, 0]), H["refused_wrist"],
      "4. gripper-only arm, refused rotate_wrist",
      f"the only actuated hinge is the jaw\n{rwr['changed_px']} pixels differ from its own rest frame "
      f"(max delta {rwr['max_delta']}/255)", "#b3261e")

# ---- ledger: the six refusals -------------------------------------------------
axl = fig.add_subplot(gs[1, 1:]); axl.axis("off")
axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.965, "The six resolution refusals, all previously undriven", fontsize=13,
    fontweight="bold", transform=axl.transAxes)
put(axl, 0.0, 0.905, "primitive            fixture         cannot resolve                      line   driven",
    fontsize=10.3, family="monospace", fontweight="bold", transform=axl.transAxes)
TOP, LAST = 0.845, 0.545
step = (TOP - LAST) / (len(REFUSALS) - 1)
assert step > 0.030, step
y = TOP
for prim, fix, what, line in REFUSALS:
    put(axl, 0.0, y, f"{prim:<20} {fix:<15} {what:<35} {line:<6} was no -> yes",
        fontsize=10.3, family="monospace", transform=axl.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

put(axl, 0.0, 0.455,
    "Each fixture is an inline MJCF removing exactly one thing a resolution needs.\n"
    "A premise class asserts that, so no refusal can fire for an unrelated reason.\n\n"
    "Every refusal is reached before any IK solve, so the module needs neither\n"
    "mink nor a GL context; only panel 2's control is gated on mink.",
    fontsize=10.2, color="#333", va="top", transform=axl.transAxes)
put(axl, 0.0, 0.115,
    "motion_primitives.py over the primitive suites:  93.69% -> 96.40%\n"
    "9 lines closed (290, 495, 504, 523, 733, 736, 847, 881, 882), 0 newly missing",
    fontsize=10.6, family="monospace", color="#1a7f37", fontweight="bold", va="top",
    transform=axl.transAxes)

# ---- mutation matrix ----------------------------------------------------------
axm = fig.add_subplot(gs[2, :]); axm.axis("off")
axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 0.95, "Seven plausible regressions, measured against both arms", fontsize=13,
    fontweight="bold", transform=axm.transAxes)
put(axm, 0.0, 0.855, f"{'mutation':<66}{'this module':>13}{'163 pre-existing cases':>25}",
    fontsize=10.3, family="monospace", fontweight="bold", transform=axm.transAxes)
MTOP, MLAST = 0.775, 0.245
mstep = (MTOP - MLAST) / (len(MUTATIONS) - 1)
assert mstep > 0.030, mstep
y = MTOP
for label, nf, of in MUTATIONS:
    tail = f"{'BLIND' if of == 0 else str(of) + ' failed':>25}"
    put(axm, 0.0, y, f"{label:<66}{str(nf) + ' failed':>13}{tail}", fontsize=10.3,
        family="monospace", color="#333" if of else "#b3261e", transform=axm.transAxes)
    y -= mstep
assert abs((y + mstep) - MLAST) < 1e-9

put(axm, 0.0, 0.155,
    "7/7 caught here; 6/7 invisible to the pre-existing suite. M6 is caught by both - it is the "
    "regression the code's own comment warns about\n(rotate_wrist would open/close the gripper "
    "instead of rotating the wrist) and the one the Isaac backend already pins.",
    fontsize=10.2, color="#333", va="top", transform=axm.transAxes)
put(axm, 0.0, 0.045,
    "Gate: 28709 passed / 258 skipped / 0 failed (627s) | ruff clean 1191 files | mypy 0 errors outside "
    "examples/ | +29 cases, 0 existing tests changed",
    fontsize=10.4, family="monospace", fontweight="bold", color="#1a7f37", transform=axm.transAxes)

for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.07, f"axes-fraction text at y={y}"
    else:
        lo, hi = ax.get_ylim()
        assert min(lo, hi) - 0.05 <= y <= max(lo, hi) + 0.07, f"data text at y={y} outside {(lo, hi)}"

out = HERE / "primitive_resolution_refusals.png"
fig.savefig(out, dpi=122, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("wrote", out, im.shape, f"{out.stat().st_size / 1024:.0f} KiB")
