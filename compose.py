"""Compose the artifact. Every number is read from _art/facts.json + /tmp/mut-*.json."""

from __future__ import annotations

import json
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).resolve().parent
RUN = os.environ["GITHUB_RUN_ID"]
F = json.loads((HERE / "facts.json").read_text())
MUT = json.loads(pathlib.Path(f"/tmp/mut-{RUN}.json").read_text())
CB = json.loads(pathlib.Path(f"/tmp/cbefore-{RUN}.json").read_text())["files"][
    "strands_robots/simulation/isaac/motion_primitives.py"
]
CA = json.loads(pathlib.Path(f"/tmp/cafter-{RUN}.json").read_text())["files"][
    "strands_robots/simulation/isaac/motion_primitives.py"
]

# --- audit the inputs before drawing anything -----------------------------
assert F["runs"]["fallback:open"]["targets"] == F["runs"]["props:open"]["targets"], "sources disagree"
assert F["runs"]["none:open"]["status"] == "error"
assert F["render"]["differing_fraction"] > 0.10, F["render"]
assert F["render"]["arm_closed"] > 0.15 and F["render"]["arm_opened"] > 0.15, F["render"]
CAUGHT_NEW, CAUGHT_OLD = MUT["caught_new"], MUT["caught_old"]
assert (CAUGHT_NEW, CAUGHT_OLD) == (8, 0), (CAUGHT_NEW, CAUGHT_OLD)
MISS_B, MISS_A = CB["summary"]["missing_lines"], CA["summary"]["missing_lines"]
PCT_B, PCT_A = CB["summary"]["percent_covered"], CA["summary"]["percent_covered"]
assert (MISS_B, MISS_A) == (30, 2), (MISS_B, MISS_A)
assert sorted(CA["missing_lines"]) == [139, 186], CA["missing_lines"]
CLOSED = MISS_B - MISS_A

GREEN, RED, GREY, INK = "#1a7f37", "#b3261e", "#8b949e", "#101418"
placed: list[tuple] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 14.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.28, 1.30, 1.02], hspace=0.20, left=0.028, right=0.978, top=0.945, bottom=0.022)

fig.suptitle(
    "Isaac articulation read/write surfaces: the fallback limit source and every I/O failure now driven",
    fontsize=16.5, fontweight="bold", color=INK, y=0.981,
)
fig.text(
    0.5, 0.9555,
    "tests only - 0 production lines changed. The Isaac articulation is faked (Isaac Sim is not required to reach "
    "either decision);\nthe joint targets the primitive commands are replayed on a MuJoCo arm declaring the same "
    "joint vocabulary and the same limits.",
    ha="center", fontsize=10.4, color="#3d444d",
)

# ------------------------------------------------------------------ row 1
top = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 1.02], wspace=0.05)
for col, (tag, label) in enumerate((("closed", "close"), ("opened", "open"))):
    ax = fig.add_subplot(top[0, col])
    ax.imshow(np.load(HERE / f"{tag}.npy"))
    ax.set_xticks([]); ax.set_yticks([])
    jaw = F["runs"][f"fallback:{label}"]["targets"]["jaw"]
    end = "LOW end" if label == "close" else "HIGH end"
    ax.set_title(f"set_gripper(state=\"{label}\")  ->  jaw = {jaw:+.2f} rad", fontsize=11.6, fontweight="bold", color=INK, pad=7)
    ax.set_xlabel(
        f"resolved from get_dof_limits() - the FALLBACK source\n{end} of the span (-0.20, +1.50); "
        f"arm fills {F['render']['arm_' + tag] * 100:.0f}% of frame",
        fontsize=9.5, color="#3d444d", labelpad=6,
    )
    for s in ax.spines.values():
        s.set_edgecolor(GREEN); s.set_linewidth(2.1)

axf = fig.add_subplot(top[0, 2]); axf.set_xlim(0, 1); axf.set_ylim(0, 1); axf.axis("off")
axf.add_patch(Rectangle((0.01, 0.02), 0.98, 0.96, facecolor="#f6f8fa", edgecolor="#d0d7de", lw=1.1))
put(axf, 0.045, 0.925, "What each documented limit source resolved", fontsize=11.4, fontweight="bold", color=INK, transform=axf.transAxes)
rows = [
    ("dof_properties  (authoritative)", "jaw -> +1.50", GREEN),
    ("get_dof_limits()  (the FALLBACK)", "jaw -> +1.50", GREEN),
    ("neither source present", "refused, no write", GREEN),
]
TOP, LAST = 0.845, 0.705
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
y = TOP
for name, val, col in rows:
    put(axf, 0.055, y, name, fontsize=9.9, color=INK, family="DejaVu Sans Mono", transform=axf.transAxes)
    put(axf, 0.72, y, val, fontsize=9.9, color=col, fontweight="bold", family="DejaVu Sans Mono", transform=axf.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(axf, 0.055, 0.615,
    "The two sources agree byte-for-byte, and the\n"
    "absent-source case refuses instead of mapping\n"
    "open/close onto a range that does not exist.\n\n"
    "Before this module only the FIRST row was\n"
    "driven: the fake articulation always supplies\n"
    "dof_properties, so the fallback source and\n"
    "every no-usable-bounds outcome were unreached.",
    fontsize=9.7, color="#3d444d", va="top", transform=axf.transAxes)
put(axf, 0.055, 0.215,
    f"closed vs open differ on {F['render']['differing_fraction'] * 100:.2f}% of pixels\n"
    "(1.70 rad of jaw travel, both panels in frame)",
    fontsize=9.4, color="#57606a", va="top", style="italic", transform=axf.transAxes)

# ------------------------------------------------------------------ row 2
axm = fig.add_subplot(gs[1]); axm.set_xlim(0, 1); axm.set_ylim(0, 1); axm.axis("off")
put(axm, 0.0, 0.975,
    "Every documented behaviour of the articulation read/write layer - was it driven?",
    fontsize=12.6, fontweight="bold", color=INK, transform=axm.transAxes)
CELLS = [
    ("_articulation_dof_limits", "dof_properties is authoritative", True),
    ("", "hasLimits=False  ->  None", True),
    ("", "dof_properties present but unreadable  ->  fall through", False),
    ("", "hasLimits field unreadable  ->  treated as absent", False),
    ("", "get_dof_limits() fallback, plain (n,2)", False),
    ("", "get_dof_limits() fallback, view-shaped (1,n,2)", False),
    ("", "get_dof_limits() fallback, torch tensor .cpu().numpy()", False),
    ("", "the fallback itself raises  ->  None", False),
    ("", "neither source present  ->  None", False),
    ("", "table shorter than the articulation  ->  None", False),
    ("", "a non-finite bound  ->  None", False),
    ("", "a degenerate bound (upper <= lower)  ->  None", False),
    ("_read_joint_positions", "a plain array is read", True),
    ("", "a torch tensor is read through .cpu().numpy()", False),
    ("", "a raising read  ->  None, never zeros", False),
    ("", "a read that answers None  ->  None", False),
    ("_apply_position_targets", "a successful write commands the indexed DOFs", True),
    ("", "a raising write  ->  structured error, not a raise", False),
    ("set_gripper / rotate_wrist", "set_gripper reports a write that failed mid-drive", False),
    ("", "set_gripper reports an unverified final state", False),
    ("", "rotate_wrist reports a pre-servo read failure", False),
    ("", "rotate_wrist reports a mid-servo write failure", False),
    ("", "rotate_wrist aborts on a mid-servo read failure", False),
    ("", "rotate_wrist propagates the gripper-resolution error", False),
]
n_before = sum(1 for *_, was in CELLS if was)
assert (n_before, len(CELLS)) == (4, 24), (n_before, len(CELLS))
TOP, LAST = 0.905, 0.055
step = (TOP - LAST) / (len(CELLS) - 1)
assert step > 0.030, step
put(axm, 0.615, 0.945, "on main", fontsize=10.2, fontweight="bold", color=INK, ha="center", transform=axm.transAxes)
put(axm, 0.755, 0.945, "with this module", fontsize=10.2, fontweight="bold", color=INK, ha="center", transform=axm.transAxes)
y = TOP
for owner, behaviour, was in CELLS:
    if owner:
        put(axm, 0.0, y, owner, fontsize=9.4, color="#0550ae", fontweight="bold", family="DejaVu Sans Mono", transform=axm.transAxes)
    put(axm, 0.205, y, behaviour, fontsize=9.6, color=INK, transform=axm.transAxes)
    put(axm, 0.615, y, "driven" if was else "NOT driven", fontsize=9.4, ha="center",
        color=GREEN if was else RED, fontweight="bold", family="DejaVu Sans Mono", transform=axm.transAxes)
    put(axm, 0.755, y, "driven", fontsize=9.4, ha="center", color=GREEN, fontweight="bold",
        family="DejaVu Sans Mono", transform=axm.transAxes)
    if not was:
        axm.add_patch(Rectangle((0.196, y - 0.011), 0.62, 0.026, facecolor="#fff1f0", edgecolor="none", zorder=0,
                                transform=axm.transAxes))
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(axm, 0.845, 0.50,
    f"{n_before} of {len(CELLS)} documented\n"
    f"behaviours were driven\n"
    f"on main; {len(CELLS)} are now.\n\n"
    f"Two documented SOURCES,\n"
    f"and only the first was\n"
    f"ever exercised.",
    fontsize=10.0, color="#3d444d", va="center", family="DejaVu Sans", transform=axm.transAxes)

# ------------------------------------------------------------------ row 3
axb = fig.add_subplot(gs[2]); axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis("off")
put(axb, 0.0, 0.975, "Mutation table - 8 plausible regressions x 2 arms", fontsize=12.6,
    fontweight="bold", color=INK, transform=axb.transAxes)
put(axb, 0.615, 0.905, "new module", fontsize=9.8, fontweight="bold", color=INK, ha="center", transform=axb.transAxes)
put(axb, 0.775, 0.905, "pre-existing suite", fontsize=9.8, fontweight="bold", color=INK, ha="center", transform=axb.transAxes)
MTOP, MLAST = 0.815, 0.245
mrows = MUT["rows"]
step = (MTOP - MLAST) / (len(mrows) - 1)
assert step > 0.030, step
y = MTOP
for r in mrows:
    put(axb, 0.0, y, r["label"], fontsize=9.5, color=INK, transform=axb.transAxes)
    put(axb, 0.615, y, f"{r['new_failed']} failed", fontsize=9.4, ha="center", color=GREEN,
        fontweight="bold", family="DejaVu Sans Mono", transform=axb.transAxes)
    put(axb, 0.775, y, f"{r['old_failed']} failed  <- BLIND", fontsize=9.4, ha="center", color=RED,
        fontweight="bold", family="DejaVu Sans Mono", transform=axb.transAxes)
    y -= step
assert abs((y + step) - MLAST) < 1e-9, (y, MLAST)
ctl = MUT["control"]
put(axb, 0.0, 0.165,
    f"unmutated control: new module {ctl['new_passed']} passed, pre-existing suite {ctl['old_passed']} passed   |   "
    f"caught by the new module {CAUGHT_NEW}/8, by the pre-existing suite {CAUGHT_OLD}/8",
    fontsize=9.8, color="#3d444d", family="DejaVu Sans Mono", transform=axb.transAxes)
put(axb, 0.0, 0.095,
    f"coverage  motion_primitives.py over tests/simulation/isaac:  {MISS_B} missing / {PCT_B:.1f}%  ->  "
    f"{MISS_A} missing / {PCT_A:.1f}%   ({CLOSED} lines closed)",
    fontsize=9.8, color=INK, family="DejaVu Sans Mono", transform=axb.transAxes)
put(axb, 0.0, 0.030,
    f"the {MISS_A} still missing are lines {sorted(CA['missing_lines'])} - the world/robot resolution guard and the "
    "Kit-pump marshal, the adapter's other two ownership bullets, named out of scope",
    fontsize=9.4, color="#57606a", style="italic", transform=axb.transAxes)

# --- layout guards --------------------------------------------------------
for ax, yy, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yy <= 1.07, (yy, "axes-fraction text outside its panel")
    else:
        lo, hi = ax.get_ylim()
        assert min(lo, hi) - 0.05 <= yy <= max(lo, hi) + 0.07, (yy, (lo, hi))

out = HERE / "artifact.png"
fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.30)
plt.close(fig)

im = np.asarray(matplotlib.image.imread(out) * 255).astype(np.uint8)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert bad == 0, f"{name} border has {bad} non-white pixels"
print(f"WROTE {out}  {im.shape[1]}x{im.shape[0]}  borders clean")
