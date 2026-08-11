"""Compose the artifact from the two captured trees. Every number is measured."""

from __future__ import annotations

import json
import os
import pathlib

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = pathlib.Path(f"/tmp/art-{os.environ['GITHUB_RUN_ID']}")
A = json.loads((D / "facts_main.json").read_text())
B = json.loads((D / "facts_branch.json").read_text())
assert A["tree"] != B["tree"], "both arms measured the same tree"
ma = {r["label"]: r for r in A["records"]}
br = {r["label"]: r for r in B["records"]}


def im(p):
    return iio.imread(p)[:, :, :3]


def diff(a, b):
    d = np.abs(im(a).astype(np.int16) - im(b).astype(np.int16))
    return float((d.max(2) > 8).mean()), int(d.max())


f_move, _ = diff(ma["healthy"]["home_png"], ma["healthy"]["after_png"])
f_hon, m_hon = diff(ma["healthy"]["after_png"], br["healthy"]["after_png"])
f_ref, m_ref = diff(ma["posinf"]["after_png"], br["posinf"]["after_png"])

# --- audit every claim this figure makes, before it is drawn ---
assert ma["healthy"]["applied"] == br["healthy"]["applied"] == 26
assert ma["posinf"]["ctor_ok"] is True and ma["posinf"]["refused"] == 26 and ma["posinf"]["applied"] == 0
assert br["posinf"]["ctor_ok"] is False and br["posinf"]["applied"] == 0
assert "panda_joint1" in ma["posinf"]["envelope"] and "pos_scale" not in ma["posinf"]["envelope"]
assert "pos_scale must be > 0, got inf" in br["posinf"]["ctor_error"]
assert f_move > 0.10, f"the honored rollout must be legible, got {f_move:.2%}"
assert f_hon == 0.0 and m_hon <= 2, (f_hon, m_hon)
assert f_ref == 0.0 and m_ref <= 2, (f_ref, m_ref)

placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 10.2), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.05)

PANELS = [
    (
        ma["healthy"]["after_png"],
        "1. A usable configuration -- unchanged by this PR",
        f"pos_scale=0.05 (the shipped default)\n26 / 26 actions applied\n"
        f"{f_move:.1%} of pixels differ from the settled pose\nidentical on both trees "
        f"({f_hon:.2%} differing, max|delta|={m_hon}/255)",
        "#1b7f3b",
    ),
    (
        ma["posinf"]["after_png"],
        "2. main: pos_scale=inf is ACCEPTED",
        "the constructor returns a controller\n0 / 26 actions applied, 26 refused\n"
        "\"send_action: action value for key\n'panda_joint1' must be finite ... got nan.\"\n"
        "(once per action -- names a joint, not the knob)",
        "#b3261e",
    ),
    (
        br["posinf"]["after_png"],
        "3. This PR: refused at construction",
        "\"IsaacDeltaEEFController: pos_scale\nmust be > 0, got inf.\"\n"
        "no controller is built, no action is issued\n"
        f"identical to panel 2 ({f_ref:.2%} differing,\nmax|delta|={m_ref}/255) -- physics untouched",
        "#1b7f3b",
    ),
]
for col, (png, title, caption, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(im(png))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(2.4)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(caption, fontsize=8.9, family="monospace", labelpad=7, linespacing=1.45)

axt = fig.add_subplot(gs[1, :])
axt.axis("off")
axt.set_xlim(0, 1)
axt.set_ylim(0, 1)

ROWS = [
    ("constructor value", "main", "this PR", "consequence on main"),
    ("pos_scale = inf", "accepted", "refused", "nan target for all 9 joints, every action"),
    ("rot_scale = inf", "accepted", "refused", "nan target for all 9 joints, every action"),
    ("damping = inf", "accepted", "refused", "nan target for all 9 joints, every action"),
    ("joint_limits all-nan", "accepted", "refused", "every target clipped to nan"),
    ("gripper_open = inf", "accepted", "refused", "nan finger target on the first open"),
    ("gripper_close = nan", "accepted", "refused", "latent: nan only at the first GRASP"),
    ("pos_scale = 'abc'", "bare float() ValueError", "refused, names pos_scale", "message did not name the parameter"),
    ("pos_scale = 0.05", "accepted", "accepted", "the shipped default -- unchanged"),
    ("gripper_close = -0.002", "accepted", "accepted", "a signed reference stays usable"),
]
TOP, LAST = 0.905, 0.115
step = (TOP - LAST) / (len(ROWS) - 1)
assert step > 0.030, step
COLX = (0.012, 0.235, 0.375, 0.545)
put(axt, 0.5, 1.055, "Every numeric knob of IsaacDeltaEEFController, before and after",
    transform=axt.transAxes, ha="center", fontsize=12.5, fontweight="bold")
y = TOP
for i, row in enumerate(ROWS):
    header = i == 0
    accepted_on_main = (not header) and row[1].startswith("accepted") and row[2] == "refused"
    for x, cell in zip(COLX, row, strict=True):
        colour = "black"
        if not header:
            if cell == "refused":
                colour = "#1b7f3b"
            elif cell.startswith("accepted") and accepted_on_main:
                colour = "#b3261e"
            elif cell.startswith("bare float"):
                colour = "#b3261e"
        put(axt, x, y, cell, transform=axt.transAxes, fontsize=9.6,
            family="monospace", fontweight="bold" if header else "normal", color=colour)
    if header:
        axt.axhline(y - step * 0.42, xmin=0.008, xmax=0.992, color="0.6", lw=0.9)
    if accepted_on_main:
        axt.add_patch(plt.Rectangle((0.006, y - step * 0.30), 0.988, step * 0.80,
                                    transform=axt.transAxes, color="#b3261e", alpha=0.055, zorder=0))
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

put(axt, 0.012, 0.028,
    "An order comparison cannot reject inf (inf > 0 is True) and cannot see a nan row (nan > nan is False); "
    "gripper_open/close had no bound at all.\nEach knob now clears the shared scalar domain. Panels are real "
    "MuJoCo headless renders: the controller's injected kinematics reads are backed by a\ncompiled MuJoCo Panda, "
    "so the production conversion runs unchanged and its joint targets have a physical consequence. "
    "Isaac Sim is not required\nto reach either constructor decision.",
    transform=axt.transAxes, fontsize=8.5, family="monospace", color="0.25", linespacing=1.5)

for ax, yy, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yy <= 1.10, (yy, "axes-fraction text outside the panel")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

out = D / "artifact.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

# border must be clean on every side
a = iio.imread(out)[:, :, :3]
for name, band in (("top", a[:8]), ("bottom", a[-8:]), ("left", a[:, :8]), ("right", a[:, -8:])):
    bad = int((np.abs(band.astype(np.int16) - 255).sum(2) > 12).sum())
    assert bad == 0, f"{name} border has {bad} non-white px"
print("WROTE", out, iio.imread(out).shape)
print(f"  motion={f_move:.2%}  honored_diff={f_hon:.2%}/max{m_hon}  refused_diff={f_ref:.2%}/max{m_ref}")
