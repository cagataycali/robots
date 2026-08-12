"""Compose the joint_limits construction-verdict figure from the two dumps."""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image

RUN = pathlib.Path("/tmp/minepath-" + __import__("os").environ["GITHUB_RUN_ID"]).read_text().strip().rsplit("-", 1)[1]
A = json.loads(pathlib.Path(f"/tmp/art-base-{RUN}.json").read_text())
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RUN}.json").read_text())
assert A["tree"] != B["tree"], "both dumps came from the same tree"

VALS = ["-1.9", "1.9", "nan", "+inf", "-inf"]
EMPTY = {(lo, hi) for lo in VALS for hi in VALS if "nan" in (lo, hi)} | {("+inf", "+inf"), ("-inf", "-inf")}
HALF_OPEN = {("-1.9", "+inf"), ("1.9", "+inf"), ("-inf", "-1.9"), ("-inf", "1.9"), ("-inf", "+inf")}

n_bad_A = sum(1 for v in A["matrix"].values() if not v["correct"])
n_bad_B = sum(1 for v in B["matrix"].values() if not v["correct"])
bad_A = {(v["low"], v["high"]) for v in A["matrix"].values() if not v["correct"]}
assert (n_bad_A, n_bad_B) == (16, 0), (n_bad_A, n_bad_B)
assert bad_A == (EMPTY | HALF_OPEN), sorted(bad_A ^ (EMPTY | HALF_OPEN))
assert len(EMPTY) == 11 and len(HALF_OPEN) == 5
assert A["huge"]["kind"] == "OverflowError" and B["huge"]["kind"] == "ValueError"
L_A, L_B = A["ledger"], B["ledger"]
assert L_A["nan max (-1.9, nan)"]["ctor"] == "accepted" and L_A["nan max (-1.9, nan)"]["applied"] == 0
assert L_B["nan max (-1.9, nan)"]["ctor"] == "refused"
# the valid control is untouched on both trees
assert L_A["valid (-1.9, 1.9)"] == L_B["valid (-1.9, 1.9)"], "the valid range changed"
assert L_A["valid (-1.9, 1.9)"]["applied"] == 5

GREEN, RED, ORANGE, GREY = "#1b7a3d", "#b3261e", "#c9691b", "#5f6368"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.36, 0.78, 0.30], hspace=0.30, wspace=0.16)

fig.suptitle(
    "joint_limits: a bound the ordering check cannot see  --  construction verdict over every (min, max) kind",
    fontsize=15.5, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.947,
    "Contract: a declared {motor: (min, max)} range must be a bounded interval able to admit a finite position.  "
    "Measured in-process on RosTelemetryBase, no ROS 2 / cyclonedds needed.",
    ha="center", fontsize=10.4, color=GREY,
)

# ---------------- row 1: the 5x5 construction-verdict matrix ----------------
for col, (dump, title) in enumerate([(A, "main @ 32bab339"), (B, "this change")]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 5); ax.set_ylim(0, 5.9); ax.axis("off")
    n_bad = sum(1 for v in dump["matrix"].values() if not v["correct"])
    ax.set_title(
        f"{title}\nwrong construction verdicts: {n_bad} of 25",
        fontsize=12.6, fontweight="bold", color=RED if n_bad else GREEN, pad=8,
    )
    for j, hi in enumerate(VALS):
        put(ax, j + 0.5, 5.22, f"max\n{hi}", ha="center", va="center", fontsize=9.6, fontweight="bold")
    for i, lo in enumerate(VALS):
        y = 4.5 - i
        put(ax, -0.12, y + 0.5, f"min {lo}", ha="right", va="center", fontsize=9.6, fontweight="bold")
        for j, hi in enumerate(VALS):
            cell = dump["matrix"][f"{lo}|{hi}"]
            if cell["correct"]:
                face, edge, label = "#e8f5ec", GREEN, {"accepted": "accept", "refused": "refuse", "raised": "raise"}[cell["verdict"]]
            elif (lo, hi) in EMPTY:
                face, edge, label = "#fdecea", RED, "accept\nadmits nothing"
            else:
                face, edge, label = "#fdf1e3", ORANGE, "accept\nhalf-open"
            if cell["verdict"] == "raised":
                face, edge, label = "#fdecea", RED, "OverflowError"
            ax.add_patch(Rectangle((j + 0.04, y + 0.04), 0.92, 0.92, facecolor=face, edgecolor=edge, lw=1.5))
            put(ax, j + 0.5, y + 0.5, label, ha="center", va="center", fontsize=8.4, color=edge, fontweight="bold")
    if col == 0:
        ax.legend(
            handles=[
                Patch(facecolor="#e8f5ec", edgecolor=GREEN, label="verdict matches the contract"),
                Patch(facecolor="#fdecea", edgecolor=RED, label="accepted, yet admits no position (11)"),
                Patch(facecolor="#fdf1e3", edgecolor=ORANGE, label="accepted as a half-open range (5, now refused)"),
            ],
            loc="lower center", bbox_to_anchor=(0.5, -0.085), ncol=1, fontsize=9.1, frameon=False,
        )

# ---------------- row 2: the consequence ledger ----------------
for col, (dump, title) in enumerate([(A, "main @ 32bab339"), (B, "this change")]):
    ax = fig.add_subplot(gs[1, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"What an operator gets  --  {title}", fontsize=11.8, fontweight="bold", pad=6)
    led = dump["ledger"]
    rows = []
    for label in ["valid (-1.9, 1.9)", "nan max (-1.9, nan)"]:
        e = led[label]
        rows.append((f"joint_limits={{'shoulder_pan': {label.split(' ', 2)[-1] if 'nan' in label else '(-1.9, 1.9)'}}}", True))
        rows.append((f"    construction        {e['ctor']}" + (f"  -> {e['err'][:78]}" if e["err"] else ""), False))
        rows.append((f"    in-range commands   {e['applied']} of {e['total']} reached the arm", False))
        if e["log"]:
            rows.append((f"    log                 {e['log'][:96]}", False))
        rows.append(("", False))
    h = dump["huge"]
    rows.append((f"joint_limits={{'j': (10**400, 1.0)}}", True))
    rows.append((f"    {h['kind']}: {h['msg'][:88]}", False))
    TOP, LAST = 0.94, 0.06
    step = (TOP - LAST) / (len(rows) - 1)
    assert step > 0.030, step
    y = TOP
    for text, bold in rows:
        colour = "#202124" if bold else GREY
        if "0 of 5" in text or "OverflowError" in text:
            colour = RED
        elif "5 of 5" in text or "must be a finite number" in text or "range of a 64-bit float" in text:
            colour = GREEN
        put(ax, 0.012, y, text, fontsize=9.0, family="monospace", color=colour,
            fontweight="bold" if bold else "normal", va="center")
        y -= step
    assert abs((y + step) - LAST) < 1e-9, (y, LAST)

# ---------------- row 3: footer ----------------
axf = fig.add_subplot(gs[2, :])
axf.set_xlim(0, 1); axf.set_ylim(0, 1); axf.axis("off")
foot = [
    "Why the ordering check is blind:  1.9 > nan is False,  nan > nan is False,  inf > inf is False  ->  (low, nan), (inf, inf) and (-inf, -inf) all pass  low > high.",
    "Each bound now goes through strands_robots.utils.finite_number_error before that comparison - the same shared domain, and the same wording, as its other callers.",
    "Both HardwareRosBridge and HardwareRtpsBridge inherit the validator, so one rule covers the rclpy and pure-RTPS transports.",
    "Gate: 28428 passed / 257 skipped / 0 failed (MUJOCO_GL=egl, full suite, 645s) | ruff + ruff format + mypy clean | pre-fix: 13 failed / 30 passed.",
]
TOP, LAST = 0.86, 0.10
step = (TOP - LAST) / (len(foot) - 1)
assert step > 0.030, step
y = TOP
for line in foot:
    put(axf, 0.006, y, line, fontsize=9.5, family="monospace",
        color="#202124" if line.startswith("Gate") else GREY, va="center")
    y -= step
assert abs((y + step) - LAST) < 1e-9

for ax, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.10, (yv, "axes-fraction text outside the panel")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.35 <= yv <= hi + 0.45, (yv, lo, hi)

out = pathlib.Path("/tmp/joint_limits_finite_bounds.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in [("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])]:
    nonwhite = int((np.abs(band - 255).sum(2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"OK {out}  size={Image.open(out).size}  wrong verdicts {n_bad_A} -> {n_bad_B}")
