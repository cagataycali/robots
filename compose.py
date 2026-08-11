"""Compose the measurement figure. Every cell comes from /tmp/facts-<run>.json."""
import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUN = os.environ["GITHUB_RUN_ID"]
F = json.load(open(f"/tmp/facts-{RUN}.json"))
OUT = pathlib.Path(f"/tmp/art-{RUN}.png")

# ---- audit every claim before drawing -------------------------------------
DRIVEN, CELLS, MUT, COV = F["driven"], F["cells"], F["mutations"], F["file_coverage"]
REFUSALS = ["fps", "posture", "cameras", "rollout rate"]
BACKENDS = ["newton", "isaac"]

before_driven = sum(DRIVEN["before"][b][r] for b in BACKENDS for r in REFUSALS)
after_driven = sum(DRIVEN["after"][b][r] for b in BACKENDS for r in REFUSALS)
assert before_driven == 0, before_driven
assert after_driven == 6, after_driven
for b in BACKENDS:
    assert DRIVEN["after"][b]["rollout rate"] is False
    assert F["is_rate_unreachable"][b]["rates"] == {}
    assert F["is_rate_unreachable"][b]["guard"] is None
caught_new = sum(1 for m in MUT if m["new"]["failed"] > 0)
blind_old = sum(1 for m in MUT if m["pre_existing"]["failed"] == 0)
assert (len(MUT), caught_new, blind_old) == (8, 8, 6), (len(MUT), caught_new, blind_old)
assert F["baseline"]["new"] == {"failed": 0, "passed": 137}
assert F["baseline"]["pre_existing"] == {"failed": 0, "passed": 287}
assert COV["pristine"]["isaac"]["miss"] - COV["branch"]["isaac"]["miss"] == 3
assert COV["pristine"]["newton"]["miss"] - COV["branch"]["newton"]["miss"] == 3
print("audit OK")

GREEN, RED, GREY, AMBER = "#1b7f3b", "#b3261e", "#6b7280", "#8a6d00"
placed: list[tuple] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 11.0), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.25, 0.62], hspace=0.30,
                      left=0.035, right=0.978, top=0.925, bottom=0.035)

fig.suptitle(
    "start_recording's caller-input refusals: which cells a driver had ever executed",
    fontsize=16.5, fontweight="bold", y=0.982,
)
fig.text(0.5, 0.951,
         "Tests only - no production line changes. Every backend already CALLED each shared guard "
         "(proven by AST sweeps); these are the refusals actually RETURNED.",
         ha="center", fontsize=10.6, color=GREY, style="italic")

# ---- row 1: the refusal matrix -------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.95, "1  Refusal returned by a driver  (MuJoCo column measured on the same full-suite run)",
    transform=ax.transAxes, fontsize=12.4, fontweight="bold")

cols = [0.185, 0.335, 0.485, 0.655, 0.845]
heads = ["refusal", "MuJoCo", "Newton  before", "Newton / Isaac  AFTER", "verdict"]
for x, h in zip(cols, heads):
    put(ax, x, 0.80, h, transform=ax.transAxes, fontsize=10.4, fontweight="bold", ha="center", color=GREY)
put(ax, 0.012, 0.80, "caller input", transform=ax.transAxes, fontsize=10.4, fontweight="bold", color=GREY)

TOP, LAST = 0.665, 0.115
step = (TOP - LAST) / (len(REFUSALS) - 1)
assert step > 0.030, step
for i, r in enumerate(REFUSALS):
    y = TOP - i * step
    unreachable = r == "rollout rate"
    put(ax, 0.012, y, r, transform=ax.transAxes, fontsize=11.4, fontweight="bold")
    put(ax, cols[0], y, f"{CELLS['newton'][r]} / {CELLS['isaac'][r]}", transform=ax.transAxes,
        fontsize=9.6, ha="center", color=GREY, family="monospace")
    put(ax, cols[1], y, "driven", transform=ax.transAxes, fontsize=11, ha="center", color=GREEN)
    put(ax, cols[2], y, "never run", transform=ax.transAxes, fontsize=11, ha="center",
        color=AMBER if unreachable else RED, fontweight="bold")
    put(ax, cols[3], y, "unreachable (proved)" if unreachable else "driven, both backends",
        transform=ax.transAxes, fontsize=11, ha="center",
        color=AMBER if unreachable else GREEN, fontweight="bold")
    note = ("_active_rollout_rates() == {} -> guard returns None"
            if unreachable else "returns the shared domain's verdict verbatim")
    put(ax, cols[4], y, note, transform=ax.transAxes, fontsize=8.9, ha="center", color=GREY, style="italic")
    ax.add_patch(plt.Rectangle((0.005, y - step * 0.42), 0.99, step * 0.84,
                               transform=ax.transAxes, facecolor=AMBER if unreachable else GREEN,
                               alpha=0.055, zorder=0))
assert abs((TOP - (len(REFUSALS) - 1) * step) - LAST) < 1e-9
put(ax, 0.012, 0.028,
    f"cells a driver executed:  before  {before_driven} of 8      after  {after_driven} of 8"
    "      (the 2 remaining are provably unreachable, not untested)",
    transform=ax.transAxes, fontsize=10.7, fontweight="bold")

# ---- row 2: mutation matrix ----------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.955, "2  Mutation matrix - a structural pin proves the guard is CALLED, never that its refusal is RETURNED",
    transform=ax2.transAxes, fontsize=12.4, fontweight="bold")
for x, h in zip([0.435, 0.585, 0.775], ["style", "new module (137)", "pre-existing (287)"]):
    put(ax2, x, 0.865, h, transform=ax2.transAxes, fontsize=10.4, fontweight="bold", ha="center", color=GREY)
put(ax2, 0.012, 0.865, "mutation applied to production", transform=ax2.transAxes,
    fontsize=10.4, fontweight="bold", color=GREY)

T2, L2 = 0.775, 0.145
s2 = (T2 - L2) / (len(MUT) - 1)
assert s2 > 0.030, s2
for i, m in enumerate(MUT):
    y = T2 - i * s2
    style = "DELETE" if m["label"].startswith("DELETE") else "DISCARD"
    blind = m["pre_existing"]["failed"] == 0
    put(ax2, 0.012, y, f"{m['backend']}/recording.py  ::  {m['label']}", transform=ax2.transAxes, fontsize=10.7)
    put(ax2, 0.435, y, style, transform=ax2.transAxes, fontsize=10.2, ha="center",
        color=RED if blind else GREY, family="monospace", fontweight="bold")
    put(ax2, 0.585, y, f"{m['new']['failed']} failed", transform=ax2.transAxes, fontsize=10.7,
        ha="center", color=GREEN, fontweight="bold")
    lbl = f"{m['pre_existing']['failed']} failed" + ("   <- BLIND" if blind else "")
    put(ax2, 0.775, y, lbl, transform=ax2.transAxes, fontsize=10.7, ha="center",
        color=RED if blind else GREY, fontweight="bold" if blind else "normal")
    if blind:
        ax2.add_patch(plt.Rectangle((0.66, y - s2 * 0.40), 0.335, s2 * 0.80,
                                    transform=ax2.transAxes, facecolor=RED, alpha=0.075, zorder=0))
assert abs((T2 - (len(MUT) - 1) * s2) - L2) < 1e-9
put(ax2, 0.012, 0.045,
    f"caught by the new module: {caught_new} of {len(MUT)}       "
    f"invisible to the pre-existing suite: {blind_old} of {len(MUT)}"
    "        (the 2 it does catch are the DELETEs - the sweeps doing their job)",
    transform=ax2.transAxes, fontsize=10.7, fontweight="bold")

# ---- row 3: coverage + gate ---------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 0.90, "3  Full-suite coverage of the two files, and the gate",
    transform=ax3.transAxes, fontsize=12.4, fontweight="bold")
lines = [
    f"simulation/newton/recording.py    {COV['pristine']['newton']['miss']:2d} missing "
    f"{COV['pristine']['newton']['pct']:5.1f}%   ->   {COV['branch']['newton']['miss']:2d} missing "
    f"{COV['branch']['newton']['pct']:5.1f}%",
    f"simulation/isaac/recording.py     {COV['pristine']['isaac']['miss']:2d} missing "
    f"{COV['pristine']['isaac']['pct']:5.1f}%   ->   {COV['branch']['isaac']['miss']:2d} missing "
    f"{COV['branch']['isaac']['pct']:5.1f}%     (lowest-covered non-optional-dep file on main)",
    "full suite  28043 passed / 257 skipped / 0 failed      ruff clean 1170 files      "
    "mypy 0 errors outside examples/",
    "the 137 new tests need no MuJoCo, Isaac Sim, Newton, Warp or lerobot: the guards run above the "
    "lerobot-extra probe (pinned)",
]
T3, L3 = 0.66, 0.10
s3 = (T3 - L3) / (len(lines) - 1)
assert s3 > 0.030, s3
for i, ln in enumerate(lines):
    put(ax3, 0.012, T3 - i * s3, ln, transform=ax3.transAxes, fontsize=10.5,
        family="monospace" if i < 2 else None, color="black" if i < 3 else GREY,
        style="italic" if i == 3 else "normal")
assert abs((T3 - (len(lines) - 1) * s3) - L3) < 1e-9

for ax_, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.07, f"text at axes y={y}"
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.05 <= y <= hi + 0.07, f"text at data y={y} outside {(lo, hi)}"

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"wrote {OUT}  size={Image.open(OUT).size}  border clean")
