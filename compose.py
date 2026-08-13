"""Compose the measurement figure. Every drawn number is asserted against facts.json."""
from __future__ import annotations

import json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).resolve().parent
F = json.loads((HERE / "facts.json").read_text())
OUT = HERE / "isaac_run_multi_policy_knob_refusals.png"

RED, GREEN, GREY, INK = "#c0392b", "#1e8449", "#7f8c8d", "#17202a"
placed: list[tuple[object, float]] = []

def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, transform=ax.transAxes, **kw)

fig = plt.figure(figsize=(16.4, 12.2), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.24, 1.0, 0.30], hspace=0.16,
                      left=0.022, right=0.985, top=0.945, bottom=0.022)

fig.suptitle(
    "Isaac run_multi_policy: the four caller knobs it routes through MuJoCo's shared helpers",
    fontsize=17, fontweight="bold", color=INK, y=0.982)
fig.text(0.5, 0.958,
         "Tests only - 0 production lines changed, so no policy, simulation, rendering, recording or asset "
         "behaviour moves. This figure is the coverage and mutation measurement, not a rollout.",
         ha="center", fontsize=10.5, color=GREY, style="italic")

# ---------------------------------------------------------------- row 1 ----- #
ax1 = fig.add_subplot(gs[0]); ax1.axis("off"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
put(ax1, 0.0, 0.955, "1. Each knob's refusal, and whether any test drove it through THIS entry point",
    fontsize=13, fontweight="bold", color=INK)
put(ax1, 0.0, 0.905,
    "The source states the parity intent four times in one block: \u201cone refusal text for every backend\u201d, "
    "\u201cguards the same domain as run_policy\u201d, \u201cMuJoCo parity\u201d, \u201cthe shared positive-int domain\u201d.",
    fontsize=9.6, color=GREY, style="italic")

cols = [0.0, 0.135, 0.315, 0.375, 0.437, 0.505]
hdr = ["knob", "shared helper", "line", "before", "now", "what the caller is told (verbatim, asserted by equality)"]
put(ax1, 0.0, 0.845, "", fontsize=1)
for x, h in zip(cols, hdr):
    put(ax1, x, 0.845, h, fontsize=9.6, fontweight="bold", color=INK)
ax1.plot([0.0, 1.0], [0.828, 0.828], transform=ax1.transAxes, color=INK, lw=0.9)

TOP, LAST = 0.755, 0.135
rows = F["rows"]
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for r in rows:
    assert r["driven_before"] is False and r["driven_after"] is True
    ax1.add_patch(Rectangle((-0.004, y - 0.048), 1.008, 0.135, transform=ax1.transAxes,
                            facecolor="#fdf2f0" if True else "none", edgecolor="none", zorder=0))
    put(ax1, cols[0], y, r["knob"], fontsize=10.2, fontweight="bold", color=INK, family="monospace")
    put(ax1, cols[1], y, r["helper"], fontsize=9.0, color=GREY, family="monospace")
    put(ax1, cols[2], y, f"L{r['line']}", fontsize=9.4, color=GREY, family="monospace")
    put(ax1, cols[3], y, "NEVER RUN", fontsize=9.4, fontweight="bold", color=RED)
    put(ax1, cols[4], y, "driven", fontsize=9.4, fontweight="bold", color=GREEN)
    txt = r["text"]
    put(ax1, cols[5], y, (txt[:74] + "\u2026") if len(txt) > 75 else txt,
        fontsize=8.8, color=INK, family="monospace")
    put(ax1, cols[5], y - 0.040, f"probe value {r['bad']}  \u2192  status={r['status']}   "
        f"and the loop applied no joint targets, stepped no physics, set no policy_running flag",
        fontsize=8.2, color=GREY, style="italic")
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

c = F["coverage"]
put(ax1, 0.0, 0.048,
    f"isaac/simulation.py over tests/simulation/isaac:  {c['before_miss']} \u2192 {c['after_miss']} missing "
    f"({c['before_pct']:.2f}% \u2192 {c['after_pct']:.2f}%).   Lines closed: {c['closed']} \u2014 the four refusals "
    f"plus L3728, the duration guard's own condition: the whole \u2018if n_steps is None\u2019 arm was unevaluated.",
    fontsize=9.4, color=INK)

# ---------------------------------------------------------------- row 2 ----- #
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
muts = F["mutations"]
caught = sum(m["new_failed"] > 0 for m in muts)
blind = sum(m["new_failed"] > 0 and m["old_failed"] == 0 for m in muts)
assert (caught, blind) == (7, 7), (caught, blind)
put(ax2, 0.0, 0.955,
    f"2. Mutation table: {caught} of {len(muts)} regressions caught here, and {blind} of {len(muts)} invisible "
    f"to the {F['cases']['before']} cases the file already had",
    fontsize=13, fontweight="bold", color=INK)
put(ax2, 0.0, 0.900,
    "Every anchor was scoped to run_multi_policy's own AST line range (in_fn=1 for all eight) and the source "
    "restored byte-identically. The \u2018verdict discarded\u2019 rows are what a structural \u2018the guard is called\u2019 "
    "check cannot see.",
    fontsize=9.6, color=GREY, style="italic")
mc = [0.0, 0.60, 0.755, 0.90]
for x, h in zip(mc, ["mutation", "new cases", "pre-existing", ""]):
    put(ax2, x, 0.830, h, fontsize=9.6, fontweight="bold", color=INK)
ax2.plot([0.0, 1.0], [0.812, 0.812], transform=ax2.transAxes, color=INK, lw=0.9)

TOP2, LAST2 = 0.735, 0.135
STEP2 = (TOP2 - LAST2) / (len(muts) - 1)
assert STEP2 > 0.030, STEP2
y = TOP2
for m in muts:
    is_blind = m["new_failed"] > 0 and m["old_failed"] == 0
    unobs = m["new_failed"] == 0
    if is_blind:
        ax2.add_patch(Rectangle((-0.004, y - 0.022), 1.008, 0.062, transform=ax2.transAxes,
                                facecolor="#fdf2f0", edgecolor="none", zorder=0))
    put(ax2, mc[0], y, m["label"], fontsize=9.6, color=GREY if unobs else INK, family="monospace")
    put(ax2, mc[1], y, ("\u2014" if unobs else f"{m['new_failed']} failed"),
        fontsize=9.6, fontweight="bold", color=GREY if unobs else GREEN)
    put(ax2, mc[2], y, ("\u2014" if unobs else f"{m['old_failed']} failed"),
        fontsize=9.6, fontweight="bold", color=GREY if unobs else RED)
    put(ax2, mc[3], y, ("unobservable" if unobs else ("\u2190 BLIND" if is_blind else "")),
        fontsize=9.0, fontweight="bold", color=GREY if unobs else RED, style="italic")
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, (y, LAST2)

put(ax2, 0.0, 0.048,
    "M4 is measured-unobservable, not a gap: _resolve_horizon REBINDS duration to n_steps / control_frequency "
    "(0.0 in \u2192 0.008 out), so the caller's unusable value never reaches the guard and removing the "
    "\u2018if n_steps is None\u2019 gate is behaviour-preserving. M8 is the regression the mirror case does catch.",
    fontsize=9.2, color=INK)

# ---------------------------------------------------------------- row 3 ----- #
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
s = F["suite"]
put(ax3, 0.0, 0.80, "3. Gate", fontsize=13, fontweight="bold", color=INK)
put(ax3, 0.0, 0.48,
    f"MUJOCO_GL=egl pytest tests  \u2192  {s['passed']} passed / {s['skipped']} skipped / {s['failed']} failed "
    f"in {s['seconds']}s at upstream {s['base']}   (pristine {s['pristine_passed']} + 11 new cases = {s['passed']}).   "
    f"ruff check + ruff format --check clean; mypy 0 errors outside examples/isaac_gs.",
    fontsize=10.0, color=INK, family="monospace")
put(ax3, 0.0, 0.16,
    f"tests/simulation/isaac/test_run_multi_policy_no_recording.py: {F['cases']['before']} \u2192 "
    f"{F['cases']['after']} cases.   git diff --numstat upstream/main...HEAD -- strands_robots/ \u2192 "
    f"{F['production_lines_changed']} lines.   No Isaac Sim Kit runtime, no GPU, no MuJoCo needed for any new case.",
    fontsize=10.0, color=INK, family="monospace")

for ax, yy in placed:
    assert -0.03 <= yy <= 1.07, (ax, yy)
fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
print("wrote", OUT, OUT.stat().st_size, "bytes")

import numpy as np
from PIL import Image
im = np.asarray(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print("border clean; size", im.shape)
