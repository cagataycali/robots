"""Compose the artifact from the two measured dumps. Every number is asserted."""
from __future__ import annotations
import json, os, pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

rid = os.environ["GITHUB_RUN_ID"]
A = json.load(open(f"/tmp/art-base-{rid}.json"))
B = json.load(open(f"/tmp/art-pr-{rid}.json"))
assert A["tree"] != B["tree"], "both arms measured the same tree"

RAISE_A = sum(1 for v in A["survey"].values() if v.startswith("raises"))
RAISE_B = sum(1 for v in B["survey"].values() if v.startswith("raises"))
N_CLS = len(A["survey"])
BUILT_DIFF = [k for k in A["built"] if A["built"][k] != B["built"][k]]
DIG_DIFF = [k for k in A["digests"] if A["digests"][k] != B["digests"][k]]
assert (N_CLS, RAISE_A, RAISE_B) == (11, 10, 0), (N_CLS, RAISE_A, RAISE_B)
assert BUILT_DIFF == [] and DIG_DIFF == []
assert "raised in repr()" in A["pytest_render"] and "node_name" in A["pytest_render"]
assert "partially constructed" in B["pytest_render"]
assert "raised in repr()" not in B["pytest_render"]
assert "node_name" not in B["pytest_render"]
assert A["refusal"] == B["refusal"] and "invalid node_name" in A["refusal"]
assert A["sim_status"] == B["sim_status"] == "success"

fa = np.asarray(Image.open(A["frame"]).convert("RGB")).astype(int)
fb = np.asarray(Image.open(B["frame"]).convert("RGB")).astype(int)
MAXD = int(np.abs(fa - fb).max())
CHANGED = int((np.abs(fa - fb).max(2) > 8).sum())
SAT = float(((fb.max(2) - fb.min(2)) > 45).mean())
assert MAXD <= 2 and CHANGED == 0, (MAXD, CHANGED)
assert SAT > 0.5, SAT

RED, GREEN, INK, MUTE = "#b3261e", "#0f7b3f", "#1b1b1f", "#5f6368"
placed: list[tuple[plt.Axes, float, bool]] = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(17.0, 13.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 1.10, 0.88], hspace=0.20, wspace=0.075,
                      left=0.022, right=0.978, top=0.925, bottom=0.028)
fig.suptitle("A constructor refusal must not be hidden by the repr of what it refused",
             fontsize=17.5, fontweight="bold", y=0.982, color=INK)
fig.text(0.5, 0.951,
         "measured on both trees by one script  -  base 2efb05fc  vs  this change",
         ha="center", fontsize=11.2, color=MUTE, style="italic")

def textpanel(ax, title, colour, lines, *, mono_from=0):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.004, 0.004), 0.992, 0.992, transform=ax.transAxes,
                           fill=False, ec=colour, lw=2.1))
    put(ax, 0.026, 0.925, title, transform=ax.transAxes, fontsize=12.4,
        fontweight="bold", color=colour, va="top")
    top, last = 0.795, 0.075
    step = (top - last) / max(len(lines) - 1, 1)
    assert step > 0.030, step
    y = top
    for i, ln in enumerate(lines):
        mono = i >= mono_from
        put(ax, 0.026, y, ln, transform=ax.transAxes, fontsize=9.3 if mono else 10.4,
            family="monospace" if mono else None, color=INK if mono else MUTE, va="top")
        y -= step
    assert y + step >= last - 1e-9

def wrap(s, w):
    return textwrap.wrap(s, w) or [""]

# ---- row 1: what a developer actually reads ---------------------------------
ax = fig.add_subplot(gs[0, 0])
textpanel(ax, "main  -  the refusal is hidden", RED,
          ["The caller passed a bad name. The constructor refused it:", "",
           *wrap(f"ValueError: {A['refusal']}", 78), "",
           "The raising frame still holds the half-built bridge, so pytest renders:", "",
           *wrap(A["pytest_render"], 78), "",
           "-> the reader is sent after `node_name`, an attribute, not the value."],
          mono_from=2)
ax = fig.add_subplot(gs[0, 1])
textpanel(ax, "this change  -  the refusal is what you read", GREEN,
          ["The caller passed a bad name. The constructor refused it:", "",
           *wrap(f"ValueError: {B['refusal']}", 78), "",
           "The same frame still holds the same instance, and pytest renders:", "",
           *wrap(B["pytest_render"], 78), "",
           "-> the lifecycle fact, naming no attribute. The message is the finding."],
          mono_from=2)

# ---- row 2: the survey over every class defining __repr__ -------------------
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.008, 0.975,
    f"repr() on an instance whose __init__ never finished  -  every class in the package that defines __repr__ ({N_CLS})",
    transform=ax.transAxes, fontsize=12.6, fontweight="bold", color=INK, va="top")
put(ax, 0.008, 0.905, "main", transform=ax.transAxes, fontsize=11.0, fontweight="bold", color=RED, va="top")
put(ax, 0.508, 0.905, "this change", transform=ax.transAxes, fontsize=11.0, fontweight="bold", color=GREEN, va="top")
keys = sorted(A["survey"])
top, last = 0.845, 0.115
step = (top - last) / (len(keys) - 1)
assert step > 0.045, step
y = top
for k in keys:
    va, vb = A["survey"][k], B["survey"][k]
    for x0, val in ((0.008, va), (0.508, vb)):
        bad = val.startswith("raises")
        ax.add_patch(Rectangle((x0, y - 0.052), 0.484, 0.058, transform=ax.transAxes,
                               fc="#fdecea" if bad else "#e9f6ee", ec="none"))
        put(ax, x0 + 0.010, y - 0.006, k, transform=ax.transAxes, fontsize=9.6,
            fontweight="bold", color=INK, va="top")
        shown = val.removeprefix("raises: ").removeprefix("ok: ")
        put(ax, x0 + 0.148, y - 0.006, shown[:62], transform=ax.transAxes, fontsize=8.9,
            family="monospace", color=RED if bad else GREEN, va="top")
    y -= step
assert y + step >= last - 1e-9
put(ax, 0.008, 0.055,
    f"raises: {RAISE_A} of {N_CLS}   ->   {RAISE_B} of {N_CLS}."
    "  IsaacSimulation already carried the tolerance; its wording now has one owner,"
    " strands_robots.utils.partial_construction_repr.",
    transform=ax.transAxes, fontsize=10.4, color=INK, va="top")

# ---- row 3: nothing else moved ----------------------------------------------
axg = fig.add_subplot(gs[2, 0])
axg.imshow(fb.astype(np.uint8)); axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_edgecolor(GREEN); sp.set_linewidth(2.1)
axg.set_xlabel(f"headless MuJoCo render, this change  -  identical to main:"
               f" max|delta| = {MAXD}/255, {CHANGED} of {fa.shape[0] * fa.shape[1]} pixels changed",
               fontsize=10.0, color=INK, labelpad=7)
axg.set_title("the simulation is untouched", fontsize=11.6, fontweight="bold", color=INK, pad=8)

ax = fig.add_subplot(gs[2, 1])
ledger = [
    ("a fully constructed instance still names its fields",
     f"{len(A['built'])} of {len(A['built'])} reprs byte-identical across trees"),
    ("nothing outside diagnostic rendering moved",
     f"{len(A['digests'])} of {len(A['digests'])} touched modules: identical AST"),
    ("   (each module's AST, with __repr__ and its import removed)", ""),
    ("the refusal message itself is unchanged", "identical on both trees"),
    ("sim.render(...) still succeeds", f"status={B['sim_status']} on both trees"),
]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0.004, 0.004), 0.992, 0.992, transform=ax.transAxes,
                       fill=False, ec=GREEN, lw=2.1))
put(ax, 0.028, 0.935, "no regression  -  measured, not asserted", transform=ax.transAxes,
    fontsize=11.9, fontweight="bold", color=GREEN, va="top")
top, last = 0.775, 0.115
step = (top - last) / (len(ledger) - 1)
assert step > 0.045, step
y = top
for label, value in ledger:
    put(ax, 0.028, y, label, transform=ax.transAxes, fontsize=10.1, color=INK, va="top")
    if value:
        put(ax, 0.560, y, value, transform=ax.transAxes, fontsize=9.4,
            family="monospace", color=GREEN, va="top")
    y -= step
assert y + step >= last - 1e-9
put(ax, 0.028, 0.058,
    "The change adds a try/except AttributeError to each __repr__ and nothing else.",
    transform=ax.transAxes, fontsize=9.8, color=MUTE, va="top", style="italic")

for a_, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.07, (yv, "axes-fraction text outside its panel")
    else:
        lo, hi = a_.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, (yv, lo, hi)

out = pathlib.Path(f"/tmp/repr-artifact-{rid}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print(f"wrote {out}  {im.shape[1]}x{im.shape[0]}")
print(f"asserted: raises {RAISE_A}->{RAISE_B} of {N_CLS}; built diff {len(BUILT_DIFF)}; "
      f"digest diff {len(DIG_DIFF)}; frame max|delta| {MAXD}, changed {CHANGED}; saturation {SAT:.3f}")
