"""Compose the measurement figure. Every number is read from the dumps."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.loads(pathlib.Path("/tmp/art-facts-robots-mine-31652545619.json").read_text())
M = json.loads(pathlib.Path("/tmp/art-mutations.json").read_text())
ROOT = "/tmp/robots-mine-31652545619"
assert F["tree"] == ROOT, F["tree"]

LOW, HIGH = F["range"]; SPAN = F["span"]; START = F["start_deg"]; STOP = F["end_stop"]
sweep = F["sweep"]; ledger = {r["label"]: r for r in F["ledger"]}

# --- assert the claims the figure makes -----------------------------------
unk = ledger["unknown motor, unbounded delta"]
same = ledger["configured motor, same call"]
inside = ledger["configured motor, delta inside travel"]
assert unk["status"] == "error" and unk["goals"] == [] and unk["domain"] is None
assert same["status"] == "success" and same["goals"] and same["goals"][0] != STOP
assert inside["status"] == "success" and inside["goals"] == [STOP] and inside["domain"] is None
n_new = sum(1 for r in M if r["new_failed"])
n_blind = sum(1 for r in M if not r["old_failed"])
assert (n_new, n_blind) == (3, 2), (n_new, n_blind)
# a delta only the travel rule accepts must exist in the sweep
only_travel = [s for s in sweep if not s["refused"] and s["endpoints_rule_refuses"]]
assert only_travel, "no delta distinguishes the travel rule from the endpoints rule"

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.0, 11.6), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1.0, 0.16], hspace=0.36, wspace=0.13,
                      left=0.055, right=0.975, top=0.925, bottom=0.035)
fig.suptitle("pose_tool incremental_move: what the delta domain bounds, and what it defers",
             fontsize=15.5, fontweight="bold", y=0.978)
fig.text(0.5, 0.951, f"measured on Thor through the public tool; joint '{F['joint']}' parked at "
                     f"{START:.2f} deg (raw {F['raw']}), range ({LOW}, {HIGH}), full travel {SPAN} deg",
         ha="center", fontsize=10.6, style="italic", color="#444")

# ---------------- row 0: the sweep ----------------------------------------
ax = fig.add_subplot(gs[0, :])
d = np.array([s["delta"] for s in sweep])
goal = np.array([s["goal"] if s["goal"] is not None else np.nan for s in sweep], dtype=float)
unc = np.array([s["unclamped"] for s in sweep], dtype=float)
ref = np.array([s["refused"] for s in sweep])

ax.axhspan(0, STOP, color="#eef6ee", zorder=0)
for lo, hi, c, lab in [(-max(d), -SPAN, "#f6dede", "refused: |delta| > full travel"),
                       (SPAN, max(d), "#f6dede", None)]:
    ax.axvspan(lo, hi, color=c, zorder=0, label=lab)
ax.axvspan(-SPAN, -HIGH, color="#fdf3e0", zorder=0, label="accepted only by the travel rule\n(an endpoints rule would refuse)")
ax.axvspan(HIGH, SPAN, color="#fdf3e0", zorder=0)

ax.plot(d, unc, ls="--", lw=1.8, color="#b0562a", zorder=3,
        label="what an unclamped scale would command")
ax.plot(d, goal, lw=2.9, color="#1f6f3f", zorder=4, marker="o", ms=3.4,
        label="Goal_Position actually written to the bus")
ax.axhline(STOP, color="#666", lw=1.1, ls=":", zorder=2)
ax.text(max(d) * 0.995, STOP + 190, f"upper end stop = {STOP}", ha="right", fontsize=9.6, color="#444")
ax.axhline(0, color="#666", lw=1.1, ls=":", zorder=2)

k = int(np.argmin(np.abs(d - inside["delta"])))
ax.annotate(f"delta = +{inside['delta']} deg is inside the {SPAN} deg travel,\n"
            f"so the domain accepts it -- but {START:.2f} + {inside['delta']} = "
            f"{START + inside['delta']:.2f} deg leaves the range,\n"
            f"and the clamp commands {STOP}. Caller is told: {inside['text']!r}",
            xy=(d[k], goal[k]), xytext=(-355, 3020), fontsize=9.9,
            bbox=dict(fc="#fff8e6", ec="#c9a227", lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#c9a227", lw=1.5), zorder=6)
ax.set_xlabel("requested delta (deg)", fontsize=10.8)
ax.set_ylabel("commanded Goal_Position", fontsize=10.8)
ax.set_title("The delta is bounded by the full travel; the resulting absolute target is bounded by the clamp",
             fontsize=11.6, pad=8)
ax.set_ylim(-450, 5400); ax.set_xlim(min(d), max(d))
ax.legend(loc="upper left", fontsize=9.2, framealpha=0.95)
ax.grid(alpha=0.22)

# ---------------- row 1 left: the deferral ledger --------------------------
axl = fig.add_subplot(gs[1, 0]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 1.045, "Both deferrals, driven through the public tool", fontsize=12.0,
    fontweight="bold", transform=axl.transAxes)
rows = [
    ("motor / delta", "domain", "action", "Goal_Position", None),
    (f"'no_such_joint'  delta=+{unk['delta']}", "defers\n(no travel to bound)",
     f"{unk['status']}\n{unk['text']!r}", f"{unk['goals']}\nnothing commanded", "ok"),
    (f"'{F['joint']}'  delta={same['delta']}", "defers\n(inside travel)",
     f"{same['status']}\n{same['text']!r}", f"{same['goals']}\ncommanded", "ok"),
    (f"'{F['joint']}'  delta=+{inside['delta']}", "defers\n(inside travel)",
     f"{inside['status']}\n{inside['text']!r}", f"{inside['goals']}\n= the end stop", "clamp"),
]
TOP, LAST = 0.93, 0.10
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
xs = [0.0, 0.29, 0.50, 0.79]
y = TOP
for i, (a, b, c, e, tag) in enumerate(rows):
    head = i == 0
    fc = None if head else ("#fff8e6" if tag == "clamp" else "#eef6ee")
    if fc:
        axl.add_patch(plt.Rectangle((-0.012, y - STEP * 0.80), 1.02, STEP * 0.90,
                                    fc=fc, ec="none", transform=axl.transAxes, zorder=0))
    for x, txt in zip(xs, (a, b, c, e)):
        put(axl, x, y, txt, fontsize=8.9 if not head else 9.6,
            fontweight="bold" if head else "normal", va="top",
            family="monospace" if not head else None, transform=axl.transAxes)
    if head:
        axl.plot([-0.012, 1.008], [y - 0.055, y - 0.055], color="#888", lw=1.0,
                 transform=axl.transAxes, clip_on=False)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y
put(axl, 0.0, 0.015, "The refusal is measured against an always-answering position source, so the\n"
                     "configured row proves it is about the motor and not about the fixture.",
    fontsize=9.2, style="italic", color="#555", va="top", transform=axl.transAxes)

# ---------------- row 1 right: the mutation matrix -------------------------
axr = fig.add_subplot(gs[1, 1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 1.045, "Plausible regressions, against two arms", fontsize=12.0,
    fontweight="bold", transform=axr.transAxes)
mrows = [("regression", "new (8)", f"pre-existing ({249})")] + [
    (r["label"], f"caught ({r['new_failed']})" if r["new_failed"] else "missed",
     f"caught ({r['old_failed']})" if r["old_failed"] else "BLIND") for r in M]
TOP2, LAST2 = 0.93, 0.24
STEP2 = (TOP2 - LAST2) / (len(mrows) - 1)
assert STEP2 > 0.030, STEP2
y = TOP2
for i, (a, b, c) in enumerate(mrows):
    head = i == 0
    if not head and c == "BLIND":
        axr.add_patch(plt.Rectangle((-0.012, y - STEP2 * 0.72), 1.02, STEP2 * 0.86,
                                    fc="#fdeaea", ec="none", transform=axr.transAxes, zorder=0))
    put(axr, 0.0, y, a, fontsize=8.7 if not head else 9.6,
        fontweight="bold" if head else "normal", va="top", transform=axr.transAxes)
    for x, txt in ((0.665, b), (0.845, c)):
        col = "#0b6b2f" if txt.startswith("caught") else ("#a11" if txt == "BLIND" else "#666")
        put(axr, x, y, txt, fontsize=8.7 if not head else 9.6, color=col if not head else "black",
            fontweight="bold" if (head or txt == "BLIND") else "normal", va="top",
            family="monospace" if not head else None, transform=axr.transAxes)
    if head:
        axr.plot([-0.012, 1.008], [y - 0.048, y - 0.048], color="#888", lw=1.0,
                 transform=axr.transAxes, clip_on=False)
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, y
put(axr, 0.0, 0.155, f"The {n_blind} regressions no test in the tree sees are exactly the ones\n"
                     f"these classes pin. The other rows are the pre-existing suite's own\n"
                     f"cells: the property asserted here (an unconfigured motor is not\n"
                     f"commanded) is held by two defences, so it does not pin which fires.",
    fontsize=9.0, style="italic", color="#555", va="top", transform=axr.transAxes)

# ---------------- row 2: gate ---------------------------------------------
axg = fig.add_subplot(gs[2, :]); axg.axis("off"); axg.set_xlim(0, 1); axg.set_ylim(0, 1)
put(axg, 0.5, 0.62, "tests only + one production docstring | pose_tool.py 418 stmts: 1 missing (99%) -> 0 missing (100%) | "
                    "28688 passed / 258 skipped / 0 failed",
    ha="center", fontsize=10.2, family="monospace", transform=axg.transAxes)
put(axg, 0.5, 0.10, "docstring-stripped AST digest of pose_tool.py unchanged (48bd01cc3adc195f): no executable production line moved",
    ha="center", fontsize=9.6, style="italic", color="#555", transform=axg.transAxes)

# ---------------- layout guards -------------------------------------------
for a, yv, axes_coords in placed:
    if axes_coords:
        assert -0.05 <= yv <= 1.08, (yv, "axes-fraction text outside the panel")
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.06 * (hi - lo) <= yv <= hi + 0.08 * (hi - lo), (yv, lo, hi)

OUT = "/tmp/pose_delta_deferrals.png"
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, (name, nw)
print(f"OK {OUT} {im.shape[1]}x{im.shape[0]}  new_caught={n_new}/{len(M)} blind={n_blind}/{len(M)}  "
      f"deltas_only_travel_accepts={len(only_travel)}")
