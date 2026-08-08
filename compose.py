import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path("/tmp/art_main.json").read_text())    # before
B = json.loads(pathlib.Path("/tmp/art_branch.json").read_text())  # after
E = json.loads(pathlib.Path("/tmp/e2e.json").read_text())         # measured runs on main
assert A["tree"] != B["tree"], (A["tree"], B["tree"])

FIELDS = ["value_loss_coef", "entropy_coef"]
ROWS = ["nan", "inf", "-inf", "True", "'1.0'", "None", "[1.0]", "0.0", "-0.5", "1.0"]
# The documented channel: a finite real is accepted; everything else is refused
# by the read-only preflight rather than reaching the loss.
WANT = {r: ("accepted" if r in ("0.0", "-0.5", "1.0") else "refused") for r in ROWS}

cells = len(ROWS) * len(FIELDS)
div_before = sum(1 for f in FIELDS for r in ROWS if A["grid"][f][r] != WANT[r])
div_after = sum(1 for f in FIELDS for r in ROWS if B["grid"][f][r] != WANT[r])
assert (cells, div_before, div_after) == (20, 14, 0), (cells, div_before, div_after)
assert A["control"]["fingerprint"]["sum"] == B["control"]["fingerprint"]["sum"]

GREEN, RED, GREY = "#1b7f3b", "#b3261e", "#5f6368"
placed: list[tuple[object, float, str]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, "axes" if kw.get("transform") is not None else "data"))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 11.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.42, 1.06, 0.60], hspace=0.30,
                      left=0.045, right=0.972, top=0.925, bottom=0.038)
fig.suptitle("An on-policy loss weight that cannot be honored is now refused by the preflight",
             fontsize=17.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.951, "value_loss_coef and entropy_coef weight the two terms of the objective PPO's update descends "
         "-- nothing judged either, and the multiplication cannot",
         ha="center", fontsize=11.4, color="#3c4043", style="italic")

# ---------------- row 1: verdict grid
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.028, "1.  Preflight verdict, measured by calling PpoTrainer.validate() on each tree",
    fontsize=13.2, fontweight="bold", transform=ax.transAxes)
COLX = {("value_loss_coef", "before"): 0.335, ("value_loss_coef", "after"): 0.495,
        ("entropy_coef", "before"): 0.680, ("entropy_coef", "after"): 0.840}
put(ax, 0.415, 0.955, "value_loss_coef", fontsize=11.8, fontweight="bold", ha="center", transform=ax.transAxes)
put(ax, 0.760, 0.955, "entropy_coef", fontsize=11.8, fontweight="bold", ha="center", transform=ax.transAxes)
put(ax, 0.012, 0.885, "the value a caller supplies", fontsize=10.6, fontweight="bold", transform=ax.transAxes)
put(ax, 0.205, 0.885, "documented channel", fontsize=10.6, fontweight="bold", transform=ax.transAxes)
for (f, k), x in COLX.items():
    put(ax, x, 0.885, "main" if k == "before" else "this change", fontsize=10.4, fontweight="bold",
        ha="center", transform=ax.transAxes)
ax.plot([0.008, 0.985], [0.862, 0.862], color="#202124", lw=1.5, transform=ax.transAxes, clip_on=False)

step = 0.0785
for i, r in enumerate(ROWS):
    y = 0.800 - i * step
    if r in ("0.0", "-0.5", "1.0"):
        ax.add_patch(Rectangle((0.008, y - 0.024), 0.977, step * 0.86, transform=ax.transAxes,
                               facecolor="#f1f3f4", edgecolor="none", zorder=0))
    put(ax, 0.012, y, f"{r}", fontsize=11.4, family="monospace", transform=ax.transAxes)
    put(ax, 0.205, y, WANT[r], fontsize=10.8, color=GREY, style="italic", transform=ax.transAxes)
    for f in FIELDS:
        for k, src in (("before", A), ("after", B)):
            got = src["grid"][f][r]
            ok = got == WANT[r]
            put(ax, COLX[(f, k)], y, got, fontsize=11.0, family="monospace", ha="center",
                color=GREEN if ok else RED, fontweight="bold" if not ok else "normal",
                transform=ax.transAxes)
put(ax, 0.012, 0.800 - len(ROWS) * step - 0.020,
    f"divergences from the documented channel:   main {div_before} of {cells} cells      "
    f"this change {div_after} of {cells}      (shaded rows: the floor is deliberately NOT decided here -- "
    "zero disables a term and a negative weight reverses its sign, and both are configurations)",
    fontsize=10.9, fontweight="bold", transform=ax.transAxes)

# ---------------- row 2: what a real run did on main
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.040, "2.  What a real 60-step run did on main, before the preflight saw the value",
    fontsize=13.2, fontweight="bold", transform=ax2.transAxes)
hdr = [(0.012, "spec"), (0.245, "run outcome"), (0.470, "checkpoint"), (0.640, "parameter sum (16 digits)")]
for x, t in hdr:
    put(ax2, x, 0.905, t, fontsize=10.6, fontweight="bold", transform=ax2.transAxes)
ax2.plot([0.008, 0.985], [0.878, 0.878], color="#202124", lw=1.5, transform=ax2.transAxes, clip_on=False)
LEDGER = [
    ("defaults (1.0 / 0.0)", "control", GREEN),
    ("entropy_coef=True", "entropy_coef=True", RED),
    ("value_loss_coef=nan", "value_loss_coef=nan", RED),
    ("entropy_coef=nan", "entropy_coef=nan", RED),
    ("value_loss_coef=inf", "value_loss_coef=inf", RED),
    ("value_loss_coef='1.0'", "value_loss_coef='1.0'", RED),
]
s2 = 0.138
for i, (label, key, colour) in enumerate(LEDGER):
    rec = E[key]; y = 0.790 - i * s2
    put(ax2, 0.012, y, label, fontsize=11.0, family="monospace", transform=ax2.transAxes)
    status = rec.get("status", "?")
    shown = "success" if status == "success" else status.replace("RAISED ", "raised ")
    put(ax2, 0.245, y, shown, fontsize=10.8, family="monospace", color=colour,
        fontweight="bold" if status != "success" or key != "control" else "normal", transform=ax2.transAxes)
    put(ax2, 0.470, y, f"{rec.get('checkpoints', 0)} written", fontsize=10.8, family="monospace",
        color=GREY, transform=ax2.transAxes)
    fp = rec.get("fingerprint")
    put(ax2, 0.640, y, fp["sum"] if fp else "--", fontsize=10.8, family="monospace",
        color=colour if fp else GREY, transform=ax2.transAxes)
    if key != "control":
        note = ("trained with a coefficient the caller did not name (the field ships defaulting to 0.0)"
                if key == "entropy_coef=True" else
                "torch, one rollout later: names neither the field nor the value -- " + rec["message"][:58] + "...")
        put(ax2, 0.030, y - 0.062, note, fontsize=9.5, color=GREY, style="italic", transform=ax2.transAxes)

# ---------------- row 3: no regression
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 1.075, "3.  No regression: the honored run is bit-identical on both trees",
    fontsize=13.2, fontweight="bold", transform=ax3.transAxes)
lines = [
    f"main          status={A['control']['status']}   params={A['control']['fingerprint']['n']}   "
    f"nan={A['control']['fingerprint']['nan']}   absmax={A['control']['fingerprint']['absmax']}   "
    f"sum={A['control']['fingerprint']['sum']}",
    f"this change   status={B['control']['status']}   params={B['control']['fingerprint']['n']}   "
    f"nan={B['control']['fingerprint']['nan']}   absmax={B['control']['fingerprint']['absmax']}   "
    f"sum={B['control']['fingerprint']['sum']}",
    "",
    "Same seed, same env, same 60 steps: the trained parameters agree to 16 digits, so the accepted domain "
    "is untouched. No policy, simulation, rendering,",
    "recording or asset behaviour changes -- the whole diff is a read-only preflight and the domain it applies.",
]
s3 = 0.152
for i, ln in enumerate(lines):
    y = 0.845 - i * s3
    put(ax3, 0.012, y, ln, fontsize=10.4,
        family="monospace" if ln.startswith(("main", "this change")) else "sans-serif",
        color="#202124" if ln.startswith(("main", "this change")) else GREY,
        style="normal" if ln.startswith(("main", "this change")) else "italic",
        transform=ax3.transAxes)

for a, y, kind in placed:
    lo, hi = (-0.05, 1.10) if kind == "axes" else a.get_ylim()
    assert lo <= y <= hi, (y, kind, lo, hi)

out = pathlib.Path("/tmp/loss_weight_domain.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
import numpy as np
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  divergences {div_before} -> {div_after} of {cells}")
