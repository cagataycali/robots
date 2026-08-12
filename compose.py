"""Compose the measured artifact: outcome matrix + mutation matrix + gate."""
import json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RID = sys.argv[1]
A = json.loads(pathlib.Path(f"/tmp/art-main-{RID}.json").read_text())     # upstream/main
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RID}.json").read_text())   # this PR
assert A["tree"] != B["tree"], (A["tree"], B["tree"])
assert A["budget_is_a_knob"] is False and B["budget_is_a_knob"] is True

CASES = ["bring-up completes", "bring-up fails", "budget expires"]
# every claim rendered below, asserted against the dumps
assert A["rows"]["budget expires"]["caller"] == "returned None"
assert A["rows"]["budget expires"]["elapsed_s"] == 30.0
assert A["rows"]["budget expires"]["reported"] is False
assert A["rows"]["budget expires"]["runtime_stored"] == "None"
assert B["rows"]["budget expires"]["caller"] == "raised TimeoutError"
assert B["rows"]["budget expires"]["reported"] is True
for unchanged in ("bring-up completes", "bring-up fails"):
    for k in ("caller", "reported", "runtime_stored"):
        assert A["rows"][unchanged][k] == B["rows"][unchanged][k], (unchanged, k)
assert all(r["operator_stdout"].startswith("arm-1 is online") for r in A["rows"].values())

MUTS = [
    ("M1  timeout arm deleted (returns None again)", 4, 0),
    ("M2  timeout checked before the recorded error", 0, 0),
    ("M3  budget re-hardcoded at the wait site", 1, 0),
    ("M4  recorded exception re-wrapped, not re-raised", 1, 0),
    ("M5  budget omitted from the refusal message", 1, 0),
    ("     (unmutated control)", 0, 0),
]
caught = sum(1 for label, n, _o in MUTS if not label.strip().startswith("(") and n > 0)
blind = sum(1 for label, n, o in MUTS if not label.strip().startswith("(") and n > 0 and o == 0)
assert (caught, blind) == (4, 4), (caught, blind)

RED, GREEN, INK, MUTED = "#b3261e", "#1b6e3c", "#202124", "#5f6368"
placed: list[tuple] = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.pop("transform", None) is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 11.3), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.42, 0.86, 0.40], hspace=0.14,
                      left=0.028, right=0.976, top=0.925, bottom=0.032)
fig.suptitle(
    "init_device_connect_sync: an expired bring-up budget was indistinguishable from success",
    fontsize=16.5, fontweight="bold", y=0.975, color=INK)
fig.text(0.5, 0.947,
         "Measured end to end through the shipped foreground runner. No policy, simulation, rendering, "
         "recording or asset behaviour changes, so the artifact is the measurement.",
         ha="center", fontsize=10.4, color=MUTED, style="italic")

# ---- row 1: the outcome matrix -------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.010, "What the caller receives, and what the operator is told",
    fontsize=13.2, fontweight="bold", color=INK, transform=ax.transAxes)
COLS = [0.0, 0.155, 0.315, 0.560, 0.760]
for x, h in zip(COLS, ["bring-up outcome", "", "upstream/main", "this pull request", "operator's only stdout line"]):
    put(ax, x, 0.905, h, fontsize=11.0, fontweight="bold", color=INK, transform=ax.transAxes)
ax.plot([0, 1], [0.878, 0.878], color="#c8ccd0", lw=1.0, transform=ax.transAxes)

TOP, LAST = 0.790, 0.130
step = (TOP - LAST) / (len(CASES) - 1)
assert step > 0.030, step
y = TOP
for case in CASES:
    a, b = A["rows"][case], B["rows"][case]
    changed = a["caller"] != b["caller"]
    if changed:
        ax.add_patch(plt.Rectangle((-0.006, y - 0.115), 1.012, 0.222, transform=ax.transAxes,
                                   facecolor="#fdecea", edgecolor="#f3b8b2", lw=1.0, zorder=0))
    put(ax, COLS[0], y + 0.048, case, fontsize=11.6, fontweight="bold", color=INK, transform=ax.transAxes)
    put(ax, COLS[0], y - 0.006, f"budget {a['budget_s']:g}s / {b['budget_s']:g}s",
        fontsize=9.2, color=MUTED, transform=ax.transAxes)
    put(ax, COLS[1], y + 0.048, "caller gets", fontsize=9.6, color=MUTED, transform=ax.transAxes)
    put(ax, COLS[1], y - 0.006, "logged", fontsize=9.6, color=MUTED, transform=ax.transAxes)
    put(ax, COLS[1], y - 0.060, "stored", fontsize=9.6, color=MUTED, transform=ax.transAxes)
    for col, side in ((COLS[2], a), (COLS[3], b)):
        ok = side["reported"] or side["caller"].startswith("returned ") and case == "bring-up completes"
        put(ax, col, y + 0.048, f"{side['caller']}  ({side['elapsed_s']:.2f}s)", fontsize=11.0,
            fontweight="bold", color=INK if ok else RED, transform=ax.transAxes)
        put(ax, col, y - 0.006,
            "reported" if side["reported"] else ("(nothing logged)" if case != "bring-up completes" else "-"),
            fontsize=10.2, color=(GREEN if side["reported"] else (RED if case != "bring-up completes" else MUTED)),
            transform=ax.transAxes)
        put(ax, col, y - 0.060, side["runtime_stored"], fontsize=9.6,
            color=RED if side["runtime_stored"] == "None" else MUTED, transform=ax.transAxes)
    put(ax, COLS[4], y + 0.048, f'"{A["rows"][case]["operator_stdout"]}"', fontsize=9.8,
        family="monospace", color=INK, transform=ax.transAxes)
    if case == "budget expires":
        put(ax, COLS[4], y - 0.006, "<- printed for a runtime that never came up",
            fontsize=9.2, color=RED, style="italic", transform=ax.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(ax, 0.0, 0.022,
    "ready.wait(timeout=...) returns False on expiry. That boolean was discarded, so the expired budget fell "
    "through to `return runtime_holder[0]` - None, past a declared -> \"DeviceRuntime\".",
    fontsize=10.0, color=INK, transform=ax.transAxes)

# ---- row 2: the mutation matrix ------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.020, "Mutation table: 5 plausible regressions x 2 arms",
    fontsize=13.2, fontweight="bold", color=INK, transform=ax2.transAxes)
for x, h in zip([0.0, 0.520, 0.660, 0.830],
                ["mutation", "new module (12)", "pre-existing (1291)", ""]):
    put(ax2, x, 0.885, h, fontsize=11.0, fontweight="bold", color=INK, transform=ax2.transAxes)
ax2.plot([0, 1], [0.845, 0.845], color="#c8ccd0", lw=1.0, transform=ax2.transAxes)
TOP2, LAST2 = 0.735, 0.115
step2 = (TOP2 - LAST2) / (len(MUTS) - 1)
assert step2 > 0.030, step2
y = TOP2
for label, nf, of in MUTS:
    control = label.strip().startswith("(")
    put(ax2, 0.0, y, label, fontsize=10.8, color=MUTED if control else INK,
        style="italic" if control else "normal", transform=ax2.transAxes)
    put(ax2, 0.520, y, f"{nf} failed", fontsize=10.8, fontweight="bold" if nf else "normal",
        color=GREEN if nf else MUTED, transform=ax2.transAxes)
    put(ax2, 0.660, y, f"{of} failed", fontsize=10.8, color=RED if (nf and not of) else MUTED,
        transform=ax2.transAxes)
    if nf and not of:
        put(ax2, 0.830, y, "<- BLIND before this PR", fontsize=10.0, color=RED, transform=ax2.transAxes)
    if label.startswith("M2"):
        put(ax2, 0.830, y, "unobservable: the event is set in a finally", fontsize=9.6,
            color=MUTED, style="italic", transform=ax2.transAxes)
    y -= step2
assert abs((y + step2) - LAST2) < 1e-9, (y, LAST2)
put(ax2, 0.0, 0.022,
    f"{caught} of 5 caught by the new module, {blind} of those invisible to the 1291 pre-existing tests. "
    "M2's two arms are mutually exclusive by construction, so their order cannot be observed.",
    fontsize=10.0, color=INK, transform=ax2.transAxes)

# ---- row 3: gate ---------------------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(plt.Rectangle((0, 0.02), 1, 0.96, transform=ax3.transAxes,
                            facecolor="#f6f8fa", edgecolor="#d6dade", lw=1.0))
GATE = [
    "Pre-fix (source reverted, tests kept): 3 failed / 5 passed in 90.51s - the 90s is 3 x the hardcoded 30s budget. "
    "Post-fix: 12 passed in 0.71s.",
    "Failures were behavioural: 'DID NOT RAISE TimeoutError' twice, and an empty operator log with "
    "'arm-1 is online. Ctrl+C to stop.' as the only captured line.",
    "device_connect/__init__.py coverage over its own suite: 88.1% -> 97.6% (10 -> 2 missing); the module's three "
    "uncovered lines were exactly this contract's failure arms.",
]
TOP3, LAST3 = 0.760, 0.180
step3 = (TOP3 - LAST3) / (len(GATE) - 1)
assert step3 > 0.030, step3
y = TOP3
for line in GATE:
    put(ax3, 0.014, y, line, fontsize=10.0, color=INK, transform=ax3.transAxes)
    y -= step3
assert abs((y + step3) - LAST3) < 1e-9, (y, LAST3)
put(ax3, 0.014, 0.075,
    "Gate: ruff check + ruff format --check clean (1186 files); mypy 0 errors outside examples/isaac_gs "
    "(14 there, byte-identical to the pristine base).",
    fontsize=9.6, color=MUTED, style="italic", transform=ax3.transAxes)

# layout self-audit
for ax_, y_, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y_ <= 1.07, (y_, "axes")
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.05 <= y_ <= hi + 0.07, (y_, lo, hi)

out = pathlib.Path(f"/tmp/artifact-{RID}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(2) > 12).sum())
    assert nonwhite == 0, (name, nonwhite)
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  caught={caught}/5 blind={blind}  border clean")
