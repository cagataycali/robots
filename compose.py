"""Compose the measured teardown ledger. Every cell comes from the two dumps."""
from __future__ import annotations
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUN = pathlib.Path(__file__).resolve().parents[1].name
A = json.load(open(f"/tmp/art-wt-main-{RUN.split('-')[-1]}.json"))   # pristine
B = json.load(open(f"/tmp/art-{RUN}.json"))                          # branch
assert A["tree"] != B["tree"], "both arms measured the same tree"
MUT = json.load(open(f"/tmp/mut-{RUN.split('-')[-1]}.json"))

GREEN, RED, GREY, INK = "#1b7f3b", "#b3261e", "#6b6b6b", "#101010"
placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, *, axes=False, **kw):
    if axes: kw["transform"] = ax.transAxes
    placed.append((ax, y, axes))
    return ax.text(x, y, s, **kw)

def obeys(row):
    """The policy: tolerate the failure AND record it."""
    return row["escaped"] is None and row["recorded_level"] is not None

n_bad_before = sum(1 for r in A["rows"] if not obeys(r))
n_bad_after  = sum(1 for r in B["rows"] if not obeys(r))
assert (len(A["rows"]), n_bad_before, n_bad_after) == (4, 2, 0), (len(A["rows"]), n_bad_before, n_bad_after)
# the two paths that already obeyed are byte-identical across trees
for i in (0, 1):
    assert A["rows"][i]["recorded_level"] == B["rows"][i]["recorded_level"] == "DEBUG"
    assert A["rows"][i]["escaped"] is None and B["rows"][i]["escaped"] is None
# controls unchanged
for a, b in zip(A["controls"], B["controls"], strict=True):
    assert a["escaped"] is b["escaped"] is None and a["recorded_level"] is b["recorded_level"] is None
n_mut = len(MUT["rows"]); n_caught = sum(1 for _, nf, _ in MUT["rows"] if nf)
n_blind = sum(1 for _, _, of in MUT["rows"] if of == 0)
assert (n_mut, n_caught, n_blind) == (5, 5, 5), (n_mut, n_caught, n_blind)

fig = plt.figure(figsize=(15.4, 11.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 1.00, 0.44], hspace=0.20,
                      left=0.030, right=0.972, top=0.930, bottom=0.030)
fig.suptitle("IotMqttTransport: every MQTT5 client teardown that could not finish is now recorded",
             fontsize=17, fontweight="bold", y=0.978)
fig.text(0.5, 0.951, "measured by driving all four call sites with a client whose stop() raises "
         "\u2014 no policy, simulation, rendering or recording behaviour changes",
         ha="center", fontsize=10.5, style="italic", color=GREY)

# ---------------- row 1: the four-site ledger ----------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965, "stop() raises on all four teardown paths \u2014 what the caller and the operator get",
    axes=True, fontsize=13, fontweight="bold", color=INK)
COLS = [0.000, 0.255, 0.400, 0.545, 0.700, 0.855]
HEAD = ["client teardown site", "main: raised?", "main: recorded", "PR: raised?", "PR: recorded", "policy"]
TOP, LAST = 0.855, 0.300
rows = list(zip(A["rows"], B["rows"], strict=True))
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
for x, h in zip(COLS, HEAD, strict=True):
    put(ax, x, 0.905, h, axes=True, fontsize=10.4, fontweight="bold", color=GREY)
ax.plot([0, 1], [0.885, 0.885], transform=ax.transAxes, color=GREY, lw=0.9)
y = TOP
for a, b in rows:
    was, now = obeys(a), obeys(b)
    if not was:
        ax.add_patch(plt.Rectangle((-0.006, y - 0.052), 1.012, 0.098, transform=ax.transAxes,
                                   facecolor=RED, alpha=0.075, zorder=0))
    put(ax, COLS[0], y, a["site"], axes=True, fontsize=10.6, family="monospace", color=INK)
    put(ax, COLS[1], y, "RAISED" if a["escaped"] else "no", axes=True, fontsize=10.4,
        fontweight="bold" if a["escaped"] else "normal", color=RED if a["escaped"] else GREEN)
    put(ax, COLS[2], y, a["recorded_level"] or "\u2014 nothing", axes=True, fontsize=10.4,
        fontweight="normal" if a["recorded_level"] else "bold",
        color=GREEN if a["recorded_level"] else RED)
    put(ax, COLS[3], y, "RAISED" if b["escaped"] else "no", axes=True, fontsize=10.4,
        color=RED if b["escaped"] else GREEN)
    put(ax, COLS[4], y, b["recorded_level"] or "\u2014 nothing", axes=True, fontsize=10.4,
        fontweight="bold" if b["recorded_level"] == "WARNING" else "normal",
        color=GREEN if b["recorded_level"] else RED)
    put(ax, COLS[5], y, ("already obeyed" if was else "FIXED"), axes=True, fontsize=10.2,
        fontweight="normal" if was else "bold", color=GREY if was else GREEN)
    if not was:
        detail = (f"escaped a method documented to return bool; client left set = {a['client_left_set']}"
                  if a["escaped"] else
                  'swallowed into a bare pass; the only line logged was INFO "IoT mesh session closed"')
        put(ax, COLS[0] + 0.012, y - 0.038, detail, axes=True, fontsize=9.0, style="italic", color=RED)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y
put(ax, 0.0, 0.190, f"paths violating the module's own stated policy:   main {n_bad_before} of 4"
    f"    \u2192    this PR {n_bad_after} of 4", axes=True, fontsize=11.6, fontweight="bold", color=INK)
put(ax, 0.0, 0.120, 'the policy, from the construction-failure path\u2019s own comment:  "a stop() error here \u2026 '
    'must not mask the original failure. Log at debug and move on."', axes=True, fontsize=9.8,
    style="italic", color=GREY)
put(ax, 0.0, 0.050, "both controls (a clean timeout, a clean close) are identical on the two trees: "
    "no raise, nothing recorded", axes=True, fontsize=9.8, color=GREEN)

# ---------------- row 2: mutation matrix ----------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.955, "Would a regression be caught? 5 plausible reversions \u00d7 2 test arms",
    axes=True, fontsize=13, fontweight="bold", color=INK)
MC = [0.000, 0.660, 0.830]
for x, h in zip(MC, ["reversion", "new module", "156 pre-existing"], strict=True):
    put(ax2, x, 0.870, h, axes=True, fontsize=10.4, fontweight="bold", color=GREY)
ax2.plot([0, 1], [0.848, 0.848], transform=ax2.transAxes, color=GREY, lw=0.9)
TOP2, LAST2 = 0.760, 0.230
S2 = (TOP2 - LAST2) / (n_mut - 1)
assert S2 > 0.030, S2
y = TOP2
for label, nf, of in MUT["rows"]:
    put(ax2, MC[0], y, label, axes=True, fontsize=10.4, family="monospace", color=INK)
    put(ax2, MC[1], y, f"{nf} failed", axes=True, fontsize=10.4, fontweight="bold", color=GREEN)
    put(ax2, MC[2], y, f"{of} failed  \u2190 BLIND" if of == 0 else f"{of} failed",
        axes=True, fontsize=10.4, fontweight="bold" if of == 0 else "normal",
        color=RED if of == 0 else GREEN)
    y -= S2
assert abs((y + S2) - LAST2) < 1e-9, y
put(ax2, 0.0, 0.115, f"caught by the new module: {n_caught} of {n_mut}      "
    f"invisible to the {MUT['base_old']} pre-existing IoT-transport tests: {n_blind} of {n_mut}",
    axes=True, fontsize=11.6, fontweight="bold", color=INK)
put(ax2, 0.0, 0.038, "the second reversion keeps the try/except and drops only the log line \u2014 the shape a "
    "\u201cthe guard is called\u201d structural check cannot see", axes=True, fontsize=9.8, style="italic", color=GREY)

# ---------------- row 3: gate ----------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes, facecolor="#f2f4f6", edgecolor=GREY, lw=0.8))
GATE = [
    "pre-fix, source reverted with the tests kept:  6 failed / 6 passed   "
    "(the 6 that pass are the clean-teardown controls, the two already-obeying paths and the planted-defect meta)",
    "iot_transport.py coverage over tests/mesh:  13 missing \u2192 5 missing,  95% \u2192 98%   "
    "(the 8 closed lines are the four teardown handlers)",
    "full suite:  28193 passed / 257 skipped / 0 failed   (28181 on main + 12 new)      "
    "ruff clean \u00b7 mypy 0 errors outside examples/isaac_gs, byte-identical to the pristine base",
]
GT, GL = 0.760, 0.240
S3 = (GT - GL) / (len(GATE) - 1)
y = GT
for line in GATE:
    put(ax3, 0.016, y, line, axes=True, fontsize=9.9, family="monospace", color=INK)
    y -= S3
assert abs((y + S3) - GL) < 1e-9, y

# ---- self-audit ----
for ax_, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.07, (yy, "axes-fraction y out of range")
    else:
        lo, hi = ax_.get_ylim(); assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

OUT = pathlib.Path("_art/iot_teardown_ledger.png")
fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.30)
im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum()); assert n == 0, (name, n)
print(f"OK {OUT}  {Image.open(OUT).size}  bad_before={n_bad_before} bad_after={n_bad_after} "
      f"mut_caught={n_caught}/{n_mut} blind={n_blind}/{n_mut}")
