import json, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.load(open("/tmp/art_main.json"))
B = json.load(open("/tmp/art_branch.json"))
assert A["tree"] != B["tree"], "both arms measured the same tree"

def stats(v):
    s = sorted(v)
    return statistics.median(s), s[int(0.95 * len(s)) - 1], s[-1], sum(1 for x in s if x > 2.0), len(s)

li = stats(A["lat_idle"] + B["lat_idle"])
ll = stats(A["lat_loaded"] + B["lat_loaded"])
a_idle = [r["failed"] for r in A["idle"]];  a_load = [r["failed"] for r in A["loaded"]]
b_idle = [r["failed"] for r in B["idle"]];  b_load = [r["failed"] for r in B["loaded"]]
assert sum(a_idle) == 0 and sum(a_load) > 0, (a_idle, a_load)
assert sum(b_idle) == 0 and sum(b_load) == 0, (b_idle, b_load)
assert li[3] == 0 and ll[3] > 0, (li, ll)

BUDGET, FLOOR = 2.0, 1000.0
placed = []
def put(ax, x, y, s, axes_coords=True, **kw):
    if axes_coords:
        kw["transform"] = ax.transAxes
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.0, 11.6), dpi=125)
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.80, 0.80], hspace=0.42, wspace=0.10,
                      left=0.065, right=0.975, top=0.905, bottom=0.035)

fig.suptitle("A live ZMQ round trip inside a 2 ms budget measures the runner, not the value reaching the socket",
             fontsize=15.5, fontweight="bold", y=0.972)
fig.text(0.5, 0.938, "fresh-socket connect + ZMQ handshake vs the budget the assertion required the answer inside "
                     "(pooled, n=120 per load state)", ha="center", fontsize=10.8, style="italic", color="#444")

# ---------------------------------------------------------------- row 1: latency
ax = fig.add_subplot(gs[0, :])
idle_all = A["lat_idle"] + B["lat_idle"]
load_all = A["lat_loaded"] + B["lat_loaded"]
ax.scatter(range(len(idle_all)), idle_all, s=16, color="#2c7fb8", label="idle host", zorder=3)
ax.scatter(range(len(load_all)), load_all, s=22, color="#d94801", marker="^",
           label="under CPU contention (16 spinners, nproc=14)", zorder=4)
ax.axhline(BUDGET, color="#cb181d", ls="--", lw=2.0, zorder=2)
ax.axhline(FLOOR, color="#238b45", ls="-.", lw=2.0, zorder=2)
ax.set_yscale("log")
ax.set_ylim(0.05, 3000)
ax.set_xlabel("fresh REQ socket, sample index", fontsize=10)
ax.set_ylabel("first call: connect + handshake + round trip (ms, log)", fontsize=10)
put(ax, 0.012, 0.30, f"2 ms - the budget the old assertion required the answer inside\n"
                     f"exceeded on {ll[3]} of {ll[4]} samples under load, never on an idle host",
    color="#cb181d", fontsize=10.4, fontweight="bold", va="center")
put(ax, 0.012, 0.90, f"MIN_ROUND_TRIP_BUDGET_MS = {int(FLOOR)} ms - the floor a round trip is asserted inside now\n"
                     f"{FLOOR / ll[2]:.0f}x the worst connect cost measured under load",
    color="#1b6d3a", fontsize=10.4, fontweight="bold", va="center")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax.grid(alpha=0.28, which="both")
ax.set_title(f"idle  p50 {li[0]:.3f} / p95 {li[1]:.3f} / max {li[2]:.3f} ms      "
             f"under load  p50 {ll[0]:.3f} / p95 {ll[1]:.3f} / max {ll[2]:.3f} ms",
             fontsize=11.4, pad=8)

# ------------------------------------------------------------- row 2: verdict grid
def verdict_cell(ax, fails, title, sub):
    ok = sum(fails) == 0
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.88, facecolor="#e8f6ec" if ok else "#fdeaea",
                               edgecolor="#238b45" if ok else "#cb181d", lw=2.4))
    put(ax, 0.5, 0.80, title, ha="center", fontsize=12.4, fontweight="bold")
    put(ax, 0.5, 0.615, sub, ha="center", fontsize=10.2, color="#555")
    runs = len(fails); bad = sum(1 for f in fails if f)
    put(ax, 0.5, 0.42, ("all %d runs pass" % runs) if ok else ("%d of %d runs FAILED" % (bad, runs)),
        ha="center", fontsize=14.2, fontweight="bold", color="#1b6d3a" if ok else "#a50f15")
    put(ax, 0.5, 0.22, "failing cases per run: " + ", ".join(str(f) for f in fails),
        ha="center", fontsize=10.0, family="monospace", color="#333")

axm = fig.add_subplot(gs[1, 0]); verdict_cell(axm, a_load, "main (bdd5fb05)", "tests/test_zmq_timeout_ms_domain.py, under load")
axb = fig.add_subplot(gs[1, 1]); verdict_cell(axb, b_load, "this change", "tests/test_zmq_timeout_ms_domain.py, under load")

# ------------------------------------------------------------------ row 3: facts
axf = fig.add_subplot(gs[2, :]); axf.axis("off"); axf.set_xlim(0, 1); axf.set_ylim(0, 1)
rows = [
    ("full suite, pristine main",                 "1 failed, 27052 passed  (one-millisecond-MoveIt2InferenceClient)"),
    ("full suite, this change",                   "27065 passed, 257 skipped, 0 failed"),
    ("target file, idle",                         f"main {len(a_idle) - sum(1 for f in a_idle if f)}/{len(a_idle)} runs pass   |   this change {len(b_idle)}/{len(b_idle)} runs pass"),
    ("target file, under contention",             f"main {sum(1 for f in a_load if f)} of {len(a_load)} runs failed   |   this change 0 of {len(b_load)}"),
    ("assertions the budget reaching the socket", "getsockopt(RCVTIMEO/SNDTIMEO), for every usable spelling - no clock"),
    ("round trip asserted inside",                f"budgets >= {int(FLOOR)} ms, derived from the usable table"),
    ("tests in the file",                         "130 -> 142   (structural guard + derivation non-vacuity)"),
]
TOP, LAST = 0.90, 0.10
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
put(axf, 0.0, 1.005, "Measured", fontsize=12.6, fontweight="bold", axes_coords=True)
y = TOP
for k, v in rows:
    put(axf, 0.005, y, k, fontsize=10.9, fontweight="bold", va="center")
    put(axf, 0.365, y, v, fontsize=10.9, family="monospace", va="center", color="#222")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

for ax_, yv, axc in placed:
    if axc:
        assert -0.05 <= yv <= 1.10, (yv, "axes-fraction text outside the panel")
    else:
        lo, hi = ax_.get_ylim()
        assert lo <= yv <= hi, (yv, lo, hi)

out = "/tmp/zmq_round_trip_budget.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.array(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert nw == 0, (name, nw)
print(f"OK {out} {im.shape[1]}x{im.shape[0]}")
print(f"    latency idle p50={li[0]:.3f} max={li[2]:.3f} over2ms={li[3]}/{li[4]}")
print(f"    latency load p50={ll[0]:.3f} p95={ll[1]:.3f} max={ll[2]:.3f} over2ms={ll[3]}/{ll[4]}")
