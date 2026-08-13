"""Compose the broker-delivery-routes figure from the measured facts."""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RUN = sys.argv[1]
F = json.load(open(f"/tmp/art-{RUN}.json"))
OUT = pathlib.Path(__file__).resolve().parent / "broker_delivery_routes.png"

GREEN, RED, GREY, BLUE = "#1a7f37", "#b02020", "#5a5a5a", "#0b5394"
MONO = {"family": "DejaVu Sans Mono"}
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.2, 12.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.00, 0.62, 1.05], hspace=0.13,
                      left=0.028, right=0.978, top=0.945, bottom=0.028)
fig.suptitle("IoT MQTT transport: the routes that decide whether a message reaches the broker",
             fontsize=15.5, fontweight="bold", y=0.984)
fig.text(0.5, 0.960, f"{F['file']}  --  {F['statements']} statements, "
         f"{F['pct_before']:.2f}% -> {F['pct_after']:.2f}% over tests/mesh   (tests only; no library line changes)",
         ha="center", fontsize=10.6, color=GREY, style="italic")

# ---- row 1: the five routes -------------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.955, "Every unexecuted line in the module, and what reaches it",
    fontsize=12.6, fontweight="bold")
cols = [0.008, 0.055, 0.315, 0.585, 0.665, 0.745]
hdr = ["line", "route", "source", "before", "after", "why it is reachable (or not)"]
put(ax, 0, 0.862, "", fontsize=1)
for x, h in zip(cols, hdr):
    put(ax, x, 0.862, h, fontsize=10.2, fontweight="bold", color=GREY)
ax.plot([0.004, 0.996], [0.836, 0.836], color=GREY, lw=0.9)
TOP, LAST = 0.775, 0.115
rows = F["routes"]
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for r in rows:
    reachable = r["kind"] == "reachable"
    band = "#eaf5ec" if r["covered_after"] and reachable else "#f1f1f6"
    ax.add_patch(Rectangle((0.004, y - 0.052), 0.992, 0.098, transform=ax.transAxes,
                           facecolor=band, edgecolor="none", zorder=0))
    put(ax, cols[0], y, str(r["line"]), fontsize=10.4, fontdict=MONO)
    put(ax, cols[1], y, r["label"], fontsize=10.4)
    put(ax, cols[2], y, r["source"][:44], fontsize=9.5, fontdict=MONO, color=BLUE)
    put(ax, cols[3], y, "no", fontsize=10.4, fontweight="bold", color=RED)
    put(ax, cols[4], y, "yes" if r["covered_after"] else "no (dead)",
        fontsize=10.4, fontweight="bold", color=GREEN if r["covered_after"] else GREY)
    put(ax, cols[5], y - 0.004, r["why"], fontsize=9.3, color=GREY, style="italic")
    if not reachable:
        put(ax, cols[5], y - 0.040, "pinned as such: the pin fails the day a reserved kind gains one",
            fontsize=8.8, color=GREY)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

# ---- row 2: the two drop mechanisms are complementary -----------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.93, "Why the DROP sentinel is not redundant with the prefix test",
    fontsize=12.6, fontweight="bold")
c2 = [0.008, 0.330, 0.470, 0.610, 0.740]
for x, h in zip(c2, ["topic", "_should_drop", "policy qos", "published?", "stopped by"]):
    put(ax2, x, 0.775, h, fontsize=10.2, fontweight="bold", color=GREY)
ax2.plot([0.004, 0.996], [0.735, 0.735], color=GREY, lw=0.9)
T2, L2 = 0.625, 0.075
S2 = (T2 - L2) / (len(F["mechanism"]) - 1)
assert S2 > 0.030, S2
y = T2
for m in F["mechanism"]:
    if not m["published"]:
        stopped = "the prefix test" if m["prefix_test"] else "the DROP sentinel  <-- only route"
        colour = GREY if m["prefix_test"] else RED
    else:
        stopped, colour = "nothing (published)", GREEN
    put(ax2, c2[0], y, m["topic"], fontsize=10.3, fontdict=MONO)
    put(ax2, c2[1], y, str(m["prefix_test"]), fontsize=10.3, fontdict=MONO)
    put(ax2, c2[2], y, str(m["policy_qos"]), fontsize=10.3, fontdict=MONO)
    put(ax2, c2[3], y, "yes" if m["published"] else "no", fontsize=10.3,
        fontweight="bold", color=GREEN if m["published"] else RED)
    put(ax2, c2[4], y, stopped, fontsize=10.0, color=colour)
    y -= S2
assert abs((y + S2) - L2) < 1e-9, (y, L2)
put(ax2, 0.008, 0.005, f"policy entries carrying the DROP sentinel: {F['drop_keys']}  --  "
    "none is reachable from a reserved top-level kind, which is why line 238 is dead",
    fontsize=9.4, color=GREY, style="italic")

# ---- row 3: mutation matrix + gate -----------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 0.955, "Mutation of each route: caught by the new module, invisible to the 470 pre-existing cases",
    fontsize=12.6, fontweight="bold")
c3 = [0.008, 0.560, 0.700, 0.845]
for x, h in zip(c3, ["regression introduced", "new module", "pre-existing", ""]):
    put(ax3, x, 0.878, h, fontsize=10.2, fontweight="bold", color=GREY)
ax3.plot([0.004, 0.996], [0.848, 0.848], color=GREY, lw=0.9)
muts = F["mutations"]
T3, L3 = 0.790, 0.190
S3 = (T3 - L3) / (len(muts) - 1)
assert S3 > 0.030, S3
y = T3
n_caught = n_blind = 0
for m in muts:
    ctrl = m["label"].startswith("(")
    caught = m["new_failed"] > 0
    blind = caught and m["old_failed"] == 0
    if not ctrl:
        n_caught += caught; n_blind += blind
    if blind:
        ax3.add_patch(Rectangle((0.004, y - 0.020), 0.992, 0.052, transform=ax3.transAxes,
                                facecolor="#eaf5ec", edgecolor="none", zorder=0))
    put(ax3, c3[0], y, m["label"][:74], fontsize=10.0, fontdict=MONO if ctrl else None,
        color=GREY if ctrl else "black")
    put(ax3, c3[1], y, f"{m['new_failed']} failed", fontsize=10.0, fontdict=MONO,
        color=(GREY if ctrl else (GREEN if caught else RED)))
    put(ax3, c3[2], y, f"{m['old_failed']} failed", fontsize=10.0, fontdict=MONO,
        color=(GREY if ctrl or m["old_failed"] == 0 else BLUE))
    if blind:
        put(ax3, c3[3], y, "<- BLIND to the suite as it stands", fontsize=9.3, color=GREEN)
    elif caught and not ctrl:
        put(ax3, c3[3], y, "<- also owned by test_iot_camera_ref_delivery.py", fontsize=9.3, color=BLUE)
    y -= S3
assert abs((y + S3) - L3) < 1e-9, (y, L3)
assert (n_caught, n_blind) == (6, 5), (n_caught, n_blind)

g = F["gate"]
foot = [
    f"caught by the new module: {n_caught}/6      invisible to the pre-existing arm: {n_blind}/6",
    f"tests/mesh: {g['subset_before_passed']} -> {g['subset_after_passed']} passed   "
    f"|   full suite: {g['pristine_passed']} + {g['new_cases']} = {g['suite_passed']} passed, "
    f"{g['suite_skipped']} skipped, {g['suite_failed']} failed",
    "ruff clean, mypy 0 non-examples errors; the AWS IoT SDK is never reached over the network",
]
TF, LF = 0.115, 0.020
SF = (TF - LF) / (len(foot) - 1)
y = TF
for i, line in enumerate(foot):
    put(ax3, 0.008, y, line, fontsize=9.8, fontdict=MONO,
        fontweight="bold" if i == 0 else None, color="black" if i == 0 else GREY)
    y -= SF

for a, yy, is_axes in placed:
    lo, hi = a.get_ylim()
    assert (lo - 0.05) <= yy <= (hi + 0.07), (yy, lo, hi)

fig.savefig(OUT, dpi=124, facecolor="white", bbox_inches="tight", pad_inches=0.30)
print("wrote", OUT)
import numpy as np
from PIL import Image
im = np.asarray(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("size:", im.shape, "border clean")
