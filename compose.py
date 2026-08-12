import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RID = os.environ["GITHUB_RUN_ID"]
MUT = json.load(open(f"/tmp/mut-{RID}.json"))["rows"]
cov_b = json.load(open(f"/tmp/c-before-{RID}.json"))["files"]["strands_robots/hardware_robot.py"]["missing_lines"]
cov_a = json.load(open(f"/tmp/c-after-{RID}.json"))["files"]["strands_robots/hardware_robot.py"]["missing_lines"]

# --- measured facts, asserted ------------------------------------------------
assert 2731 in cov_b and 2731 not in cov_a, "L2731 must go from missing to covered"
assert 2660 not in cov_b and 2660 not in cov_a, "L2660 (publish) was already covered in both arms"
by = {r["label"].split()[0]: r for r in MUT}
assert len(by) == 5, by.keys()
caught_new = sum(1 for r in MUT if r["new"][0] > 0)
blind_old = sum(1 for r in MUT if r["old"][0] == 0)
assert (caught_new, blind_old) == (4, 5), (caught_new, blind_old)
assert by["M3"]["new"][0] == 0 and by["M3"]["old"][0] == 0, "M3 is invisible to both arms"

GREEN, RED, AMBER, INK = "#1F7A3D", "#B3261E", "#8A6A00", "#1A3B5C"
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 11.5), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.00, 1.02, 0.40], hspace=0.20,
                      left=0.035, right=0.978, top=0.935, bottom=0.030)
fig.suptitle("The teleop stream-replacement contract: one hole in a four-cell matrix",
             fontsize=15.5, fontweight="bold", color=INK, y=0.982)
fig.text(0.5, 0.955, "Both teleop registration surfaces stop the stream already registered under a key before "
                     "installing its replacement. Tests only; no production line changes.",
         ha="center", fontsize=9.6, color="#444", style="italic")

# ---------------- row 1: the 2x2 matrix, before | after ---------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.055, "Which cells of the contract are pinned", fontsize=12.4,
    fontweight="bold", color=INK, transform=ax.transAxes)

CELLS = {
    ("publish", "refused"): ("PINNED", "test_rejected_publish_leaves_a_live_stream_running", "PINNED"),
    ("publish", "accepted"): ("PINNED", "test_a_usable_rate_replaces_the_live_publisher  (L2660)", "PINNED"),
    ("receive", "refused"): ("PINNED", "test_rejected_receive_leaves_a_live_stream_running", "PINNED"),
    ("receive", "accepted"): ("NOT PINNED", "L2731 never executed", "PINNED"),
}
for pi, (panel, title) in enumerate([("before", "main"), ("after", "this change")]):
    x0 = 0.02 + pi * 0.505
    put(ax, x0, 0.965, f"{'ON MAIN' if panel=='before' else 'WITH THIS CHANGE'}   ({title})",
        fontsize=10.6, fontweight="bold", color=INK)
    for ci, col in enumerate(["refused -> live stream survives", "accepted -> live stream replaced"]):
        put(ax, x0 + 0.145 + ci * 0.170, 0.905, col, fontsize=8.1, color="#333", ha="left")
    for ri, surface in enumerate(["publish", "receive"]):
        y = 0.780 - ri * 0.300
        put(ax, x0, y + 0.075, f"start_teleop_{surface}", fontsize=9.5,
            fontweight="bold", color=INK, family="monospace")
        for ci, kind in enumerate(["refused", "accepted"]):
            state = CELLS[(surface, kind)][0 if panel == "before" else 2]
            detail = CELLS[(surface, kind)][1]
            ok = state == "PINNED"
            cx = x0 + 0.145 + ci * 0.170
            ax.add_patch(Rectangle((cx, y - 0.135), 0.162, 0.195, transform=ax.transAxes,
                                   facecolor=(GREEN if ok else RED), alpha=0.13,
                                   edgecolor=(GREEN if ok else RED), lw=1.5))
            put(ax, cx + 0.008, y + 0.040, ("PINNED" if ok else "NOT PINNED"),
                fontsize=9.6, fontweight="bold", color=(GREEN if ok else RED))
            body = detail if (panel == "before" or ok) else detail
            if panel == "after" and (surface, kind) == ("receive", "accepted"):
                body = "4 cases drive the accepted path\n(L2731 now covered)"
            put(ax, cx + 0.008, y - 0.005, body, fontsize=6.9, color="#333",
                family="monospace", wrap=True)
put(ax, 0.02, 0.130,
    'The receive refusal cell asserts a live receiver is NOT stopped. That is equally true of a body that tears\n'
    'nothing down, so on its own it ruled nothing out - the accepted cell is what gives it content.',
    fontsize=8.6, color=AMBER, style="italic")

# ---------------- row 2: the mutation matrix -------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.055, "Plausible regressions in start_teleop_receive, against two test arms",
    fontsize=12.4, fontweight="bold", color=INK, transform=ax2.transAxes)
hdr = [(0.015, "regression introduced into the production body"),
       (0.560, "the 4 new cases"), (0.720, "the 175 pre-existing cases"), (0.900, "verdict")]
for x, h in hdr:
    put(ax2, x, 0.955, h, fontsize=8.4, fontweight="bold", color="#333")
TOP, LAST = 0.850, 0.230
step = (TOP - LAST) / (len(MUT) - 1)
assert step > 0.030, step
LABELS = {
 "M1": "M1  delete the teardown entirely",
 "M2": "M2  keep the key lookup, drop the .stop() call",
 "M3": "M3  tear down before validating the identifiers",
 "M4": "M4  key the registry on device_name alone",
 "M5": "M5  never register the replacement in the slot",
}
y = TOP
for r in MUT:
    m = r["label"].split()[0]
    nf, of = r["new"][0], r["old"][0]
    put(ax2, 0.015, y, LABELS[m], fontsize=9.0, family="monospace", color=INK)
    put(ax2, 0.560, y, f"{nf} failed" if nf else "0 failed", fontsize=9.0,
        family="monospace", fontweight="bold", color=(GREEN if nf else AMBER))
    put(ax2, 0.720, y, f"{of} failed" if of else "0 failed  <- BLIND", fontsize=9.0,
        family="monospace", fontweight="bold", color=(GREEN if of else RED))
    put(ax2, 0.900, y, "caught" if nf else "not observable", fontsize=8.6,
        color=(GREEN if nf else AMBER))
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(ax2, 0.015, 0.150,
    f"Caught by the new cases: {caught_new} of {len(MUT)}.   Invisible to the pre-existing cases: {blind_old} of {len(MUT)}.\n"
    "M3 is not observable on this surface and cannot be made so: both refusable arguments are part of the registry key, so a\n"
    "refused value can never name a registered entry. Ordering IS observable on publish - hz is refused while device_name still\n"
    "names the live stream - and is pinned there with the rate guard. Recorded rather than papered over.",
    fontsize=8.5, color="#333")

# ---------------- row 3: gate ---------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(Rectangle((0.008, 0.06), 0.984, 0.88, transform=ax3.transAxes,
                        facecolor="#F4F7FA", edgecolor="#CBD6E2", lw=1.0))
lines = [
 f"coverage   strands_robots/hardware_robot.py  L2731 (receive replacement)  missing -> covered      "
 f"[same file list, before-arm deselects the new class: {len(cov_b)} -> {len(cov_a)} missing]",
 "           L2660 (publish replacement) already covered in both arms - the matrix confirmation, not a change",
 "gate       28539 passed / 257 skipped / 0 failed (622s, MUJOCO_GL=egl)   base 28535 + 4 new cases",
 "           ruff check + ruff format --check clean (1185 files); mypy 0 errors outside examples/isaac_gs",
 "           git diff --numstat upstream/main...HEAD -- strands_robots/  ->  0 lines",
]
GTOP, GLAST = 0.845, 0.150
gstep = (GTOP - GLAST) / (len(lines) - 1)
assert gstep > 0.030, gstep
assert GLAST > 0.05, GLAST
gy = GTOP
for i, s in enumerate(lines):
    put(ax3, 0.022, gy, s, fontsize=8.2, family="monospace",
        color=(INK if i in (0, 2) else "#444"))
    gy -= gstep
assert abs((gy + gstep) - GLAST) < 1e-9, (gy, GLAST)

for a, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.10, (yy, "axes-fraction out of band")
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.10, (yy, lo, hi)

out = pathlib.Path("_art/teleop_receive_replacement.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  borders clean")
print(f"    asserted: caught_new={caught_new} blind_old={blind_old}  L2731 {2731 in cov_b}->{2731 in cov_a}")
