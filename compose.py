from __future__ import annotations
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = pathlib.Path("_art")
f = json.loads((A / "facts.json").read_text())
home = np.load(A / "home.npy"); honored = np.load(A / "honored.npy"); refused = np.load(A / "refused.npy")
rel = f["relations"]
sp, mp = f["start_policy"], f["run_multi_policy"]

# ---- claims re-derived here, so the figure cannot ship a stale cell ---------
assert rel["honored_moved_frac"] > 0.10
assert rel["refused_changed_pixels_over_threshold"] == 0
assert rel["joints_identical_across_the_refusals"] is True
assert sp["status"] == "error" and mp["status"] == "error"
assert sp["threads"] == 0 and mp["threads"] == 0 and sp["policy_running"] is False
assert sp["envelope_verbatim"] and mp["envelope_verbatim"]
assert sp["frames_on_disk"] == 0 and mp["frames_on_disk"] == 0
assert "1.667x" in sp["text"] and "declares 30 fps" in sp["text"]

placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

MUTATIONS = [
    ("start_policy", "drop the guard entirely", "5 failed / 10 passed", "1 failed / 78 passed"),
    ("start_policy", "keep the call, discard the refusal", "5 failed / 10 passed", "79 passed  <- invisible"),
    ("run_multi_policy", "drop the guard entirely", "4 failed / 11 passed", "1 failed / 78 passed"),
    ("run_multi_policy", "keep the call, discard the refusal", "4 failed / 11 passed", "79 passed  <- invisible"),
]
LEDGER = [
    ("entry point", "status", "worker", "robot marked running", "recorder frames", "rows on disk", "envelope"),
    ("start_policy (async)", sp["status"], f'{sp["threads"]} submitted', str(sp["policy_running"]),
     str(sp["recorder_frames"]), str(sp["frames_on_disk"]), "shared helper verbatim"),
    ("run_multi_policy", mp["status"], f'{mp["threads"]} submitted', str(mp["policy_running"]),
     str(mp["recorder_frames"]), str(mp["frames_on_disk"]), "shared helper verbatim"),
]

fig = plt.figure(figsize=(16.4, 12.2), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[3.05, 1.30, 1.05], hspace=0.30, wspace=0.06,
                      left=0.022, right=0.978, top=0.925, bottom=0.028)
fig.suptitle(
    "A rollout must not capture into a dataset whose declared rate it disagrees with -\n"
    "the two entry points that exist only on the MuJoCo backend, pinned behaviourally",
    fontsize=15.5, fontweight="bold", y=0.982,
)

panels = [
    (home, "1. at rest", "the scene before anything is driven", "#444444"),
    (honored, "2. honored rollout (no recording open)",
     f'start_policy -> success, worker runs\n{rel["honored_moved_frac"]:.1%} of pixels differ from (1)', "#1a7f37"),
    (refused, "3. after BOTH refusals (30 fps recording open, rollout at 50 Hz)",
     f'0 of {rel["total_pixels"]:,} pixels changed beyond renderer noise\njoint state identical to (1)', "#b3261e"),
]
for col, (img, title, sub, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_edgecolor(colour); sp_.set_linewidth(2.4)
    ax.set_title(title, fontsize=11.4, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(sub, fontsize=10.2, color="#333333", labelpad=7)

# ---- ledger ---------------------------------------------------------------
axl = fig.add_subplot(gs[1, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 1.055, "Measured on both entry points: the refusal is returned synchronously and costs nothing",
    transform=axl.transAxes, fontsize=12.6, fontweight="bold")
COLS = [0.005, 0.185, 0.265, 0.365, 0.545, 0.675, 0.785]
TOP, LAST = 0.86, 0.50
rows = LEDGER
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.045, step
for i, row in enumerate(rows):
    y = TOP - i * step
    head = i == 0
    for x, cell in zip(COLS, row, strict=True):
        put(axl, x, y, cell, transform=axl.transAxes, fontsize=10.5,
            family="monospace" if not head else "sans-serif",
            fontweight="bold" if head else "normal",
            color="#111111" if head else ("#b3261e" if cell == "error" else "#333333"))
    if head:
        axl.plot([0.0, 0.995], [y - step * 0.42] * 2, color="#bbbbbb", lw=0.9, transform=axl.transAxes)
msg = sp["text"].split(". Align")[0]
put(axl, 0.005, 0.30, "refusal text (start_policy; run_multi_policy differs only in the method name):",
    transform=axl.transAxes, fontsize=10.4, fontweight="bold", color="#444444")
wrapped = msg[:150] + "\n" + msg[150:300] + "\n" + msg[300:]
put(axl, 0.005, 0.055, wrapped, transform=axl.transAxes, fontsize=9.3, family="monospace", color="#555555")

# ---- mutation table -------------------------------------------------------
axm = fig.add_subplot(gs[2, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 1.02,
    "Why a structural pin was not enough: it asserts the guard is CALLED, not that its refusal is RETURNED",
    transform=axm.transAxes, fontsize=12.6, fontweight="bold")
MCOLS = [0.005, 0.175, 0.520, 0.720]
mrows = [("entry point", "mutation applied to the guard", "the 15 new cases", "the 79 pre-existing cases")] + [
    (m, lab, new, old) for m, lab, new, old in MUTATIONS
]
MTOP, MLAST = 0.80, 0.10
mstep = (MTOP - MLAST) / (len(mrows) - 1)
assert mstep > 0.045, mstep
for i, row in enumerate(mrows):
    y = MTOP - i * mstep
    head = i == 0
    for x, cell in zip(MCOLS, row, strict=True):
        colour = "#111111"
        if not head and "invisible" in cell:
            colour = "#b3261e"
        elif not head and cell.startswith(("5 failed", "4 failed")):
            colour = "#1a7f37"
        put(axm, x, y, cell, transform=axm.transAxes, fontsize=10.4,
            family="monospace" if not head else "sans-serif",
            fontweight="bold" if (head or "invisible" in cell) else "normal", color=colour)
    if head:
        axm.plot([0.0, 0.995], [y - mstep * 0.42] * 2, color="#bbbbbb", lw=0.9, transform=axm.transAxes)

for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.10, (y, "axes-fraction text outside the panel")
    else:
        lo, hi = ax.get_ylim(); assert lo - 0.05 <= y <= hi + 0.07, (y, lo, hi)

out = A / "rate_refusal.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(plt.imread(out) * 255, dtype=np.int16)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("wrote", out, im.shape)
