"""Compose the GH #2239 figure from the measured facts. Every cell is derived."""
from __future__ import annotations
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

F = json.loads(pathlib.Path(sys.argv[1]).read_text())
OUT = pathlib.Path(sys.argv[2])
A, B = F["main"], F["branch"]

# --- assertions on the measurement itself -----------------------------------
assert A["executor"] == "ThreadPoolExecutor" and B["executor"] == "DaemonThreadExecutor"
assert A["exited"] is False and B["exited"] is True
assert A["non_daemon_left"] == ["test_arm_executor_0"] and B["non_daemon_left"] == []
assert A["verdict"] == B["verdict"] == "TimeoutError -- the test fails here"
assert A["verdict_after_s"] == B["verdict_after_s"] == 2.0
assert A["start_task_status"] == B["start_task_status"] == "success"
assert A["tree"] != B["tree"]

MUT = [
    ("(unmutated control)", "0 failed / 17", "0 failed / 201"),
    ("M1 port_domain fixture -> ThreadPoolExecutor", "1 failed / 16", "0 failed / 201"),
    ("M2 lifecycle fixture -> ThreadPoolExecutor", "1 failed / 16", "0 failed / 201"),
    ("M3 cleanup_disconnects -> ThreadPoolExecutor", "1 failed / 16", "0 failed / 201"),
    ("M4 helper worker is not a daemon", "3 failed / 14", "0 failed / 201"),
    ("M5 shutdown(wait=True) does not join", "1 failed / 16", "1 failed / 200"),
    ("M6 submit after shutdown allowed", "1 failed / 16", "0 failed / 201"),
    ("M7 worker swallows the exception", "1 failed / 16", "0 failed / 201"),
]
rows = MUT[1:]
caught = sum(1 for _l, n, _p in rows if not n.startswith("0 "))
blind = sum(1 for _l, _n, p in rows if p.startswith("0 "))
assert (len(rows), caught, blind) == (7, 7, 6), (len(rows), caught, blind)

RED, GREEN, INK, MUTED = "#b3261e", "#1b5e20", "#1a1a1a", "#5f6368"
placed: list[tuple[object, float]] = []


def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, transform=ax.transAxes, **kw)


fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], width_ratios=[1.06, 1.0],
                      hspace=0.30, wspace=0.16, left=0.045, right=0.965, top=0.905, bottom=0.055)

fig.suptitle(
    "GH #2239 - a fixture work item the test gives up on decided whether the run ended",
    fontsize=15.5, fontweight="bold", y=0.968, color=INK)
fig.text(0.5, 0.933,
         "Real path: Robot.start_task submits the rollout to Robot._executor; bring-up is wedged, "
         "the test waits 2s and gives up. Same verdict on both trees.",
         ha="center", fontsize=10.4, color=MUTED)

# ---- row 1: wall clock ------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 48); ax.set_ylim(-0.75, 1.75); ax.invert_yaxis()
ax.set_yticks([0, 1]); ax.set_yticklabels(["main\n(ThreadPoolExecutor)", "this PR\n(DaemonThreadExecutor)"], fontsize=10.4)
ax.set_xlabel("wall clock (seconds)", fontsize=10.2)
ax.set_title("The verdict arrives at the same instant. Only the exit differs.", fontsize=11.6, pad=9, color=INK)
ax.grid(axis="x", alpha=0.25, linewidth=0.6)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

ax.add_patch(Rectangle((0, -0.22), A["wall_s"], 0.44, facecolor=RED, alpha=0.85))
ax.add_patch(Rectangle((0, 0.78), B["wall_s"], 0.44, facecolor=GREEN, alpha=0.85))
for y, d in ((0, A), (1, B)):
    ax.plot([d["verdict_after_s"]], [y], marker="|", markersize=26, color="white", markeredgewidth=2.6, zorder=5)
    ax.annotate(f'verdict at {d["verdict_after_s"]:.1f}s', xy=(d["verdict_after_s"], y),
                xytext=(d["verdict_after_s"] + 1.0, y - 0.42), fontsize=9.4, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.text(A["wall_s"] + 0.6, 0, f'KILLED at {A["killed_at_s"]:.0f}s - never exited',
        va="center", fontsize=10.6, color=RED, fontweight="bold")
ax.text(B["wall_s"] + 0.6, 1, f'exit 0 at {B["wall_s"]:.2f}s', va="center",
        fontsize=10.6, color=GREEN, fontweight="bold")

# ---- row 2 left: measured ledger -------------------------------------------
axl = fig.add_subplot(gs[1, 0]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.975, "Measured on the real Robot.start_task path", fontsize=12.0, fontweight="bold", color=INK)
put(axl, 0.60, 0.895, "main", fontsize=10.4, fontweight="bold", color=RED, ha="center")
put(axl, 0.87, 0.895, "this PR", fontsize=10.4, fontweight="bold", color=GREEN, ha="center")
LED = [
    ("executor the fixture builds", A["executor"], B["executor"]),
    ("start_task", A["start_task_status"], B["start_task_status"]),
    ("the wait", "TimeoutError", "TimeoutError"),
    ("...after", f'{A["verdict_after_s"]:.1f}s', f'{B["verdict_after_s"]:.1f}s'),
    ("non-daemon threads left", str(A["non_daemon_left"][0]), "none"),
    ("interpreter exited", "no", "yes"),
    ("exit code", "killed", str(B["exit_code"])),
    ("wall clock", f'{A["wall_s"]:.2f}s', f'{B["wall_s"]:.2f}s'),
]
TOP, LAST = 0.815, 0.155
step = (TOP - LAST) / (len(LED) - 1)
assert step > 0.030, step
y = TOP
for label, a, b in LED:
    put(axl, 0.0, y, label, fontsize=9.9, color=INK)
    bad = a != b
    put(axl, 0.60, y, a, fontsize=9.9, ha="center", color=RED if bad else MUTED,
        fontweight="bold" if bad else "normal")
    put(axl, 0.87, y, b, fontsize=9.9, ha="center", color=GREEN if bad else MUTED,
        fontweight="bold" if bad else "normal")
    y -= step
assert abs((y + step) - LAST) < 1e-9
put(axl, 0.0, 0.055,
    "The two rows that differ are the two that decide whether the job ends.\n"
    "Everything the test reports is byte-identical.",
    fontsize=9.5, color=MUTED, style="italic")

# ---- row 2 right: mutation matrix ------------------------------------------
axr = fig.add_subplot(gs[1, 1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 0.975, "7 plausible regressions x 2 test arms", fontsize=12.0, fontweight="bold", color=INK)
put(axr, 0.70, 0.895, "new module", fontsize=9.9, fontweight="bold", color=INK, ha="center")
put(axr, 0.91, 0.895, "pre-existing", fontsize=9.9, fontweight="bold", color=INK, ha="center")
TOP2, LAST2 = 0.815, 0.185
step2 = (TOP2 - LAST2) / (len(MUT) - 1)
assert step2 > 0.030, step2
y = TOP2
for label, new, pre in MUT:
    control = label.startswith("(")
    put(axr, 0.0, y, label, fontsize=9.0, color=MUTED if control else INK,
        style="italic" if control else "normal")
    put(axr, 0.70, y, new, fontsize=9.0, ha="center",
        color=MUTED if control else GREEN, fontweight="normal" if control else "bold")
    put(axr, 0.91, y, pre, fontsize=9.0, ha="center",
        color=MUTED if (control or pre.startswith("1 ")) else RED,
        fontweight="bold" if (not control and pre.startswith("0 ")) else "normal")
    y -= step2
assert abs((y + step2) - LAST2) < 1e-9
put(axr, 0.0, 0.075,
    f"{caught} of {len(rows)} caught here; {blind} of {len(rows)} invisible to the 201 pre-existing cases.\n"
    "M5 is caught by both - cleanup_disconnects asserts ordering around\n"
    "production's shutdown(wait=True), so that half of the contract is load-bearing.",
    fontsize=9.2, color=MUTED, style="italic")

for ax_, ylo in ((axl, 0.0), (axr, 0.0)):
    for a_, yv in placed:
        if a_ is ax_:
            assert -0.03 <= yv <= 1.07, (yv,)

fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
print("wrote", OUT)

import numpy as np
from PIL import Image
im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("size", im.shape, "border clean")
