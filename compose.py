"""Compose the artifact from the two measured dumps. Every number is read, not typed."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

A = json.loads(pathlib.Path("/tmp/art_main/facts.json").read_text())
B = json.loads(pathlib.Path("/tmp/art_branch/facts.json").read_text())
assert A["tree"] != B["tree"], "both dumps came from the same tree"
assert A["reference_joints"] == B["reference_joints"], "the reference rollout differs"

ROWS = list(zip(A["rows"], B["rows"], strict=True))
for ra, rb in ROWS:
    assert ra["label"] == rb["label"]
N_ABANDONED_MAIN = sum(r["worker_abandoned"] for r in A["rows"])
N_ABANDONED_PR = sum(r["worker_abandoned"] for r in B["rows"])
N_DROPPED_MAIN = sum(r["dropped_records"] for r in A["rows"])
N_DROPPED_PR = sum(r["dropped_records"] for r in B["rows"])
assert (N_ABANDONED_MAIN, N_ABANDONED_PR) == (6, 0), (N_ABANDONED_MAIN, N_ABANDONED_PR)
assert (N_DROPPED_MAIN, N_DROPPED_PR) == (1, 0), (N_DROPPED_MAIN, N_DROPPED_PR)
assert not [r for r in A["rows"] + B["rows"] if r["raised"]], "cleanup raised somewhere"
assert all(r["world_released"] for r in A["rows"] + B["rows"]), "a teardown did not release the world"

img_a = np.array(Image.open("/tmp/art_main/reference.png").convert("RGB"), dtype=int)
img_b = np.array(Image.open("/tmp/art_branch/reference.png").convert("RGB"), dtype=int)
delta = int(np.abs(img_a - img_b).max())
diff_px = int((np.abs(img_a - img_b).sum(2) > 0).sum())
total_px = img_a.shape[0] * img_a.shape[1]
sat = float(((img_a.max(2) - img_a.min(2)) > 45).mean())
assert delta <= 2, delta
assert sat > 0.05, sat

GREEN, RED, GREY = "#1b7f3b", "#b3261e", "#5f6368"
placed: list[tuple] = []


def put(ax, x, y, s, axes_coords=False, **kw):
    if axes_coords:
        kw["transform"] = ax.transAxes
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 11.6), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0], width_ratios=[1.22, 1.0], hspace=0.16, wspace=0.10)

# ---- row 1: measured verdict grid ---------------------------------------
ax = fig.add_subplot(gs[0, :])
ax.axis("off")
n = len(ROWS)
TOP, BOTTOM = n + 1.55, -0.35
ax.set_xlim(0, 1)
ax.set_ylim(BOTTOM, TOP)

put(ax, 0.0, n + 1.12, "cleanup(policy_stop_timeout=X) against a live 50 Hz rollout", fontsize=15.5, fontweight="bold")
put(
    ax,
    0.0,
    n + 0.70,
    "Each row starts a rollout, sleeps 0.5 s, then calls cleanup once. "
    "'worker awaited' is the join actually completing before the world is freed.",
    fontsize=10.6,
    color=GREY,
    style="italic",
)

COLS = [0.005, 0.150, 0.255, 0.400, 0.615, 0.720, 0.865]
HEAD = ["budget", "waited", "worker awaited", "reported", "waited", "worker awaited", "reported"]
for x, h in zip(COLS, HEAD, strict=True):
    put(ax, x, n + 0.24, h, fontsize=10.4, fontweight="bold", color=GREY)
put(ax, 0.150, n + 0.62, "main", fontsize=12.4, fontweight="bold", color=RED)
put(ax, 0.615, n + 0.62, "this change", fontsize=12.4, fontweight="bold", color=GREEN)
ax.plot([0.0, 0.565], [n + 0.05, n + 0.05], color=GREY, lw=0.9)
ax.plot([0.600, 1.0], [n + 0.05, n + 0.05], color=GREY, lw=0.9)
ax.plot([0.583, 0.583], [BOTTOM + 0.1, n + 0.80], color="#c9ccd1", lw=1.2)

for i, (ra, rb) in enumerate(ROWS):
    y = n - 1 - i
    bad = ra["worker_abandoned"] or ra["dropped_records"]
    if bad:
        ax.add_patch(Rectangle((0.0, y - 0.30), 0.565, 0.68, facecolor=RED, alpha=0.10, lw=0))
    ax.add_patch(Rectangle((0.600, y - 0.30), 0.400, 0.68, facecolor=GREEN, alpha=0.10, lw=0))
    put(ax, COLS[0], y, ra["label"], fontsize=11.2, family="monospace", fontweight="bold")
    for col, r, base in ((1, ra, 0), (4, rb, 3)):
        put(ax, COLS[col], y, f"{r['waited_s']:.4f} s", fontsize=10.4, family="monospace")
        awaited = not r["worker_abandoned"]
        put(
            ax,
            COLS[col + 1],
            y,
            "yes" if awaited else "NO - abandoned",
            fontsize=10.6,
            fontweight="bold",
            color=GREEN if awaited else RED,
        )
        if r["dropped_records"]:
            txt, colour = f"record DROPPED x{r['dropped_records']}", RED
        elif r["reported"]:
            txt, colour = "yes", GREEN
        else:
            txt, colour = "n/a (usable)", GREY
        put(ax, COLS[col + 2], y, txt, fontsize=10.2, color=colour)

put(
    ax,
    0.0,
    -0.20,
    f"a live worker was abandoned in {N_ABANDONED_MAIN} of {n} cases on main and {N_ABANDONED_PR} of {n} here;"
    f"  log records dropped: {N_DROPPED_MAIN} -> {N_DROPPED_PR};  cleanup raised in 0 of {2 * n} runs",
    fontsize=11.4,
    fontweight="bold",
)

# ---- row 2 left: the reference rollout ----------------------------------
ax_img = fig.add_subplot(gs[1, 0])
ax_img.imshow(img_b.astype(np.uint8))
ax_img.set_xticks([])
ax_img.set_yticks([])
ax_img.set_title("A normal rollout + teardown - unchanged", fontsize=12.6, fontweight="bold", pad=8)
ax_img.set_xlabel(
    f"run_policy(1.2 s @ 50 Hz) then cleanup(policy_stop_timeout=2.0).\n"
    f"Every joint identical to 6 dp on both trees; render max|delta| = {delta}/255 over "
    f"{diff_px} of {total_px} px (renderer noise).",
    fontsize=9.9,
    color=GREY,
)

# ---- row 2 right: fact ledger ------------------------------------------
ax_txt = fig.add_subplot(gs[1, 1])
ax_txt.axis("off")
ax_txt.set_xlim(0, 1)
ax_txt.set_ylim(0, 1)
put(ax_txt, 0.0, 0.965, "Why each budget could not be honored", fontsize=12.6, fontweight="bold")
LINES = [
    ("Future.result measures its wait as", GREY),
    ("time.monotonic() + timeout, so:", GREY),
    ("", GREY),
    ("0 / negative / nan   expire it before the first check", None),
    ("inf                  raises OverflowError from that sum", None),
    ("'5' / [5]            raise TypeError - and the join's", None),
    ("                     own %.1f warning then raises too,", None),
    ("                     so the record is dropped", None),
    ("True                 is a silent 1 s cap on a 5 s default", None),
    ("", GREY),
    ("Each abandons a worker that may still be inside", GREY),
    ("mj_step on the world cleanup is about to free.", GREY),
    ("", GREY),
    ("The budget is now held to the same positive-finite", None),
    ("domain every other span of time is, reported against", None),
    ("its parameter, and resolved to the documented", None),
    ("default - what None already means. cleanup is the", None),
    ("release path __exit__ and the finalizer call, so", None),
    ("refusing would leak the world for a value error.", None),
]
TOP_T, LAST_T = 0.905, 0.035
STEP = (TOP_T - LAST_T) / (len(LINES) - 1)
assert STEP > 0.030, STEP
for j, (line, colour) in enumerate(LINES):
    y = TOP_T - j * STEP
    put(
        ax_txt,
        0.0,
        y,
        line,
        fontsize=9.9,
        family="monospace",
        color=colour or "#202124",
        style="italic" if colour == GREY else "normal",
    )
assert TOP_T - (len(LINES) - 1) * STEP >= LAST_T - 1e-9

for ax_, y, axes_coords in placed:
    lo, hi = (-0.03, 1.07) if axes_coords else ax_.get_ylim()
    assert lo - 0.06 * (hi - lo) <= y <= hi + 0.06 * (hi - lo), (y, lo, hi)

out = pathlib.Path("/tmp/artifact_cleanup_budget.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.array(Image.open(out).convert("RGB"), dtype=int)
for name, band in (
    ("top", im[:8]),
    ("bottom", im[-8:]),
    ("left", im[:, :8]),
    ("right", im[:, -8:]),
):
    non_white = int((np.abs(band - 255).sum(2) > 12).sum())
    assert non_white == 0, f"{name} border has {non_white} non-white px"
print(f"OK {out} {im.shape[1]}x{im.shape[0]} px; borders clean; {len(placed)} text placements verified")
