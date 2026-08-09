"""Compose the artifact from the two measured trees. Every number is read from JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = json.loads(Path("/tmp/art_main/facts.json").read_text())    # upstream/main
B = json.loads(Path("/tmp/art_branch/facts.json").read_text())  # this change
assert A["tree"] != B["tree"], "both dumps came from the same tree"

ra = {r["tag"]: r for r in A["rows"]}
rb = {r["tag"]: r for r in B["rows"]}

# ---- facts the figure asserts before it is drawn ---------------------------
assert ra["nonfinite"]["status"] == "success" and ra["nonfinite"]["steps"] == 100
assert ra["nonfinite"]["captured"] == 0 and ra["nonfinite"]["failures"] == 100
assert ra["nonfinite"]["warnings_emitted"] == 0, "main emitted no warning for nan"
assert rb["nonfinite"]["status"] == "refused" and rb["nonfinite"]["steps"] == 0
for tag in ("healthy", "honored"):  # unchanged halves
    for k in ("status", "steps", "captured", "failures", "warnings_emitted", "aborted"):
        assert ra[tag][k] == rb[tag][k], f"{tag}.{k} differs across trees"
assert ra["healthy"]["captured"] == 100 and ra["honored"]["aborted"] is True

heal_a = np.load("/tmp/art_main/healthy_frame.npy")
heal_b = np.load("/tmp/art_branch/healthy_frame.npy")
delta = int(np.abs(heal_a.astype(int) - heal_b.astype(int)).max())
assert delta <= 2, f"the healthy capture differs across trees by {delta}"

nan_main = np.load("/tmp/art_main/nonfinite_final.npy")
nan_branch = np.load("/tmp/art_branch/nonfinite_final.npy")
moved = float((np.abs(nan_main.astype(int) - nan_branch.astype(int)).sum(2) > 24).mean())
assert moved > 0.05, f"the two nan panels differ on only {moved:.2%} of pixels"

placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    is_axes = kw.get("transform") is not None
    placed.append((ax, y, is_axes))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(16.4, 13.6), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.42, 0.72, 0.72], hspace=0.30, wspace=0.07)

fig.suptitle(
    "max_onframe_failures is the ceiling that stops a broken on_frame hook capturing nothing behind a "
    '"successful" rollout',
    fontsize=15.5, fontweight="bold", y=0.982,
)
fig.text(
    0.5, 0.958,
    "MuJoCo headless (MUJOCO_GL=egl), so100, 2.0 s at 50 Hz = 100 control steps. The hook renders one frame "
    "per step; in the two right-hand runs it raises every call.",
    ha="center", fontsize=10.6, style="italic", color="#333333",
)

PANELS = [
    (
        np.load("/tmp/art_main/healthy_frame.npy"),
        "A  reference: a working capture hook",
        f"captured {ra['healthy']['captured']} of {ra['healthy']['steps']} frames  |  status={ra['healthy']['status']}\n"
        "byte-identical on both trees (max delta "
        f"{delta}/255) - the honored path is untouched",
        "#1a7f37",
    ),
    (
        nan_main,
        "B  main: max_onframe_failures=nan",
        f"{ra['nonfinite']['steps']} steps ran, {ra['nonfinite']['failures']} hook failures, "
        f"{ra['nonfinite']['captured']} frames captured\n"
        f"status={ra['nonfinite']['status']}  |  warnings emitted: {ra['nonfinite']['warnings_emitted']}  "
        "- the abort never fired",
        "#b42318",
    ),
    (
        nan_branch,
        "C  this change: max_onframe_failures=nan",
        f"refused before step 1 ({rb['nonfinite']['steps']} steps, "
        f"{rb['nonfinite']['captured']} frames)\n"
        "the arm is still at its start pose",
        "#1a7f37",
    ),
]
for col, (img, title, sub, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colour); spine.set_linewidth(3.0)
    ax.set_title(title, fontsize=12.4, fontweight="bold", color=colour, pad=8)
    ax.set_xlabel(sub, fontsize=9.7, color="#222222", labelpad=8)

# ---- row 2: what the operator could see -----------------------------------
ax = fig.add_subplot(gs[1, :])
for tag, label, colour, style in [
    ("healthy", f"working hook (limit=None): {ra['healthy']['captured']} frames captured", "#1a7f37", "-"),
    ("honored", f"broken hook, limit=3: aborted after {ra['honored']['failures']} failures", "#0969da", "-"),
    ("nonfinite", f"broken hook, limit=nan (main): {ra['nonfinite']['steps']} steps, 0 captured", "#b42318", "-"),
]:
    y = ra[tag]["per_step_captured"]
    ax.plot(range(1, len(y) + 1), y, style, color=colour, lw=2.6, label=label)
    ax.plot([len(y)], [y[-1] if y else 0], "o", color=colour, ms=9)
ax.axvline(ra["honored"]["failures"], color="#0969da", ls=":", lw=1.8)
put(ax, ra["honored"]["failures"] + 1.5, 52, "limit=3 aborts here\nand warns the operator",
    fontsize=9.6, color="#0969da", va="center")
put(ax, 52, 8, "limit=nan: 100 consecutive failures, no abort, no warning, status=success",
    fontsize=10.4, color="#b42318", fontweight="bold", va="center")
ax.set_xlabel("control step", fontsize=10.6)
ax.set_ylabel("frames captured\n(cumulative)", fontsize=10.6)
ax.set_xlim(0, 104); ax.set_ylim(-4, 104)
ax.grid(alpha=0.26)
ax.legend(loc="upper left", fontsize=9.8, framealpha=0.94)
ax.set_title("The hook fails identically in both broken runs; only the limit differs",
             fontsize=11.6, fontweight="bold", pad=7)

# ---- row 3: measured verdict table -----------------------------------------
ax = fig.add_subplot(gs[2, :]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
COLS = [0.012, 0.145, 0.245, 0.395, 0.505, 0.625, 0.775]
HEAD = ["max_onframe_failures", "tree", "status", "steps run", "frames captured",
        "warnings", "abort fired?"]
TOP, LAST = 0.90, 0.10
ROWS = [
    ("3  (usable)", "both", ra["honored"]["status"], ra["honored"]["steps"],
     ra["honored"]["captured"], ra["honored"]["warnings_emitted"], "yes", "#1a7f37"),
    ("None -> 5", "both", ra["healthy"]["status"], ra["healthy"]["steps"],
     ra["healthy"]["captured"], ra["healthy"]["warnings_emitted"], "n/a (no failures)", "#1a7f37"),
    ("nan", "main", ra["nonfinite"]["status"], ra["nonfinite"]["steps"],
     f"{ra['nonfinite']['captured']} of 100", ra["nonfinite"]["warnings_emitted"], "NO", "#b42318"),
    ("nan", "this change", "refused", rb["nonfinite"]["steps"],
     rb["nonfinite"]["captured"], "-", "n/a (never ran)", "#1a7f37"),
]
STEP = (TOP - LAST) / (len(ROWS))
assert STEP > 0.10, STEP
for x, h in zip(COLS, HEAD, strict=True):
    put(ax, x, TOP, h, fontsize=10.4, fontweight="bold", color="#111111", transform=ax.transAxes)
y = TOP
for label, tree, status, steps, captured, warns, aborted, colour in ROWS:
    y -= STEP
    ax.add_patch(plt.Rectangle((0.004, y - 0.036), 0.992, 0.088,
                               facecolor=colour, alpha=0.075, transform=ax.transAxes, zorder=0))
    for x, cell in zip(COLS, [label, tree, status, steps, captured, warns, aborted], strict=True):
        put(ax, x, y, str(cell), fontsize=10.1, color=colour,
            fontweight="bold" if colour == "#b42318" else "normal", transform=ax.transAxes)
assert y > 0.04, y
put(ax, 0.012, 0.018,
    'main\'s abort text reads "aborting episode to avoid silent dataset corruption" - a non-finite limit made '
    "the comparison it depends on false for every counter value, and broke the %d-formatted warning as well.",
    fontsize=10.0, style="italic", color="#333333", transform=ax.transAxes)

for ax_, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.07, f"axes-fraction text at y={yy}"
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.06 * (hi - lo) <= yy <= hi + 0.08 * (hi - lo), f"data text at y={yy} outside {(lo, hi)}"

out = Path("/tmp/art_onframe.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(matplotlib.image.imread(out) * 255).astype(np.uint8)[:, :, :3]
for name, band in [("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])]:
    nz = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nz == 0, f"{name} border has {nz} non-white pixels"
print(f"OK  {out}  {im.shape[1]}x{im.shape[0]}")
print(f"    healthy capture identical across trees: max delta {delta}/255")
print(f"    panels B vs C differ on {moved:.2%} of pixels")
