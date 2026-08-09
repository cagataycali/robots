"""Compose the artifact from the two capture runs. Every cell is measured."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = json.loads(Path("/tmp/art_main/facts.json").read_text())   # upstream/main
B = json.loads(Path("/tmp/art_pr/facts.json").read_text())     # this change
assert A["tree"] != B["tree"], (A["tree"], B["tree"])

ra = {r["tag"]: r for r in A["rows"]}
rb = {r["tag"]: r for r in B["rows"]}

# --- the measured claims this figure rests on ---
assert ra["scene_falsy"]["status"] == "success" and ra["scene_falsy"]["scene_loaded"] is False
assert ra["instr_int"]["instruction_seen"] == "42" and ra["instr_int"]["instruction_type"] == "int"
assert ra["honored"]["scene_loaded"] is True and ra["honored"]["instruction_seen"] == "'pick up the crate'"
for tag in ("scene_falsy", "instr_int", "instr_list", "name_int"):
    assert rb[tag]["constructed"] is False, tag
assert rb["honored"]["status"] == "success" and rb["honored"]["scene_loaded"] is True

hon_a = np.asarray(iio.imread("/tmp/art_main/honored.png")).astype(int)
hon_b = np.asarray(iio.imread("/tmp/art_pr/honored.png")).astype(int)
skipped = np.asarray(iio.imread("/tmp/art_main/scene_falsy.png")).astype(int)

# the honored path is untouched
d_hon = int(np.abs(hon_a - hon_b).max())
assert d_hon <= 2, d_hon
# and the silently-skipped scene really is a different world
frac = float((np.abs(hon_a - skipped).sum(2) > 12).mean())
assert frac > 0.10, frac
print(f"honored across trees max|delta|={d_hon}   honored vs skipped-scene differ={frac:.2%}")

placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 11.0), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.62, 1.0], hspace=0.14, wspace=0.05)

fig.suptitle(
    "DeclarativeBenchmark: a declared scene and instruction now mean the same thing on both construction paths",
    fontsize=15.5, y=0.975, fontweight="bold",
)
fig.text(
    0.5, 0.938,
    "One benchmark, evaluated over a two-link arm. `from_dict` refused every value below; the constructor stored each one raw.",
    ha="center", fontsize=10.6, style="italic", color="#333333",
)

# ---- row 1: three real renders ----
panels = [
    (hon_a.astype(np.uint8),
     "A - the scene the benchmark declares",
     "scene='scene.xml'  ->  status=success, scene loaded\ninstruction reaches the policy as 'pick up the crate'\nidentical on both trees (max|delta| = %d/255)" % d_hon,
     "#1f7a35"),
    (skipped.astype(np.uint8),
     "B - on main: scene=[]  ->  status=\"success\"",
     "the declared scene was never loaded: `on_episode_start`\nreads it under `if self._scene:`, and [] is falsy.\nDiffers from A on %.1f%% of pixels." % (frac * 100),
     "#b3261e"),
    (None,
     "C - with this change: scene=[] is refused",
     "refused at construction, before any episode runs:\n\n%s\n\nThe world above never existed either way -\nthe difference is that the caller is now told." % rb["scene_falsy"]["refusal"],
     "#1f7a35"),
]
for col, (img, title, cap, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    if img is None:
        ax.set_facecolor("#12161c")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        put(ax, 0.5, 0.56, cap, ha="center", va="center", fontsize=10.4, color="#e8eef6",
            family="monospace", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(colour)
            sp.set_linewidth(2.2)
        ax.set_title(title, fontsize=11.6, fontweight="bold", color=colour, pad=8)
    else:
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(colour)
            sp.set_linewidth(2.2)
        ax.set_title(title, fontsize=11.6, fontweight="bold", color=colour, pad=8)
        ax.set_xlabel(cap, fontsize=9.6, color="#333333", labelpad=7)

# ---- row 2: the measured table ----
ax = fig.add_subplot(gs[1, :])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

COLS = (0.028, 0.230, 0.520, 0.792)
HEAD = ("value passed to the constructor", "on main", "what the caller was told", "with this change")
TOP = 0.925
put(ax, 0.028, 0.988, "Measured: the same value through the constructor, before and after",
    fontsize=12.4, fontweight="bold", transform=ax.transAxes)
for x, h in zip(COLS, HEAD):
    put(ax, x, TOP, h, fontsize=10.4, fontweight="bold", color="#1a1a1a", transform=ax.transAxes)

ROWS = [
    ("instruction=42", f"accepted; policy received {ra['instr_int']['instruction_seen']} ({ra['instr_int']['instruction_type']})",
     'status="success"', "refused", True),
    ("instruction=['pick']", f"accepted; policy received {ra['instr_list']['instruction_seen']}",
     'status="success"', "refused", True),
    ("scene=[]", "accepted; declared scene never loaded",
     'status="success"', "refused", True),
    ("scene=42", "accepted; load_scene(42) then failed",
     'status="error" (the one loud case)', "refused", False),
    ("name=7", "accepted; advertised as the benchmark id",
     'status="success"', "refused", True),
    ("default_robot=''", "accepted", 'status="success"', "refused", True),
    ("instruction='pick up the crate'", "accepted and honored", 'status="success"', "accepted and honored", False),
    ("scene='scene.xml'", "accepted and loaded", 'status="success"', "accepted and loaded", False),
]
LAST = 0.075
STEP = (TOP - 0.085 - LAST) / (len(ROWS) - 1)
assert STEP > 0.045, STEP
y = TOP - 0.085
for value, before, told, after, silent in ROWS:
    band = "#fdeceb" if silent else ("#eef7f0" if "honored" in after or "loaded" in after else "#fff8e6")
    ax.add_patch(plt.Rectangle((0.018, y - 0.030), 0.966, 0.062, transform=ax.transAxes,
                               facecolor=band, edgecolor="none", zorder=0))
    put(ax, COLS[0], y, value, fontsize=9.9, family="monospace", transform=ax.transAxes)
    put(ax, COLS[1], y, before, fontsize=9.9, transform=ax.transAxes,
        color="#b3261e" if silent else "#333333")
    put(ax, COLS[2], y, told, fontsize=9.9, family="monospace", transform=ax.transAxes,
        color="#b3261e" if silent else "#333333")
    put(ax, COLS[3], y, after, fontsize=9.9, fontweight="bold", transform=ax.transAxes,
        color="#1f7a35")
    y -= STEP
assert y + STEP >= LAST - 1e-9, (y, LAST)

n_silent = sum(1 for r in ROWS if r[4])
put(ax, 0.028, 0.018,
    f"{n_silent} of the {len(ROWS)} rows were accepted and reported success on main. "
    "The accepted side is unchanged: instruction='' and scene=''/None still declare neither.",
    fontsize=10.2, style="italic", color="#333333", transform=ax.transAxes)

for ax_, yv, is_axes in placed:
    if is_axes:
        assert -0.03 <= yv <= 1.07, (yv, "axes-fraction out of range")
    else:
        lo, hi = ax_.get_ylim()
        assert min(lo, hi) - 0.05 <= yv <= max(lo, hi) + 0.07, (yv, lo, hi)

OUT = Path("/tmp/art_out/benchmark_string_field_domains.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(iio.imread(OUT))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band.astype(int) - 255).sum(-1) > 12).sum())
    assert bad == 0, (name, bad)
print("WROTE", OUT, im.shape, "border clean")
