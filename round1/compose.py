"""Compose the measurement figure. Every rendered number is asserted against facts.json."""
from __future__ import annotations
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent
F = json.loads((HERE / "facts.json").read_text())
print("TREE:", F["tree"])

VARIANTS = F["variants"]
TABLE, SCORE, MIRROR = F["table"], F["score"], F["mirror"]

# ---- self-audit of every claim the figure makes ----
assert VARIANTS == ["except BaseException", "no handler", "except Exception"], VARIANTS
assert SCORE["except BaseException"] == {"collected": 5, "escaped": 0}, SCORE
assert SCORE["no handler"] == {"collected": 2, "escaped": 4}, SCORE
assert SCORE["except Exception"] == {"collected": 5, "escaped": 4}, SCORE
winners = [n for n in VARIANTS if SCORE[n]["collected"] == 5 and SCORE[n]["escaped"] == 4]
assert winners == ["except Exception"], winners
assert MIRROR["before"] == [MIRROR["published_alert"]], MIRROR
assert MIRROR["after"] == [], MIRROR
assert len(TABLE) == 9 and sum(1 for r in TABLE if r["kind"] == "control") == 4
assert F["prefix_proof"] == {"failed": 4, "passed": 3, "of": 7}

GREEN, RED, GREY = "#1a7f37", "#b42318", "#57606a"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(13.4, 9.0), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[3.05, 1.0, 0.78], hspace=0.30)
fig.suptitle(
    "The verdict classifier's handler width: three candidates, nine outcomes",
    fontsize=14.5, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.943,
    "`_renders_a_half_built_instance` turns a repr outcome into a verdict the survey compares "
    "against every class in the package.\nA failure inside repr is one of those answers; an interrupt is not.",
    ha="center", fontsize=9.6, color=GREY,
)

# ---------- row 1: the matrix ----------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
COL_X = [0.315, 0.545, 0.775]
TOP, LAST = 0.885, 0.115
STEP = (TOP - LAST) / (len(TABLE) - 1)
assert STEP > 0.030, STEP

put(ax, 0.012, 0.965, "outcome the classifier is handed", fontsize=10, fontweight="bold", transform=ax.transAxes)
for cx, name in zip(COL_X, VARIANTS, strict=True):
    put(ax, cx, 0.965, name, fontsize=10, fontweight="bold", ha="center", transform=ax.transAxes,
        color=GREEN if name == "except Exception" else "#24292f")
ax.plot([0.008, 0.99], [0.935, 0.935], color="#d0d7de", lw=1.1, transform=ax.transAxes)

y = TOP
for row in TABLE:
    is_control = row["kind"] == "control"
    if is_control:
        ax.add_patch(Rectangle((0.008, y - 0.030), 0.982, 0.062, transform=ax.transAxes,
                               facecolor="#fff8f0", edgecolor="none", zorder=0))
    put(ax, 0.012, y, row["label"], fontsize=9.9, va="center", transform=ax.transAxes,
        fontweight="bold" if is_control else "normal")
    put(ax, 0.245, y, "control flow" if is_control else "library", fontsize=8.4, va="center",
        color=GREY, ha="right", transform=ax.transAxes, style="italic")
    for cx, name, cell in zip(COL_X, VARIANTS, row["cells"], strict=True):
        out = cell["outcome"]
        # A cell is right when a library outcome became a verdict and control flow escaped.
        good = (not is_control and not out.startswith("ESCAPES")) or (is_control and out.startswith("ESCAPES"))
        put(ax, cx, y, out, fontsize=9.4, va="center", ha="center", transform=ax.transAxes,
            color=GREEN if good else RED, fontweight="bold" if not good else "normal")
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

# ---------- row 2: the score ----------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.012, 0.93, "Score", fontsize=11, fontweight="bold", transform=ax2.transAxes)
put(ax2, 0.315, 0.93, "library outcome collected", fontsize=9.6, ha="center", color=GREY, transform=ax2.transAxes)
put(ax2, 0.545, 0.93, "control flow reaches the runner", fontsize=9.6, ha="center", color=GREY, transform=ax2.transAxes)
put(ax2, 0.80, 0.93, "verdict", fontsize=9.6, ha="center", color=GREY, transform=ax2.transAxes)
S_TOP, S_LAST = 0.66, 0.16
sstep = (S_TOP - S_LAST) / (len(VARIANTS) - 1)
assert sstep > 0.030, sstep
sy = S_TOP
for name in VARIANTS:
    s = SCORE[name]
    ok = s["collected"] == 5 and s["escaped"] == 4
    put(ax2, 0.012, sy, name, fontsize=10, va="center", transform=ax2.transAxes,
        fontweight="bold" if ok else "normal", color=GREEN if ok else "#24292f")
    put(ax2, 0.315, sy, f"{s['collected']}/5", fontsize=10.4, va="center", ha="center", transform=ax2.transAxes,
        color=GREEN if s["collected"] == 5 else RED, fontweight="bold")
    put(ax2, 0.545, sy, f"{s['escaped']}/4", fontsize=10.4, va="center", ha="center", transform=ax2.transAxes,
        color=GREEN if s["escaped"] == 4 else RED, fontweight="bold")
    put(ax2, 0.80, sy, "satisfies both halves" if ok else "loses one half", fontsize=9.4, va="center",
        ha="center", transform=ax2.transAxes, color=GREEN if ok else RED,
        fontweight="bold" if ok else "normal")
    sy -= sstep
assert abs((sy + sstep) - S_LAST) < 1e-9, (sy, S_LAST)

# ---------- row 3: mirror + proof ----------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
b = MIRROR["before"][0]
lines = [
    f"Validated rule mirror (py/catch-base-exception over tests/): reproduces the published alert at "
    f"line {b[0]}, columns {b[1]}-{b[2]} -> now {len(MIRROR['after'])} hits in the file.",
    f"Pre-fix proof: with only the handler reverted and the new tests kept, "
    f"{F['prefix_proof']['failed']} of {F['prefix_proof']['of']} fail; the "
    f"{F['prefix_proof']['passed']} that pass are the library direction the wide handler also collects.",
    f"Full suite: {F['suite']['passed']:,} passed, {F['suite']['skipped']} skipped, "
    f"{F['suite']['failed']} failed ({F['suite']['seconds']} s). Tests only - no production line changes.",
]
L_TOP, L_LAST = 0.80, 0.16
lstep = (L_TOP - L_LAST) / (len(lines) - 1)
assert lstep > 0.030, lstep
ly = L_TOP
for text in lines:
    put(ax3, 0.012, ly, text, fontsize=9.5, va="center", transform=ax3.transAxes, family="monospace")
    ly -= lstep
assert abs((ly + lstep) - L_LAST) < 1e-9, (ly, L_LAST)

for ax_obj, yv, is_axes in placed:
    if is_axes:
        assert -0.05 <= yv <= 1.10, (yv, "axes-fraction out of range")
    else:
        lo, hi = ax_obj.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, (yv, lo, hi)

OUT = HERE / "handler-width-decision-table.png"
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white px"
print(f"OK {OUT} {im.shape[1]}x{im.shape[0]}  texts={len(placed)}")
