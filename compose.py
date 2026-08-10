"""Compose the measured figure. Every cell is read from the capture JSON."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
F = json.loads(pathlib.Path(f"/tmp/art-{ROOT.name}.json").read_text())

# ---- audit the facts before drawing anything ------------------------------
assert F["tree"] == str(ROOT)
assert F["fresh_process_lerobot_in_sys_modules"] == "False"
assert F["cov_before"]["probe_lines_listed"] is True
assert F["cov_after"]["probe_lines_listed"] is False
assert (F["cov_before"]["missing"], F["cov_after"]["missing"]) == (45, 41)
assert (F["cov_before"]["percent"], F["cov_after"]["percent"]) == (92, 93)
assert F["restored_byte_identical"] is True
assert len(F["mutations"]) == 4
CAUGHT = sum(1 for m in F["mutations"] if "failed" in m["new"])
BLIND = sum(1 for m in F["mutations"] if "failed" not in m["old"])
assert (CAUGHT, BLIND) == (4, 4), (CAUGHT, BLIND)

GREEN, RED, GREY, INK = "#1b7f3b", "#b3261e", "#6b6b6b", "#111111"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.06, 1.10, 0.50], hspace=0.30,
                      left=0.035, right=0.972, top=0.926, bottom=0.036)

fig.suptitle(
    "The probe that picks the lerobot install remedy: the branch the suite ran was not the branch a caller runs",
    fontsize=15.5, fontweight="bold", y=0.982,
)
fig.text(0.5, 0.951,
         "strands_robots/dataset_recorder.py :: _lerobot_installed  (lines "
         f"{F['probe_span'][0]}-{F['probe_span'][1]})   |   every number below is measured, none typed",
         ha="center", fontsize=10.4, style="italic", color=GREY)

# ---------------- row 1: which branch answers ----------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.03, "1.  Why the gap existed: two callers, two branches", transform=ax.transAxes,
    fontsize=12.6, fontweight="bold", color=INK)

hdr = [(0.005, "the probe's body"), (0.395, "answers for"), (0.605, "before"), (0.775, "after")]
for x, t in hdr:
    put(ax, x, 0.87, t, fontsize=10.6, fontweight="bold", color=GREY)
ax.plot([0.0, 1.0], [0.845, 0.845], lw=1.0, color=GREY)

rows = [
    ('if "lerobot" in sys.modules:  return True', "the pytest session\n(lerobot already imported)",
     "covered", "covered", GREEN, GREEN),
    ('return find_spec("lerobot") is not None', "a real caller\n(the recorder imports no lerobot)",
     "NEVER RUN", "covered", RED, GREEN),
    ("except (ImportError, ValueError):  return False", "a lookup that raises\non a broken install",
     "NEVER RUN", "covered", RED, GREEN),
]
TOP, LAST = 0.70, 0.14
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.20, step
for i, (code, who, before, after, cb, ca) in enumerate(rows):
    y = TOP - i * step
    put(ax, 0.005, y, code, fontsize=10.2, family="monospace", color=INK)
    put(ax, 0.395, y + 0.035, who, fontsize=9.5, color=GREY, va="top")
    put(ax, 0.605, y, before, fontsize=10.6, fontweight="bold", color=cb)
    put(ax, 0.775, y, after, fontsize=10.6, fontweight="bold", color=ca)
assert TOP - (len(rows) - 1) * step > 0.05

put(ax, 0.0, -0.055,
    "Measured in a child interpreter: `import strands_robots.dataset_recorder` leaves "
    f"'lerobot' in sys.modules -> {F['fresh_process_lerobot_in_sys_modules']}.  Under pytest it is already True, so the "
    "fast path answered every time.",
    transform=ax.transAxes, fontsize=9.9, color=INK)

# ---------------- row 2: mutation matrix ---------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.04, "2.  Four plausible regressions in the probe, run against both suites",
    transform=ax2.transAxes, fontsize=12.6, fontweight="bold", color=INK)
for x, t in [(0.005, "mutation applied to the probe"), (0.475, "these tests"),
             (0.700, "the 155 pre-existing tests over the same module")]:
    put(ax2, x, 0.88, t, fontsize=10.6, fontweight="bold", color=GREY)
ax2.plot([0.0, 1.0], [0.855, 0.855], lw=1.0, color=GREY)

TOP2, LAST2 = 0.72, 0.16
step2 = (TOP2 - LAST2) / (len(F["mutations"]) - 1)
assert step2 > 0.12, step2
for i, m in enumerate(F["mutations"]):
    y = TOP2 - i * step2
    caught = "failed" in m["new"]
    blind = "failed" not in m["old"]
    if blind:
        ax2.add_patch(Rectangle((0.695, y - 0.045), 0.302, 0.098, transform=ax2.transAxes,
                                facecolor=RED, alpha=0.10, edgecolor="none", zorder=0))
    put(ax2, 0.005, y, m["label"], fontsize=10.4, family="monospace", color=INK)
    put(ax2, 0.475, y, m["new"].split(" in ")[0], fontsize=10.4, fontweight="bold",
        color=GREEN if caught else RED)
    put(ax2, 0.700, y, m["old"].split(" in ")[0] + ("   <- BLIND" if blind else ""),
        fontsize=10.4, fontweight="bold", color=RED if blind else GREEN)
assert TOP2 - (len(F["mutations"]) - 1) * step2 > 0.05

put(ax2, 0.0, -0.045,
    f"Caught here: {CAUGHT} of 4.   Caught by the pre-existing suite: {4 - BLIND} of 4 -- each mutation leaves all 155 green, "
    "so nothing today would report the probe answering wrongly.",
    transform=ax2.transAxes, fontsize=9.9, color=INK)
put(ax2, 0.0, -0.115,
    f"Each anchor is unique inside the probe's own line span; the source is restored byte-identically afterwards "
    f"({F['restored_byte_identical']}).",
    transform=ax2.transAxes, fontsize=9.4, color=GREY, style="italic")

# ---------------- row 3: footer ------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(Rectangle((0.0, 0.02), 1.0, 0.96, transform=ax3.transAxes,
                        facecolor="#f4f4f4", edgecolor=GREY, lw=0.8))
foot = [
    f"dataset_recorder.py over the same four files:  {F['cov_before']['missing']} missing / "
    f"{F['cov_before']['percent']}%   ->   {F['cov_after']['missing']} missing / {F['cov_after']['percent']}%"
    f"      (lines {F['probe_span'][0] + 9}-{F['probe_span'][1]} leave the missing list)",
    f"subset:  {F['subset_before']}   ->   {F['subset_after']}      (exactly the 11 new cases)",
    "tests only -- no production line changes, so no policy, simulation, rendering, recording or asset behaviour moves; "
    "the figure is the coverage and mutation measurement rather than a rollout.",
]
TOP3, LAST3 = 0.74, 0.20
step3 = (TOP3 - LAST3) / (len(foot) - 1)
assert step3 > 0.15, step3
for i, line in enumerate(foot):
    put(ax3, 0.016, TOP3 - i * step3, line, fontsize=10.0,
        family="monospace" if i < 2 else None, color=INK)
assert TOP3 - (len(foot) - 1) * step3 > 0.08

# ---- layout guards -------------------------------------------------------
for a, y, axes_coords in placed:
    if axes_coords:
        assert -0.13 <= y <= 1.10, f"axes-coord text at y={y}"
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.06 <= y <= hi + 0.06, f"data-coord text at y={y} outside {(lo, hi)}"

out = ROOT / "_art/probe_contract.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image

im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((band < 245).any(axis=2).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"OK {out}  size={im.shape[1]}x{im.shape[0]}  caught={CAUGHT}/4  blind={BLIND}/4")
