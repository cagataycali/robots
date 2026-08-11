"""Compose the measured figure. Every number is read from art_facts.json."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.loads(pathlib.Path("/tmp/art_facts.json").read_text())
ROOT = pathlib.Path(__file__).resolve().parents[1]
assert F["tree"] == str(ROOT), f"facts came from {F['tree']}, composing in {ROOT}"

BR, MUT = F["branches"], F["mutations"]
# --- self-audit: the claims this figure makes ---
assert len(BR) == 5, len(BR)
assert all(b["missing_before"] and not b["missing_after"] for b in BR), "a branch was not uncovered->covered"
assert sum(len(b["lines"]) for b in BR) == 10, "the five branches are not 10 lines"
assert F["cov_before"]["missing"] and not F["cov_after"]["missing"]
assert len(F["cov_before"]["missing"]) == 10
assert len(MUT) == 6, len(MUT)
CAUGHT_NEW = sum(1 for m in MUT if m[3] > 0)
CAUGHT_OLD = sum(1 for m in MUT if m[5] > 0)
assert (CAUGHT_NEW, CAUGHT_OLD) == (6, 0), (CAUGHT_NEW, CAUGHT_OLD)
PREEXISTING = MUT[0][6]          # collected cases in the pre-existing arm
assert PREEXISTING == 71, PREEXISTING
assert all(m[6] == PREEXISTING for m in MUT), "the pre-existing arm ran a different set per mutation"
assert F["healthy_kind_readable"] == ["a grasp that does not move the object is empty"]

GREEN, RED, AMBER = "#1b7f3b", "#b3261e", "#8a6100"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(16.2, 11.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 1.00, 0.30], hspace=0.13,
                      left=0.014, right=0.986, top=0.945, bottom=0.022)

fig.suptitle(
    "harness_memory: the five branches that report an unusable input or store were its entire uncovered set",
    fontsize=15.5, fontweight="bold", y=0.983,
)
fig.text(0.5, 0.958,
         "Four refuse and one degrades. Each is a documented contract; none was exercised. "
         "Production diff is docstrings only.",
         ha="center", fontsize=10.6, style="italic", color="#333333")

# ---------------- row 1: the five branches ----------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965, "1.  Every branch, the reason a caller actually gets, and its coverage",
    fontsize=12.6, fontweight="bold", transform=ax.transAxes)
COLS = [0.0, 0.175, 0.400, 0.878, 0.940]
for x, h in zip(COLS, ["function", "when", "the reason it reports (measured)", "main", "this PR"]):
    put(ax, x, 0.895, h, fontsize=10.3, fontweight="bold", color="#444444", transform=ax.transAxes)
ax.plot([0.0, 1.0], [0.876, 0.876], lw=1.0, color="#999999", transform=ax.transAxes)

TOP, LAST = 0.800, 0.115
STEP = (TOP - LAST) / (len(BR) - 1)
assert STEP > 0.100, STEP
for i, b in enumerate(BR):
    y = TOP - i * STEP
    put(ax, COLS[0], y, b["fn"], fontsize=10.0, family="monospace", transform=ax.transAxes)
    put(ax, COLS[1], y, b["trigger"], fontsize=9.7, color="#333333", transform=ax.transAxes)
    reason = b["reason"]
    reason = reason if len(reason) <= 92 else reason[:89] + "..."
    put(ax, COLS[2], y, reason, fontsize=9.5, family="monospace",
        color=(AMBER if b["kind"] == "degrade" else RED), transform=ax.transAxes)
    tag = "degrades" if b["kind"] == "degrade" else "refuses"
    put(ax, COLS[2], y - 0.052, f"L{b['lines'][0]}-{b['lines'][1]}   ({tag})",
        fontsize=8.7, color="#777777", family="monospace", transform=ax.transAxes)
    put(ax, COLS[3], y, "unreached", fontsize=9.6, color=RED, fontweight="bold", transform=ax.transAxes)
    put(ax, COLS[4], y, "pinned", fontsize=9.6, color=GREEN, fontweight="bold", transform=ax.transAxes)
    if i < len(BR) - 1:
        ax.plot([0.0, 1.0], [y - 0.083, y - 0.083], lw=0.5, color="#e2e2e2", transform=ax.transAxes)
assert TOP - (len(BR) - 1) * STEP - 0.052 > 0.02

# ---------------- row 2: the mutation matrix ----------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.955,
    "2.  Six plausible regressions, run against both arms  (the production source is restored byte-identically after each)",
    fontsize=12.6, fontweight="bold", transform=ax2.transAxes)
M = [0.0, 0.560, 0.760]
for x, h in zip(M, ["mutation applied to the branch", "new tests", f"the {PREEXISTING} cases already there"]):
    put(ax2, x, 0.862, h, fontsize=10.3, fontweight="bold", color="#444444", transform=ax2.transAxes)
ax2.plot([0.0, 1.0], [0.842, 0.842], lw=1.0, color="#999999", transform=ax2.transAxes)

MT, ML = 0.755, 0.115
MS = (MT - ML) / (len(MUT) - 1)
assert MS > 0.090, MS
for i, m in enumerate(MUT):
    label, _in_fn, _in_file, nf, _np, of, _op = m
    y = MT - i * MS
    put(ax2, M[0], y, label, fontsize=9.9, family="monospace", transform=ax2.transAxes)
    put(ax2, M[1], y, f"{nf} failed", fontsize=9.9, color=GREEN, fontweight="bold", transform=ax2.transAxes)
    put(ax2, M[2], y, f"{of} failed  <- BLIND", fontsize=9.9, color=RED, fontweight="bold",
        transform=ax2.transAxes)
    ax2.add_patch(plt.Rectangle((M[2] - 0.010, y - 0.026), 0.245, 0.058, transform=ax2.transAxes,
                                facecolor=RED, alpha=0.075, zorder=0))
ax2.plot([0.0, 1.0], [ML - 0.045, ML - 0.045], lw=1.0, color="#999999", transform=ax2.transAxes)
put(ax2, M[0], ML - 0.100, "caught", fontsize=10.2, fontweight="bold", transform=ax2.transAxes)
put(ax2, M[1], ML - 0.100, f"{CAUGHT_NEW} of {len(MUT)}", fontsize=10.2, color=GREEN,
    fontweight="bold", transform=ax2.transAxes)
put(ax2, M[2], ML - 0.100, f"{CAUGHT_OLD} of {len(MUT)}", fontsize=10.2, color=RED,
    fontweight="bold", transform=ax2.transAxes)
assert ML - 0.100 > 0.005

# ---------------- row 3: footer ----------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
foot = [
    f"coverage   strands_robots/tools/harness_memory.py   {F['cov_before']['pct']}% "
    f"({len(F['cov_before']['missing'])} missing)  ->  {F['cov_after']['pct']}% (0 missing)     "
    f"same file list, the new module deselected for the before arm",
    "all-or-nothing   the readable kind returns "
    f"{F['healthy_kind_readable']!r} on its own, and load_rules still refuses -- a per-kind fallback",
    "                 would present a store that could not be read as a kind with no rules. Now stated in "
    "load_rules' and append_rule's Raises: entries.",
    "no artifact rollout   nothing in policy, simulation, rendering, recording or asset handling changes; "
    "the production diff is docstrings only and the",
    "                      docstring-stripped AST digest is unchanged at 819de46739e52f52. This figure is "
    "the coverage and mutation measurement.",
]
FT, FL = 0.90, 0.06
FS = (FT - FL) / (len(foot) - 1)
assert FS > 0.150, FS
for i, line in enumerate(foot):
    put(ax3, 0.0, FT - i * FS, line, fontsize=9.6, family="monospace", color="#222222",
        transform=ax3.transAxes)
assert FT - (len(foot) - 1) * FS > 0.03

# ---------------- layout guards ----------------
for a, y, is_axes in placed:
    if is_axes:
        assert -0.03 <= y <= 1.10, f"axes-fraction y={y} out of band"
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= y <= hi + 0.07, f"data y={y} outside {(lo, hi)}"

OUT = pathlib.Path("/tmp/harness_memory_unusable_inputs.png")
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.array(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {im.shape[1]}x{im.shape[0]}  texts={len(placed)}  borders clean")
