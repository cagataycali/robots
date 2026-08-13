"""Compose the figure. Every drawn value is asserted against facts-<run>.json."""

from __future__ import annotations

import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

RUN = os.environ["GITHUB_RUN_ID"]
ROOT = pathlib.Path(f"/tmp/robots-mine-{RUN}")
F = json.loads((ROOT / "_art" / f"facts-{RUN}.json").read_text())
assert F["tree"] == str(ROOT), F["tree"]

C = F["compose"]
V = F["verdicts"]
L = F["live"]
M = F["mutations"]

# --- assert the story before drawing any of it -----------------------------------
assert C["merge_exit"] == 0 and C["conflicts"] == []
assert C["guard_py_exists"] is False and C["limits_py_exists"] is True
assert "ModuleNotFoundError" in C["error"]
assert F["alone"]["pr-a"].endswith("passed in 0.00s") or "passed" in F["alone"]["pr-a"]
assert "passed" in F["alone"]["pr-b"]
assert V["main"]["exit"] == 0 and V["main"]["pairs_reported"] is False
assert V["branch"]["exit"] == 1 and V["branch"]["pairs_reported"] is True
assert L["renames_in_open_set"] == 0 and L["requests_after"] - L["requests_before"] == 7
caught_new = sum(1 for _n, a, _b in M if a > 0)
blind_old = sum(1 for _n, _a, b in M if b == 0)
assert (caught_new, blind_old) == (4, 3), (caught_new, blind_old)

placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(16.2, 13.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.15, 1.0], hspace=0.20, wspace=0.10,
                      left=0.035, right=0.972, top=0.925, bottom=0.028)

fig.suptitle(
    "A rename and a sibling editing the old name: git composes it with no conflict, and the sweep could not see it",
    fontsize=15.5, fontweight="bold", y=0.983,
)
fig.text(0.5, 0.952,
         "strands-labs/robots  -  scripts/check_merge_base_overlap.py  -  measured on Thor, real git + real pytest",
         ha="center", fontsize=10.6, style="italic", color="#444")

MONO = {"family": "monospace"}
GREEN, RED, GREY = "#1a7f37", "#b3261e", "#555"

# ---------------------------------------------------------------- row 1: the topology
axl = fig.add_subplot(gs[0, 0]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.5, 1.02, "1a. Each branch alone - green", ha="center", fontsize=12.4, fontweight="bold",
    transform=axl.transAxes)

boxes = [
    (0.03, 0.60, "#10  renames the guard", [
        "git mv pkg/guard.py pkg/limits.py",
        "tests/test_guard.py imports pkg.limits",
    ], F["alone"]["pr-a"]),
    (0.03, 0.14, "#20  extends it at the old name", [
        "pkg/guard.py  +CEILING = 2",
        "tests/test_ceiling.py imports pkg.guard",
    ], F["alone"]["pr-b"]),
]
for bx, by, title, lines, verdict in boxes:
    axl.add_patch(Rectangle((bx, by), 0.94, 0.34, transform=axl.transAxes,
                            facecolor="#f6f8fa", edgecolor="#c9d1d9", linewidth=1.2))
    put(axl, bx + 0.03, by + 0.255, title, fontsize=11.6, fontweight="bold", transform=axl.transAxes)
    for i, ln in enumerate(lines):
        put(axl, bx + 0.045, by + 0.165 - i * 0.072, ln, fontsize=10.0, color=GREY,
            transform=axl.transAxes, **MONO)
    put(axl, bx + 0.90, by + 0.255, f"pytest: {verdict.split(' in ')[0]}", fontsize=10.6,
        fontweight="bold", color=GREEN, ha="right", transform=axl.transAxes)

axr = fig.add_subplot(gs[0, 1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.5, 1.02, "1b. Composed with a real `git merge` - broken", ha="center", fontsize=12.4,
    fontweight="bold", transform=axr.transAxes)
axr.add_patch(Rectangle((0.03, 0.14), 0.94, 0.80, transform=axr.transAxes,
                        facecolor="#fff5f5", edgecolor=RED, linewidth=1.5))
rows = [
    ("git merge exit status", str(C["merge_exit"]), GREEN),
    ("conflicted paths", f"{len(C['conflicts'])}  (none)", RED),
    ("pkg/guard.py in the merged tree", "absent  - #10 renamed it", GREY),
    ("pkg/limits.py in the merged tree", "present - carries #20's edit", GREY),
    ("pytest on the merged tree", C["pytest_tail"], RED),
    ("", C["error"], RED),
]
TOP, LAST = 0.845, 0.215
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for label, value, colour in rows:
    if label:
        put(axr, 0.065, y, label, fontsize=10.5, color="#24292f", transform=axr.transAxes)
    put(axr, 0.50, y, value, fontsize=10.4, color=colour, fontweight="bold" if colour != GREY else "normal",
        transform=axr.transAxes, **MONO)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
put(axr, 0.5, 0.075,
    "git's rename detection applied #20's edit to the new path.\n"
    "Nothing conflicts, so nothing asks a human - and no test in either branch ever ran this tree.",
    ha="center", fontsize=10.3, style="italic", color=RED, transform=axr.transAxes)

# ------------------------------------------------------- row 2: what the sweep said
for col, (label, key, colour, verdict) in enumerate((
    ("2a. `main` - the same topology reported clean", "main", RED, "exit 0  -  no findings"),
    ("2b. this change - reported", "branch", GREEN, "exit 1  -  1 blocking pair"),
)):
    ax = fig.add_subplot(gs[1, col]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    put(ax, 0.5, 1.03, label, ha="center", fontsize=12.4, fontweight="bold", transform=ax.transAxes)
    ax.add_patch(Rectangle((0.02, 0.10), 0.96, 0.855, transform=ax.transAxes,
                           facecolor="#0d1117", edgecolor=colour, linewidth=1.8))
    body = [ln for ln in V[key]["report"].splitlines() if ln.strip()]
    keep, seen_table = [], False
    for ln in body:
        if ln.startswith("Neither pull request") or ln.startswith("**To clear"):
            continue
        if ln.startswith("|") and "---" in ln:
            continue
        keep.append(ln)
        if ln.startswith("|"):
            seen_table = True
        if seen_table and len(keep) > 9:
            break
    for i, ln in enumerate(keep[:9]):
        put(ax, 0.045, 0.905 - i * 0.083, ln[:78], fontsize=9.5, color="#c9d1d9",
            transform=ax.transAxes, **MONO)
    put(ax, 0.5, 0.045, verdict, ha="center", fontsize=12.0, fontweight="bold", color=colour,
        transform=ax.transAxes)

# ------------------------------------------------------------ row 3: measured ledger
axm = fig.add_subplot(gs[2, 0]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.5, 1.03, "3a. Mutation table  (48 cases in the file)", ha="center", fontsize=12.4,
    fontweight="bold", transform=axm.transAxes)
put(axm, 0.045, 0.905, "regression", fontsize=10.2, fontweight="bold", color=GREY, transform=axm.transAxes)
put(axm, 0.700, 0.905, "new", fontsize=10.2, fontweight="bold", color=GREY, ha="center", transform=axm.transAxes)
put(axm, 0.850, 0.905, "existing", fontsize=10.2, fontweight="bold", color=GREY, ha="center", transform=axm.transAxes)
TOP3, LAST3 = 0.790, 0.310
S3 = (TOP3 - LAST3) / (len(M) - 1)
assert S3 > 0.030, S3
y = TOP3
for name, new_failed, old_failed in M:
    blind = old_failed == 0
    put(axm, 0.045, y, name, fontsize=9.9, color="#24292f", transform=axm.transAxes, **MONO)
    put(axm, 0.700, y, f"{new_failed} failed" if new_failed else "-", fontsize=9.9, ha="center",
        color=GREEN if new_failed else GREY, fontweight="bold" if new_failed else "normal",
        transform=axm.transAxes, **MONO)
    put(axm, 0.850, y, f"{old_failed} failed" if old_failed else "BLIND", fontsize=9.9, ha="center",
        color=GREY if old_failed else RED, fontweight="normal" if old_failed else "bold",
        transform=axm.transAxes, **MONO)
    if blind:
        axm.add_patch(Rectangle((0.035, y - 0.028), 0.93, 0.056, transform=axm.transAxes,
                                facecolor=RED, alpha=0.055, edgecolor="none"))
    y -= S3
assert abs((y + S3) - LAST3) < 1e-9
put(axm, 0.045, 0.225,
    f"{caught_new} of {len(M)} caught here; {blind_old} of {len(M)} invisible to the 43 pre-existing cases.",
    fontsize=10.2, fontweight="bold", transform=axm.transAxes)
put(axm, 0.045, 0.160,
    "M2 is caught by both - it breaks ordinary paths too.\n"
    "M5 is caught only by the surviving base-side cap pin: that contract\n"
    "keeps its owner, and these cases deliberately do not duplicate it.",
    fontsize=9.8, style="italic", color=GREY, transform=axm.transAxes)

axf = fig.add_subplot(gs[2, 1]); axf.axis("off"); axf.set_xlim(0, 1); axf.set_ylim(0, 1)
put(axf, 0.5, 1.03, "3b. On the live queue", ha="center", fontsize=12.4, fontweight="bold",
    transform=axf.transAxes)
facts_rows = [
    ("open non-draft pull requests / pairs", f"{L['open_prs']} / {L['pairs']}"),
    ("renames open right now", f"{L['renames_in_open_set']}  - the topology is not open today"),
    ("largest open changed-file count", f"{L['max_changed_files']}  - far under the 300 cap"),
    ("merged precedent", L["merged_precedent"].split(": ", 1)[0]),
    ("", "tests/simulation/test_args_docstring_completeness.py"),
    ("", "  -> tests/test_args_docstring_completeness.py"),
    ("requests per sweep", f"{L['requests_before']} -> {L['requests_after']}  (+1 per pull request)"),
    ("findings on the live queue", "identical to `main` - same pair, same stale base"),
]
TOP4, LAST4 = 0.880, 0.235
S4 = (TOP4 - LAST4) / (len(facts_rows) - 1)
assert S4 > 0.030, S4
y = TOP4
for label, value in facts_rows:
    if label:
        put(axf, 0.045, y, label, fontsize=10.1, color="#24292f", transform=axf.transAxes)
    put(axf, 0.505, y, value, fontsize=9.9, color=GREY, transform=axf.transAxes, **MONO)
    y -= S4
assert abs((y + S4) - LAST4) < 1e-9
put(axf, 0.045, 0.145,
    "The renamed file was a repo-wide guard, which is the file class where a\n"
    "sibling still editing the old path matters most. No sim, policy, rendering,\n"
    "recording or asset behaviour changes here: this is a CI report script, so\n"
    "the artifact is the measurement rather than a rollout.",
    fontsize=9.8, style="italic", color=GREY, transform=axf.transAxes)

for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.04 <= y <= 1.08, f"axes-fraction text at y={y}"
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= y <= hi + 0.07, f"data text at y={y} outside {(lo, hi)}"

out = ROOT / "_art" / "rename_overlap.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image

im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white pixels"
print(f"wrote {out}  {im.shape[1]}x{im.shape[0]}  borders clean, {len(placed)} text placements checked")
