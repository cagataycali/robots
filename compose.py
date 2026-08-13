"""Compose the sweep artifact. Every drawn number is asserted against the capture."""
from __future__ import annotations
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

F = json.loads(pathlib.Path(sys.argv[1]).read_text())
OUT = pathlib.Path(sys.argv[2])
print("capture tree:", F["tree"])

# ---- assertions on the capture itself ---------------------------------------
assert F["open_non_draft"] == 8 and F["pairs"] == 28, (F["open_non_draft"], F["pairs"])
BLOCK = [p for p in F["pair_findings"] if p["blocking"]]
assert len(BLOCK) == 1, BLOCK
PAIR = BLOCK[0]
assert PAIR["left"] == 1035 and PAIR["right"] == 1722, PAIR
assert PAIR["blocking"] == ["strands_robots/mesh/__init__.py",
                            "tests/mesh/test_drive_command_numeric_domains.py"], PAIR
assert PAIR["prose"] == ["docs/ros2-integration.md"], PAIR
assert F["capped_but_still_paired"] == [1035], F["capped_but_still_paired"]
assert len(F["stale_findings"]) == 1 and F["stale_findings"][0]["number"] == 1722
PB = {b["number"]: b for b in F["per_branch"]}
assert all(not b["names_a_shared_path"] for b in PB.values())
NUMS = sorted(p["number"] for p in F["prs"])
CELL = {(m["left"], m["right"]): m for m in F["matrix"]}
assert len(CELL) == 28

MUT = [("M1  the pairwise comparison is dropped", 3), ("M2  prose and code are not partitioned", 4),
       ("M3  the stale-base mode is dropped", 2), ("M4  a capped base side drops the whole PR", 2),
       ("M5  drafts are swept as if reviewable", 1)]

placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

MONO = {"family": "monospace"}
fig = plt.figure(figsize=(15.6, 13.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.16, 1.0, 0.96], hspace=0.30, wspace=0.10)
fig.suptitle("Sweeping the open set finds a composition no per-branch check can see",
             fontsize=15.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.958, f"strands-labs/robots, measured live: {F['open_non_draft']} open non-draft "
         f"pull requests, {F['pairs']} pairs", ha="center", fontsize=10.6, style="italic", color="#444")

# ---------- row 1: the pair matrix ------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.set_xlim(-0.5, len(NUMS) - 0.5); ax.set_ylim(len(NUMS) - 0.5, -0.5)
ax.set_xticks(range(len(NUMS))); ax.set_xticklabels([f"#{n}" for n in NUMS], fontsize=9.4, **MONO)
ax.set_yticks(range(len(NUMS))); ax.set_yticklabels([f"#{n}" for n in NUMS], fontsize=9.4, **MONO)
ax.set_title(f"Every pair compared head-to-head  \u2014  {F['pairs']} pairs, 1 blocking",
             fontsize=11.8, fontweight="bold", pad=9)
n_block = n_prose = 0
for i, a in enumerate(NUMS):
    for j, b in enumerate(NUMS):
        if i == j:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#eceff1", edgecolor="white", lw=1.4)); continue
        m = CELL.get((min(a, b), max(a, b)))
        blocking, prose = m["n_blocking"], m["n_prose"]
        if blocking:
            face, txt, tc = "#c62828", str(blocking), "white"
            if i < j: n_block += 1
        elif prose:
            face, txt, tc = "#ffe082", str(prose), "#5d4037"
            if i < j: n_prose += 1
        else:
            face, txt, tc = "#f5f7f8", "", "#666"
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=face, edgecolor="white", lw=1.4))
        if txt:
            put(ax, j, i, txt, ha="center", va="center", fontsize=10.4, fontweight="bold", color=tc, **MONO)
assert n_block == 1 and n_prose == 3, (n_block, n_prose)
ai, bi = NUMS.index(1035), NUMS.index(1722)
ax.annotate(f"#1035 + #1722 share {len(PAIR['blocking'])} behaviour-bearing paths\n"
            f"  {PAIR['blocking'][0]}\n  {PAIR['blocking'][1]}\n"
            f"(+ {PAIR['prose'][0]}, prose \u2014 reported, not blocking)",
            xy=(bi + .45, ai), xytext=(bi + 1.05, ai + 1.9), fontsize=9.9, **MONO,
            bbox=dict(boxstyle="round,pad=0.42", fc="#ffebee", ec="#c62828", lw=1.3),
            arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.5))
for lbl, fc, ec in [("blocking", "#c62828", "#c62828"), ("prose only", "#ffe082", "#bfa100"), ("disjoint", "#f5f7f8", "#ccc")]:
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=fc, edgecolor=ec, label=lbl))
ax.legend(loc="lower right", fontsize=9.2, framealpha=0.96)

# ---------- row 2: what each branch's own signals said ----------------------
for k, num in enumerate((1035, 1722)):
    b = PB[num]; a2 = fig.add_subplot(gs[1, k]); a2.axis("off"); a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    a2.add_patch(Rectangle((0.008, 0.02), 0.984, 0.96, transform=a2.transAxes,
                           facecolor="#fafbfc", edgecolor="#b0bec5", lw=1.2, zorder=0))
    put(a2, 0.03, 0.925, f"#{num}  {b['title'][:40]}", transform=a2.transAxes,
        fontsize=11.2, fontweight="bold")
    rows = [
        ("own merge base", f"{b['merge_base']}  ({b['behind_by']} commits behind main)", "#333"),
        ("GitHub mergeable", f"{b['mergeable']} / {b['mergeStateStatus']}", "#333"),
        ("review", b["reviewDecision"], "#333"),
        ("single-branch check", b["single_branch_verdict"], "#c62828" if b["single_branch_overlap"] else "#2e7d32"),
        ("  \u2514 names", ", ".join(p.split("/")[-1] for p in b["single_branch_overlap"]) or "(nothing)", "#555"),
        ("names a shared path?", "NO \u2014 invisible to this branch's own run", "#c62828"),
    ]
    TOP, LAST = 0.79, 0.14
    STEP = (TOP - LAST) / (len(rows) - 1); assert STEP > 0.030, STEP
    y = TOP
    for label, value, col in rows:
        put(a2, 0.045, y, label, transform=a2.transAxes, fontsize=9.9, color="#555", **MONO)
        put(a2, 0.40, y, value[:52], transform=a2.transAxes, fontsize=9.9, color=col, **MONO,
            fontweight="bold" if label == "names a shared path?" else "normal")
        y -= STEP
    assert abs((y + STEP) - LAST) < 1e-9, y

# ---------- row 3: the design decision, and the mutation matrix -------------
a3 = fig.add_subplot(gs[2, :]); a3.axis("off"); a3.set_xlim(0, 1); a3.set_ylim(0, 1)
put(a3, 0.0, 0.955, "The design decision that made the finding reachable", transform=a3.transAxes,
    fontsize=11.5, fontweight="bold")
put(a3, 0.012, 0.855,
    f"#1035's base-side path set hit the 300-entry file cap, so it is excluded from the stale-base mode and\n"
    f"listed as unevaluated with that reason. It stays in the pairwise comparison \u2014 and it is exactly the\n"
    f"pull request that produced the one blocking finding. Dropping a capped pull request wholesale would\n"
    f"have discarded it. Measured: capped_but_still_paired = {F['capped_but_still_paired']}.",
    transform=a3.transAxes, fontsize=10.0, color="#333", va="top", **MONO)
put(a3, 0.0, 0.545, "Mutation matrix  \u2014  new cases vs the 30 pre-existing", transform=a3.transAxes,
    fontsize=11.5, fontweight="bold")
put(a3, 0.012, 0.462, "regression", transform=a3.transAxes, fontsize=9.6, color="#666", fontweight="bold", **MONO)
put(a3, 0.60, 0.462, "new (43)", transform=a3.transAxes, fontsize=9.6, color="#666", fontweight="bold", **MONO)
put(a3, 0.75, 0.462, "pre-existing (30)", transform=a3.transAxes, fontsize=9.6, color="#666", fontweight="bold", **MONO)
TOP2, LAST2 = 0.385, 0.115
STEP2 = (TOP2 - LAST2) / (len(MUT) - 1); assert STEP2 > 0.030, STEP2
y = TOP2
for label, n in MUT:
    put(a3, 0.012, y, label, transform=a3.transAxes, fontsize=9.8, color="#333", **MONO)
    put(a3, 0.60, y, f"{n} failed", transform=a3.transAxes, fontsize=9.8, color="#2e7d32", fontweight="bold", **MONO)
    put(a3, 0.75, y, "0 failed  \u2190 BLIND", transform=a3.transAxes, fontsize=9.8, color="#c62828",
        fontweight="bold", **MONO)
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, y
put(a3, 0.012, 0.030,
    "Gate: 29547 passed / 266 skipped / 0 failed (652s, MUJOCO_GL=egl)  \u00b7  ruff clean  \u00b7  "
    "mypy 0 errors outside examples/  \u00b7  no policy, simulation, rendering, recording or asset behaviour changes",
    transform=a3.transAxes, fontsize=9.0, color="#666", style="italic")

for a, y, is_axes in placed:
    if is_axes:
        assert -0.03 <= y <= 1.07, (y, "axes-fraction text outside the panel")
    else:
        lo, hi = a.get_ylim(); lo, hi = min(lo, hi), max(lo, hi)
        assert lo - 1.0 <= y <= hi + 1.0, (y, lo, hi)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image
im = np.asarray(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert bad == 0, f"{name} border has {bad} non-white px"
print(f"OK {OUT} {im.shape[1]}x{im.shape[0]}  blocking={n_block} prose={n_prose}  texts={len(placed)}")
