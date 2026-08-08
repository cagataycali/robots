"""Compose the intake-repository figure from the two captured JSON dumps.

Every cell is re-derived from the dumps; the generator asserts each claim it
renders, that the two dumps came from different trees, and that the figure has a
clean border, so a stale panel cannot ship.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MAIN = json.loads(Path("/tmp/art_main.json").read_text(encoding="utf-8"))
BRANCH = json.loads(Path("/tmp/art_branch.json").read_text(encoding="utf-8"))
assert MAIN["tree"] != BRANCH["tree"], (MAIN["tree"], BRANCH["tree"])

TARGET = "strands-labs/robots"
GREEN, RED, GREY = "#1a7f37", "#b3261e", "#57606a"

# ------------------------------------------------------------------ claims
m, b = MAIN["rows"], BRANCH["rows"]
assert MAIN["documented_argv"] == ["--issue", "2029"], MAIN["documented_argv"]
assert BRANCH["documented_argv"] == ["--repo", TARGET, "--issue", "2029"], BRANCH["documented_argv"]
assert m["documented"]["subject"] == "huggingface/lerobot#2029" and m["documented"]["compared"] == 405
assert b["documented"]["subject"] == f"{TARGET}#2029" and b["documented"]["compared"] == 4
assert m["inferred_intake"]["exit"] == 0 and not m["inferred_intake"]["refused_inference"]
assert b["inferred_intake"]["exit"] == 2 and b["inferred_intake"]["refused_inference"]
for unchanged in ("explicit_intake", "review_mode", "nothing_to_infer"):
    left = {k: v for k, v in m[unchanged].items() if k != "blob"}
    right = {k: v for k, v in b[unchanged].items() if k != "blob"}
    assert left == right, (unchanged, left, right)

ROWS = (
    ("the command AGENTS.md step 1 prints", "documented", True),
    ("--issue 2029  (repository inferred)", "inferred_intake", True),
    ("--repo strands-labs/robots --issue 2029", "explicit_intake", False),
    ("--pr 2028  (repository inferred)", "review_mode", False),
    ("--issue 2029  (nothing to infer)", "nothing_to_infer", False),
)


def cell(row: dict[str, object]) -> tuple[str, str, bool]:
    """Return (text, colour, whether it read the intended repository)."""
    if row["refused_inference"]:
        return "exit 2  refused: name the repository", GREEN, True
    if row["required_flag"]:
        return "exit 2  --repo is required", GREY, True
    subject = str(row["subject"])
    right = subject.startswith(TARGET)
    compared = f"  {row['compared']} compared" if row["compared"] is not None else ""
    return f"exit {row['exit']}  {row['outcome']}  {subject}{compared}", GREEN if right else RED, right


wrong_before = sum(1 for _, key, _ in ROWS if not cell(m[key])[2])
wrong_after = sum(1 for _, key, _ in ROWS if not cell(b[key])[2])
assert (wrong_before, wrong_after) == (2, 0), (wrong_before, wrong_after)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(16.0, 9.6), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.00], hspace=0.16, wspace=0.05,
                      left=0.012, right=0.988, top=0.905, bottom=0.028)
placed: list[tuple[plt.Axes, float]] = []


def put(ax: plt.Axes, x: float, y: float, s: str, **kw: object) -> None:
    placed.append((ax, y))
    ax.text(x, y, s, transform=ax.transAxes, **kw)  # type: ignore[arg-type]


fig.suptitle(
    "Duplicate-claim intake: the repository the check reads, measured on both trees",
    fontsize=17, fontweight="bold", y=0.972,
)
fig.text(
    0.5, 0.928,
    f"ambient $GITHUB_REPOSITORY = huggingface/lerobot   |   target = {TARGET}   |   "
    f"main {MAIN['head']}  vs  this branch {BRANCH['head']}",
    ha="center", fontsize=11.5, color="#444444",
)

grid = fig.add_subplot(gs[0, :])
grid.set_xlim(0, 1); grid.set_ylim(0, 1); grid.axis("off")
X_INV, X_MAIN, X_BR = 0.008, 0.352, 0.672
put(grid, X_INV, 0.945, "invocation", fontsize=12.5, fontweight="bold")
put(grid, X_MAIN, 0.945, f"main  {MAIN['head']}", fontsize=12.5, fontweight="bold")
put(grid, X_BR, 0.945, f"this branch  {BRANCH['head']}", fontsize=12.5, fontweight="bold")
grid.plot([0.004, 0.996], [0.915, 0.915], color="#333333", lw=1.3, transform=grid.transAxes, clip_on=False)

TOP, FLOOR, PAD = 0.855, 0.055, 0.030
STEP = (TOP - FLOOR - PAD * len(ROWS)) / len(ROWS)
assert STEP > 0.030, STEP
y = TOP
for label, key, headline in ROWS:
    if headline:
        grid.add_patch(Rectangle((0.002, y - STEP + 0.006), 0.996, STEP + 0.020, transform=grid.transAxes,
                                 facecolor="#fff8e1", edgecolor="none", zorder=0))
    put(grid, X_INV, y - 0.012, label, fontsize=11.4, family="monospace",
        fontweight="bold" if headline else "normal")
    for x, row in ((X_MAIN, m[key]), (X_BR, b[key])):
        text, colour, _ok = cell(row)
        put(grid, x, y - 0.012, text, fontsize=11.4, family="monospace", color=colour,
            fontweight="bold" if headline else "normal")
    y -= STEP + PAD
assert y > 0.030, y
put(grid, X_INV, 0.012,
    f"read the intended repository:  main {len(ROWS) - wrong_before} of {len(ROWS)}"
    f"     this branch {len(ROWS) - wrong_after} of {len(ROWS)}",
    fontsize=12.0, fontweight="bold")

for col, (title, blob, colour) in enumerate((
    (f"main {MAIN['head']}: the documented command's report", m["documented"]["blob"], RED),
    (f"this branch {BRANCH['head']}: the same documented command", b["documented"]["blob"], GREEN),
)):
    ax = fig.add_subplot(gs[1, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.012, 0.02), 0.976, 0.86, transform=ax.transAxes,
                           facecolor="#f6f8fa", edgecolor=colour, lw=1.6))
    put(ax, 0.012, 0.935, title, fontsize=12.2, fontweight="bold", color=colour)
    put(ax, 0.032, 0.815, str(blob), fontsize=10.4, family="monospace", va="top", color="#24292f")

fig.text(0.5, 0.004,
         "Both reports say unique-claim and exit 0, so on main the wrong answer is shaped exactly like the right one.",
         ha="center", fontsize=11.0, style="italic", color="#444444")

for ax, yv in placed:
    lo, hi = ax.get_ylim()
    assert lo - 0.05 <= yv <= hi + 0.09, (yv, lo, hi)

out = Path(sys.argv[1])
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert bad == 0, (name, bad)
print(f"WROTE {out}  {im.shape[1]}x{im.shape[0]}  border clean  divergences {wrong_before} -> {wrong_after}")
