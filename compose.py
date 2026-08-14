"""Compose the docstring-sweep figure from the measured facts."""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.loads(pathlib.Path(sys.argv[1]).read_text())
OUT = pathlib.Path(sys.argv[2])

# --- self-audit on the measurements ------------------------------------------
assert F["markers_after"] == 0, F["markers_after"]
assert F["markers_before"] == 43, F["markers_before"]
assert F["digests_equal"] == F["n_touched"] == 15, (F["digests_equal"], F["n_touched"])
assert all(r["ast_identical"] and r["text_differs"] for r in F["rows"]), "every file: text differs, AST identical"
assert F["guard_on_base"]["failed"] == 9 and F["guard_on_head"]["failed"] == 0
assert F["pkg_markers_before"] == 50 and F["pkg_markers_after"] == 7

PLACED: list = []


def put(ax, x, y, s, **kw):
    PLACED.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(16.2, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 1.0, 0.30], hspace=0.20, wspace=0.13)

fig.suptitle(
    "Review-history markers removed from shipped docstrings  --  43 -> 0 across 15 modules, every AST digest unchanged",
    fontsize=15.5,
    fontweight="bold",
    y=0.975,
)

# --- row 1: what a reader of the docstring actually sees ---------------------
BEFORE = [
    ("simulation/models.py  (public dataclass field)", "``mesh`` / ``peer_id`` (post-PR #101): when the parent\n``Simulation`` is a mesh peer, ..."),
    ("benchmarks/libero/adapter.py", "The reviewer's variant-B keeps the bind state\nprocess-shared ..."),
    ("mesh/security.py", "#: Symmetric with :data:`_POLICY_HOST_ENTRY_RE` so a\n#: reviewer reading this can ..."),
    ("simulation/models.py  (comment)", "# Physics state checkpoints (used by save_state in PR #85).\n# Kept top-level - requested by @yinsong1986 during review"),
]
AFTER = [
    ("simulation/models.py  (public dataclass field)", "``mesh`` / ``peer_id``: populated when the parent\n``Simulation`` is a mesh peer, ..."),
    ("benchmarks/libero/adapter.py", "Keeping the bind state process-shared lets the\nrenderer reuse one context ..."),
    ("mesh/security.py", "#: Symmetric with :data:`_POLICY_HOST_ENTRY_RE` so both\n#: forms accept the same host spellings ..."),
    ("simulation/models.py  (comment)", "# Physics state checkpoints, read by save_state /\n# restore_state. Top-level so a checkpoint survives ..."),
]

for col, (title, rows, tint) in enumerate(
    [("main: docstring carries review history", BEFORE, "#fdecea"), ("this change: docstring carries the reason", AFTER, "#e8f5e9")]
):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=tint, zorder=0))
    put(ax, 0.5, 1.035, title, transform=ax.transAxes, ha="center", fontsize=12.6, fontweight="bold")
    TOP, LAST = 0.905, 0.075
    step = (TOP - LAST) / (len(rows) - 1)
    assert step > 0.10, step
    y = TOP
    for name, body in rows:
        put(ax, 0.035, y, name, transform=ax.transAxes, fontsize=9.4, fontweight="bold", color="#37474f")
        put(ax, 0.045, y - 0.052, body, transform=ax.transAxes, fontsize=9.0, family="monospace", va="top")
        y -= step
    assert abs((y + step) - LAST) < 1e-9, (y, step, LAST)

# --- row 2 left: AST identity per module ------------------------------------
axd = fig.add_subplot(gs[1, 0])
axd.set_xlim(0, 1)
axd.set_ylim(0, 1)
axd.axis("off")
put(
    axd,
    0.5,
    1.035,
    "Docstring-stripped AST digest: identical for all 15 modules\n(so no executable line moved)",
    transform=axd.transAxes,
    ha="center",
    fontsize=12.2,
    fontweight="bold",
)
rows = F["rows"][:11]
TOP, LAST = 0.905, 0.115
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
put(axd, 0.035, 0.965, "module", transform=axd.transAxes, fontsize=9.0, fontweight="bold", color="#546e7a")
put(axd, 0.605, 0.965, "markers", transform=axd.transAxes, fontsize=9.0, fontweight="bold", color="#546e7a")
put(axd, 0.755, 0.965, "AST digest (before = after)", transform=axd.transAxes, fontsize=9.0, fontweight="bold", color="#546e7a")
y = TOP
for r in rows:
    put(axd, 0.035, y, r["path"], transform=axd.transAxes, fontsize=8.8, family="monospace")
    put(axd, 0.615, y, f"{r['markers_before']} -> 0", transform=axd.transAxes, fontsize=8.8, family="monospace", color="#2e7d32")
    put(axd, 0.755, y, r["digest_before"], transform=axd.transAxes, fontsize=8.8, family="monospace", color="#1565c0")
    put(axd, 0.945, y, "OK", transform=axd.transAxes, fontsize=8.8, fontweight="bold", color="#2e7d32")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, step, LAST)
put(
    axd,
    0.035,
    0.040,
    f"+ {F['n_touched'] - len(rows)} more modules, same result   |   text differs in 15/15, AST identical in {F['digests_equal']}/15",
    transform=axd.transAxes,
    fontsize=9.2,
    style="italic",
    color="#455a64",
)

# --- row 2 right: guard verdict + what deliberately stays -------------------
axg = fig.add_subplot(gs[1, 1])
axg.set_xlim(0, 1)
axg.set_ylim(0, 1)
axg.axis("off")
put(axg, 0.5, 1.035, "The guard, and what deliberately stays", transform=axg.transAxes, ha="center", fontsize=12.2, fontweight="bold")
lines = [
    ("tests/test_source_no_review_history_markers.py", "", "#37474f", True),
    ("  on main's source", f"{F['guard_on_base']['failed']} failed / {F['guard_on_base']['passed']} passed", "#c62828", False),
    ("  on this change", f"{F['guard_on_head']['failed']} failed / {F['guard_on_head']['passed']} passed", "#2e7d32", False),
    ("  first names", "adapter.py:1418, :1667, :1735, ...", "#c62828", False),
    ("", "", "#000000", False),
    ("Package-wide marker census", f"{F['pkg_markers_before']} -> {F['pkg_markers_after']}", "#37474f", True),
    ("  3x  AGENTS.md > Review Learnings (PR #92)", "section exists, L1290", "#1565c0", False),
    ("  3x  lerobot PR #3604", "declared dependency", "#1565c0", False),
    ("  1x  \"preview's frame period\"", "an ordinary word", "#1565c0", False),
    ("", "", "#000000", False),
    ("A reference a reader can follow is kept.", "", "#2e7d32", True),
    ("A name only the review thread knew is not.", "", "#2e7d32", True),
]
TOP, LAST = 0.925, 0.055
step = (TOP - LAST) / (len(lines) - 1)
assert step > 0.030, step
y = TOP
for label, value, colour, bold in lines:
    if label:
        put(axg, 0.035, y, label, transform=axg.transAxes, fontsize=9.4, family="monospace" if label.startswith(" ") else None, fontweight="bold" if bold else None, color=colour)
    if value:
        put(axg, 0.585, y, value, transform=axg.transAxes, fontsize=9.4, family="monospace", color=colour)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, step, LAST)

# --- row 3: gate ------------------------------------------------------------
axf = fig.add_subplot(gs[2, :])
axf.set_xlim(0, 1)
axf.set_ylim(0, 1)
axf.axis("off")
axf.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axf.transAxes, facecolor="#eceff1", zorder=0))
foot = [
    f"Gate at base {F['base_sha']} (head {F['head_sha']}):  full suite 29813 passed / 266 skipped / 0 failed (714s)  |  ruff clean 1221 files  |  mypy 0 non-examples errors",
    "Docstrings and comments only.  No policy, simulation, rendering, recording, dataset or asset behaviour changes -- the AST digests above are the mechanical proof.",
    "Prose is the deliverable, so the artifact is the reader-facing diff plus the identity proof rather than a rollout.",
]
TOP, LAST = 0.76, 0.20
step = (TOP - LAST) / (len(foot) - 1)
y = TOP
for line in foot:
    put(axf, 0.014, y, line, transform=axf.transAxes, fontsize=9.6, family="monospace", color="#263238")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, step, LAST)

for ax, y, is_axes in PLACED:
    if is_axes:
        assert -0.05 <= y <= 1.08, (y, "axes-fraction out of range")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.03 <= y <= hi + 0.06, (y, lo, hi)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print(f"OK {OUT} {Image.open(OUT).size}")
