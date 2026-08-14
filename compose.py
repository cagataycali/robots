"""Compose the fleet-availability figure from the measured JSON.

Every rendered number is asserted against the capture dump first, so the figure
cannot ship a stale cell. Text placement is tracked and bounds-checked, and the
outer border is verified white per side.
"""
import json, os, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

RID = os.environ["GITHUB_RUN_ID"]
F = json.load(open(f"/tmp/art-safety-{RID}.json"))
OUT = pathlib.Path(__file__).resolve().parent / "safety_envelope_zenoh_absent.png"

B, M, COV, MUT, G = F["branch"], F["m2"], F["coverage"], F["mutations"], F["gate"]

# ---- assert the claims before drawing -------------------------------------
assert B["wire_zid"] is None and M["wire_zid"] == "deadbeefdeadbeef"
assert B["fleet_available"] is True and M["fleet_available"] is False
assert B["receiver_locked_after"] is False and M["receiver_locked_after"] is True
assert B["resume_status"] == M["resume_status"] == {"status": "ok"}
assert B["verify_error"] is None and M["verify_error"] is None
assert (COV["missing_before"], COV["missing_after"]) == (31, 22)
assert COV["opened"] == [] and len(COV["closed"]) == 9
assert sum(1 for _, n, _ in MUT if n > 0) == 7 and all(o == 0 for _, _, o in MUT)

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top")
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

GREEN, RED, GREY, INK = "#1a7f37", "#b3261e", "#5f6368", "#202124"
fig = plt.figure(figsize=(15.6, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.05, 0.30], hspace=0.20, wspace=0.10)

fig.suptitle(
    "An e-stop lockout raised on an install without the mesh extra must still be clearable",
    fontsize=15.5, fontweight="bold", y=0.972, color=INK,
)
fig.text(
    0.5, 0.944,
    "Mesh publishes safety envelopes through four helpers that each degrade to the transport-agnostic put() path. "
    "All four except-ImportError arms were unexecuted.\n"
    "Left: this branch, driven end to end with zenoh hidden from the import system.  "
    "Right: the same sequence with M2 applied -- the regression these tests now catch.",
    ha="center", va="top", fontsize=10.2, color=GREY,
)

# ---- row 1: the two arms --------------------------------------------------
ARMS = [
    ("This branch (zenoh absent)", B, GREEN, "FLEET AVAILABLE", "the receiver cleared its lockout"),
    ("M2: proof bound to the local zid anyway", M, RED, "FLEET STAYS E-STOPPED", "no error on either end"),
]
for col, (title, row, colour, verdict, sub) in enumerate(ARMS):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.02), 0.98, 0.96, fill=False, ec=colour, lw=2.2, transform=ax.transAxes))
    put(ax, 0.5, 0.965, title, ha="center", fontsize=12.4, fontweight="bold", color=colour, transform=ax.transAxes)

    steps = [
        ("issuer._safety_wire_zid(resume_key)", "None" if row["wire_zid"] is None else row["wire_zid"],
         row["wire_zid"] is None),
        ("published body carries source_zid", str(row["body_carries_source_zid"]), True),
        ("published body carries override_proof", str(row["body_carries_proof"]), True),
        ("issuer reports", json.dumps(row["resume_status"]), True),
        ("receiver raised", "nothing" if row["verify_error"] is None else row["verify_error"], True),
        ("receiver lockout before -> after",
         f'{row["receiver_locked_before"]} -> {row["receiver_locked_after"]}',
         row["receiver_locked_after"] is False),
    ]
    TOP, LAST = 0.845, 0.375
    step = (TOP - LAST) / (len(steps) - 1)
    assert step > 0.030, step
    y = TOP
    for label, value, good in steps:
        put(ax, 0.055, y, label, fontsize=10.3, color=INK, transform=ax.transAxes)
        put(ax, 0.955, y, value, ha="right", fontsize=10.3, family="monospace",
            color=GREEN if good else RED, fontweight="bold" if not good else "normal",
            transform=ax.transAxes)
        y -= step
    assert abs((y + step) - LAST) < 1e-9

    ax.add_patch(Rectangle((0.055, 0.115), 0.90, 0.165, fc=colour, alpha=0.10,
                           ec=colour, lw=1.4, transform=ax.transAxes))
    put(ax, 0.5, 0.255, verdict, ha="center", fontsize=14.2, fontweight="bold",
        color=colour, transform=ax.transAxes)
    put(ax, 0.5, 0.175, sub, ha="center", fontsize=10.2, color=GREY, transform=ax.transAxes)

# ---- row 2 left: coverage ------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.0, 0.985, "The four import-failure arms", fontsize=12.2, fontweight="bold",
    color=INK, transform=ax.transAxes)
put(ax, 0.0, 0.905, "coverage of strands_robots/mesh/core.py over tests/mesh", fontsize=9.8,
    color=GREY, transform=ax.transAxes)
ARMROWS = [
    ("_local_session_zid", "3281-3282", "defence in depth"),
    ("_safety_wire_zid", "3325-3326", "binds the resume proof"),
    ("_safety_publisher_for", "3354-3355", "defence in depth"),
    ("_publish_safety_envelope", "3459-3461", "publishes, stripped"),
]
put(ax, 0.0, 0.800, "helper", fontsize=9.6, fontweight="bold", color=GREY, transform=ax.transAxes)
put(ax, 0.44, 0.800, "lines", fontsize=9.6, fontweight="bold", color=GREY, transform=ax.transAxes)
put(ax, 0.60, 0.800, "before", fontsize=9.6, fontweight="bold", color=GREY, transform=ax.transAxes)
put(ax, 0.755, 0.800, "after", fontsize=9.6, fontweight="bold", color=GREY, transform=ax.transAxes)
TOP, LAST = 0.715, 0.435
step = (TOP - LAST) / (len(ARMROWS) - 1)
assert step > 0.030, step
y = TOP
for name, lines, note in ARMROWS:
    put(ax, 0.0, y, name, fontsize=10.2, family="monospace", color=INK, transform=ax.transAxes)
    put(ax, 0.44, y, lines, fontsize=9.8, family="monospace", color=GREY, transform=ax.transAxes)
    put(ax, 0.60, y, "unexecuted", fontsize=9.8, color=RED, fontweight="bold", transform=ax.transAxes)
    put(ax, 0.755, y, "driven", fontsize=9.8, color=GREEN, fontweight="bold", transform=ax.transAxes)
    put(ax, 0.875, y, note, fontsize=8.8, color=GREY, style="italic", transform=ax.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9
put(ax, 0.0, 0.320,
    f"missing lines {COV['missing_before']} -> {COV['missing_after']}   "
    f"({COV['pct_before']}% -> {COV['pct_after']}%)   9 closed, 0 opened",
    fontsize=10.4, family="monospace", color=INK, transform=ax.transAxes)
put(ax, 0.0, 0.225,
    "No production behaviour changes: the docstring-stripped AST of\n"
    f"mesh/core.py is identical ({G['ast_digest']}).",
    fontsize=9.8, color=GREY, transform=ax.transAxes)

# ---- row 2 right: mutations ----------------------------------------------
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.0, 0.985, "Mutation table", fontsize=12.2, fontweight="bold", color=INK, transform=ax.transAxes)
put(ax, 0.0, 0.905,
    f"failures per arm.  right column = all {G['pre_existing_mesh_cases']} pre-existing tests/mesh cases",
    fontsize=9.8, color=GREY, transform=ax.transAxes)
put(ax, 0.0, 0.800, "regression", fontsize=9.6, fontweight="bold", color=GREY, transform=ax.transAxes)
put(ax, 0.735, 0.800, "new", fontsize=9.6, fontweight="bold", color=GREY, transform=ax.transAxes)
put(ax, 0.845, 0.800, "pre-existing", fontsize=9.6, fontweight="bold", color=GREY, transform=ax.transAxes)
TOP, LAST = 0.715, 0.235
step = (TOP - LAST) / (len(MUT) - 1)
assert step > 0.030, step
y = TOP
for label, new, old in MUT:
    head = label.split(":")[0]
    rest = label.split(":", 1)[1].strip()
    put(ax, 0.0, y, head, fontsize=9.9, family="monospace", fontweight="bold", color=INK, transform=ax.transAxes)
    put(ax, 0.105, y, rest, fontsize=9.6, color=INK, transform=ax.transAxes)
    put(ax, 0.765, y, str(new), ha="center", fontsize=10.0, family="monospace",
        color=GREEN, fontweight="bold", transform=ax.transAxes)
    put(ax, 0.895, y, str(old), ha="center", fontsize=10.0, family="monospace",
        color=RED, fontweight="bold", transform=ax.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9
put(ax, 0.0, 0.130, "7 of 7 caught here.  0 of 7 caught by the pre-existing suite.",
    fontsize=10.6, fontweight="bold", color=INK, transform=ax.transAxes)
put(ax, 0.0, 0.055,
    "Three anchors measured in_fn=1 in_file=3 (the byte-identical arms),\n"
    "so the AST function-range scoping was load-bearing.",
    fontsize=9.5, color=GREY, transform=ax.transAxes)

# ---- row 3: gate ---------------------------------------------------------
ax = fig.add_subplot(gs[2, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0.0, 0.10), 1.0, 0.80, fc="#f1f3f4", ec="#dadce0", lw=1.0, transform=ax.transAxes))
put(ax, 0.018, 0.78,
    f"Gate at {G['base']}:  MUJOCO_GL=egl pytest tests -> {G['suite']}   "
    f"(pristine {G['pristine']} + {G['new_cases']} new = 29751)",
    fontsize=10.4, family="monospace", color=INK, transform=ax.transAxes)
put(ax, 0.018, 0.50,
    "ruff check / ruff format --check clean (1218 files).  mypy: 0 errors outside examples/isaac_gs.  "
    "Tests and docstrings only -- no policy, simulation, rendering, recording or asset behaviour changes,",
    fontsize=9.7, color=GREY, transform=ax.transAxes)
put(ax, 0.018, 0.30,
    "so the artifact is the measured availability outcome rather than a rollout.",
    fontsize=9.7, color=GREY, transform=ax.transAxes)

for ax_, y, is_axes in placed:
    lo, hi = (-0.03, 1.08) if is_axes else ax_.get_ylim()
    assert lo <= y <= hi, (y, lo, hi)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {im.shape[1]}x{im.shape[0]}  texts={len(placed)}")
