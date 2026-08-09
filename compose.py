"""Compose the measured verdict figure from the two capture dumps."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(Path("/tmp/art_main.json").read_text())   # upstream/main
B = json.loads(Path("/tmp/art_branch.json").read_text())  # this change
assert A["tree"] != B["tree"], (A["tree"], B["tree"])

GREEN, RED, INK, MUTE = "#1b7f3b", "#b3261e", "#101418", "#5b6670"
BAND_OK, BAND_BAD = "#e8f5ec", "#fdeceb"

placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.pop("axes_coords", False)
    if axes_coords:
        kw["transform"] = ax.transAxes
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


# ---- verdict classification -------------------------------------------------
DECLARED = {"True": True, "False": False}


def estop_ok(row):
    """True when the outcome honours the flag's declared posture."""
    lab = row["label"]
    if lab in DECLARED:
        return row["verdict"] == "provisioned" and row["grants_estop"] is DECLARED[lab]
    # every other spelling is not a boolean: the only honest outcome is a
    # refusal that provisions nothing
    return row["verdict"] == "refused" and row["touched"] is False


def boot_ok(row):
    lab = row["label"]
    if lab == "confirm=True, dry_run=False":
        return row["entered_create"] is True
    if lab == "dry_run=True (default)":
        return row["verdict"] == "previewed"
    if lab == "confirm=False, dry_run=False":
        return row["verdict"] == "refused"
    return row["verdict"] == "refused"


n_bad_before = sum(1 for r in A["estop"] if not estop_ok(r)) + sum(1 for r in A["boot"] if not boot_ok(r))
n_bad_after = sum(1 for r in B["estop"] if not estop_ok(r)) + sum(1 for r in B["boot"] if not boot_ok(r))
n_cells = len(A["estop"]) + len(A["boot"])
# Structure only - the counts are derived from the two dumps, never typed.
assert n_cells == len(A["estop"]) + len(A["boot"]) == 14, n_cells
assert n_bad_after == 0, [r["label"] for r in B["estop"] if not estop_ok(r)]
assert n_bad_before > 0, "nothing to fix on main"

fig = plt.figure(figsize=(16.2, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 1.02, 0.62], hspace=0.20, wspace=0.06)

fig.suptitle(
    "A provisioning posture flag is checked, not read by truthiness",
    fontsize=17.5, fontweight="bold", y=0.975, color=INK,
)
fig.text(
    0.5, 0.945,
    "mesh.iot has two public entry points carrying bool flags. Each selects a posture, and each was read by "
    "truthiness -\nso every non-boolean spelling of off is truthy and selected the permissive branch.",
    ha="center", va="top", fontsize=11.0, color=MUTE,
)


def table(ax, title, rows, cols, widths, okfn, subtitle):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    put(ax, 0.0, 1.045, title, fontsize=13.2, fontweight="bold", color=INK, axes_coords=True)
    put(ax, 0.0, 0.985, subtitle, fontsize=9.6, color=MUTE, axes_coords=True)

    TOP, LAST, PAD = 0.895, 0.075, 0.055
    step = (TOP - PAD - LAST) / len(rows)
    assert step > 0.045, step
    # header
    for x, c in zip(widths, cols, strict=True):
        put(ax, x, TOP, c, fontsize=9.6, fontweight="bold", color=INK, axes_coords=True)
    ax.plot([0.0, 1.0], [TOP - 0.022, TOP - 0.022], lw=1.0, color="#c8ced6",
            transform=ax.transAxes, clip_on=False)

    y = TOP - PAD
    for r in rows:
        ok = okfn(r)
        ax.add_patch(Rectangle((-0.012, y - step * 0.34), 1.024, step * 0.80,
                               transform=ax.transAxes, facecolor=BAND_OK if ok else BAND_BAD,
                               edgecolor="none", zorder=0, clip_on=False))
        cells = r["_cells"]
        for i, (x, txt) in enumerate(zip(widths, cells, strict=True)):
            col = INK if i == 0 else (GREEN if ok else RED)
            put(ax, x, y, txt, fontsize=9.9, color=col,
                fontweight="bold" if i in (0, len(cells) - 1) else "normal",
                family="monospace" if i else "sans-serif", axes_coords=True)
        y -= step
    assert y + step > 0.040, y + step


def estop_cells(r):
    if r["verdict"] == "refused":
        return [r["label"], "refused", "-", "-", "no"]
    return [r["label"], "provisioned", r["policy"], "YES" if r["grants_estop"] else "no",
            "yes" if r["touched"] else "no"]


def boot_cells(r):
    v = {"ENTERED CREATE PATH": "created", "previewed": "previewed", "refused": "refused"}.get(
        r["verdict"], r["verdict"])
    return [r["label"], v, "YES" if r["entered_create"] else "no"]


ECOLS = ["allow_estop_publish=", "outcome", "policy attached to the cert", "grants estop", "AWS touched"]
EW = [0.0, 0.205, 0.325, 0.685, 0.855]
BCOLS = ["bootstrap_account(...)", "outcome", "account create path entered"]
BW = [0.0, 0.375, 0.560]

for r in A["estop"] + B["estop"]:
    r["_cells"] = estop_cells(r)
for r in A["boot"] + B["boot"]:
    r["_cells"] = boot_cells(r)

ax = fig.add_subplot(gs[0, 0])
table(ax, "main  (614a96ef)", A["estop"], ECOLS, EW, estop_ok,
      "provision_robot(allow_estop_publish=...)  -  a security opt-out")
ax = fig.add_subplot(gs[0, 1])
table(ax, "this change", B["estop"], ECOLS, EW, estop_ok,
      "the same eight calls, same recording IoT client")

ax = fig.add_subplot(gs[1, 0])
table(ax, "main  (614a96ef)", A["boot"], BCOLS, BW, boot_ok,
      'bootstrap_account(...)  -  "Must be True to actually create resources"')
ax = fig.add_subplot(gs[1, 1])
table(ax, "this change", B["boot"], BCOLS, BW, boot_ok, "the same six calls")

# ---- footer -----------------------------------------------------------------
ax = fig.add_subplot(gs[2, :])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f4f6f8",
                       edgecolor="#c8ced6", lw=1.0))

lines = [
    ("Outcomes that do not honour the flag's own declared posture:  "
     f"main {n_bad_before} of {n_cells}   ->   this change {n_bad_after} of {n_cells}", True),
    ("", False),
    ("Unchanged - the two declared spellings still select the two policies they always did:", True),
    ("    allow_estop_publish=True   -> strands-robot           (AllowSafetyEstop present)   on both trees", False),
    ("    allow_estop_publish=False  -> strands-robot-no-estop  (AllowSafetyEstop absent)    on both trees", False),
    ("    confirm=True, dry_run=False -> account create path entered; confirm=False, dry_run=False -> the "
     "documented refusal; dry_run=True -> preview", False),
    ("", False),
    ("A refused call provisions nothing: no Thing, no policy, no certificate, and boto3 is never resolved - "
     "the check precedes every AWS call.", True),
    ("No policy, simulation, rendering, recording or asset behaviour changes, so the artifact is a measured "
     "verdict table rather than a rollout.", False),
]
TOP_F, PAD_F = 0.885, 0.030
step_f = (TOP_F - PAD_F) / len(lines)
assert step_f > 0.075, step_f
y = TOP_F
for txt, bold in lines:
    if txt:
        put(ax, 0.014, y, txt, fontsize=10.2 if bold else 9.7,
            fontweight="bold" if bold else "normal",
            color=INK if bold else MUTE,
            family="sans-serif" if bold else "monospace", axes_coords=True)
    y -= step_f
assert y + step_f > 0.020, y + step_f

# ---- self-audit -------------------------------------------------------------
for ax_, y_, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y_ <= 1.07, (y_, "axes-fraction text outside the panel")
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.05 <= y_ <= hi + 0.07, (y_, lo, hi)

OUT = Path("/tmp/artifact.png")
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image

im = np.asarray(Image.open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nonwhite == 0, (name, nonwhite)
print(f"OK {OUT} {im.shape[1]}x{im.shape[0]}  divergences {n_bad_before} -> {n_bad_after} of {n_cells}")
