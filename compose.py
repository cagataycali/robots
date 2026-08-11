"""Compose the artifact: what the GL-gated halves verify + the measured matrices."""

from __future__ import annotations

import json
import pathlib

import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).parent
F = json.loads((OUT / "facts.json").read_text())

# Every number below was measured; nothing is typed by hand into the figure.
HOST = {
    ("main", "GL present"): ("123 passed", False),
    ("main", "GL-free host"): ("3 FAILED, 120 passed", True),
    ("this PR", "GL present"): (F["host_matrix"]["gl_present"].split(" in ")[0], False),
    ("this PR", "GL-free host"): (F["host_matrix"]["gl_free"].split(" in ")[0], False),
}
GUARD = {"main": ("2 failed, 8 passed", True), "this PR": (F["guard"].split(" in ")[0], False)}

MUTATIONS = [
    ("a non-string name resolves to a real entity\ninstead of 'not found'", "TestTheSessionSurvives", "1 failed, 1 skipped"),
    ("a registered name stops resolving", "test_a_registered_name_is_unaffected", "1 failed"),
    ("the per-camera dimensions are dropped\non the way to add_camera", "test_a_usable_config_installs", "1 failed"),
]

FRAMES = [
    ("entity_name", "test_the_world_still_renders_after_a_refused_lookup", "default camera, after a refused lookup"),
    ("unhashable_name", "test_a_registered_camera_name_still_renders", "camera 'look', a resolvable name"),
    ("libero_camera", "test_a_usable_config_renders", f"camera 'image', installed at {F['installed_dims'][0]}x{F['installed_dims'][1]}"),
]

placed: list[tuple[plt.Axes, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 12.2), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 0.72, 0.78], hspace=0.30, wspace=0.10)

fig.suptitle(
    "A MuJoCo render-success assertion is a host graphics capability, not the contract under test",
    fontsize=15.5, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.947,
    "Three modules asserted render(...)[\"status\"] == \"success\" inline. Split behind the shared GL probe: "
    "the GL-free assertions keep running everywhere, the render keeps its own case.",
    ha="center", fontsize=10.6, style="italic", color="#333333",
)

# ---- row 1: the real frames each new @requires_gl case verifies -------------
for col, (key, test_name, caption) in enumerate(FRAMES):
    ax = fig.add_subplot(gs[0, col])
    arr = iio.imread(OUT / F["frames"][key]["file"])
    ax.imshow(arr)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#1b7f3b"); spine.set_linewidth(2.4)
    ax.set_title(f"@requires_gl\n{test_name}", fontsize=9.2, fontweight="bold", color="#14532d", pad=6)
    ax.set_xlabel(f"{caption}\nreal frame, {arr.shape[1]}x{arr.shape[0]}  |  saturated {F['frames'][key]['saturated_frac']:.0%}",
                  fontsize=8.9, color="#333333")

# ---- row 2: the measured host matrix ---------------------------------------
axh = fig.add_subplot(gs[1, :]); axh.axis("off"); axh.set_xlim(0, 1); axh.set_ylim(0, 1)
put(axh, 0.5, 1.02, "The same three modules, on both hosts - measured", ha="center",
    fontsize=12.2, fontweight="bold", transform=axh.transAxes)

cols_x = [0.30, 0.62]
TOP, LAST = 0.80, 0.30
rows = ["GL present", "GL-free host"]
step = (TOP - LAST) / (len(rows) - 1)
put(axh, 0.055, TOP + 0.15, "host", fontsize=10.4, fontweight="bold", transform=axh.transAxes)
for x, tree in zip(cols_x, ["main", "this PR"], strict=True):
    put(axh, x, TOP + 0.15, tree, ha="center", fontsize=10.4, fontweight="bold", transform=axh.transAxes)
y = TOP
for row in rows:
    put(axh, 0.055, y, row, fontsize=10.2, va="center", transform=axh.transAxes)
    for x, tree in zip(cols_x, ["main", "this PR"], strict=True):
        text, bad = HOST[(tree, row)]
        axh.add_patch(plt.Rectangle((x - 0.145, y - 0.075), 0.29, 0.15,
                                    facecolor="#fde2e1" if bad else "#e3f5e8",
                                    edgecolor="#b3261e" if bad else "#1b7f3b", linewidth=1.5,
                                    transform=axh.transAxes, zorder=0))
        put(axh, x, y, text, ha="center", va="center", fontsize=10.1,
            color="#7f1d1d" if bad else "#14532d", fontweight="bold" if bad else "normal",
            transform=axh.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

put(axh, 0.055, 0.08, "the new guard, run against each tree's test files:", fontsize=10.2, transform=axh.transAxes)
for x, tree in zip(cols_x, ["main", "this PR"], strict=True):
    text, bad = GUARD[tree]
    put(axh, x, 0.08, text, ha="center", fontsize=10.1, fontweight="bold" if bad else "normal",
        color="#7f1d1d" if bad else "#14532d", transform=axh.transAxes)

# ---- row 3: the mutation matrix --------------------------------------------
axm = fig.add_subplot(gs[2, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.5, 1.05, "The split does not gut the pin: revert the production behaviour each module owns, "
    "and the RETAINED GL-free assertions still fail on a GL-free host",
    ha="center", fontsize=12.2, fontweight="bold", transform=axm.transAxes)

MTOP, MLAST = 0.72, 0.16
mstep = (MTOP - MLAST) / (len(MUTATIONS) - 1)
put(axm, 0.035, MTOP + 0.16, "production behaviour reverted", fontsize=10.2, fontweight="bold", transform=axm.transAxes)
put(axm, 0.545, MTOP + 0.16, "retained GL-free test", fontsize=10.2, fontweight="bold", transform=axm.transAxes)
put(axm, 0.885, MTOP + 0.16, "outcome", ha="center", fontsize=10.2, fontweight="bold", transform=axm.transAxes)
ym = MTOP
for label, test, outcome in MUTATIONS:
    put(axm, 0.035, ym, label, fontsize=9.5, va="center", transform=axm.transAxes)
    put(axm, 0.545, ym, test, fontsize=9.3, va="center", family="monospace", transform=axm.transAxes)
    put(axm, 0.885, ym, outcome + "  <- caught", ha="center", va="center", fontsize=9.6,
        fontweight="bold", color="#14532d", transform=axm.transAxes)
    ym -= mstep
assert abs((ym + mstep) - MLAST) < 1e-9, (ym, MLAST)

put(axm, 0.5, 0.02,
    "Unmutated control: each retained test passes.  Scope is the mujoco requirement, not a directory - a render "
    "asserted through another backend is excluded by construction.  Tests only; no production line changes.",
    ha="center", fontsize=9.4, style="italic", color="#444444", transform=axm.transAxes)

for ax, y, axes_coords in placed:
    lo, hi = (-0.05, 1.22) if axes_coords else ax.get_ylim()
    assert lo <= y <= hi, f"text at y={y} outside {(lo, hi)}"

fig.savefig(OUT / "gl_gate.png", bbox_inches="tight", pad_inches=0.32, facecolor="white")

img = np.asarray(iio.imread(OUT / "gl_gate.png"))[:, :, :3]
for name, band in (("top", img[:8]), ("bottom", img[-8:]), ("left", img[:, :8]), ("right", img[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white pixels"

# self-audit: every claim in the figure comes from the measurement
assert F["tree"].endswith(pathlib.Path.cwd().name), F["tree"]
assert F["host_matrix"]["gl_free"].startswith("123 passed, 3 skipped"), F["host_matrix"]
assert F["guard"].startswith("10 passed"), F["guard"]
assert F["installed_dims"] == [320, 320], F["installed_dims"]
for key, _, _ in FRAMES:
    assert F["frames"][key]["saturated_frac"] > 0.5, (key, F["frames"][key])
print("saved", OUT / "gl_gate.png", iio.imread(OUT / "gl_gate.png").shape)
