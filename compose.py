import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(sys.argv[1])
A = json.loads((OUT / "facts-main.json").read_text())
B = json.loads((OUT / "facts-pr.json").read_text())
assert A["tree"] != B["tree"], (A["tree"], B["tree"])
MUT = json.loads(pathlib.Path(sys.argv[2]).read_text())
ROBOTS = ["so101", "unitree_g1", "unitree_go2"]

# --- audited facts ---------------------------------------------------------
assert A["scenario_nocleanup"]["cleanup_warnings"] == 3 and B["scenario_nocleanup"]["cleanup_warnings"] == 0
assert A["scenario_cleanup"]["cleanup_warnings"] == 3 and B["scenario_cleanup"]["cleanup_warnings"] == 0
assert A["shutdown_trace"] == ["enter", "raised:ImportError"], A["shutdown_trace"]
assert B["shutdown_trace"] == ["enter", "_shutdown_ros_bridge", "_close_main_thread_renderers", "returned"]
for r in ROBOTS:
    assert A["scenario_nocleanup"]["ok"] and B["scenario_nocleanup"]["ok"]
maxd = {}
for r in ROBOTS:
    a = np.load(OUT / f"{r}-main.npy").astype(int); b = np.load(OUT / f"{r}-pr.npy").astype(int)
    maxd[r] = int(np.abs(a - b).max())
    assert maxd[r] <= 2, (r, maxd[r])           # the sim path is untouched
    assert B["renders"][r]["content_frac"] > 0.15

placed: list[tuple] = []
def put(ax, x, y, s, axes_coords=True, **kw):
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, transform=(ax.transAxes if axes_coords else ax.transData), **kw)

fig = plt.figure(figsize=(15.4, 13.4), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.02, 1.02, 0.72], hspace=0.30, wspace=0.10)
fig.suptitle(
    "MuJoCoSimEngine.cleanup() opened with a function-local import, so the finalizer\n"
    "raised before its first teardown step and reported a failure naming the interpreter",
    fontsize=14.5, fontweight="bold", y=0.982)

# Row 1: the three sims still render identically -- the sim path is untouched
for i, r in enumerate(ROBOTS):
    ax = fig.add_subplot(gs[0, i]); ax.axis("off")
    ax.imshow(np.load(OUT / f"{r}-pr.npy"))
    ax.set_title(f"{r}", fontsize=11.5, fontweight="bold", pad=5)
    ax.set_xlabel(f"identical on both trees   max|delta| = {maxd[r]}/255\n"
                  f"content {B['renders'][r]['content_frac']:.2%}",
                  fontsize=9.2, labelpad=6)
    ax.xaxis.set_visible(True); ax.set_xticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

# Row 2 left: the reporter's 3-sim scenario
axl = fig.add_subplot(gs[1, :2]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.965, "The reported scenario: three MuJoCo sims in one process",
    fontsize=12.5, fontweight="bold", va="top")
put(axl, 0.0, 0.895, "cleanup-failure warnings printed at exit, per process",
    fontsize=9.6, style="italic", color="#444", va="top")
rows = [
    ("3 sims built, cleanup() never called",
     f"{A['scenario_nocleanup']['cleanup_warnings']} warnings",
     f"{B['scenario_nocleanup']['cleanup_warnings']} warnings", False),
    ("3 sims built, cleanup() called on each",
     f"{A['scenario_cleanup']['cleanup_warnings']} warnings",
     f"{B['scenario_cleanup']['cleanup_warnings']} warnings", False),
    ("finalizer trace at real shutdown",
     " -> ".join(A["shutdown_trace"]), " -> ".join(B["shutdown_trace"]), True),
    ("ROS bridge released at exit", "no", "yes", True),
    ("renderers released at exit", "no", "yes", True),
    ("all three sims render + step", "yes", "yes", False),
]
TOP, LAST = 0.775, 0.085
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
put(axl, 0.015, TOP + 0.075, "", fontsize=1)
put(axl, 0.015, TOP + 0.058, "measurement", fontsize=9.6, fontweight="bold", color="#333")
put(axl, 0.470, TOP + 0.058, "main", fontsize=9.6, fontweight="bold", color="#a11")
put(axl, 0.740, TOP + 0.058, "this PR", fontsize=9.6, fontweight="bold", color="#161")
y = TOP
for label, mv, pv, severe in rows:
    axl.add_patch(plt.Rectangle((0.005, y - 0.026), 0.99, 0.050,
                                facecolor=("#fdecec" if severe else "#f5f5f5"),
                                edgecolor="none", transform=axl.transAxes, zorder=0))
    put(axl, 0.015, y, label, fontsize=9.3, va="center")
    put(axl, 0.470, y, mv, fontsize=9.0, va="center", color="#a11",
        family="monospace", fontweight="bold" if severe else "normal")
    put(axl, 0.740, y, pv, fontsize=9.0, va="center", color="#161",
        family="monospace", fontweight="bold" if severe else "normal")
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

# Row 2 right: the one-line cause
axr = fig.add_subplot(gs[1, 2]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 0.965, "The cause", fontsize=12.5, fontweight="bold", va="top")
cause = [
    ("def cleanup(self, ...):", "#333"),
    ('    """..."""', "#888"),
    ("    import contextlib as _cl", "#a11"),
    ("    #  ^ statement 1 of 17", "#a11"),
    ("", "#333"),
    ("    ... 9 teardown steps ...", "#666"),
    ("      ROS bridge, teleop,", "#666"),
    ("      per-robot mesh, mesh,", "#666"),
    ("      world, renderers,", "#666"),
    ("      executor", "#666"),
]
cy = 0.855
for text, col in cause:
    put(axr, 0.02, cy, text, fontsize=8.6, family="monospace", color=col, va="top")
    cy -= 0.062
put(axr, 0.0, 0.185,
    "9 module-level `import contextlib`\nin the package; this was the only\nfunction-local one.",
    fontsize=8.9, va="top", color="#333")
put(axr, 0.0, 0.045, "simulation/base.py, which declares\nthe contract, imports it at module\nscope.",
    fontsize=8.9, va="top", style="italic", color="#555")

# Row 3: mutation matrix
axm = fig.add_subplot(gs[2, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 0.94, "Mutation table  -  every regression the new tests catch, and what the suite saw before",
    fontsize=12.0, fontweight="bold", va="top")
labels = [k for k in MUT if k != "M0 unmutated control"]
n_blind = sum(1 for k in labels if MUT[k]["pre_existing"]["failed"] == 0 and MUT[k]["new"]["failed"] > 0)
n_caught = sum(1 for k in labels if MUT[k]["new"]["failed"] > 0)
assert (n_caught, n_blind) == (3, 3), (n_caught, n_blind)
MTOP, MLAST = 0.70, 0.20
MSTEP = (MTOP - MLAST) / (len(labels) - 1)
assert MSTEP > 0.030, MSTEP
put(axm, 0.015, MTOP + 0.115, "regression", fontsize=9.4, fontweight="bold", color="#333")
put(axm, 0.640, MTOP + 0.115, "new tests", fontsize=9.4, fontweight="bold", color="#161")
put(axm, 0.800, MTOP + 0.115, "the 19 pre-existing", fontsize=9.4, fontweight="bold", color="#a11")
my = MTOP
for k in labels:
    nf = MUT[k]["new"]["failed"]; pf = MUT[k]["pre_existing"]["failed"]
    blind = pf == 0 and nf > 0
    axm.add_patch(plt.Rectangle((0.005, my - 0.055), 0.99, 0.105,
                                facecolor=("#fdecec" if blind else "#f5f5f5"),
                                edgecolor="none", transform=axm.transAxes, zorder=0))
    put(axm, 0.015, my, k, fontsize=9.0, va="center")
    put(axm, 0.640, my, f"{nf} failed" if nf else "not caught",
        fontsize=9.0, va="center", family="monospace",
        color=("#161" if nf else "#888"), fontweight="bold" if nf else "normal")
    put(axm, 0.800, my, f"{pf} failed" if pf else "0 failed  <- BLIND" if blind else "0 failed",
        fontsize=9.0, va="center", family="monospace", color="#a11" if blind else "#888")
    my -= MSTEP
assert abs((my + MSTEP) - MLAST) < 1e-9, (my, MLAST)
put(axm, 0.0, 0.085,
    f"{n_caught} of {len(labels)} caught here, {n_blind} of them invisible to the 19 cases already in this module. "
    "The fourth removes the docstring paragraph that\nexplains why the import sits at module scope: prose, so it is "
    "not pinned - the structural scan enforces the behaviour itself.",
    fontsize=9.0, va="top", color="#333")
put(axm, 0.0, 0.005,
    "Gate: 28897 passed / 258 skipped / 0 failed (main 28680 + 217).  "
    "Pre-fix with the source at upstream/main: 4 failed / 232 passed.",
    fontsize=8.8, va="top", family="monospace", color="#555")

# layout guards
for ax, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.13, (yv, "axes")
    else:
        lo, hi = ax.get_ylim()
        assert min(lo, hi) - 0.05 <= yv <= max(lo, hi) + 0.07, (yv, ax.get_ylim())

path = OUT / "finalizer-teardown.png"
fig.savefig(path, bbox_inches="tight", pad_inches=0.32, facecolor="white")
plt.close(fig)
import imageio.v3 as iio
im = iio.imread(path)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((band.reshape(-1, band.shape[-1])[:, :3].min(axis=1) < 235).sum())
    assert nonwhite == 0, (name, nonwhite)
print(f"OK {path} {im.shape}  maxd={maxd}  caught={n_caught} blind={n_blind}")
