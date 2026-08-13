from __future__ import annotations
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.loads(pathlib.Path(sys.argv[1]).read_text())
OUT = pathlib.Path(sys.argv[2])
placed: list[tuple[object, float, bool]] = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

cells, cfg, ren, muts, gate = F["cells"], F["config"], F["render"], F["mutations"], F["gate"]
assert len(cells) == 6 and len(cfg) == 4 and len(muts) == 8
assert all(c["verdict"] == "refused" for c in cells), cells
n_before = sum(c["driven_before"] for c in cells)
n_after = len(cells)
assert (n_before, n_after) == (2, 6), (n_before, n_after)
assert ren["built"] == "success" and abs(ren["timestep_installed"] - 0.002) < 1e-12
assert ren["saturated"] > 0.10
assert sum(1 for c in cfg if c["config_constructs"]) == 3, cfg
assert all(c["create_world"] == "refused" for c in cfg)
assert all(m[1] > 0 and m[2] == 0 for m in muts)

fig = plt.figure(figsize=(16.6, 14.4), dpi=122)
gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 0.62, 0.80], width_ratios=[0.86, 1.14],
                      hspace=0.20, wspace=0.10, left=0.035, right=0.972, top=0.938, bottom=0.028)
fig.suptitle("create_world installs the physics timestep on every backend - the domain was pinned on one",
             fontsize=15.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.958, "Tests only; no library behaviour changes. Every cell below is measured on the shipped code.",
         ha="center", fontsize=10.4, style="italic", color="#444")

# --- row 1 left: the real render -------------------------------------------
axr = fig.add_subplot(gs[0, 0]); axr.axis("off")
axr.imshow(np.asarray(Image.open(ren["path"])))
axr.set_title("A world genuinely built through create_world(timestep=0.002)", fontsize=11.4, fontweight="bold", pad=7)
axr.set_xlabel(f"so101 in MuJoCo, headless (MUJOCO_GL=egl).  dt installed = {ren['timestep_installed']}s\n"
               f"400 steps -> sim clock t = {ren['sim_time']:.3f}s (400 x 0.002), saturated {ren['saturated']*100:.1f}%\n"
               "This is the reference cell that was already pinned - the other four now are too.",
               fontsize=9.5, labelpad=8)

# --- row 1 right: the six-cell matrix --------------------------------------
axm = fig.add_subplot(gs[0, 1]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 1.028, "The claim: every backend's create_world validates the EFFECTIVE dt",
    transform=axm.transAxes, fontsize=11.6, fontweight="bold")
put(axm, 0.0, 0.972, "and names whichever knob the value came from.   dt = -0.002 in every cell.",
    transform=axm.transAxes, fontsize=9.8, style="italic", color="#555")
hdr_y = 0.895
put(axm, 0.005, hdr_y, "backend", transform=axm.transAxes, fontsize=9.6, fontweight="bold")
put(axm, 0.135, hdr_y, "knob supplying the dt", transform=axm.transAxes, fontsize=9.6, fontweight="bold")
put(axm, 0.560, hdr_y, "verdict", transform=axm.transAxes, fontsize=9.6, fontweight="bold")
put(axm, 0.700, hdr_y, "names the knob", transform=axm.transAxes, fontsize=9.6, fontweight="bold")
put(axm, 0.885, hdr_y, "driven before", transform=axm.transAxes, fontsize=9.6, fontweight="bold")
TOP, LAST = 0.800, 0.130
STEP = (TOP - LAST) / (len(cells) - 1)
assert STEP > 0.045, STEP
y = TOP
for c in cells:
    before = c["driven_before"]
    axm.add_patch(plt.Rectangle((0.0, y - 0.038), 1.0, 0.088, transform=axm.transAxes,
                                facecolor="#e8f5e9" if before else "#fff3e0", edgecolor="none", zorder=0))
    put(axm, 0.005, y, c["backend"], transform=axm.transAxes, fontsize=10.2, family="monospace", fontweight="bold")
    put(axm, 0.135, y, c["knob"], transform=axm.transAxes, fontsize=9.8, family="monospace")
    put(axm, 0.560, y, "refused", transform=axm.transAxes, fontsize=9.8, family="monospace",
        color="#1b5e20", fontweight="bold")
    knob = c["knob"].split(" ")[0]
    put(axm, 0.700, y, knob if knob in c["names"] else ",".join(c["names"]),
        transform=axm.transAxes, fontsize=9.4, family="monospace", color="#1b5e20")
    put(axm, 0.885, y, "yes" if before else "NO  <- now", transform=axm.transAxes, fontsize=9.4,
        family="monospace", fontweight="bold", color="#1b5e20" if before else "#bf360c")
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9
put(axm, 0.005, 0.038, f"cells driven before: {n_before} of {n_after}   ->   after: {n_after} of {n_after}",
    transform=axm.transAxes, fontsize=10.4, fontweight="bold")
put(axm, 0.005, -0.020, "The two green rows were driven by an importorskip(\"mujoco\")-gated, MuJoCo-only module.",
    transform=axm.transAxes, fontsize=9.0, style="italic", color="#555")

# --- row 2: the config guard cannot see these ------------------------------
axc = fig.add_subplot(gs[1, :]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
put(axc, 0.0, 1.020, "Why the effective-dt check is load-bearing rather than defensive",
    transform=axc.transAxes, fontsize=11.6, fontweight="bold")
put(axc, 0.0, 0.905, "IsaacConfig.__post_init__ tests `physics_dt <= 0` - a bare comparison, which is False for nan and inf "
    "and lets a bool through.", transform=axc.transAxes, fontsize=9.8, style="italic", color="#555")
hy = 0.760
for x, t in ((0.005, "IsaacConfig(physics_dt=...)"), (0.300, "config constructs?"),
             (0.520, "create_world verdict"), (0.760, "the only guard between it and the world")):
    put(axc, x, hy, t, transform=axc.transAxes, fontsize=9.6, fontweight="bold")
CTOP, CLAST = 0.610, 0.110
CSTEP = (CTOP - CLAST) / (len(cfg) - 1)
assert CSTEP > 0.045, CSTEP
y = CTOP
for c in cfg:
    ok = c["config_constructs"]
    axc.add_patch(plt.Rectangle((0.0, y - 0.045), 1.0, 0.105, transform=axc.transAxes,
                                facecolor="#ffebee" if ok else "#e8f5e9", edgecolor="none", zorder=0))
    put(axc, 0.005, y, c["value"], transform=axc.transAxes, fontsize=10.0, family="monospace", fontweight="bold")
    put(axc, 0.300, y, "YES - it constructs" if ok else "no - refused",
        transform=axc.transAxes, fontsize=9.8, family="monospace",
        color="#b71c1c" if ok else "#1b5e20", fontweight="bold")
    put(axc, 0.520, y, c["create_world"], transform=axc.transAxes, fontsize=9.8, family="monospace",
        color="#1b5e20", fontweight="bold")
    put(axc, 0.760, y, "create_world" if ok else "the config guard", transform=axc.transAxes,
        fontsize=9.6, family="monospace")
    y -= CSTEP
assert abs((y + CSTEP) - CLAST) < 1e-9
put(axc, 0.005, 0.020, "3 of 4: only create_world stands between an unusable engine default and a world built on it.",
    transform=axc.transAxes, fontsize=10.0, fontweight="bold")

# --- row 3: mutation table + gate -----------------------------------------
axx = fig.add_subplot(gs[2, :]); axx.axis("off"); axx.set_xlim(0, 1); axx.set_ylim(0, 1)
put(axx, 0.0, 1.010, "Mutation table - would a regression be caught?", transform=axx.transAxes,
    fontsize=11.6, fontweight="bold")
hy = 0.905
put(axx, 0.005, hy, "regression applied to the shipped guard", transform=axx.transAxes, fontsize=9.6, fontweight="bold")
put(axx, 0.660, hy, "new cells", transform=axx.transAxes, fontsize=9.6, fontweight="bold")
put(axx, 0.810, hy, "this module before the PR", transform=axx.transAxes, fontsize=9.6, fontweight="bold")
MTOP, MLAST = 0.790, 0.240
MSTEP = (MTOP - MLAST) / (len(muts) - 1)
assert MSTEP > 0.045, MSTEP
y = MTOP
for label, a, b in muts:
    put(axx, 0.005, y, label, transform=axx.transAxes, fontsize=9.5, family="monospace")
    put(axx, 0.660, y, f"{a} failed", transform=axx.transAxes, fontsize=9.5, family="monospace",
        color="#1b5e20", fontweight="bold")
    put(axx, 0.810, y, f"{b} failed  <- BLIND", transform=axx.transAxes, fontsize=9.5, family="monospace",
        color="#b71c1c", fontweight="bold")
    y -= MSTEP
assert abs((y + MSTEP) - MLAST) < 1e-9
put(axx, 0.005, 0.145, f"8 of 8 caught by the new cells; 0 of 8 by the 67 cases this module already had.",
    transform=axx.transAxes, fontsize=10.4, fontweight="bold")
put(axx, 0.005, 0.075, f"Gate: {gate['suite']} in {gate['elapsed']} (MUJOCO_GL=egl).  This file: {gate['file']}.",
    transform=axx.transAxes, fontsize=9.8, family="monospace")
put(axx, 0.005, 0.012, f"{gate['lint']}.  Every new cell runs with no Newton, Warp, Isaac Sim or GL.",
    transform=axx.transAxes, fontsize=9.8, family="monospace")

fig.savefig(OUT, dpi=122, bbox_inches="tight", pad_inches=0.30, facecolor="white")

# ---- layout guards -------------------------------------------------------
for ax, yy, axes_coords in placed:
    if axes_coords:
        assert -0.05 <= yy <= 1.06, f"axes-fraction text at y={yy}"
    else:
        lo, hi = ax.get_ylim()
        assert min(lo, hi) - 0.05 <= yy <= max(lo, hi) + 0.07, f"data text at y={yy} vs {(lo, hi)}"
im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {Image.open(OUT).size}  cells {n_before}->{n_after}")
