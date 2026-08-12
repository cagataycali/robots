"""Compose the mesh-teardown artifact from the two measured trees."""
import json, os, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RID = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/art-base-{RID}.json").read_text())    # upstream/main
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RID}.json").read_text())  # this PR
assert A["tree"] != B["tree"], "both arms measured the same tree"

# --- self-audit of every claim the figure makes ------------------------------
assert A["script"]["timed_out"] is True and A["script"]["exit"] is None
assert B["script"]["timed_out"] is False and B["script"]["exit"] == 0
assert A["attrs"]["example_reads"] == "_mesh" and A["attrs"]["read_returns_none"] is True
assert B["attrs"]["example_reads"] == "mesh" and B["attrs"]["read_returns_none"] is False
assert A["attrs"]["factory_sets_mesh"] is True and B["attrs"]["factory_sets_mesh"] is True
assert A["threads"]["surviving_non_daemon"] == 6, A["threads"]
assert B["threads"]["surviving_non_daemon"] == 0, B["threads"]

fa = np.asarray(Image.open(f"/tmp/art-base-{RID}.png").convert("RGB")).astype(int)
fb = np.asarray(Image.open(f"/tmp/art-branch-{RID}.png").convert("RGB")).astype(int)
assert fa.shape == fb.shape
dmax = int(np.abs(fa - fb).max())
assert dmax <= 2, f"the simulation the example builds must be unchanged, max|delta|={dmax}"
sat = float(((fb.max(2) - fb.min(2)) > 45).mean())
assert sat > 0.05, f"render has no content, saturated={sat:.3f}"

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 9.6); kw.setdefault("va", "top")
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.0, 11.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 0.86, 0.62], width_ratios=[1.0, 1.06],
                      hspace=0.22, wspace=0.13)
fig.suptitle("examples/04_mesh_peer_discovery.py released the Zenoh session through an attribute the SDK never sets",
             fontsize=14.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.955, "Measured on Thor (aarch64, MUJOCO_GL=egl). Left: the simulation the example builds - "
         "byte-comparable on both trees. Right: what the script actually did.",
         ha="center", fontsize=10.2, style="italic", color="#444")

# --- row 1 left: the real render --------------------------------------------
axr = fig.add_subplot(gs[0, 0]); axr.imshow(fb.astype(np.uint8)); axr.axis("off")
axr.set_title("Robot(\"so100\", mode=\"sim\") - the world the example creates", fontsize=10.8, fontweight="bold")
axr.set_xlabel(f"identical on both trees: max|delta| = {dmax}/255   (saturated {sat:.0%})\n"
               "this PR changes teardown only - no simulation, policy or rendering behaviour",
               fontsize=9.4, color="#2d6a2d")

# --- row 1 right: exit timeline ----------------------------------------------
axt = fig.add_subplot(gs[0, 1])
axt.barh([1], [A["script"]["wall_s"]], color="#c0392b", height=0.42, label="main: killed, never exits")
axt.barh([0], [B["script"]["wall_s"]], color="#2d6a2d", height=0.42, label="this PR: exit 0")
axt.axvline(3.0, ls="--", lw=1.4, color="#555")
axt.text(3.15, 1.55, 'docstring promise: "~3 seconds"', fontsize=9.2, color="#555")
axt.set_yticks([0, 1]); axt.set_yticklabels(["this PR", "upstream/main"], fontsize=10)
axt.set_xlabel("wall-clock seconds to terminate", fontsize=9.8)
axt.set_xlim(0, 34); axt.set_ylim(-0.55, 1.9)
axt.text(A["script"]["wall_s"] - 0.5, 1, f"  timeout {A['script']['wall_s']}s (SIGKILL)",
         va="center", ha="right", fontsize=9.4, color="white", fontweight="bold")
axt.text(B["script"]["wall_s"] + 0.5, 0, f"{B['script']['wall_s']}s, exit 0", va="center", fontsize=9.4,
         color="#2d6a2d", fontweight="bold")
axt.set_title("The example as a user runs it", fontsize=10.8, fontweight="bold")
axt.legend(loc="upper right", fontsize=8.8, frameon=False)
for s in ("top", "right"): axt.spines[s].set_visible(False)

# --- row 2: the measured ledger ---------------------------------------------
axl = fig.add_subplot(gs[1, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
axl.set_title("What the cleanup line does, measured in-process", fontsize=11, fontweight="bold", loc="left")
rows = [
    ("Robot factory assigns sim.mesh (robot.py:392)", "yes", "yes", True),
    ("SDK teardown reads getattr(instance, \"mesh\") (robot.py:468)", "yes", "yes", True),
    ("attribute the example reads", '"_mesh"', '"mesh"', False),
    ("that read returns None (cleanup silently skipped)", "YES", "no", False),
    ("Mesh.stop() actually called", "no", "yes", False),
    ("surviving non-daemon threads after cleanup",
     f'{A["threads"]["surviving_non_daemon"]}  (pyo3-closure)', str(B["threads"]["surviving_non_daemon"]), False),
    ("script terminates", "no", "yes", False),
]
TOP, LAST = 0.86, 0.10
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
put(axl, 0.015, 0.975, "measured property", fontweight="bold", transform=axl.transAxes)
put(axl, 0.615, 0.975, "upstream/main", fontweight="bold", color="#c0392b", transform=axl.transAxes)
put(axl, 0.815, 0.975, "this PR", fontweight="bold", color="#2d6a2d", transform=axl.transAxes)
y = TOP
for label, a, b, same in rows:
    if not same:
        axl.add_patch(plt.Rectangle((0.008, y - 0.035), 0.984, 0.052, transform=axl.transAxes,
                                    facecolor="#fdf1f0", edgecolor="none", zorder=0))
    put(axl, 0.015, y, label, transform=axl.transAxes)
    put(axl, 0.615, y, a, color="#c0392b" if not same else "#333",
        fontweight="bold" if not same else "normal", family="monospace", transform=axl.transAxes)
    put(axl, 0.815, y, b, color="#2d6a2d" if not same else "#333",
        fontweight="bold" if not same else "normal", family="monospace", transform=axl.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

# --- row 3: guard + gate -----------------------------------------------------
axg = fig.add_subplot(gs[2, :]); axg.axis("off"); axg.set_xlim(0, 1); axg.set_ylim(0, 1)
axg.set_title("Guard and gate", fontsize=11, fontweight="bold", loc="left")
lines = [
    "Guard derives the attribute from strands_robots/robot.py (the init_mesh -> sim.mesh chain), so a rename tracks it:",
    "   pre-fix  2 failed / 84 passed  -  names 'line 43: getattr(..., \"_mesh\")' and its consequence      post-fix  86 passed",
    "Precision: run against open PR #2198's head, examples/fleet/dashboard.py PASSES - it assigns its own self._mesh and reads",
    "   it back (own private state, not a missing SDK attribute), so the rule does not over-reach on legitimate code.",
    "Gate at 0b7d1ed5:  28633 passed / 257 skipped / 0 failed (672s, MUJOCO_GL=egl)  -  28547 pristine + 86 new cases.",
    "   ruff check + ruff format --check clean (1186 files);  mypy 0 errors outside examples/isaac_gs (byte-identical to base).",
]
GT, GL = 0.80, 0.07
gstep = (GT - GL) / (len(lines) - 1)
assert gstep > 0.030, gstep
gy = GT
for ln in lines:
    put(axg, 0.012, gy, ln, fontsize=9.5, family="monospace", transform=axg.transAxes)
    gy -= gstep
assert abs((gy + gstep) - GL) < 1e-9

for ax, yv, is_axes in placed:
    if is_axes:
        assert -0.03 <= yv <= 1.07, (yv, "axes-fraction text outside panel")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, (yv, lo, hi)

out = pathlib.Path(f"/tmp/mesh-teardown-{RID}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", out, im.shape, f"render max|delta|={dmax}")
