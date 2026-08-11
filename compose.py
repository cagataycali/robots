import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np

OUT = pathlib.Path("_art")
F = json.loads((OUT / "facts.json").read_text())
ROOT = pathlib.Path.cwd()
assert F["tree"] == str(ROOT), (F["tree"], ROOT)

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.2, 14.4), dpi=118)
gs = fig.add_gridspec(3, 2, height_ratios=[1.42, 0.66, 0.62], hspace=0.20, wspace=0.06)

fig.suptitle("VeraPolicy: the lazy MinkIKBridge build - the branch every eef-delta rollout takes once",
             fontsize=15.5, fontweight="bold", y=0.972)
fig.text(0.5, 0.947, "Tests only; no production line changes. The renders show the capability the newly covered "
         "two statements serve, headless MuJoCo (MUJOCO_GL=egl).",
         ha="center", fontsize=10.4, style="italic", color="#333333")

# ---- row 1: the real renders -------------------------------------------------
for col, (fn, title, sub) in enumerate([
    ("home.png", "Before the first inference",
     f"policy._ik_bridge = {F['bridge_before_first_inference']}   hand z = {F['hand_home'][2]:.4f} m"),
    ("driven.png", f"After {F['actions_applied']} actions through the built bridge",
     f"MinkIKBridge(frame={F['frame_name']}, type={F['frame_type']}, qp={F['qp_solver']})   "
     f"hand z = {F['hand_driven'][2]:.4f} m"),
]):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(iio.imread(OUT / fn)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12.4, fontweight="bold", pad=7)
    ax.set_xlabel(sub, fontsize=9.6, family="monospace", labelpad=7)

# ---- row 2: coverage ---------------------------------------------------------
axc = fig.add_subplot(gs[1, :]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
put(axc, 0.0, 0.94, "strands_robots/policies/vera/provider.py  -  the build was never executed",
    fontsize=12.6, fontweight="bold", transform=axc.transAxes)
rows = [
    ("", "missing lines", "coverage", "vera suite", ""),
    ("before  (this file ignored)", "7   incl. 725-727", "98%", f"{F['suite']['before']} passed",
     "the two statements that construct a bridge"),
    ("after   (this file included)", "5", "99%", f"{F['suite']['after']} passed",
     "0 of the build's lines remain"),
]
TOP, LAST = 0.70, 0.16
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
xs = [0.0, 0.30, 0.455, 0.565, 0.695]
y = TOP
for i, r in enumerate(rows):
    head = i == 0
    for x, cell in zip(xs, r):
        put(axc, x, y, cell, fontsize=10.5 if not head else 10.2, family="monospace",
            fontweight="bold" if head else "normal", color="#555555" if head else "#111111",
            transform=axc.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(axc, 0.0, 0.03, "Every existing contract injects a bridge (policy._ik_bridge = FakeBridge()), so "
    "`if self._ik_bridge is None` is always False.", fontsize=9.9, style="italic",
    color="#444444", transform=axc.transAxes)

# ---- row 3: mutations --------------------------------------------------------
axm = fig.add_subplot(gs[2, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 0.95, "Plausible regressions in the build, run against both arms",
    fontsize=12.6, fontweight="bold", transform=axm.transAxes)
muts = [("mutation of _ensure_ik_bridge", "these tests", "pre-existing vera (728)")] + [tuple(m) for m in F["mutations"]]
TOP2, LAST2 = 0.78, 0.13
step2 = (TOP2 - LAST2) / (len(muts) - 1)
assert step2 > 0.030, step2
y = TOP2
n_blind = 0
for i, (label, a, b) in enumerate(muts):
    head = i == 0
    blind = (not head) and label.strip().startswith("M") and b.startswith("0 failed")
    if blind: n_blind += 1
    put(axm, 0.0, y, label, fontsize=10.4, family="monospace",
        fontweight="bold" if head else "normal", color="#555555" if head else "#111111",
        transform=axm.transAxes)
    put(axm, 0.545, y, a, fontsize=10.4, family="monospace",
        fontweight="bold" if head else "normal", color="#555555" if head else "#0b6623",
        transform=axm.transAxes)
    put(axm, 0.700, y, b, fontsize=10.4, family="monospace",
        fontweight="bold" if head else "normal",
        color="#555555" if head else ("#b00020" if blind else "#111111"), transform=axm.transAxes)
    if blind:
        put(axm, 0.928, y, "<- BLIND", fontsize=10.4, family="monospace",
            fontweight="bold", color="#b00020", transform=axm.transAxes)
    y -= step2
assert abs((y + step2) - LAST2) < 1e-9, (y, LAST2)
assert n_blind == 3, n_blind
put(axm, 0.0, 0.02, f"3 of 4 are invisible to the whole pre-existing vera suite. Descent driven through the "
    f"built bridge: {F['descent_cm']:.2f} cm over {F['actions_applied']} actions "
    f"({F['infer_calls']} inferences).", fontsize=9.9, style="italic", color="#444444", transform=axm.transAxes)

for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.08, (y, "axes-fraction text outside the panel")

fig.savefig(OUT / "vera_ik_bridge_lazy_build.png", bbox_inches="tight", pad_inches=0.30,
            facecolor="white")
img = iio.imread(OUT / "vera_ik_bridge_lazy_build.png")
h, w = img.shape[:2]
for name, band in (("top", img[:8]), ("bottom", img[-8:]), ("left", img[:, :8]), ("right", img[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(-1) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"figure {w}x{h}  borders clean  blind={n_blind}  descent={F['descent_cm']}cm")
