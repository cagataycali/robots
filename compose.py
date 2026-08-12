import json, os, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, imageio.v3 as iio

D = pathlib.Path(f"/tmp/art-{os.environ['GITHUB_RUN_ID']}")
M = json.load(open(D / "facts_main.json")); P = json.load(open(D / "facts_pr.json"))
assert M["tree"] != P["tree"], "the two arms must resolve different trees"
MUT = json.load(open(f"/tmp/mut-{os.environ['GITHUB_RUN_ID']}.json"))

m1, m2 = M["runs"]; p1, p2 = P["runs"]
t_m1, t_m2 = np.array(m1["trace"]), np.array(m2["trace"])
t_p1, t_p2 = np.array(p1["trace"]), np.array(p2["trace"])
ptp = lambda t: t.max() - t.min()

# --- audited claims -------------------------------------------------------
assert np.allclose(t_m1, t_p1), "run 1 must be identical on both trees"
assert ptp(t_m2) > 3.5 * ptp(t_m1), (ptp(t_m2), ptp(t_m1))
assert ptp(t_p2) <= ptp(t_p1), (ptp(t_p2), ptp(t_p1))
def cmp(a, b):
    A, B = iio.imread(D/a).astype(int), iio.imread(D/b).astype(int)
    d = np.abs(A - B)
    return (d.sum(2) > 8).mean() * 100, int(d.max())
r1_pct, r1_max = cmp("run1_main.png", "run1_pr.png")
r2_pct, r2_max = cmp("run2_main.png", "run2_pr.png")
assert r1_pct == 0.0 and r1_max <= 2, (r1_pct, r1_max)
assert r2_pct > 10.0, r2_pct
blind = sum(1 for lbl, a, b in MUT["rows"] if a[0] > 0 and b[0] == 0)
caught = sum(1 for lbl, a, b in MUT["rows"] if a[0] > 0)
assert (caught, len(MUT["rows"])) == (7, 7), (caught, len(MUT["rows"]))

fig = plt.figure(figsize=(15.6, 13.4), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 0.92, 0.80], hspace=0.30, wspace=0.13)
fig.suptitle("Two consecutive WBC balance rollouts on one sim: the auto-installed torque shim's cleanup",
             fontsize=15.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.958, "Unitree G1, real SONIC GR00T-WholeBodyControl-Balance.onnx, MuJoCo headless (EGL), "
                     "50 Hz x 150 ticks per rollout, target_velocity = 0",
         ha="center", fontsize=10.2, style="italic", color="#444")

panels = [
    ("run1_pr.png", "Rollout 1 - both trees\nBYTE-COMPARABLE across main and this PR",
     f"pelvis end {t_p1[-1]:.4f} m   excursion {ptp(t_p1):.4f} m\n"
     f"{r1_pct:.2f}% of pixels differ (max|delta| = {r1_max})", "#1a7f37"),
    ("run2_main.png", "Rollout 2 - main", 
     f"pelvis end {t_m2[-1]:.4f} m   excursion {ptp(t_m2):.4f} m  ({ptp(t_m2)/ptp(t_m1):.1f}x rollout 1)\n"
     f"stale controller still registered; actuators back to position servos", "#b42318"),
    ("run2_pr.png", "Rollout 2 - this PR",
     f"pelvis end {t_p2[-1]:.4f} m   excursion {ptp(t_p2):.4f} m  ({ptp(t_p2)/ptp(t_p1):.1f}x rollout 1)\n"
     f"cleanup unregistered it, so this rollout installed its own shim", "#1a7f37"),
]
for col, (fn, title, cap, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(iio.imread(D / fn)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(colour); s.set_linewidth(2.4)
    ax.set_title(title, fontsize=11.4, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(cap, fontsize=9.1, color="#333", labelpad=7)

axt = fig.add_subplot(gs[1, :])
x = np.arange(len(t_m1)) / 50.0
axt.plot(x, t_m1, color="#888", lw=2.6, label=f"rollout 1 (both trees) - excursion {ptp(t_m1):.4f} m")
axt.plot(x, t_m2, color="#b42318", lw=2.0, label=f"rollout 2, main - excursion {ptp(t_m2):.4f} m")
axt.plot(x, t_p2, color="#1a7f37", lw=2.0, ls="--", label=f"rollout 2, this PR - excursion {ptp(t_p2):.4f} m")
axt.axhspan(t_m2.min(), t_m2.max(), color="#b42318", alpha=0.07)
axt.axhspan(t_p2.min(), t_p2.max(), color="#1a7f37", alpha=0.10)
axt.set_xlabel("rollout time (s)", fontsize=10); axt.set_ylabel("pelvis height (m)", fontsize=10)
axt.set_title("Pelvis height. Rollout 1 is identical on both trees; only the SECOND rollout diverges.",
              fontsize=11.3, fontweight="bold", pad=6)
axt.legend(fontsize=9.4, loc="upper left"); axt.grid(alpha=0.28)

axl = fig.add_subplot(gs[2, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
placed = []
def put(x, y, s, **kw):
    placed.append(y); axl.text(x, y, s, transform=axl.transAxes, **kw)

rows = [
    ("what the hook decides on rollout 2", "declines: 'a manually-installed controller wins'", "declines is correct - but the entry is gone, so it installs"),
    ("registry after rollout 1's cleanup", "still holds rollout 1's controller", "cleared"),
    ("driven actuator bias type then", "AFFINE (position servo, gains restored)", "AFFINE (position servo, gains restored)"),
    ("so rollout 2 is driven by", "the STALE controller: PD torques into position servos", "a freshly installed shim matched to torque actuators"),
    ("rollout 2 pelvis excursion", f"{ptp(t_m2):.4f} m  ({ptp(t_m2)/ptp(t_m1):.1f}x rollout 1)", f"{ptp(t_p2):.4f} m  ({ptp(t_p2)/ptp(t_p1):.1f}x rollout 1)"),
    ("rollout 2 render vs the other tree", f"{r2_pct:.2f}% of pixels differ (max|delta| = {r2_max})", "-"),
]
TOP, LAST = 0.90, 0.30
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
put(0.005, 0.975, "", fontsize=1)
placed.pop()
axl.text(0.005, 0.965, "", transform=axl.transAxes)
put(0.245, 0.975, "main", ha="center", fontsize=11.2, fontweight="bold", color="#b42318")
put(0.72, 0.975, "this PR", ha="center", fontsize=11.2, fontweight="bold", color="#1a7f37")
y = TOP
for label, a, b in rows:
    put(0.005, y, label, fontsize=9.5, fontweight="bold", color="#222", va="center")
    put(0.245, y, a, ha="center", fontsize=9.3, color="#b42318", va="center")
    put(0.72, y, b, ha="center", fontsize=9.3, color="#1a7f37", va="center")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(0.005, 0.185, "Regression test: 1 of the 2 new cleanup cases fails on main "
                  "(`assert 'action_controller' not in ...`); the other is the over-reach control "
                  "(a foreign entry must survive), which passes on both.",
    fontsize=9.2, color="#333")
put(0.005, 0.125, f"Mutations of the hook's five no-op conditions: {caught} of {len(MUT['rows'])} caught by the new cases, "
                  f"{blind} of {len(MUT['rows'])} invisible to the 490 pre-existing wbc tests "
                  f"(including reverting the cleanup itself).",
    fontsize=9.2, color="#333")
put(0.005, 0.065, "Gate: 28541 passed / 257 skipped / 0 failed (full suite, MUJOCO_GL=egl) | ruff clean | "
                  "mypy 0 errors outside examples/isaac_gs | simulation.py 18 -> 14 missing lines (98.84% -> 99.10%).",
    fontsize=9.2, color="#333")
assert all(-0.03 <= v <= 1.07 for v in placed), [v for v in placed if not -0.03 <= v <= 1.07]

out = D / "wbc_auto_torque_cleanup.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.array(iio.imread(out))[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"WROTE {out}  {im.shape[1]}x{im.shape[0]}")
print(f"audited: run1 {r1_pct:.2f}% differ / run2 {r2_pct:.2f}% differ / "
      f"excursion main {ptp(t_m2):.4f} vs pr {ptp(t_p2):.4f} / mutations {caught}-{blind}")
