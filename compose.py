"""Compose the artifact from the two measured captures. Every number is asserted."""
import json, os, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

RID = os.environ["GITHUB_RUN_ID"]; D = pathlib.Path(f"/tmp/art-{RID}")
A = json.loads((D / "facts_main.json").read_text())
B = json.loads((D / "facts_branch.json").read_text())
assert A["tree"] != B["tree"], "both arms measured the same tree"

img = {f"{l}_{n}": np.load(D / f"{l}_{n}.npy") for l in ("main", "branch") for n in ("spawn", "control")}
def diff_frac(x, y):
    return float((np.abs(x.astype(int) - y.astype(int)).sum(2) > 8).mean())
def max_delta(x, y):
    return int(np.abs(x.astype(int) - y.astype(int)).max())

SPAWN_DIFF = diff_frac(img["main_spawn"], img["branch_spawn"])
CTL_DIFF = diff_frac(img["main_control"], img["branch_control"])
CTL_MAX = max_delta(img["main_control"], img["branch_control"])
# The claims this figure makes.
assert A["spawn"]["base_pos"][1:] == [0.0, 0.0] and B["spawn"]["base_pos"] == [0.0, 0.8, 0.445]
assert A["spawn"]["quat_norm"] < 0.01 and abs(B["spawn"]["quat_norm"] - 1.0) < 1e-9
assert A["spawn"]["thigh"] == B["spawn"]["thigh"] == 0.9
assert A["arms_at_spawn"] == B["arms_at_spawn"], "the arms must be untouched by the change"
assert A["control_one_prior"] == B["control_one_prior"], "the one-prior-robot layout must be unchanged"
assert SPAWN_DIFF > 0.10, SPAWN_DIFF
assert CTL_DIFF == 0.0 and CTL_MAX <= 2, (CTL_DIFF, CTL_MAX)
for im in img.values():
    assert ((im.max(2).astype(int) - im.min(2).astype(int)) > 45).mean() > 0.5, "frame has no scene in it"

fig = plt.figure(figsize=(15.4, 11.6), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], hspace=0.14, wspace=0.05,
                      left=0.015, right=0.985, top=0.925, bottom=0.02)
fig.suptitle("A floating base added behind two robots: spawned at an undefined pose vs at its declared pose",
             fontsize=15.5, fontweight="bold", y=0.977)
fig.text(0.5, 0.945, "MuJoCo headless (EGL). Same script, same scene, same seed - the only variable is the tree. "
         "Rendered at t=0, the instant add_robot returns.", ha="center", fontsize=10.5, style="italic", color="#333333")

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 9.4); kw.setdefault("va", "top"); kw.setdefault("family", "monospace")
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

def panel(ax, key, title, colour, caption):
    ax.imshow(img[key]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(3.2)
    ax.set_title(title, fontsize=12.4, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(caption, fontsize=9.7, family="monospace", labelpad=7)

sa, sb = A["spawn"], B["spawn"]
panel(fig.add_subplot(gs[0, 0]), "main_spawn", "main - Go2 added behind two robots", "#b3261e",
      f"base_pos = {[round(v, 4) for v in sa['base_pos']]}   |quat| = {sa['quat_norm']:.6f}\n"
      f"add_robot returned status='success'; FL_thigh_joint = {sa['thigh']}")
panel(fig.add_subplot(gs[0, 1]), "branch_spawn", "this change - the same call", "#1b7f37",
      f"base_pos = {[round(v, 4) for v in sb['base_pos']]}   |quat| = {sb['quat_norm']:.6f}\n"
      f"the pose the model declares; FL_thigh_joint = {sb['thigh']}")
panel(fig.add_subplot(gs[1, 0]), "branch_control", "control: the same Go2 behind ONE robot", "#2d5b9a",
      f"byte-comparable across trees: {CTL_DIFF * 100:.2f}% of pixels differ, max|delta| = {CTL_MAX}/255\n"
      "the layout where the leftovers happened to be right is untouched")

ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.0, "Measured", fontsize=12.6, fontweight="bold", family="sans-serif", transform=ax.transAxes)
rows = [
    ("free-joint qpos, 2 prior robots", f"{[round(v, 6) for v in sa['qpos']]}", f"{[round(v, 6) for v in sb['qpos']]}"),
    ("|quaternion|", f"{sa['quat_norm']:.6f}  (not a rotation)", f"{sb['quat_norm']:.6f}"),
    ("base_pos.z", f"{sa['base_pos'][2]:.4f} m", f"{sb['base_pos'][2]:.4f} m"),
    ("keyframe hinge FL_thigh_joint", f"{sa['thigh']}  (correct)", f"{sb['thigh']}  (correct)"),
    ("add_robot status", "success", "success"),
    ("same Go2, ONE prior robot", f"z = {A['control_one_prior']['base_pos'][2]:.4f} m",
     f"z = {B['control_one_prior']['base_pos'][2]:.4f} m"),
    ("the two parked arms", "unchanged", "unchanged"),
]
TOP, LAST = 0.905, 0.300
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for label, left, right in rows:
    put(ax, 0.0, y, label, fontsize=9.6, family="sans-serif", fontweight="bold", transform=ax.transAxes)
    put(ax, 0.03, y - 0.052, f"main    {left}", color="#b3261e", transform=ax.transAxes)
    put(ax, 0.03, y - 0.096, f"branch  {right}", color="#1b7f37", transform=ax.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
foot = (
    f"Renders differ on {SPAWN_DIFF * 100:.2f}% of pixels (top row).  The residue is not reliably zero: the same\n"
    f"slice has held denormals ({sa['quat_norm']:.6f} here comes from ~1e-3 leftovers; 9.269e-310 seen elsewhere),\n"
    "so the spawn pose was undefined rather than merely wrong."
)
put(ax, 0.0, 0.165, foot, fontsize=9.3, color="#333333", transform=ax.transAxes)

for a_, y_, is_axes in placed:
    lo, hi = (-0.03, 1.10) if is_axes else a_.get_ylim()
    assert lo <= y_ <= hi, (y_, lo, hi)

out = D / "free_base_spawn_pose.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.3, facecolor="white")
im = np.asarray(plt.imread(out) * 255).astype(int)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print(f"OK {out}  {im.shape}  spawn_diff={SPAWN_DIFF*100:.2f}%  ctl_diff={CTL_DIFF*100:.2f}% max={CTL_MAX}")
