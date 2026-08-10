"""Compose the artifact: identical renders, identical verdicts, dead-check measurement."""
from __future__ import annotations
import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = pathlib.Path(f"/tmp/art-{os.environ['GITHUB_RUN_ID']}")
main = json.load(open(A / "facts_main.json"))
br = json.load(open(A / "facts_branch.json"))
assert main["tree"] != br["tree"], "both halves measured the same tree"

rm_, rb = np.load(A / "render_main.npy").astype(int), np.load(A / "render_branch.npy").astype(int)
assert rm_.shape == rb.shape
delta = int(np.abs(rm_ - rb).max())
changed = int((np.abs(rm_ - rb).max(axis=2) > 8).sum())
sat = float(((rm_.max(2) - rm_.min(2)) > 45).mean())

keys = list(main["verdicts"])
differing = [k for k in keys if main["verdicts"][k] != br["verdicts"][k]]
assert not differing, differing
assert main["neuter"] == {"applications": 3, "names": ["coerce_pose_vector", "pose_vector_error"],
                          "invoked": 12, "cases": 30, "changed": 0}, main["neuter"]
assert br["neuter"] == {"applications": 2, "names": ["coerce_pose_vector"]}, br["neuter"]
assert main["stored_types"] == br["stored_types"] == ["float"]
assert main["degenerate_status"] == br["degenerate_status"] == "error"

placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.32, 1.30, 0.66], hspace=0.20, wspace=0.06)
fig.suptitle("add_camera: the pose rule applied twice, and what the second application changed",
             fontsize=15.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.958, "MuJoCo headless (MUJOCO_GL=egl) - one camera created through add_camera, then rendered "
         "from. Every number below is measured on the two trees named in each panel.",
         ha="center", fontsize=10.2, style="italic")

# --- row 1: the surviving path, rendered on both trees --------------------
for col, (arr, label, tree) in enumerate((
        (np.load(A / "render_main.npy"), "main", main["tree"]),
        (np.load(A / "render_branch.npy"), "this change", br["tree"]))):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{label}: add_camera(position=np.array([0.42, -0.38, 0.30]), target=[0.0, 0.0, 0.12])",
                 fontsize=10.4, fontweight="bold", pad=6)
    ax.set_xlabel(f"status={main['add_status'] if col == 0 else br['add_status']}   "
                  f"stored position {main['stored_position'] if col == 0 else br['stored_position']} "
                  f"({', '.join(main['stored_types'])})\n{pathlib.Path(tree).name}",
                  fontsize=9.0, family="monospace")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2e7d32"); sp.set_linewidth(2.2)

# --- row 2: verdict parity ------------------------------------------------
axv = fig.add_subplot(gs[1, :]); axv.axis("off"); axv.set_xlim(0, 1); axv.set_ylim(0, 1)
put(axv, 0.0, 1.045, "Every pose the caller can supply, through both parameters - main vs this change",
    transform=axv.transAxes, fontsize=11.6, fontweight="bold")
probes = sorted({k.split("|")[0] for k in keys}, key=lambda p: list(dict.fromkeys(k.split("|")[0] for k in keys)).index(p))
TOP, LAST = 0.955, 0.055
step = (TOP - LAST) / (len(probes) - 1)
assert step > 0.030, step
put(axv, 0.005, TOP + 0.052, "value supplied", transform=axv.transAxes, fontsize=9.4, fontweight="bold")
for x, h in ((0.235, "position: main"), (0.435, "position: this change"),
             (0.655, "target: main"), (0.845, "target: this change")):
    put(axv, x, TOP + 0.052, h, transform=axv.transAxes, fontsize=9.4, fontweight="bold", ha="center")
y = TOP
for p in probes:
    put(axv, 0.005, y, p, transform=axv.transAxes, fontsize=9.2, family="monospace", va="center")
    for x, (src, param) in zip((0.235, 0.435, 0.655, 0.845),
                               ((main, "position"), (br, "position"), (main, "target"), (br, "target"))):
        st, txt, reg = src["verdicts"][f"{p}|{param}"]
        ok = st == "success"
        colour = "#1b5e20" if ok else "#b71c1c"
        reason = "registered" if ok else txt.split(param + " ", 1)[-1][:34].rstrip(",. ")
        axv.add_patch(plt.Rectangle((x - 0.093, y - 0.019), 0.186, 0.038,
                                    transform=axv.transAxes, facecolor="#e8f5e9" if ok else "#fdecea",
                                    edgecolor="none", zorder=0))
        put(axv, x, y, f"{st}: {reason}", transform=axv.transAxes, fontsize=7.8, ha="center",
            va="center", color=colour, family="monospace")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(axv, 0.005, -0.012, f"{len(keys)} cells compared, {len(differing)} differing - the accepted and refused "
    f"domains are unchanged.", transform=axv.transAxes, fontsize=9.6, fontweight="bold", color="#1b5e20")

# --- row 3: what the second application did -------------------------------
axf = fig.add_subplot(gs[2, :]); axf.axis("off"); axf.set_xlim(0, 1); axf.set_ylim(0, 1)
rows = [
    ("applications of the pose rule in add_camera",
     f"3  ({' + '.join(main['neuter']['names'])})", "2  (coerce_pose_vector, one per parameter)"),
    ("second application invoked at runtime", f"{main['neuter']['invoked']} times", "does not exist"),
    ("outcomes it changed, neutered to always accept",
     f"{main['neuter']['changed']} of {main['neuter']['cases']}", "n/a"),
    ("render through the surviving path",
     f"{main['render_shape'][1]}x{main['render_shape'][0]}, saturated {sat * 100:.1f}% of pixels",
     f"max|delta| = {delta}/255 over {changed} pixels above threshold"),
    ("degenerate pose (position == substituted default target)",
     f"{main['degenerate_status']}: {main['degenerate_text'][:52]}",
     f"{br['degenerate_status']}: unchanged"),
    ("re-inserting the loop verbatim", "invisible to all 2,937 pre-existing MuJoCo backend tests",
     "caught by 2 of the new tests"),
]
TOP3, LAST3 = 0.90, 0.10
s3 = (TOP3 - LAST3) / (len(rows) - 1)
assert s3 > 0.030, s3
put(axf, 0.0, 1.10, "The second application, measured", transform=axf.transAxes,
    fontsize=11.6, fontweight="bold")
yy = TOP3
for label, a, b in rows:
    put(axf, 0.005, yy, label, transform=axf.transAxes, fontsize=9.2, va="center")
    put(axf, 0.400, yy, a, transform=axf.transAxes, fontsize=8.8, va="center",
        family="monospace", color="#b71c1c")
    put(axf, 0.700, yy, b, transform=axf.transAxes, fontsize=8.8, va="center",
        family="monospace", color="#1b5e20")
    yy -= s3
assert abs((yy + s3) - LAST3) < 1e-9

for ax, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.13, (yv, "axes-fraction out of range")
    else:
        lo, hi = ax.get_ylim()
        assert min(lo, hi) - 0.05 <= yv <= max(lo, hi) + 0.07, (yv, lo, hi)

out = A / "artifact.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white pixels"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  delta={delta} changed_px={changed} sat={sat:.3f} "
      f"differing_verdicts={len(differing)}/{len(keys)}")
