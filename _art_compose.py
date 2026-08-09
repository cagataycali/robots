import json, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("/tmp/art244")
F = json.loads((OUT / "facts.json").read_text())
a, b = np.load(OUT / "as_written.npy"), np.load(OUT / "fixed.npy")

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(14.4, 8.9), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[3.05, 1.28], hspace=0.30, wspace=0.06)

for col, (tag, title, colour) in enumerate([
    ("as_written", "main: add_camera(camera_name=...)", "#b3261e"),
    ("fixed", "this change: add_camera(name=...)", "#1b7f3b"),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(a if tag == "as_written" else b)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(3.0)
    ax.set_title(title, fontsize=12.5, color=colour, fontweight="bold", pad=8)
    f = F[tag]
    ax.set_xlabel(
        f"add_camera -> {f['add_status']}   |   cameras in observation: {f['image_keys']}\n"
        f"view handed to the policy: '{f['camera_served_to_policy']}'"
        + ("   (the 'front' view it was trained on)" if f["front_camera_exists"]
           else "   ('front' never existed - the request was refused and discarded)"),
        fontsize=10.2, color=colour, labelpad=9,
    )

axt = fig.add_subplot(gs[1, :]); axt.axis("off")
axt.set_xlim(0, 1); axt.set_ylim(0, 1)
rows = [
    ("dispatch verdict for add_camera", f"error: unknown parameter 'camera_name'", "success: Camera 'front' added"),
    ("was the error surfaced to the caller?", "no - the envelope was discarded", "yes - _must() raises on it"),
    ("cameras present in the observation", "['default']", "['default', 'front']"),
    ("get_observation parameter (line 77)", "camera_name -> also refused", "removed (takes robot_name/skip_images)"),
    ("where the failure appears", "later, as a policy image-keys complaint", "at the call that was refused"),
]
TOP, LAST = 0.90, 0.14
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
put(axt, 0.008, 0.985, "measured on Thor (MuJoCo, MUJOCO_GL=egl) -- one tree; the only variable is the parameter name",
    transform=axt.transAxes, fontsize=10.4, style="italic", color="#333333")
y = TOP
for label, left, right in rows:
    put(axt, 0.008, y, label, transform=axt.transAxes, fontsize=10.6, fontweight="bold")
    put(axt, 0.345, y, left, transform=axt.transAxes, fontsize=10.6, color="#b3261e", family="monospace")
    put(axt, 0.700, y, right, transform=axt.transAxes, fontsize=10.6, color="#1b7f3b", family="monospace")
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
put(axt, 0.008, 0.035,
    f"panels differ on {F['diff_fraction'] * 100:.1f}% of pixels; both frames are real renders "
    f"(saturated-pixel fraction {F['as_written']['saturation']:.2f} / {F['fixed']['saturation']:.2f})",
    transform=axt.transAxes, fontsize=10.0, color="#333333")

fig.suptitle("examples/vla/molmoact2_sim_pickplace.py: the view the policy is actually fed",
             fontsize=14.2, fontweight="bold", y=0.985)

for ax, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.10, (yv,)
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, (yv, lo, hi)

p = OUT / "camera_param_244.png"
fig.savefig(p, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border not clean: {n}"
print("wrote", p, im.shape)
