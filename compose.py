import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

F = json.loads(pathlib.Path("_probe/artifact.json").read_text())
rep, cor = F["reported"], F["correct"]
# audit every number the figure will draw
assert (rep["calls"], rep["ok"], rep["err"], rep["cylinder_geoms"], rep["listed"]) == (14, 10, 4, 0, 10), rep
assert (cor["calls"], cor["ok"], cor["err"], cor["cylinder_geoms"], cor["listed"]) == (14, 14, 0, 4, 14), cor
assert F["crop_diff_frac"] > 0.10

fig = plt.figure(figsize=(13.6, 8.9), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.32, 1.0], hspace=0.16, wspace=0.07)
fig.suptitle("strands-robots #256: list_objects() does NOT drop cylinders -- the 4 cylinders were never created",
             fontsize=13.5, fontweight="bold", y=0.975)

for col, (tag, lab, sub) in enumerate([
    ("reported", "size=[radius, height]  (2 components)",
     "add_object refused 4 of 14 calls -> 0 cylinder geoms in the model"),
    ("correct", "size=[diameter, unused, full height]  (3 components)",
     "14 of 14 accepted -> 4 cylinder geoms, all 14 listed by list_objects()")]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(Image.open(f"_probe/{tag}_crop.png")); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#b3261e" if tag == "reported" else "#1b7f3b"); s.set_linewidth(2.6)
    ax.set_title(lab, fontsize=10.6, fontweight="bold",
                 color="#b3261e" if tag == "reported" else "#1b7f3b", pad=7)
    ax.set_xlabel(sub, fontsize=9.4)

axt = fig.add_subplot(gs[1, :]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
placed = []
def put(x, y, s, **kw):
    placed.append(y); axt.text(x, y, s, transform=axt.transAxes, **kw)

rows = [
    ("Reported claim", "Measured on Thor (mujoco 3.11.0, strands-robots @ upstream/main)", "Verdict"),
    ("list_objects() silently drops cylinders",
     "a correctly-sized cylinder IS listed: '- can: cylinder at [0.3, 0.1, 0.06], 0.1kg'", "FALSE"),
    ("cylinders 'render fine' but are missing from introspection",
     f"reporter's spelling compiles {rep['cylinder_geoms']} cylinder geoms -- never created, so never rendered", "FALSE"),
    ("agent under-counts (10 of 14)",
     f"add_object accepted exactly {rep['ok']} of {rep['calls']} calls; the agent narrated the truth", "NOT A BUG"),
    ("shape-specific filtering in list_objects",
     "list_objects reads world.objects and formats obj.shape -- no shape branch exists", "FALSE"),
    ("silent failure",
     "each refused call returned status='error' naming shape, required count, layout and the value", "NOT SILENT"),
    ("agent path differs from direct path",
     "sim(action='add_object', ...) returns a byte-identical refusal", "IDENTICAL"),
]
put(0.012, 0.945, "Per-claim verdict", fontsize=11.4, fontweight="bold", va="top")
TOP, LAST = 0.845, 0.115
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
y = TOP
for i, (claim, measured, verdict) in enumerate(rows):
    head = i == 0
    col = "#333333" if head else ("#1b7f3b" if verdict in ("NOT A BUG", "IDENTICAL", "NOT SILENT") else "#b3261e")
    put(0.012, y, claim, fontsize=9.5, fontweight="bold" if head else "normal", va="top",
        family="DejaVu Sans", color="#333333")
    put(0.335, y, measured, fontsize=9.0, fontweight="bold" if head else "normal", va="top",
        family="DejaVu Sans Mono" if not head else "DejaVu Sans", color="#333333")
    put(0.885, y, verdict, fontsize=9.4, fontweight="bold", va="top", color=col)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(0.012, 0.048,
    f"Root cause: MuJoCo add_object documents cylinder size as [diameter, unused, full height] (3 components); "
    f"the reporter passed MJCF's [radius, half-length] (2).  Crop diff {F['crop_diff_frac']*100:.1f}% of the densest "
    f"changed window; the 4 cylinders are {F['panel_diff_frac']*100:.1f}% of the full frame.",
    fontsize=8.5, va="top", color="#555555", style="italic")
for yy in placed:
    assert -0.03 <= yy <= 1.06, yy

out = pathlib.Path("_probe/verdict_2256.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
import numpy as np
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    assert int((np.abs(band.astype(int) - 255).sum(2) > 12).sum()) == 0, name
print("wrote", out, im.shape)
