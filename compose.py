import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F = json.load(open("/tmp/art_facts.json"))
home = np.load("/tmp/art_home.npy"); honored = np.load("/tmp/art_honored.npy"); refused = np.load("/tmp/art_refused.npy")
rows = F["rows"]

# --- self-audit: every claim the figure makes ---
assert len(rows) == 5, rows
assert all(r["status"] == "error" for r in rows), "every probe must be refused"
assert all(r["qpos_same"] and r["ctrl_same"] and r["clock_same"] for r in rows), "a refusal touched the model"
moved = float(np.mean(np.any(home != honored, axis=2)))
delta = int(np.abs(honored.astype(int) - refused.astype(int)).max())
assert abs(moved - F["moved_frac"]) < 1e-9 and delta == F["refusal_max_delta"]
assert moved > 0.10, f"honored panel must be legible, got {moved:.2%}"
assert delta <= 1, f"refused panel must match the honored one, got {delta}"
assert F["arm_frac"] > 0.30, F["arm_frac"]
assert "reached" in F["honored_msg"]

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.2, 8.6), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[2.35, 1.0], hspace=0.20, wspace=0.06)
fig.suptitle(
    "Motion-primitive numeric guards run before the primitive touches the model",
    fontsize=15.5, fontweight="bold", y=0.975,
)
fig.text(0.5, 0.937,
         "MuJoCo headless (MUJOCO_GL=egl), the primitives' own inline MJCF arm. Tests only - no production line changes.",
         ha="center", fontsize=10.4, style="italic", color="#333333")

panels = [
    (home, "1. home", "the arm as the scene is built", "#555555"),
    (honored, "2. after a honored call", F["honored_msg"].strip(), "#1a7f37"),
    (refused, "3. after 5 refused calls", "qpos, ctrl and the sim clock bit-identical to panel 2", "#0b5cad"),
]
for col, (img, title, cap, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=12.6, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel("\n".join(cap[i:i+62] for i in range(0, len(cap), 62)), fontsize=9.0, color="#222222", labelpad=6)

axt = fig.add_subplot(gs[1, :]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
put(axt, 0.0, 0.955,
    f"panel 1 -> 2: {moved:.2%} of pixels differ (the honored call really moves the arm)   |   "
    f"panel 2 -> 3: max|delta| = {delta}/255 (byte-identical)",
    transform=axt.transAxes, fontsize=11.4, fontweight="bold", va="top", color="#111111")
put(axt, 0.0, 0.845, "the five refused calls between panel 2 and panel 3:",
    transform=axt.transAxes, fontsize=10.6, va="top", color="#333333")

TOP, LAST = 0.735, 0.135
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
for i, r in enumerate(rows):
    y = TOP - i * STEP
    put(axt, 0.0, y, r["call"], transform=axt.transAxes, fontsize=9.5, family="monospace",
        va="center", color="#111111")
    put(axt, 0.335, y, f"status={r['status']}", transform=axt.transAxes, fontsize=9.5,
        family="monospace", va="center", color="#b3261e", fontweight="bold")
    put(axt, 0.435, y, r["message"].strip(), transform=axt.transAxes, fontsize=9.3,
        family="monospace", va="center", color="#111111")
assert abs((TOP - (len(rows) - 1) * STEP) - LAST) < 1e-9
put(axt, 0.0, 0.035,
    "Each refusal names the field and the value it was given, and nothing was written: "
    "qpos unchanged, ctrl unchanged, clock unchanged, on all five.",
    transform=axt.transAxes, fontsize=10.0, va="center", style="italic", color="#0b5cad")

for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.10, (y, "axes-fraction text outside the panel")
    else:
        lo, hi = ax.get_ylim(); assert lo - 0.05 <= y <= hi + 0.07, (y, lo, hi)

out = pathlib.Path("/tmp/art_primitive_guards.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(matplotlib.image.imread(out) * 255, dtype=np.uint8)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int(np.sum(np.abs(band.astype(int) - 255).sum(axis=2) > 12))
    assert bad == 0, f"{name} border has {bad} non-white px"
print("OK", out, im.shape, f"moved={moved:.2%} delta={delta} arm={F['arm_frac']:.1%}")
