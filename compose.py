import json, pathlib, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = pathlib.Path("_art")
F = json.loads((A / "facts.json").read_text())
M = json.loads((A / "mutations.json").read_text())
frame = np.load(A / "frame.npy")

# ---- audit the measurement before drawing anything ----
assert F["frame_shape"] == [520, 640, 3], F["frame_shape"]
assert F["saturated_frac"] > 0.5, f"frame looks blank: {F['saturated_frac']}"
assert F["consumers"]["render"]["kind"] == "envelope"
assert F["consumers"]["render_depth"]["kind"] == "envelope"
assert F["consumers"]["get_frame"]["kind"] == "raise"
assert F["extra"]["HybridCompositor.render"]["kind"] == "raise"
assert F["extra"]["get_world_point"]["kind"] == "envelope"
assert F["unpack_hazard"]["binds"] == ["status", "content"]
assert all(r["new"] > 0 for r in M["rows"]), "a mutation the new tests miss"
assert all(r["old"] == 0 for r in M["rows"]), "a mutation the old suite catches"
N_MUT = len(M["rows"])
for name in ("OpenGL",):
    for k in ("get_frame",):
        assert name in F["consumers"][k]["text"]

placed = []
def put(ax, x, y, s, axes_coords=True, **kw):
    placed.append((ax, y, axes_coords))
    if axes_coords:
        kw["transform"] = ax.transAxes
    return ax.text(x, y, s, **kw)

MONO = {"family": "monospace"}
fig = plt.figure(figsize=(16.0, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.26, 1.02, 0.20], width_ratios=[1.0, 1.16],
                      hspace=0.16, wspace=0.10)

# ---------- row 1 left: the real frame ----------
axf = fig.add_subplot(gs[0, 0]); axf.imshow(frame); axf.set_xticks([]); axf.set_yticks([])
axf.set_title("What get_frame returns on a GL-capable host\n(unchanged by this PR)",
              fontsize=12.5, fontweight="bold", pad=8)
axf.set_xlabel(
    f"real headless MuJoCo render via sim.get_frame('look', 640, 520)\n"
    f"rgb {tuple(F['frame_shape'])} uint8   depth {tuple(F['depth_shape'])} float32, "
    f"{F['depth_min']}-{F['depth_max']} m   saturated {F['saturated_frac']*100:.1f}%\n"
    "On a host with no EGL/OSMesa there is no frame at all -- what a caller gets instead is the contract below.",
    fontsize=9.4)

# ---------- row 1 right: the consumer matrix ----------
axm = fig.add_subplot(gs[0, 1]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 1.045, "The four consumers of _get_renderer, when it returns None",
    fontsize=12.5, fontweight="bold")
hdr_y, last_y = 0.90, 0.30
rows = [
    ("render",               "status=error envelope", "pinned",   "pinned"),
    ("render_depth",         "status=error envelope", "pinned",   "pinned"),
    ("_get_sim_observation", "skips the camera",      "pinned",   "pinned"),
    ("get_frame",            "raises RuntimeError",   "UNPINNED", "pinned"),
]
step = (hdr_y - last_y) / (len(rows) - 1)
assert step > 0.030, step
cols = (0.0, 0.315, 0.635, 0.815)
for x, t in zip(cols, ("consumer", "no-GL channel", "before", "after")):
    put(axm, x, hdr_y + 0.075, t, fontsize=10.4, fontweight="bold", color="#333333")
y = hdr_y
for name, channel, before, after in rows:
    hole = before == "UNPINNED"
    if hole:
        axm.add_patch(Rectangle((-0.012, y - 0.031), 1.02, 0.072, transform=axm.transAxes,
                                facecolor="#ffe2e2", edgecolor="none", zorder=0))
    put(axm, cols[0], y, name, fontsize=10.6, fontweight="bold" if hole else "normal", **MONO)
    put(axm, cols[1], y, channel, fontsize=10.0)
    put(axm, cols[2], y, before, fontsize=10.2, fontweight="bold" if hole else "normal",
        color="#b00020" if hole else "#1b7a3d")
    put(axm, cols[3], y, after, fontsize=10.2, color="#1b7a3d", fontweight="bold" if hole else "normal")
    y -= step
assert abs((y + step) - last_y) < 1e-9, (y, last_y)

put(axm, 0.0, 0.185, "Why the channels must differ (measured, not asserted):", fontsize=10.4, fontweight="bold")
put(axm, 0.0, 0.118,
    "a two-key envelope unpacks with no complaint at a consumer's\n"
    f"  rgb, depth = sim.get_frame(...)   ->  binds {tuple(F['unpack_hazard']['binds'])}, "
    f"np.asarray('status').shape == {tuple(F['unpack_hazard']['asarray_shape'])}",
    fontsize=9.5, **MONO)
put(axm, 0.0, 0.020,
    "so 'harmonising' get_frame onto the envelope channel fails far from the missing GL context.",
    fontsize=9.6, style="italic", color="#444444")

# ---------- row 2 left: the two in-process consumers ----------
axc = fig.add_subplot(gs[1, 0]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
put(axc, 0.0, 1.03, "The raise is what makes both documented consumers actionable",
    fontsize=12.0, fontweight="bold")
lines = [
    ("HybridCompositor.render", F["extra"]["HybridCompositor.render"]["type"] + " propagates:",
     F["extra"]["HybridCompositor.render"]["text"]),
    ("get_world_point", "converts it into its own error envelope:",
     F["extra"]["get_world_point"]["text"]),
]
top, floor = 0.88, 0.12
per = (top - floor) / len(lines)
assert per > 0.24, per
yy = top
for name, kind, text in lines:
    put(axc, 0.0, yy, name, fontsize=10.8, fontweight="bold", **MONO)
    put(axc, 0.0, yy - 0.085, kind, fontsize=9.8, color="#444444")
    wrapped = text if len(text) < 62 else text[:60] + "\n  " + text[60:]
    if len(wrapped) > 130:
        head, tail = wrapped.split("\n  ", 1)
        wrapped = head + "\n  " + tail[:60] + "\n  " + tail[60:]
    put(axc, 0.02, yy - 0.185, wrapped, fontsize=8.9, color="#1b3a5c", **MONO)
    yy -= per
assert yy + per > floor - 1e-9

# ---------- row 2 right: the mutation matrix ----------
axx = fig.add_subplot(gs[1, 1]); axx.axis("off"); axx.set_xlim(0, 1); axx.set_ylim(0, 1)
put(axx, 0.0, 1.03, f"Mutation table -- would a regression be caught? ({N_MUT} plausible regressions)",
    fontsize=12.0, fontweight="bold")
mh, ml = 0.88, 0.20
mstep = (mh - ml) / (N_MUT - 1)
assert mstep > 0.030, mstep
mcols = (0.0, 0.055, 0.735, 0.885)
for x, t in zip(mcols, ("", "regression introduced", "new tests", f"suite as it\nstands ({M['arm_old_size']})")):
    put(axx, x, mh + 0.085, t, fontsize=9.8, fontweight="bold", color="#333333")
my = mh
for r in M["rows"]:
    axx.add_patch(Rectangle((0.715, my - 0.028), 0.30, 0.062, transform=axx.transAxes,
                            facecolor="#fdecec", edgecolor="none", zorder=0))
    put(axx, mcols[0], my, r["id"], fontsize=10.0, fontweight="bold", **MONO)
    put(axx, mcols[1], my, r["what"], fontsize=9.5)
    put(axx, mcols[2], my, f"{r['new']} failed", fontsize=10.0, color="#1b7a3d", fontweight="bold")
    put(axx, mcols[3], my, "0  <- BLIND", fontsize=10.0, color="#b00020", fontweight="bold", **MONO)
    my -= mstep
assert abs((my + mstep) - ml) < 1e-9
put(axx, 0.0, 0.075,
    f"{N_MUT} of {N_MUT} caught by the new tests, 0 of {N_MUT} by the {M['arm_old_size']} tests already covering\n"
    "get_frame, the renderer-None family and the compositor -- and each mutation is caught\n"
    "by a different subset, so the five tests are not copies of one another.",
    fontsize=9.6, style="italic", color="#444444")

# ---------- row 3: footer ----------
axg = fig.add_subplot(gs[2, :]); axg.axis("off"); axg.set_xlim(0, 1); axg.set_ylim(0, 1)
put(axg, 0.0, 0.62,
    "Tests only -- no production line changes, so no policy / simulation / rendering / recording / asset behaviour moves.",
    fontsize=10.4, fontweight="bold")
put(axg, 0.0, 0.14,
    "rendering.py line 1302 (the raise) covered for the first time  |  all 7 tests in the module pass on a "
    "GL-free host  |  ruff + mypy clean  |  full suite green",
    fontsize=9.8, color="#333333", **MONO)

fig.suptitle("No-OpenGL-context contract: the one renderer consumer that raises was unpinned",
             fontsize=14.6, fontweight="bold", y=0.985)

# ---- layout guards ----
for ax, y, axes_coords in placed:
    if axes_coords:
        assert -0.05 <= y <= 1.10, f"text at axes-y {y}"
    else:
        lo, hi = ax.get_ylim(); assert lo <= y <= hi, (y, lo, hi)

out = A / "no_gl_context_contract.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(plt.imread(out) * 255).astype(int)[:, :, :3]
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}")
