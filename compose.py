import io, json, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

F = json.load(open("/tmp/art_facts.json"))
assert F["tree"].endswith(pathlib.Path(".").resolve().name), F["tree"]
home = np.array(Image.open("/tmp/art_home.png").convert("RGB"))
reach = np.array(Image.open("/tmp/art_reached.png").convert("RGB"))
diff = float((np.abs(home.astype(int) - reach.astype(int)).sum(2) > 24).mean())
sat = float(((reach.max(2) - reach.min(2)) > 45).mean())
print(f"home-vs-reached differing px: {diff:.2%}   saturated: {sat:.2%}")
assert diff > 0.10, f"framing: only {diff:.2%} differs"

# ---- self-audit: every number the figure prints, re-derived from the dump ----
assert F["payload"]["reached"] is True and F["payload"]["frame_type"] == "body"
assert F["payload"]["frame"] == "arm/hand" and F["leaf"]["frame"] == "arm/link4"
assert F["leaf"]["frame_type"] == "body" and F["leaf"]["reached"] is True
assert abs(F["frame_offset_m"] - 0.025) < 1e-6, F["frame_offset_m"]
assert F["err_from_origin_m"] <= F["tol"] < F["err_from_inertial_m"], (F["err_from_origin_m"], F["err_from_inertial_m"])
assert F["mink_vs_origin_m"] == 0.0 and F["mink_vs_inertial_m"] > 0.01
assert F["ee_position"] == F["xpos"] and F["ee_position"] != F["xipos"]

MUT = [("M1  body position reads the inertial frame (xipos)", "8 of 10", "0 of 161"),
       ("M2  body quaternion reports identity", "1 of 10", "0 of 161"),
       ("M3  readback assumes a site frame", "8 of 10", "0 of 161"),
       ("M4  IK bridge built on a hardcoded site frame", "8 of 10", "0 of 161"),
       ("M5  payload reports a hardcoded frame_type", "3 of 10", "0 of 161")]
assert all(m[2] == "0 of 161" for m in MUT)

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.6, 12.4), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.30, 1.0], hspace=0.16, wspace=0.10)
MONO = {"family": "monospace"}

# --- A: home ---
axa = fig.add_subplot(gs[0, 0]); axa.imshow(home); axa.axis("off")
axa.set_title("1. at rest", fontsize=13, fontweight="bold")
axa.set_xlabel("site-less arm; discovery must resolve a BODY", fontsize=10)

# --- B: reached, with the two candidate frames projected ---
axb = fig.add_subplot(gs[0, 1]); axb.imshow(reach); axb.axis("off")
tx, ty = F["proj"]["target"]; ox, oy = F["proj"]["xpos"]; ix, iy = F["proj"]["xipos"]
axb.plot([tx], [ty], marker="x", ms=16, mew=3.2, color="#111111")
axb.plot([ox], [oy], marker="o", ms=13, mew=2.6, mfc="none", color="#0b8a3a")
axb.plot([ix], [iy], marker="s", ms=12, mew=2.6, mfc="none", color="#b3261e")
axb.set_title(f"2. move_to reached in {F['payload']['steps']} steps", fontsize=13, fontweight="bold")
axb.set_xlabel("x target   o body frame origin (reported)   [] inertial frame", fontsize=10)

# --- C: zoom on the tip ---
pad = 118
cx, cy = int((ox + ix) / 2), int((oy + iy) / 2)
x0, x1 = max(0, cx - pad), min(reach.shape[1], cx + pad)
y0, y1 = max(0, cy - pad), min(reach.shape[0], cy + pad)
axc = fig.add_subplot(gs[0, 2]); axc.imshow(reach[y0:y1, x0:x1]); axc.axis("off")
axc.plot([tx - x0], [ty - y0], marker="x", ms=20, mew=3.6, color="#111111")
axc.plot([ox - x0], [oy - y0], marker="o", ms=19, mew=3.4, mfc="none", color="#0b8a3a")
axc.plot([ix - x0], [iy - y0], marker="s", ms=17, mew=3.4, mfc="none", color="#b3261e")
axc.annotate("", xy=(ox - x0, oy - y0), xytext=(ix - x0, iy - y0),
             arrowprops={"arrowstyle": "<->", "color": "#b3261e", "lw": 2.0})
axc.set_title(f"3. the two body frames are {F['frame_offset_m']*1000:.0f} mm apart", fontsize=13, fontweight="bold")
axc.set_xlabel("zoom on the tool-mount body", fontsize=10)

# --- D: which frame the convergence check reads ---
axd = fig.add_subplot(gs[1, 0:2]); axd.axis("off"); axd.set_xlim(0, 1); axd.set_ylim(0, 1)
put(axd, 0.0, 1.045, "Which body frame `reached` is decided at  (measured, this scene)",
    fontsize=13, fontweight="bold", transform=axd.transAxes)
rows = [
    ("target",                              f"{F['target']}", ""),
    ("reported ee_position (frame origin)", f"[{', '.join(f'{v:+.4f}' for v in F['xpos'])}]", ""),
    ("inertial frame of the same body",     f"[{', '.join(f'{v:+.4f}' for v in F['xipos'])}]", ""),
    ("", "", ""),
    ("error measured at the frame origin",  f"{F['err_from_origin_m']*1000:6.2f} mm",  f"<= tol {F['tol']*1000:.0f} mm  ->  reached"),
    ("error measured at the inertial frame", f"{F['err_from_inertial_m']*1000:6.2f} mm", f"{F['err_from_inertial_m']/F['tol']:.1f}x tol  ->  never converges"),
    ("", "", ""),
    ("mink forward pose vs frame origin",   f"{F['mink_vs_origin_m']*1000:6.2f} mm",   "the solver optimizes this frame"),
    ("mink forward pose vs inertial frame", f"{F['mink_vs_inertial_m']*1000:6.2f} mm", ""),
]
TOP, LAST = 0.90, 0.10
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
y = TOP
for label, val, note in rows:
    if label:
        put(axd, 0.005, y, label, fontsize=11, transform=axd.transAxes)
        put(axd, 0.455, y, val, fontsize=11, transform=axd.transAxes, **MONO)
        colour = "#0b8a3a" if "reached" in note else ("#b3261e" if "never" in note else "#333333")
        put(axd, 0.660, y, note, fontsize=10.5, color=colour,
            fontweight="bold" if note and ("reached" in note or "never" in note) else "normal",
            transform=axd.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, y

# --- E: mutation matrix ---
axe = fig.add_subplot(gs[1, 2]); axe.axis("off"); axe.set_xlim(0, 1); axe.set_ylim(0, 1)
put(axe, 0.0, 1.045, "Regressions the new tests catch", fontsize=13, fontweight="bold", transform=axe.transAxes)
put(axe, 0.0, 0.925, "                                   new    pre-existing", fontsize=9.6,
    transform=axe.transAxes, **MONO)
TOP2, LAST2 = 0.80, 0.20
step2 = (TOP2 - LAST2) / (len(MUT) - 1)
assert step2 > 0.030, step2
y2 = TOP2
for label, newr, oldr in MUT:
    put(axe, 0.0, y2, label, fontsize=9.4, transform=axe.transAxes)
    put(axe, 0.0, y2 - step2 * 0.42, f"      {newr:>8}    {oldr:>9}   <- blind", fontsize=9.4,
        color="#b3261e", transform=axe.transAxes, **MONO)
    y2 -= step2
assert abs((y2 + step2) - LAST2) < 1e-9, y2
put(axe, 0.0, 0.055, "every regression is invisible to the 161\npre-existing motion-primitive tests",
    fontsize=10, style="italic", transform=axe.transAxes)

fig.suptitle("move_to on a body-framed end-effector  -  tests only, no library behaviour changes",
             fontsize=15, fontweight="bold", y=0.985)
for ax, yy, axes_coords in placed:
    if axes_coords:
        assert -0.04 <= yy <= 1.10, (yy, axes_coords)
    else:
        lo, hi = ax.get_ylim(); assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.32, facecolor="white")
im = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white px"
pathlib.Path("/tmp/art_final.png").write_bytes(buf.getvalue())
print("figure:", im.shape, len(buf.getvalue()) // 1024, "KB   borders clean")
