import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

A = "/tmp/relnotes/assets"
B = json.load(open(f"{A}/options_before.json")); Af = json.load(open(f"{A}/options_after.json"))
fb, fa = B["facts"], Af["facts"]
tb, ta = B["trace"], Af["trace"]
nb = np.load(f"{A}/options_before.npz"); na = np.load(f"{A}/options_after.npz")

RED, GREEN, INK, GREY = "#b91c1c", "#057a3d", "#111827", "#6b7280"

fig, ax = plt.subplots(1, 2, figsize=(13.4, 3.9), dpi=100)
for a, key, ylab, title in (
    (ax[0], "max_joint_vel", "max |joint velocity|  (rad/s)",
     "Residual motion at a held set-point"),
    (ax[1], "max_track_err_rad", "max |tracking error|  (rad)",
     "Position-servo tracking error"),
):
    a.plot([t["i"] for t in tb], [t[key] for t in tb], color=RED, lw=1.5,
           label=f'Euler  (declaration discarded)')
    a.plot([t["i"] for t in ta], [t[key] for t in ta], color=GREEN, lw=1.5,
           label=f'implicitfast  (as panda.xml declares)')
    a.set_yscale("symlog", linthresh=1e-3)
    a.set_xlabel("control tick  (50 Hz, 10 physics substeps each)", fontsize=9)
    a.set_ylabel(ylab, fontsize=9)
    a.set_title(title, fontsize=12, fontweight="bold", color=INK)
    a.grid(alpha=0.25, lw=0.6)
    a.axvspan(200, 260, color="#94a3b8", alpha=0.16)
    a.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    a.tick_params(labelsize=8)
ax[0].annotate(f'never settles:\n{fb["settled_jitter_rad_s"]:.4f} rad/s', xy=(232, max(1e-3, fb["settled_jitter_rad_s"])),
               xytext=(120, 3.2), fontsize=9, color=RED, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax[0].annotate(f'settles to exactly 0', xy=(232, 0.0), xytext=(118, 0.00013), fontsize=9,
               color=GREEN, fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
plt.tight_layout()
plt.savefig(f"{A}/_opt_trace.png", bbox_inches="tight", pad_inches=0.14, facecolor="white")
plt.close()

def font(sz, bold=False):
    p = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try: return ImageFont.truetype(p, sz)
    except Exception: return ImageFont.load_default()
def mono(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", sz)
    except Exception: return ImageFont.load_default()

RGB_RED, RGB_GREEN, RGB_INK, RGB_GREY = (185,28,28), (5,122,61), (17,24,39), (107,114,128)
PAD = 12

def cell(img, title, tcol, rows):
    caph = 26 + 20*len(rows)
    im = Image.new("RGB", (img.shape[1], img.shape[0]+caph), "white")
    im.paste(Image.fromarray(img), (0,0)); d = ImageDraw.Draw(im); y0 = img.shape[0]
    d.rectangle([0,y0,im.width,y0+caph], fill=(247,248,250)); d.line([0,y0,im.width,y0], fill=(203,207,213))
    d.text((10,y0+5), title, font=font(17,True), fill=tcol)
    for i,(t,c) in enumerate(rows): d.text((10,y0+27+20*i), t, font=mono(13), fill=c)
    return im

c1 = cell(nb["final"], "BEFORE  -  at #1687^", RGB_RED, [
    (f'compiled integrator : {fb["integrator_compiled"]}', RGB_RED),
    (f'panda.xml declares  : {fb["integrator_declared_by_panda_xml"]}', RGB_INK),
    (f'residual jitter     : {fb["settled_jitter_rad_s"]:.4f} rad/s  (never rests)', RGB_RED),
    (f'tracking error      : {fb["settled_track_err_rad"]:.6f} rad', RGB_RED),
])
c2 = cell(na["final"], "AFTER  -  the model's own declaration", RGB_GREEN, [
    (f'compiled integrator : {fa["integrator_compiled"]}', RGB_GREEN),
    (f'panda.xml declares  : {fa["integrator_declared_by_panda_xml"]}', RGB_INK),
    (f'residual jitter     : {fa["settled_jitter_rad_s"]:.4f} rad/s  (at rest)', RGB_GREEN),
    (f'tracking error      : {fa["settled_track_err_rad"]:.6f} rad', RGB_GREEN),
])
top = Image.new("RGB", (c1.width+c2.width+PAD, max(c1.height,c2.height)), "white")
top.paste(c1,(0,0)); top.paste(c2,(c1.width+PAD,0))

tr = Image.open(f"{A}/_opt_trace.png").convert("RGB")
scale = top.width / tr.width
tr = tr.resize((top.width, int(tr.height*scale)), Image.LANCZOS)

TOPH = 68
fig2 = Image.new("RGB", (top.width+2*PAD, TOPH+top.height+PAD+tr.height+PAD), "white")
d = ImageDraw.Draw(fig2)
d.text((PAD+2,14), "Fixed: a robot is simulated under the solver settings its own model declares  (#1687)",
       font=font(25,True), fill=RGB_INK)
d.text((PAD+2,45), "MuJoCo <option> is model-global and does not survive spec.attach(), so every setting a robot MJCF "
                   "declared for itself was discarded when add_robot merged it into a scene.",
       font=font(13), fill=RGB_GREY)
fig2.paste(top,(PAD,TOPH)); fig2.paste(tr,(PAD,TOPH+top.height+PAD))
fig2.save(f"{A}/fix_options.png")

# ---- audit ----
a = np.asarray(fig2)
bd = np.concatenate([a[:6].reshape(-1,3),a[-6:].reshape(-1,3),a[:,:6].reshape(-1,3),a[:,-6:].reshape(-1,3)])
nw = int((np.abs(bd.astype(int)-255).sum(1)>12).sum())
panel_diff = float((np.abs(nb["final"].astype(int)-na["final"].astype(int)).sum(2)>30).mean())
print("size", fig2.size, "border_nonwhite", nw)
print("declaration honored  before/after :", fb["declaration_honored"], "/", fa["declaration_honored"])
print("settled jitter rad/s before/after :", fb["settled_jitter_rad_s"], "/", fa["settled_jitter_rad_s"])
print("tracking err   before/after       :", fb["settled_track_err_rad"], "/", fa["settled_track_err_rad"])
print("final-pose differing-pixel frac   :", round(panel_diff,4))
assert nw == 0, nw
assert fb["declaration_honored"] is False and fa["declaration_honored"] is True
assert fa["settled_jitter_rad_s"] == 0.0 and fb["settled_jitter_rad_s"] > 0.1
assert fa["settled_track_err_rad"] < fb["settled_track_err_rad"]
print("AUDIT OK")
