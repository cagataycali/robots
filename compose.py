import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

B = json.loads(pathlib.Path("/tmp/art_before/facts.json").read_text())
A = json.loads(pathlib.Path("/tmp/art_after/facts.json").read_text())

# --- self-audit: the two halves really came from different trees -------------
assert A["tree"] != B["tree"], (A["tree"], B["tree"])
bc, ac = B["cases"], A["cases"]
assert bc["0"]["result"] == "success" and bc["0"]["view_shape"] == [128, 128, 3] and bc["0"]["stored"] == "0"
assert bc["2.7"]["result"] == "success" and bc["2.7"]["view_shape"] == [2, 2, 3]
assert bc["True"]["result"] == "success" and bc["True"]["view_shape"] == [1, 1, 3]
for k in ("0", "2.7", "True"):
    assert ac[k]["config"] == "refused" and "render_width must be a positive whole number" in ac[k]["result"], ac[k]
assert bc["128"]["view_widths"] == ac["128"]["view_widths"] == [128]

frame = np.load("/tmp/art_before/sim_frame.npy")
v128_b = np.load("/tmp/art_before/view_128.npy")
v128_a = np.load("/tmp/art_after/view_128.npy")
delta128 = int(np.abs(v128_b.astype(int) - v128_a.astype(int)).max())
assert delta128 == 0, f"the honored view must be byte-identical across trees, got {delta128}"

def up(a, size=176):
    """Nearest-neighbour upscale so a 2x2 view reads as a 2x2 view."""
    h, w = a.shape[:2]
    ys = (np.arange(size) * h // size).clip(0, h - 1)
    xs = (np.arange(size) * w // size).clip(0, w - 1)
    return a[ys][:, xs]

CASES = [("128", "128"), ("0", "0"), ("2.7", "2.7"), ("True", "True")]
GREEN, RED, GREY = "#1a7f37", "#b3261e", "#57606a"

fig = plt.figure(figsize=(15.0, 9.6), dpi=124)
gs = fig.add_gridspec(3, 5, height_ratios=[1.30, 1.0, 1.0], width_ratios=[1.22, 1, 1, 1, 1],
                      hspace=0.30, wspace=0.16)
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.pop("_cs", "axes")))
    return ax.text(x, y, s, transform=ax.transAxes, **kw)

fig.suptitle("VeraConfig.render_width sizes every camera view the WAN/DFoT planner is shown",
             fontsize=15.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.949, "Each tile is the view the provider actually put on the wire, captured through the real "
         "`_extract_frame` and upscaled nearest-neighbour so a 2x2 view reads as 2x2.",
         ha="center", fontsize=10.2, color=GREY)

# ---- the source frame ------------------------------------------------------
axf = fig.add_subplot(gs[0, 0])
axf.imshow(frame); axf.set_xticks([]); axf.set_yticks([])
axf.set_title("camera frame from the sim\n(MuJoCo headless, 480x480)", fontsize=10.5, fontweight="bold")
for s in axf.spines.values():
    s.set_color(GREY); s.set_linewidth(1.6)

# ---- row 0 cols 1..4 : main ------------------------------------------------
for i, (label, req) in enumerate(CASES):
    ax = fig.add_subplot(gs[0, i + 1])
    rec = bc[label]
    ok = label == "128"
    view = np.load(f"/tmp/art_before/view_{label}.npy")
    ax.imshow(up(view), interpolation="nearest")
    n = rec["view_shape"][0]
    ax.set_title(f"render_width={req}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"sent {n}x{n} px  ·  view_widths={rec['view_widths']}",
                  fontsize=9.6, color=(GREEN if ok else RED),
                  fontweight="normal" if ok else "bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GREEN if ok else RED); s.set_linewidth(2.6)
    if not ok:
        note = "the request was\nreplaced by 128" if label == "0" else f"{n}x{n} view,\nunder success"
        put(ax, 0.5, 0.5, note, ha="center", va="center", fontsize=10.5, color="white",
            fontweight="bold", bbox=dict(facecolor=RED, edgecolor="none", alpha=0.86, pad=4.5))

axm = fig.add_subplot(gs[1, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 0.72, "on main  ·  every row above returned status=success", fontsize=12.6,
    fontweight="bold", color=RED, va="center")
put(axm, 0.0, 0.30, "The two silent rows are silent in the direction that cannot be noticed: a 1x1 or 2x2 view is a "
    "successful rollout against the planner,\nso the mistake presents as a policy that does not solve rather than as "
    "an error. The remaining values (-1, nan, inf, \"abc\", [128]) raised\nout of `_extract_frame` instead - per frame, "
    "after the server subprocess had already been launched, naming neither the field nor the class.",
    fontsize=10.4, color="#24292f", va="center", linespacing=1.55)

# ---- row 2 : this PR ------------------------------------------------------
axa = fig.add_subplot(gs[2, 0]); axa.axis("off"); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
put(axa, 0.0, 0.80, "with this change", fontsize=12.6, fontweight="bold", color=GREEN, va="center")
put(axa, 0.0, 0.44, "`render_width` takes the shared\nmedia pixel domain at the config\nfunnel, so an unusable width is\n"
    "refused once, by name, before\nany client or server is built.", fontsize=10.2, color="#24292f",
    va="center", linespacing=1.6)

for i, (label, req) in enumerate(CASES):
    ax = fig.add_subplot(gs[2, i + 1]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if label == "128":
        ax.imshow(up(v128_a), interpolation="nearest", extent=(0, 1, 0, 1))
        ax.set_xlabel("sent 128x128 px  ·  byte-identical to main", fontsize=9.6, color=GREEN)
        for s in ax.spines.values():
            s.set_color(GREEN); s.set_linewidth(2.6)
    else:
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f6f8fa", edgecolor="none"))
        put(ax, 0.5, 0.66, "REFUSED", ha="center", va="center", fontsize=13.5, fontweight="bold", color=GREEN)
        put(ax, 0.5, 0.34, f"VeraConfig: render_width\nmust be a positive whole\nnumber, got {req}.",
            ha="center", va="center", fontsize=8.9, color="#24292f", family="monospace", linespacing=1.5)
        ax.set_xlabel("nothing was built", fontsize=9.6, color=GREEN)
        for s in ax.spines.values():
            s.set_color(GREEN); s.set_linewidth(2.6)

for ax, y, cs in placed:
    lo, hi = ax.get_ylim() if cs == "data" else (0.0, 1.0)
    assert lo - 0.05 <= y <= hi + 0.09, f"text at y={y} escapes {cs} range [{lo},{hi}]"

out = pathlib.Path("/tmp/vera_render_width.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

from PIL import Image as _PILImage
im = np.asarray(_PILImage.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print(f"OK {out}  size={im.shape[1]}x{im.shape[0]}  honored-view delta={delta128}")
print("before tree:", B["tree"], "| after tree:", A["tree"])
