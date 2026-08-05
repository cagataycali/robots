import json, pathlib, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.loads(pathlib.Path("/tmp/art/main/facts.json").read_text())
B = json.loads(pathlib.Path("/tmp/art/branch/facts.json").read_text())
assert A["tree"] != B["tree"], "both halves came from the same tree"

ha = np.load("/tmp/art/main/honored.npy"); hb = np.load("/tmp/art/branch/honored.npy")
delta = int(np.abs(ha.astype(int) - hb.astype(int)).max())
assert delta <= 2, f"honored view differs across trees by {delta}"
sat = float(((ha.max(2).astype(int) - ha.min(2).astype(int)) > 45).mean())
assert sat > 0.05, f"reference view is not a real scene (saturation {sat:.4f})"

# claims
assert A["typo"]["install"] == "returned normally" and A["typo"]["camera_in_world"] is False
assert A["typo"]["render_status"] == "error" and len(A["typo"]["warnings"]) == 1
assert B["typo"]["install"].startswith("ValueError: LiberoAdapter cameras['image']")
assert B["typo"]["warnings"] == []
for t in (A, B):
    assert t["honored"]["camera_in_world"] and t["honored"]["published_dims"] == [320, 320]
    assert t["honored"]["render_status"] == "success"

placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 8.0), dpi=125)
gs = fig.add_gridspec(2, 3, height_ratios=[3.05, 1.0], hspace=0.30, wspace=0.10,
                      left=0.022, right=0.978, top=0.885, bottom=0.045)

fig.suptitle(
    "LiberoAdapter(cameras=...): a per-camera key add_camera cannot accept "
    "used to drop the policy's view silently",
    fontsize=15.5, fontweight="bold", y=0.968)
fig.text(0.5, 0.916,
         "MuJoCo headless (EGL). One config typo: \"heigth\" instead of \"height\". "
         "Left = what the config asks for; centre/right = what each tree does with the typo.",
         ha="center", fontsize=10.6, style="italic", color="#333333")

# --- panel 1: the honored view
ax = fig.add_subplot(gs[0, 0]); ax.imshow(ha); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("cameras={\"image\": {..., \"height\": 320}}\nhonored on both trees",
             fontsize=11.4, fontweight="bold", color="#1a7a3c")
ax.set_xlabel("world.cameras['image'] = 320x320\nrender(camera_name='image') -> success",
              fontsize=9.9, family="monospace")

def textpanel(ax, title, colour, rows, badge):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.005, 0.005), 0.99, 0.99, transform=ax.transAxes,
                           facecolor="#f6f6f4", edgecolor=colour, linewidth=2.4))
    ax.set_title(title, fontsize=11.4, fontweight="bold", color=colour)
    put(ax, 0.5, 0.905, badge, ha="center", fontsize=12.4, fontweight="bold",
        color="white", family="monospace",
        bbox=dict(boxstyle="round,pad=0.42", facecolor=colour, edgecolor="none"))
    y = 0.775
    for label, val, mono in rows:
        put(ax, 0.045, y, label, fontsize=9.8, fontweight="bold", color="#222222")
        for line in textwrap.wrap(val, 46):
            y -= 0.062
            put(ax, 0.075, y, line, fontsize=9.0,
                family="monospace" if mono else "sans-serif", color="#333333")
        y -= 0.085
    return ax

ax2 = fig.add_subplot(gs[0, 1])
textpanel(ax2, "main: the typo", "#b3261e", [
    ("install call", "_install_libero_cameras() returned normally", True),
    ("what the caller saw", "nothing - one WARNING in the log", False),
    ("log", "add_camera('image') raised: unexpected keyword argument 'heigth'. Did you mean 'height'?", False),
    ("world.cameras", "'image' ABSENT - the eval runs on without it", True),
    ("render('image')", "error: the view the policy reads does not exist", True),
], "NO VIEW  (eval proceeds)")

ax3 = fig.add_subplot(gs[0, 2])
textpanel(ax3, "this change: the typo", "#1a7a3c", [
    ("install call", "raises ValueError before any camera is added", True),
    ("message", B["typo"]["install"].removeprefix("ValueError: "), True),
    ("world.cameras", "unchanged - nothing half-installed", True),
    ("fix", "rename 'heigth' -> 'height'; the left panel is the result", False),
], "REFUSED  (named + fixable)")

# --- fact table
axt = fig.add_subplot(gs[1, :]); axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off")
cols = [0.035, 0.315, 0.575, 0.815]
hdr = ["measured on a real MuJoCo world", "config as written", "main", "this change"]
for x, h in zip(cols, hdr):
    put(axt, x, 0.90, h, fontsize=10.2, fontweight="bold", color="#111111")
axt.plot([0.025, 0.975], [0.845, 0.845], color="#999999", lw=1.0)
rows = [
    ("_install_libero_cameras()", "returns", "returns (typo swallowed)", "raises, names the key"),
    ("'image' in world.cameras", "yes, 320x320", "NO", "no - refused, not dropped"),
    ("render(camera_name='image')", "success", "error", "error (never reached)"),
    ("caller-visible signal", "-", "WARNING only", "ValueError + accepted keys"),
]
y = 0.72
for r in rows:
    for x, cell in zip(cols, r):
        put(axt, x, y, cell, fontsize=9.6, family="monospace",
            color="#b3261e" if cell in ("NO", "error", "WARNING only", "returns (typo swallowed)") else "#222222")
    y -= 0.185
put(axt, 0.035, 0.035,
    f"Left panel byte-comparable across trees (max|delta| = {delta}/255 renderer noise) - "
    "a usable config is untouched. Accepted key set read from the sim's add_camera "
    "(MuJoCo/Newton declare parent_body, Isaac does not).",
    fontsize=9.2, style="italic", color="#444444")

for ax_, y_ in placed:
    lo, hi = ax_.get_ylim()
    assert lo - 0.03 <= y_ <= hi + 0.06, f"text at y={y_} outside {ax_.get_ylim()}"

out = pathlib.Path("/tmp/art/libero_camera_config.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.28, facecolor="white")
plt.close(fig)

im = np.array(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 20).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", out, im.shape, "delta", delta, "sat", round(sat, 4))
