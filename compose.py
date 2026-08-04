import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

R = json.load(open("/tmp/art/render.json"))
M = json.load(open("/tmp/art/main.json"))
B = json.load(open("/tmp/art/branch.json"))

# ---- self-audit: every claim below is measured ----
assert M["True"]["newton"]["status"] == "success", "main must report success for True"
assert M["True"]["newton"]["applied"] == [0.0, 0.0, 1.0], M["True"]["newton"]["applied"]
assert M["True"]["newton"]["rebuilt"] == 1, "main must have rebuilt the model"
assert B["True"]["newton"]["status"] == "error"
assert B["True"]["newton"]["applied"] == [0.0, 0.0, -9.81]
assert B["True"]["newton"]["rebuilt"] == 0, "a refused gravity must not rebuild"
assert M["True"]["isaac"]["cleared"] is True and B["True"]["isaac"]["cleared"] is False
assert M["np.array([0,0,-3.7])"]["isaac"]["cleared"] is False
assert B["np.array([0,0,-3.7])"]["isaac"]["cleared"] is True
assert R["main_true"]["crate_z"] == [0.78, 0.8604, 1.1008], R["main_true"]["crate_z"]
assert R["pr_kept"]["crate_z"] == [0.78, 0.1589, 0.1699], R["pr_kept"]["crate_z"]

def load(p):
    return np.asarray(Image.open(p).convert("RGB"))

top = [load(p) for p in R["main_true"]["frames"]]
bot = [load(p) for p in R["pr_kept"]["frames"]]
# The t=0 rows are the same scene rendered by two engine instances: identical
# up to renderer noise, which is what proves only the gravity differs.
rig = np.abs(top[0].astype(int) - bot[0].astype(int))
RIG_MAX, RIG_PX = int(rig.max()), int((rig.sum(2) > 0).sum())
assert RIG_MAX <= 2, f"t=0 rows differ by {RIG_MAX} - not the same rig"
diff = (np.abs(top[-1].astype(int) - bot[-1].astype(int)).sum(2) > 24).mean()
assert diff > 0.09, f"final frames differ on only {diff:.2%} of pixels"
for f in top + bot:
    assert ((f[:, :, 0].astype(int) - f[:, :, 2]) > 60).mean() > 0.04, "the crate is not in frame"

# divergence counts over the measured cases (MuJoCo is the reference domain)
CASES = list(M)
def verdicts(tree):
    out = {}
    for k in CASES:
        out[k] = (tree[k]["newton"]["status"] == "success", tree[k]["isaac"]["cleared"])
    return out
# The reference verdict is MuJoCo's, which already routed through the shared
# domain: accept a real number or a 3-vector of them, refuse everything else.
REF = {"-3.7": True, "[0,0,-3.7]": True, "True": False, "False": False, "[0,0,True]": False,
       "np.bool_(True)": False, "np.float32(-3.7)": True, "np.float64(-3.7)": True,
       "np.array([0,0,-3.7])": True, "nan": False, "[0,0]": False, "'heavy'": False}
def divergences(tree):
    v = verdicts(tree)
    return sum(1 for k in CASES for got in v[k] if got != REF[k])
DIV_A, DIV_B = divergences(M), divergences(B)
assert (DIV_A, DIV_B) == (9, 0), (DIV_A, DIV_B)

placed = []
def put(ax, x, y, s, **kw):
    placed.append(y)
    return ax.text(x, y, s, transform=ax.transAxes, **kw)

fig = plt.figure(figsize=(15.6, 11.9), facecolor="white")
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.16], hspace=0.30, wspace=0.05,
                      left=0.035, right=0.972, top=0.905, bottom=0.035)

fig.suptitle("A gravity surface that did not use the shared gravity domain applied a boolean as a magnitude",
             fontsize=16.5, weight="bold", y=0.978)
fig.text(0.5, 0.945,
         "NewtonSimEngine.set_gravity(True) -> status=\"success\", world.gravity=[0, 0, +1.0]  |  "
         "the stored vector, replayed in a MuJoCo world (headless)",
         ha="center", fontsize=11.5, style="italic", color="#333333")

LABELS = ["t = 0.0 s  (release)", "t = 0.4 s", "t = 0.8 s"]
for row, (tag, imgs, colour, title) in enumerate((
    ("main_true", top, "#c62828",
     "main:  set_gravity(True) accepted  ->  gravity [0, 0, +1.0]  -  the crate falls UPWARD"),
    ("pr_kept", bot, "#2e7d32",
     "this change:  set_gravity(True) refused  ->  gravity unchanged [0, 0, -9.81]"),
)):
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(imgs[col]); ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor(colour); side.set_linewidth(3.0)
        ax.set_xlabel(f"{LABELS[col]}    crate z = {R[tag]['crate_z'][col]:.3f} m",
                      fontsize=10.5, color=colour, weight="bold", labelpad=5)
        if col == 0:
            ax.set_title(title, fontsize=12.5, color=colour, weight="bold", loc="left", pad=9)

# ---- bottom left: verdict matrix ----
axm = fig.add_subplot(gs[2, :2]); axm.axis("off")
axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 0.96, "Verdict per gravity value  (MuJoCo, which already used the shared domain, is the reference)",
    fontsize=12.3, weight="bold")
hdr = f"{'value passed as gravity':24s} {'reference':>10s} {'newton':>18s} {'isaac':>18s}"
put(axm, 0.0, 0.875, hdr, fontsize=10.0, family="monospace", weight="bold", color="#444444")
put(axm, 0.0, 0.828, "-" * 76, fontsize=10.0, family="monospace", color="#999999")
y = 0.775
for k in CASES:
    ref = "accept" if REF[k] else "refuse"
    nm = "accept" if M[k]["newton"]["status"] == "success" else "refuse"
    nb = "accept" if B[k]["newton"]["status"] == "success" else "refuse"
    im = "accept" if M[k]["isaac"]["cleared"] else "refuse"
    ib = "accept" if B[k]["isaac"]["cleared"] else "refuse"
    def cell(main_v, pr_v):
        if main_v != ref:
            return f"{main_v} -> {pr_v}", ("#c62828" if pr_v != ref else "#2e7d32")
        return f"{main_v}    {'  ok':>4s}", "#555555"
    ntxt, ncol = cell(nm, nb)
    itxt, icol = cell(im, ib)
    put(axm, 0.0, y, f"{k:24s} {ref:>10s}", fontsize=10.0, family="monospace", color="#222222")
    put(axm, 0.485, y, f"{ntxt:>18s}", fontsize=10.0, family="monospace", color=ncol,
        weight=("bold" if ncol != "#555555" else "normal"))
    put(axm, 0.735, y, f"{itxt:>18s}", fontsize=10.0, family="monospace", color=icol,
        weight=("bold" if icol != "#555555" else "normal"))
    y -= 0.0555
put(axm, 0.0, y - 0.012, "-" * 76, fontsize=10.0, family="monospace", color="#999999")
put(axm, 0.0, y - 0.082,
    f"verdicts disagreeing with the reference:  main {DIV_A} of {len(CASES) * 2}"
    f"    ->    this change {DIV_B} of {len(CASES) * 2}",
    fontsize=11.6, weight="bold", color="#1b5e20")

# ---- bottom right: ledger ----
axl = fig.add_subplot(gs[2, 2]); axl.axis("off")
axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.96, "set_gravity(True) on Newton", fontsize=12.3, weight="bold")
rows_l = [
    ("", "main", "this change"),
    ("status", M["True"]["newton"]["status"], B["True"]["newton"]["status"]),
    ("world.gravity z", f"{M['True']['newton']['applied'][2]:+.2f}", f"{B['True']['newton']['applied'][2]:+.2f}"),
    ("model rebuilt", str(M["True"]["newton"]["rebuilt"]), str(B["True"]["newton"]["rebuilt"])),
    ("crate z @ 0.8 s", f"{R['main_true']['crate_z'][-1]:.3f} m", f"{R['pr_kept']['crate_z'][-1]:.3f} m"),
]
yy = 0.865
for name, a, bb in rows_l:
    bold = name == ""
    put(axl, 0.0, yy, f"{name:15s}", fontsize=10.2, family="monospace",
        weight=("bold" if bold else "normal"), color="#444444")
    put(axl, 0.46, yy, f"{a:>11s}", fontsize=10.2, family="monospace",
        weight=("bold" if bold else "normal"), color=("#444444" if bold else "#c62828"))
    put(axl, 0.75, yy, f"{bb:>11s}", fontsize=10.2, family="monospace",
        weight=("bold" if bold else "normal"), color=("#444444" if bold else "#2e7d32"))
    yy -= 0.095
put(axl, 0.0, yy - 0.03,
    "The blue sphere on the post marks the\nrelease height, so the displacement is\nreadable off the frame.\n\n"
    "Newton and Isaac are exercised directly\n(no solver installed); the frames replay\nthe gravity Newton stored\n"
    "into a MuJoCo world.",
    fontsize=9.6, color="#333333", va="top")

assert all(-0.30 <= v <= 0.99 for v in placed), f"text outside its axes: {sorted(placed)[:3]}"
fig.savefig("/tmp/art/gravity_domain_parity.png", dpi=115, facecolor="white",
            bbox_inches="tight", pad_inches=0.28)

img = np.asarray(Image.open("/tmp/art/gravity_domain_parity.png").convert("RGB"))
border = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3),
                         img[:, :8].reshape(-1, 3), img[:, -8:].reshape(-1, 3)])
nonwhite = int((border < 250).any(1).sum())
print(f"size={img.shape}  non-white border px={nonwhite}  final-frame diff={diff:.2%}  "
      f"t=0 rig max|delta|={RIG_MAX} over {RIG_PX} px")
assert nonwhite == 0, f"{nonwhite} non-white border pixels - content is clipped"
print("OK")
