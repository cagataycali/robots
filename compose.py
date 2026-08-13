import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.load(open("_art/facts-main.json"))      # main
B = json.load(open("_art/facts-branch.json"))     # branch
assert A["tree"] != B["tree"], "both arms measured the same tree"

# --- derive every number ---
def verdict(row):
    return row["outcome"] if row["outcome"] == "refused" else f"accepted->{'sandbox' if row['in_sandbox'] else 'CWD'}"
pairs = list(zip(A["rows"], B["rows"], strict=True))
for a, b in pairs:
    assert a["input"] == b["input"]
changed = [(a, b) for a, b in pairs if verdict(a) != verdict(b)]
same    = [(a, b) for a, b in pairs if verdict(a) == verdict(b)]
assert len(changed) == 1 and changed[0][0]["input"] == "frame.png", changed
assert len(same) == 5, len(same)
assert A["render"]["status"] == "error" and A["render"]["file_exists"] is False
assert B["render"]["status"] == "success" and B["render"]["file_exists"] is True
assert "symlink" not in A["symlink_at_anchored_dest"], A["symlink_at_anchored_dest"]
assert "is a symlink" in B["symlink_at_anchored_dest"], B["symlink_at_anchored_dest"]
N_CHANGED, N_SAME = len(changed), len(same)

img = np.array(Image.open("_art/render-robots-mine-31660517728.png").convert("RGB"))
sat = float((img.max(2).astype(int) - img.min(2).astype(int) > 45).mean())
assert sat > 0.10, f"render looks empty (sat={sat:.3f})"

fig = plt.figure(figsize=(15.6, 12.4), dpi=124)
gs = fig.gridspec = fig.add_gridspec(3, 2, height_ratios=[1.02, 1.06, 0.42],
                                     hspace=0.30, wspace=0.14,
                                     left=0.035, right=0.972, top=0.925, bottom=0.028)
placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    ax.text(x, y, s, **kw)

fig.suptitle("render(output_path=\"frame.png\") - a bare filename is written into the sandbox instead of refused",
             fontsize=15.5, fontweight="bold", y=0.972)
fig.text(0.5, 0.945, "validate_output_path is the single owner of the guards behind render / run_policy(video=) / "
         "start_cameras_recording. Measured on Thor, headless (MUJOCO_GL=egl).",
         ha="center", fontsize=10.2, style="italic", color="#444")

# --- row 1 left: main has NO file to show ---
axL = fig.add_subplot(gs[0, 0]); axL.axis("off")
axL.set_xlim(0, 1); axL.set_ylim(0, 1)
axL.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, facecolor="#1d1f22", edgecolor="#b3282d", lw=2.4))
put(axL, 0.5, 0.90, "main  -  no file written", ha="center", fontsize=13.5, fontweight="bold",
    color="#ff8f93", transform=axL.transAxes)
put(axL, 0.5, 0.80, 'sim.render(..., output_path="frame.png")', ha="center", fontsize=10.6,
    family="monospace", color="#e6e6e6", transform=axL.transAxes)
msg = A["render"]["text"]
body = [msg[i:i+62] for i in range(0, min(len(msg), 248), 62)]
y = 0.665
for ln in body:
    put(axL, 0.055, y, ln, fontsize=8.9, family="monospace", color="#ffb3b6", transform=axL.transAxes)
    y -= 0.062
put(axL, 0.055, 0.31, f'status          = {A["render"]["status"]!r}', fontsize=10.2, family="monospace",
    color="#ff8f93", transform=axL.transAxes)
put(axL, 0.055, 0.24, f'file on disk    = {A["render"]["file_exists"]}', fontsize=10.2, family="monospace",
    color="#ff8f93", transform=axL.transAxes)
put(axL, 0.055, 0.14, "The quoted path is the process CWD.\nThe caller named no directory at all.",
    fontsize=9.8, color="#cfcfcf", style="italic", transform=axL.transAxes)

# --- row 1 right: the real frame the branch writes ---
axR = fig.add_subplot(gs[0, 1])
axR.imshow(img); axR.set_xticks([]); axR.set_yticks([])
for sp in axR.spines.values():
    sp.set_edgecolor("#2e7d32"); sp.set_linewidth(2.4)
axR.set_title("this change  -  the PNG lands in the sandbox", fontsize=13.5, fontweight="bold",
              color="#1b5e20", pad=8)
axR.set_xlabel(f'saved_path = {pathlib.Path(B["render"]["saved_path"]).parent.name}/'
               f'{pathlib.Path(B["render"]["saved_path"]).name}     '
               f'file on disk = {B["render"]["file_exists"]}     {B["render"]["bytes"]} bytes     '
               f'saturated pixels = {sat:.2%}', fontsize=9.6, family="monospace", labelpad=7)

# --- row 2: verdict table ---
axT = fig.add_subplot(gs[1, :]); axT.axis("off")
axT.set_xlim(0, 1); axT.set_ylim(0, 1)
put(axT, 0.0, 0.965, f"Every case: {N_CHANGED} verdict changes, {N_SAME} are byte-identical to main",
    fontsize=12.6, fontweight="bold", transform=axT.transAxes)
cols = [0.005, 0.145, 0.305, 0.635]
hdr = ["input", "confinement", "main", "this change"]
put(axT, cols[0], 0.885, hdr[0], fontsize=10.4, fontweight="bold", family="monospace", transform=axT.transAxes)
for c, h in zip(cols[1:], hdr[1:], strict=True):
    put(axT, c, 0.885, h, fontsize=10.4, fontweight="bold", family="monospace", transform=axT.transAxes)
axT.plot([0.0, 1.0], [0.858, 0.858], color="#999", lw=1.0, transform=axT.transAxes)

rows = []
for a, b in pairs:
    conf = "sandbox" if a["confined"] else "guards-only"
    def cell(r):
        if r["outcome"] == "refused":
            reason = r["reason"].split(":")[-1].strip() if "unsafe" in r["reason"] else "outside the sandbox"
            return f"refused ({reason})", "#b3282d"
        return (f"-> sandbox root", "#1b5e20") if r["in_sandbox"] else ("-> process CWD", "#555")
    rows.append((a["input"], conf, cell(a), cell(b), verdict(a) != verdict(b)))
rows.append(("evil.png -> symlink", "sandbox",
             ("refused (outside the sandbox)", "#8a6d00"), ("refused (is a symlink)", "#1b5e20"), True))

TOP, LAST = 0.795, 0.115
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for inp, conf, (ma, mc), (ba, bc), diff in rows:
    if diff:
        axT.add_patch(plt.Rectangle((0.0, y - 0.030), 1.0, 0.062, facecolor="#fff8e1",
                                    edgecolor="none", zorder=0, transform=axT.transAxes))
    put(axT, cols[0], y, inp, fontsize=9.9, family="monospace", transform=axT.transAxes)
    put(axT, cols[1], y, conf, fontsize=9.5, family="monospace", color="#555", transform=axT.transAxes)
    put(axT, cols[2], y, ma, fontsize=9.9, family="monospace", color=mc, transform=axT.transAxes)
    put(axT, cols[3], y, ba, fontsize=9.9, family="monospace", color=bc, transform=axT.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
put(axT, 0.0, 0.045,
    "The one-component rule is what keeps it narrow: a separator, a '..', an absolute path outside the root and a\n"
    "metacharacter are all refused exactly as before, and guards-only mode stays CWD-relative.",
    fontsize=9.6, style="italic", color="#333", transform=axT.transAxes)

# --- row 3: gate ---
axG = fig.add_subplot(gs[2, :]); axG.axis("off")
axG.set_xlim(0, 1); axG.set_ylim(0, 1)
axG.add_patch(plt.Rectangle((0.0, 0.04), 1.0, 0.92, facecolor="#f4f6f8", edgecolor="#c8ccd0", lw=1.0))
gl = [
    "Gate (upstream/main 83cc5272, Thor, MUJOCO_GL=egl):  28689 passed / 258 skipped / 0 failed in 657s  "
    "(pristine main 28680 + 9 new cases)",
    "Pre-fix (source reverted to main, tests kept): 3 failed / 51 passed - the 51 passing are the security controls.",
    "Mutations: 4 of 5 regressions caught by the new cases, 0 of 5 by the 45 pre-existing ones. "
    "M2 (anchor/traversal order) is provably unobservable:",
    "the traversal check scans every part, so joining the root cannot hide a '..' - reported rather than padded.",
    "ruff clean (1190 files). mypy: 0 errors outside examples/isaac_gs (14 there, byte-identical to a pristine-base worktree).",
]
GT, GL_ = 0.80, 0.14
GS_ = (GT - GL_) / (len(gl) - 1)
assert GS_ > 0.030, GS_
gy = GT
for ln in gl:
    put(axG, 0.014, gy, ln, fontsize=9.2, family="monospace", color="#222", transform=axG.transAxes)
    gy -= GS_
assert abs((gy + GS_) - GL_) < 1e-9

for ax, yv, is_axes in placed:
    if is_axes:
        assert -0.03 <= yv <= 1.07, (ax, yv)
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, (ax, yv)

out = pathlib.Path("_art/bare-filename-sandbox.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.array(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bot", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  changed={N_CHANGED} same={N_SAME} sat={sat:.2%}")
