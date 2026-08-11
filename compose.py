import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ART = pathlib.Path("_art")
F = json.load(open(ART / "facts.json"))
assert F["tree"].endswith(pathlib.Path.cwd().name), F["tree"]
D, H = F["diagnoses"], F["honored"]
BK = ["mujoco", "newton", "isaac"]
CAUSES = ["absent-lerobot-extra", "module-did-not-import", "module-supplied-no-recorder"]
# which cells had ever run before this change (from the full-suite coverage JSON)
BEFORE = {("mujoco", CAUSES[0]): 1, ("mujoco", CAUSES[1]): 1, ("newton", CAUSES[0]): 1}
ADDED = sum(1 for b in BK for c in CAUSES if (b, c) not in BEFORE)
assert ADDED == 6, ADDED
MUT = [("M1 mujoco: collapse the partial-install diagnosis", 1, 0),
       ("M2 newton: collapse the partial-install diagnosis", 1, 0),
       ("M3 isaac:  collapse the partial-install diagnosis", 1, 0),
       ("M4 isaac:  drop the no-recorder-symbol check", 2, 0),
       ("M5 newton: drop the no-recorder-symbol check", 2, 0),
       ("M6 isaac:  narrow the import guard to ModuleNotFoundError", 4, 0),
       ("M7 newton: recommend a fallback it does not implement", 1, 0)]
assert all(n > 0 and o == 0 for _, n, o in MUT)
COV = [("mujoco", 2, 1, "98%", "99%"), ("newton", 5, 2, "97%", "99%"), ("isaac", 19, 15, "88%", "91%")]

# every cell reported, three distinct diagnoses, correct fallback, nothing created
for b in BK:
    assert len({D[b][c]["marker"] for c in CAUSES}) == 3, b
    for c in CAUSES:
        d = D[b][c]
        assert d["status"] == "error" and not d["session_open"] and not d["root_created"], (b, c, d)
    want = "run_policy(video=...)" if b == "newton" else "start_cameras_recording"
    assert all(D[b][c]["fallback"] == want for c in CAUSES), b
assert (H["start_status"], H["rollout_status"], H["stop_status"]) == ("success",) * 3
assert H["episodes"] == 1 and H["frames"] == 24 and H["decoded_frames"] == 24

frame = np.load(ART / "frame.npy")
sat = float(((frame.max(2).astype(int) - frame.min(2).astype(int)) > 45).mean())
assert sat > 0.03, sat

GREEN, RED, AMBER, INK = "#1a7f37", "#b3261e", "#9a6700", "#1f2328"
placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_co = kw.pop("transform", None) is not None
    t = ax.text(x, y, s, transform=ax.transAxes if axes_co else ax.transData, **kw)
    placed.append((ax, y, axes_co)); return t

fig = plt.figure(figsize=(16.6, 13.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 1.32, 1.22], width_ratios=[1.0, 1.28],
                      hspace=0.30, wspace=0.16)
fig.suptitle("start_recording: every backend's report that the LeRobot dataset stack is unavailable",
             fontsize=17, fontweight="bold", y=0.982)
fig.text(0.5, 0.958, "three causes x three backends = nine cells; three had ever run. "
         "Tests only - no production line changes.", ha="center", fontsize=11.4, style="italic", color="#57606a")

# ---- row 1: the 9-cell matrix ------------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.955, "Which of the nine cells had ever executed", transform=ax.transAxes,
    fontsize=13, fontweight="bold", color=INK)
xs = [0.40, 0.585, 0.77]
for x, b in zip(xs, BK, strict=True):
    put(ax, x, 0.79, b, transform=ax.transAxes, fontsize=11.6, fontweight="bold", ha="center", color=INK)
TOP, LAST = 0.62, 0.20
step = (TOP - LAST) / (len(CAUSES) - 1)
assert step > 0.030, step
y = TOP
for c in CAUSES:
    put(ax, 0.0, y, c, transform=ax.transAxes, fontsize=11, family="monospace", color=INK)
    for x, b in zip(xs, BK, strict=True):
        was = (b, c) in BEFORE
        put(ax, x, y, "driven" if was else "ADDED", transform=ax.transAxes, fontsize=11,
            ha="center", fontweight="normal" if was else "bold", color="#57606a" if was else GREEN)
    y -= step
assert abs((y + step) - LAST) < 1e-9, y
put(ax, 0.0, 0.055, f"6 of 9 cells were unreached; all 6 are now driven "
    f"(Isaac's unavailability report had never executed at all)",
    transform=ax.transAxes, fontsize=11.4, fontweight="bold", color=GREEN)

# ---- row 2 left: the honored path still records ------------------------------
axf = fig.add_subplot(gs[1, 0]); axf.imshow(frame); axf.set_xticks([]); axf.set_yticks([])
for s in axf.spines.values(): s.set_edgecolor(GREEN); s.set_linewidth(2.4)
axf.set_title("the honored path: a real dataset, read back", fontsize=12.4, fontweight="bold", color=GREEN)
axf.set_xlabel(f"decoded out of the dataset's own MP4 - {H['episodes']} episode, {H['frames']} frames "
               f"@ {H['fps']}fps, {H['decoded_frames']} frames read back", fontsize=10.2, color="#57606a")

# ---- row 2 right: the three verbatim diagnoses -------------------------------
ax2 = fig.add_subplot(gs[1, 1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.985, "Three causes, three remedies - the diagnosis a caller actually gets",
    transform=ax2.transAxes, fontsize=12.6, fontweight="bold", color=INK)
def wrap(s, n=74):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n: out.append(line); line = w
        else: line = f"{line} {w}".strip()
    out.append(line); return out
lines: list[tuple[str, str, str]] = []
for b in BK:
    lines.append(("hdr", b, f"{b}   (plain-MP4 fallback it names: {D[b][CAUSES[0]]['fallback']})"))
    for c in CAUSES:
        for i, seg in enumerate(wrap(D[b][c]["marker"])):
            lines.append(("body", c, ("  " if i else "  * ") + seg))
TOP2, FLOOR2 = 0.925, 0.035
step2 = (TOP2 - FLOOR2) / len(lines)
assert step2 > 0.014, step2
y = TOP2
for kind, key, s in lines:
    if kind == "hdr":
        put(ax2, 0.0, y, s, transform=ax2.transAxes, fontsize=10.6, fontweight="bold", color=INK)
    else:
        col = {CAUSES[0]: "#0550ae", CAUSES[1]: AMBER, CAUSES[2]: "#8250df"}[key]
        put(ax2, 0.012, y, s, transform=ax2.transAxes, fontsize=8.7, family="monospace", color=col)
    y -= step2
assert y > 0.010, y

# ---- row 3: mutation + coverage ---------------------------------------------
ax3 = fig.add_subplot(gs[2, :]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 0.985, "Every plausible regression in that block, against both test sets",
    transform=ax3.transAxes, fontsize=13, fontweight="bold", color=INK)
put(ax3, 0.615, 0.885, "this module", transform=ax3.transAxes, fontsize=10.8, fontweight="bold", ha="center", color=INK)
put(ax3, 0.815, 0.885, "312 pre-existing", transform=ax3.transAxes, fontsize=10.8, fontweight="bold", ha="center", color=INK)
TOP3, LAST3 = 0.80, 0.30
step3 = (TOP3 - LAST3) / (len(MUT) - 1)
assert step3 > 0.045, step3
y = TOP3
for label, nf, of in MUT:
    put(ax3, 0.0, y, label, transform=ax3.transAxes, fontsize=10.2, family="monospace", color=INK)
    put(ax3, 0.615, y, f"{nf} failed", transform=ax3.transAxes, fontsize=10.2, ha="center", fontweight="bold", color=GREEN)
    put(ax3, 0.815, y, f"{of} failed", transform=ax3.transAxes, fontsize=10.2, ha="center", fontweight="bold", color=RED)
    y -= step3
assert abs((y + step3) - LAST3) < 1e-9, y
ax3.add_patch(Rectangle((0.735, 0.255), 0.16, 0.60, transform=ax3.transAxes,
                        facecolor=RED, alpha=0.10, edgecolor=RED, lw=1.3, zorder=0))
put(ax3, 0.90, 0.545, "<- BLIND", transform=ax3.transAxes, fontsize=11.6, fontweight="bold", color=RED)
put(ax3, 0.0, 0.195, "caught by this module: 7 of 7      caught by the pre-existing suite: 0 of 7      "
    "unmutated control: 0 failed on both (37 / 312 passed)",
    transform=ax3.transAxes, fontsize=11, fontweight="bold", color=INK)
cov = "   ".join(f"{b} recording.py {mb}->{ma} missing ({pb}->{pa})" for b, mb, ma, pb, pa in COV)
put(ax3, 0.0, 0.115, cov, transform=ax3.transAxes, fontsize=10.4, family="monospace", color="#0550ae")
put(ax3, 0.0, 0.035, "gate: 28174 passed / 257 skipped / 0 failed  -  ruff clean  -  mypy 0 errors outside examples/",
    transform=ax3.transAxes, fontsize=10.6, color="#57606a")

for ax_, yv, axes_co in placed:
    if axes_co: assert -0.03 <= yv <= 1.10, (yv, ax_)
    else:
        lo, hi = ax_.get_ylim(); assert lo - 0.05 <= yv <= hi + 0.07, (yv, lo, hi)

out = ART / "dataset_stack_unavailable.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(plt.imread(out)[:, :, :3] * 255).astype(np.uint8)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, (name, nw)
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  saturated={sat:.3f}  added_cells={ADDED}")
