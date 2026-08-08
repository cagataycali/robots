"""Compose the artifact: which controller a caller asking for walk=False got."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

A = json.loads(open("/tmp/art_main/facts.json").read())     # upstream/main
B = json.loads(open("/tmp/art_branch/facts.json").read())   # this change
assert A["tree"] != B["tree"], "both halves came from the same tree"

# The measured claims, asserted rather than typed.
assert A["kwargs_walk"] is True and B["kwargs_walk"] is False
assert A["walk_session_runs"] == 400 and A["main_session_runs"] == 0
assert B["walk_session_runs"] == 0 and B["main_session_runs"] == 400
assert B["walk_session_loaded"] is False and A["walk_session_loaded"] is True
assert A["status"] == B["status"] == "success"
assert A["travel_x"] - B["travel_x"] > 0.5, (A["travel_x"], B["travel_x"])

TIMES = [0.0, 2.5, 5.0, 7.5]
R0, R1 = 170, 470          # drop empty sky; keep the whole corridor width

def frame(facts, t):
    p = next(f["path"] for f in facts["frames"] if abs(f["t"] - t) < 1e-9)
    return np.asarray(Image.open(p).convert("RGB"))[R0:R1]

# Same rig: the two trees must be pixel-identical before the policy diverges.
d0 = np.abs(frame(A, 0.0).astype(int) - frame(B, 0.0).astype(int)).max()
assert d0 <= 2, f"t=0 frames differ by {d0}"

tA = np.load("/tmp/art_main/trace.npy")
tB = np.load("/tmp/art_branch/trace.npy")

placed: list[tuple[object, float]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, transform=ax.transAxes, **kw)

fig = plt.figure(figsize=(16.4, 11.6))
gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 1.30],
              hspace=0.30, wspace=0.05, left=0.035, right=0.978, top=0.905, bottom=0.045)

fig.suptitle(
    "build_policy_kwargs(\"wbc\", checkpoint=..., walk=False, target_velocity=[0.6, 0, 0])\n"
    "walk=False is documented as \"only the main policy\" - the caller's instruction not to enter locomotion",
    fontsize=14.5, fontweight="bold", y=0.975,
)

ROW = [
    (A, "on main: kwargs came back walk=True - the WALK network ran", "#b3261e"),
    (B, "with this change: walk=False honoured - the MAIN network ran", "#1b6b2f"),
]
for r, (facts, label, colour) in enumerate(ROW):
    for c, t in enumerate(TIMES):
        ax = fig.add_subplot(gs[r, c])
        im = frame(facts, t)
        sat = ((im.max(2).astype(int) - im.min(2).astype(int)) > 45).mean()
        assert sat > 0.05, f"blank frame at t={t}"
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(2.4)
        ax.set_xlabel(f"t = {t:.1f} s   pelvis x = {np.interp(t, facts['n_ticks'] and tA[:, 0] if r == 0 else tB[:, 0], (tA if r == 0 else tB)[:, 1]):.2f} m",
                      fontsize=10.5)
        if c == 0:
            ax.set_ylabel(label, fontsize=11, color=colour, fontweight="bold", labelpad=8)

# ── trace ─────────────────────────────────────────────────────────────
axt = fig.add_subplot(gs[2, 0:2])
for m in (1, 2, 3, 4):
    axt.axhline(m, color="#4a6fb5", ls=":", lw=1.1, zorder=1)
    axt.annotate(f"{m} m post", (0.02, m), xycoords=("axes fraction", "data"),
                 fontsize=8.5, color="#3a5a97", va="bottom")
axt.plot(tA[:, 0], tA[:, 1], color="#b3261e", lw=2.4,
         label=f"main: walk network, {A['travel_x']:.2f} m")
axt.plot(tB[:, 0], tB[:, 1], color="#1b6b2f", lw=2.4,
         label=f"this change: main network, {B['travel_x']:.2f} m")
axt.annotate("", xy=(8.0, A["travel_x"]), xytext=(8.0, B["travel_x"]),
             arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6))
axt.annotate(f"{A['travel_x'] - B['travel_x']:.2f} m apart\nat t = 8 s",
             xy=(7.55, (A["travel_x"] + B["travel_x"]) / 2), fontsize=9.5,
             ha="right", va="center", color="#333")
axt.set_xlabel("time (s)", fontsize=11)
axt.set_ylabel("pelvis x (m)", fontsize=11)
axt.set_title("Same robot, same velocity command, same seed - different network",
              fontsize=12, fontweight="bold")
axt.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
axt.grid(alpha=0.25)
axt.set_xlim(0, 8.6)

# ── verdict table ─────────────────────────────────────────────────────
axv = fig.add_subplot(gs[2, 2:4])
axv.axis("off"); axv.set_xlim(0, 1); axv.set_ylim(0, 1)
put(axv, 0.0, 0.965, "What the caller asked for, and what ran",
    fontsize=12.5, fontweight="bold")

ROWS = [
    ("caller asked for", "walk=False", "walk=False", None),
    ("kwargs['walk'] returned", str(A["kwargs_walk"]), str(B["kwargs_walk"]), "bad_a"),
    ("walk network loaded", str(A["walk_session_loaded"]), str(B["walk_session_loaded"]), "bad_a"),
    ("main-network inferences", f"{A['main_session_runs']} of 400", f"{B['main_session_runs']} of 400", "bad_a"),
    ("walk-network inferences", f"{A['walk_session_runs']} of 400", f"{B['walk_session_runs']} of 400", "bad_a"),
    ("run_policy status", A["status"], B["status"], None),
    ("distance walked (8 s)", f"{A['travel_x']:.3f} m", f"{B['travel_x']:.3f} m", None),
    ("pelvis height at 8 s", f"{A['z_end']:.4f} m", f"{B['z_end']:.4f} m", None),
]
TOP, FLOOR, PAD = 0.885, 0.055, 0.012
STEP = (TOP - FLOOR - PAD * len(ROWS)) / len(ROWS)
assert STEP > 0.030, STEP
put(axv, 0.42, TOP + 0.045, "main", fontsize=11, fontweight="bold", color="#b3261e", ha="center")
put(axv, 0.78, TOP + 0.045, "this change", fontsize=11, fontweight="bold", color="#1b6b2f", ha="center")
y = TOP
for label, va, vb, flag in ROWS:
    if flag == "bad_a":
        axv.add_patch(plt.Rectangle((0.28, y - STEP * 0.30), 0.28, STEP * 0.95,
                                    color="#f6d3d0", zorder=0, transform=axv.transAxes))
        axv.add_patch(plt.Rectangle((0.64, y - STEP * 0.30), 0.28, STEP * 0.95,
                                    color="#d4ecd9", zorder=0, transform=axv.transAxes))
    put(axv, 0.0, y, label, fontsize=10.5)
    put(axv, 0.42, y, va, fontsize=10.5, family="monospace", ha="center",
        color="#b3261e" if flag else "#222")
    put(axv, 0.78, y, vb, fontsize=10.5, family="monospace", ha="center",
        color="#1b6b2f" if flag else "#222")
    y -= STEP + PAD
assert y > 0.030, y
put(axv, 0.0, y - 0.005,
    "The registry declares wbc defaults {\"walk\": true}.  The defaults were merged\n"
    "before the caller's extra kwargs, and the extra loop skipped keys already\n"
    "present - so the default it had just inserted won.  t = 0 s frames are\n"
    "pixel-identical across both trees (max|delta| = 1/255): same scene, same seed.",
    fontsize=9.3, va="top", color="#444")

for ax, yy in placed:
    lo, hi = ax.get_ylim()
    if ax is axv:
        assert -0.03 <= yy <= 1.07, (yy, "axes-fraction out of range")

out = "/tmp/artifact_kwarg_precedence.png"
fig.savefig(out, dpi=118, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out} size={Image.open(out).size}")
