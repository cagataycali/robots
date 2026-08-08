"""Compose the measured evidence figure. Every cell is read from the two dumps."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

A = json.load(open("/tmp/A_main.json")); B = json.load(open("/tmp/B_branch.json"))
assert A["tree"] != B["tree"], "before/after came from the same tree"

CASES = ["5 (default)", "0", "-3", "True", "2.7", "None"]
NEVER = A["never_trained"]["fp"]["sum"]
HONORED = A["cases"]["5 (default)"]["fp"]["sum"]
# --- claims asserted before anything is drawn -------------------------------
assert A["cases"]["0"]["status"] == "success" and A["cases"]["0"]["steps"] == 0
assert A["cases"]["-3"]["status"] == "success" and A["cases"]["-3"]["steps"] == 0
assert A["cases"]["0"]["fp"]["sum"] == A["cases"]["-3"]["fp"]["sum"] == NEVER, "not the untrained net"
assert set(A["cases"]["0"]["losses"].values()) == {0.0}
assert A["cases"]["True"]["steps"] == 12 and A["cases"]["5 (default)"]["steps"] == 60
assert A["cases"]["2.7"]["status"] == "TypeError" and A["cases"]["None"]["status"] == "TypeError"
assert B["cases"]["5 (default)"]["fp"]["sum"] == HONORED, "honored run changed"
assert all(B["cases"][c]["refused"] for c in CASES if c != "5 (default)")
assert not B["cases"]["5 (default)"]["refused"]
GREEN, RED, AMBER, GREY = "#1b7f4b", "#b3261e", "#a05a00", "#4a4a4a"

placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 10.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.00, 1.28, 0.50], hspace=0.20, wspace=0.06,
                      left=0.035, right=0.972, top=0.915, bottom=0.035)
fig.suptitle("PPO's optimization-epoch count is the loop bound of the entire optimizer step",
             fontsize=16.5, fontweight="bold", y=0.973)
fig.text(0.5, 0.941, "measured over a real 60-step PPO run on the SO-100 reach task "
         "(seed 0, rollout_steps=20, num_mini_batches=4)",
         ha="center", fontsize=10.4, style="italic", color=GREY)

# ---------------- row 1: verdict matrix ------------------------------------
for col, (tree, label, colour) in enumerate(
        [(A, "main", RED), (B, "this change", GREEN)]):
    ax = fig.add_subplot(gs[0, col]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    put(ax, 0.5, 1.045, label, ha="center", fontsize=13, fontweight="bold", color=colour,
        transform=ax.transAxes)
    put(ax, 0.015, 0.92, "num_learning_epochs", fontsize=9.6, fontweight="bold", color=GREY)
    put(ax, 0.40, 0.92, "preflight", fontsize=9.6, fontweight="bold", color=GREY)
    put(ax, 0.60, 0.92, "run", fontsize=9.6, fontweight="bold", color=GREY)
    put(ax, 0.80, 0.92, "grad steps", fontsize=9.6, fontweight="bold", color=GREY)
    ax.plot([0.01, 0.99], [0.885, 0.885], color=GREY, lw=0.9, transform=ax.transAxes)
    rows = len(CASES); top, floor = 0.83, 0.05
    step = (top - floor) / rows
    assert step > 0.030, step
    for i, c in enumerate(CASES):
        r = tree["cases"][c]; y = top - i * step
        ok = (c == "5 (default)")
        honors = ok if tree is A else True
        if tree is A:
            honors = ok or r["status"] == "TypeError"      # a raise at least does not lie
        band = (GREEN if honors else RED)
        if tree is A and r["status"] == "TypeError":
            band = AMBER
        ax.add_patch(Rectangle((0.008, y - step * 0.42), 0.984, step * 0.80,
                               facecolor=band, alpha=0.10, edgecolor="none",
                               transform=ax.transAxes))
        put(ax, 0.015, y - 0.012, c, fontsize=10.4, family="monospace",
            fontweight="bold" if not ok else "normal")
        pf = "refused" if r["refused"] else "silent"
        run = ("-" if r["refused"] else
               ("TypeError" if r["status"] == "TypeError" else r["status"]))
        put(ax, 0.40, y - 0.012, pf, fontsize=10.2, color=(GREEN if r["refused"] else band))
        put(ax, 0.60, y - 0.012, run, fontsize=10.2, color=band)
        put(ax, 0.80, y - 0.012, "-" if r["refused"] else str(r["steps"]),
            fontsize=10.2, family="monospace", color=band,
            fontweight="bold" if (not r["refused"] and r["steps"] == 0 and r["status"] == "success") else "normal")

# ---------------- row 2 left: the consequence on main ----------------------
ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.0, 1.020, "What a silently-accepted count produced on main",
    fontsize=12.4, fontweight="bold", color=RED, transform=ax.transAxes)
lines = [
    ("num_learning_epochs = 0   ->  status = \"success\"", RED, "bold"),
    ("    optimizer steps taken .............  0   (of 60 at the default)", RED, "normal"),
    ("    reported surrogate_loss / value_loss / entropy .....  0.0 / 0.0 / 0.0", RED, "normal"),
    ("    checkpoint written .....  policy.pt, 34701 parameters", RED, "normal"),
    ("", GREY, "normal"),
    ("The metrics read 0.0 rather than blank because the update averages", GREY, "italic"),
    ("its accumulators through  max(1, n_updates)  -- so an epoch count that", GREY, "italic"),
    ("ran no minibatch reports plausible losses for a run that learned nothing.", GREY, "italic"),
    ("", GREY, "normal"),
    ("Bit-exact actor-critic parameter sum of the checkpoint each run wrote:", GREY, "bold"),
    (f"    num_learning_epochs = 0 ....  {A['cases']['0']['fp']['sum']}", RED, "mono"),
    (f"    num_learning_epochs = -3 ...  {A['cases']['-3']['fp']['sum']}", RED, "mono"),
    (f"    update() replaced by a no-op  {NEVER}", GREY, "mono"),
    ("    ^ identical to 16 digits: the checkpoint IS the untrained network", RED, "bold"),
    (f"    num_learning_epochs = 5 ....  {HONORED}   (trained)", GREEN, "mono"),
]
top, floor = 0.955, 0.02
lh = (top - floor) / len(lines)
assert lh > 0.045, lh
for i, (txt, col, sty) in enumerate(lines):
    y = top - i * lh
    kw = {"fontsize": 9.9, "color": col}
    if sty == "mono": kw.update(family="monospace", fontsize=9.1)
    elif sty == "bold": kw.update(fontweight="bold")
    elif sty == "italic": kw.update(style="italic")
    put(ax, 0.0, y, txt, **kw)

# ---------------- row 2 right: what the change reports --------------------
ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.0, 1.020, "What the preflight reports instead",
    fontsize=12.4, fontweight="bold", color=GREEN, transform=ax.transAxes)
msgs = [(c, B["cases"][c]["message"]) for c in CASES if c != "5 (default)"]
rlines = [("Read-only, before the environment or the networks are built:", GREY, "bold")]
rlines += [("    " + m, GREEN, "mono") for _, m in msgs]
rlines += [
    ("", GREY, "normal"),
    ("train() is fail-closed on validate(), so no rollout is collected and no", GREY, "italic"),
    ("checkpoint is written for a value the optimizer loop cannot honor.", GREY, "italic"),
    ("", GREY, "normal"),
    ("No regression -- the honored run is bit-identical across both trees:", GREY, "bold"),
    (f"    main .........  {HONORED}", GREY, "mono"),
    (f"    this change ..  {B['cases']['5 (default)']['fp']['sum']}", GREEN, "mono"),
    (f"    optimizer steps: {A['cases']['5 (default)']['steps']} on both", GREEN, "mono"),
    ("", GREY, "normal"),
    ("Scoped to the on-policy backend: FastSAC optimizes per gradient step", GREY, "italic"),
    ("from a replay buffer and has no epoch loop, so it stays silent.", GREY, "italic"),
]
lh2 = (top - floor) / len(rlines)
assert lh2 > 0.045, lh2
for i, (txt, col, sty) in enumerate(rlines):
    y = top - i * lh2
    kw = {"fontsize": 9.9, "color": col}
    if sty == "mono": kw.update(family="monospace", fontsize=8.6)
    elif sty == "bold": kw.update(fontweight="bold")
    elif sty == "italic": kw.update(style="italic")
    put(ax, 0.0, y, txt, **kw)

# ---------------- row 3: the premise --------------------------------------
ax = fig.add_subplot(gs[2, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0.0, 0.06), 1.0, 0.86, facecolor="#f2f2f2", edgecolor=GREY, lw=0.7))
put(ax, 0.014, 0.76, "Why the field is not merely a scale factor  (PpoTrainer.update)",
    fontsize=11.6, fontweight="bold", color=GREY)
put(ax, 0.014, 0.50, "for _ in range(spec.num_learning_epochs):        # <- the caller's value bounds this loop",
    fontsize=10.0, family="monospace", color=GREY)
put(ax, 0.014, 0.30, "    for start in range(0, n, mb_size):  ...  self.optimizer.step()   # <- every gradient step is inside it",
    fontsize=10.0, family="monospace", color=RED)
put(ax, 0.014, 0.13, "So a non-positive count does not reduce the optimization -- it removes all of it.",
    fontsize=10.2, style="italic", fontweight="bold", color=RED)

for ax_, y in placed:
    assert -0.06 <= y <= 1.07, (y,)
out = pathlib.Path("/tmp/epochs_evidence.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
import numpy as np
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white px"
print("OK", out, im.shape[1], "x", im.shape[0])
