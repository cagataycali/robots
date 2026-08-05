import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np

A = json.load(open("/tmp/before_1949.json"))   # upstream/main
B = json.load(open("/tmp/after_1949.json"))    # this change
assert A["tree"] != B["tree"], "same tree - measurement would be fake"

# --- assert every claim this figure makes -----------------------------------
assert A["with_torch"]["None"]["kind"] == "raised" and A["with_torch"]["None"]["random_moved"]
assert A["without_torch"]["None"]["kind"] == "accepted" and A["without_torch"]["None"]["random_moved"]
assert A["with_torch"]["None"]["numpy_moved"] and A["without_torch"]["None"]["numpy_moved"]
for k in ("with_torch", "without_torch"):
    assert B[k]["None"]["kind"] == "refused"
    assert not B[k]["None"]["random_moved"] and not B[k]["None"]["numpy_moved"]
for c in ("2.5", "7", "MAX+1"):
    for k in ("with_torch", "without_torch"):
        assert A[k][c]["kind"] == B[k][c]["kind"], c
        assert A[k][c]["random_moved"] == B[k][c]["random_moved"], c
assert A["default_msgs"] == B["default_msgs"] and A["default_ceiling"] == B["default_ceiling"]
assert A["facade_none_accepted"] and B["facade_none_accepted"]
assert A["randomize_none_accepted"] and B["randomize_none_accepted"]
assert all(B["valid_seeds_reproducible"].values())
assert "or None" in A["domain"]["2.5"] and "or None" in A["domain"]["MAX+1"]

RED, GREEN, AMBER, INK, MUTED = "#c0392b", "#1e8449", "#b9770e", "#17202a", "#5d6d7e"
placed: list[tuple] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 11.4))
gs = fig.add_gridspec(3, 3, height_ratios=[1.42, 0.80, 0.72], hspace=0.30, wspace=0.11,
                      left=0.028, right=0.972, top=0.905, bottom=0.035)

fig.suptitle("set_eval_seed(None): one value, two outcomes, and a process-wide RNG side effect either way",
             fontsize=16.5, fontweight="bold", y=0.972)
fig.text(0.5, 0.933, "measured by one script run in a worktree at upstream/main and on this branch "
         "- every cell below is read from those two JSON dumps",
         ha="center", fontsize=10.3, color=MUTED, style="italic")

# ---------------- Row 0: the three outcomes for None ------------------------
COLS = [
    ("main - torch installed", A["with_torch"]["None"], RED,
     "torch.manual_seed(None) raises after\nrandom + NumPy are already reseeded"),
    ("main - no torch (minimal install)", A["without_torch"]["None"], RED,
     "the ImportError handler swallows the\nonly thing that objected: silent success"),
    ("this change (both installs)", B["with_torch"]["None"], GREEN,
     "refused ahead of every RNG, so the\nrefusal has no side effect either"),
]
for i, (title, rec, colour, note) in enumerate(COLS):
    ax = fig.add_subplot(gs[0, i]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                           facecolor=colour, alpha=0.055, edgecolor=colour, lw=2.0, zorder=0))
    put(ax, 0.5, 0.945, title, ha="center", fontsize=12.2, fontweight="bold", color=colour)
    verdict = {"raised": "RAISED  (bare TypeError)", "accepted": "ACCEPTED  (no error at all)",
               "refused": "REFUSED  (named ValueError)"}[rec["kind"]]
    put(ax, 0.5, 0.858, verdict, ha="center", fontsize=11.6, fontweight="bold", color=colour)
    msg = rec["msg"] or "(returned normally)"
    if rec["kind"] == "raised":
        body = "TypeError: int() argument must be a\nstring, a bytes-like object or a real\nnumber, not 'NoneType'\n\n-> names neither the parameter\n   nor the method that accepted it"
    elif rec["kind"] == "accepted":
        body = "(nothing raised, nothing logged)\n\n-> the caller is told the seed was\n   applied; there was no seed"
    else:
        body = ("set_eval_seed: seed is required;\nNone is the absence of a seed, not\na seed to apply. To leave the RNGs\nuntouched, do not call\nset_eval_seed - reseeding them from\nentropy is a global side effect an\nunseeded rollout must not acquire.")
    put(ax, 0.055, 0.775, body, ha="left", va="top", fontsize=8.9, family="monospace", color=INK)
    # RNG ledger
    y0 = 0.315
    put(ax, 0.5, y0 + 0.075, "process RNG state after the call", ha="center", fontsize=9.7,
        fontweight="bold", color=MUTED)
    for j, (label, key) in enumerate((("random (Python global)", "random_moved"), ("numpy.random (global)", "numpy_moved"))):
        moved = rec[key]
        c = RED if moved else GREEN
        txt = "RESEEDED FROM ENTROPY" if moved else "untouched"
        put(ax, 0.075, y0 - j * 0.072, f"{label}", ha="left", fontsize=9.4, color=INK)
        put(ax, 0.955, y0 - j * 0.072, txt, ha="right", fontsize=9.4, fontweight="bold", color=c)
    put(ax, 0.5, 0.115, note, ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

# ---------------- Row 1: nothing else changed -------------------------------
ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.5, 0.985, "Every other value: unchanged on both installs  (the refusal is scoped to the one value that has no seed to apply)",
    ha="center", fontsize=12.4, fontweight="bold", color=INK)
rows = [("None", "the defect"), ("2.5", "already refused"), ("7", "a usable seed"), ("MAX+1", "above the applier's ceiling")]
xs = [0.055, 0.235, 0.435, 0.635, 0.835]
hdr = ["seed", "main +torch", "main -torch", "this change", ""]
for x, h in zip(xs, hdr):
    put(ax, x, 0.845, h, ha="left", fontsize=10.4, fontweight="bold", color=MUTED)
for r, (label, note) in enumerate(rows):
    y = 0.735 - r * 0.135
    changed = (label == "None")
    put(ax, xs[0], y, label, ha="left", fontsize=10.8, family="monospace",
        fontweight="bold" if changed else "normal", color=RED if changed else INK)
    for k, x in ((("with_torch"), xs[1]), (("without_torch"), xs[2])):
        rec = A[k][label]
        s = {"raised": "raised", "accepted": "accepted", "refused": "refused"}[rec["kind"]]
        s += "  / RNG moved" if rec["random_moved"] else "  / RNG untouched"
        ok = not changed
        put(ax, x, y, s, ha="left", fontsize=9.9, family="monospace", color=INK if ok else RED)
    rec = B["with_torch"][label]
    s = {"raised": "raised", "accepted": "accepted", "refused": "refused"}[rec["kind"]]
    s += "  / RNG moved" if rec["random_moved"] else "  / RNG untouched"
    put(ax, xs[3], y, s, ha="left", fontsize=9.9, family="monospace", color=GREEN if changed else INK)
    put(ax, xs[4], y, ("<- the only row that differs" if changed else "identical"), ha="left",
        fontsize=9.4, color=RED if changed else MUTED, style="italic")

# ---------------- Row 2: the message band + no-regression -------------------
ax = fig.add_subplot(gs[2, :2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0.005, 0.02), 0.99, 0.96, transform=ax.transAxes, facecolor=AMBER,
                       alpha=0.05, edgecolor=AMBER, lw=1.5, zorder=0))
put(ax, 0.5, 0.925, "...and the reasons stop advertising a value this destination refuses",
    ha="center", fontsize=11.8, fontweight="bold", color=AMBER)
put(ax, 0.03, 0.735, "main, set_eval_seed(2.5):", ha="left", fontsize=9.6, color=MUTED)
put(ax, 0.03, 0.625, A["domain"]["2.5"], ha="left", fontsize=8.7, family="monospace", color=RED)
put(ax, 0.03, 0.520, "^ offers None at the one destination that cannot apply it", ha="left",
    fontsize=8.9, color=RED, style="italic")
put(ax, 0.03, 0.370, "this change, set_eval_seed(2.5):", ha="left", fontsize=9.6, color=MUTED)
put(ax, 0.03, 0.260, B["domain"]["2.5"], ha="left", fontsize=8.7, family="monospace", color=GREEN)
put(ax, 0.03, 0.150, "allow_none=False drops it - the same per-destination shape as the existing max_seed ceiling",
    ha="left", fontsize=8.9, color=GREEN, style="italic")

ax = fig.add_subplot(gs[2, 2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax.transAxes, facecolor=GREEN,
                       alpha=0.05, edgecolor=GREEN, lw=1.5, zorder=0))
put(ax, 0.5, 0.925, "No regression (measured)", ha="center", fontsize=11.3, fontweight="bold", color=GREEN)
checks = [
    ("randomize(seed=None)", "still accepted"),
    ("run_policy(seed=None)", "still accepted"),
    ("default-path messages", "byte-identical"),
    ("ceiling message", "byte-identical"),
    ("seeds 0 / 7 / MAX", "reproducible"),
]
for i, (k, v) in enumerate(checks):
    y = 0.760 - i * 0.152
    put(ax, 0.055, y, k, ha="left", fontsize=9.5, color=INK)
    put(ax, 0.955, y, f"OK  {v}", ha="right", fontsize=9.2, fontweight="bold", color=GREEN)

for ax_, y in placed:
    lo, hi = ax_.get_ylim()
    assert lo - 0.03 <= y <= hi + 0.07, f"text at y={y} outside {ax_.get_ylim()}"

out = "/tmp/artifact_1949.png"
fig.savefig(out, dpi=125, bbox_inches="tight", pad_inches=0.28, facecolor="white")
plt.close(fig)
im = np.array(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", out, im.shape)
