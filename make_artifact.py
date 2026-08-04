"""Compose the measured run-size verdict figure from the two JSON dumps."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

A = json.load(open("/tmp/A_main.json"))
B = json.load(open("/tmp/B_branch.json"))
assert A["tree"] != B["tree"], "both halves came from the same tree"

# measured downstream consequence for each value main let through
CONSEQ = {
 ("steps", "10000"): ("honored", "10000 optimizer steps"),
 ("steps", "0"): ("ok", "refused at preflight"),
 ("steps", "-5"): ("ok", "refused at preflight"),
 ("steps", "True"): ("bad", "len(range(0, True)) == 1  ->  a SILENT one-step run"),
 ("steps", "2.7"): ("bad", "TypeError in range() after dataset + model load"),
 ("steps", "nan"): ("bad", "TypeError in range() after dataset + model load"),
 ("steps", "inf"): ("bad", "TypeError in range() after dataset + model load"),
 ("steps", "'1000'"): ("bad", "TypeError out of validate() itself"),
 ("steps", "None"): ("bad", "TypeError out of validate() itself"),
 ("global_batch_size", "32"): ("honored", "batches of 32"),
 ("global_batch_size", "0"): ("bad", "ValueError from DataLoader, dataset already materialized"),
 ("global_batch_size", "-8"): ("bad", "ValueError from DataLoader, dataset already materialized"),
 ("global_batch_size", "True"): ("bad", "ValueError from DataLoader, dataset already materialized"),
 ("global_batch_size", "2.7"): ("bad", "ValueError from DataLoader, dataset already materialized"),
 ("global_batch_size", "nan"): ("bad", "ValueError from DataLoader, dataset already materialized"),
 ("global_batch_size", "'32'"): ("bad", "ValueError from DataLoader, dataset already materialized"),
}
rows = [(a["field"], a["value"], a["uniform"], b["uniform"]) for a, b in zip(A["rows"], B["rows"])]
for a in A["rows"]:
    assert a["uniform"] != "SPLIT", a          # all four backends agreed, before and after
for b in B["rows"]:
    assert b["uniform"] != "SPLIT", b
assert A["local_copies"] == 4 and A["gate_call_sites"] == 0
assert B["local_copies"] == 0 and B["gate_call_sites"] == 4
n_defect = sum(1 for f, v, m, _ in rows if CONSEQ[(f, v)][0] == "bad")
n_after = sum(1 for f, v, _, p in rows if CONSEQ[(f, v)][0] == "bad" and p != "refused")
assert (n_defect, n_after) == (12, 0), (n_defect, n_after)
assert all(p == "accepted" for f, v, m, p in rows if CONSEQ[(f, v)][0] == "honored")

RED, GRN, GREY = "#c0392b", "#1e7a45", "#5a5a5a"
fig = plt.figure(figsize=(15.6, 9.4))
gs = fig.add_gridspec(2, 1, height_ratios=[7.0, 1.85], hspace=0.12,
                      left=0.012, right=0.988, top=0.945, bottom=0.02)

placed: list[tuple[object, float]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig.suptitle("TrainSpec run size: 4 duplicated comparisons, 1 unchecked factor  ->  one shared positive-count domain",
             fontsize=15.5, fontweight="bold", y=0.985)
fig.text(0.5, 0.958, "every cell measured by one script run in two trees (upstream/main 3804e647 and this branch); "
                     "verdicts were identical across mock / cosmos3 / groot / lerobot_local",
         ha="center", fontsize=9.6, style="italic", color=GREY)

ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
COLS = [(0.012, "TrainSpec field"), (0.175, "value"), (0.275, "main"),
        (0.375, "what main then did with it"), (0.845, "this PR")]
for x, h in COLS:
    put(ax, x, 0.972, h, fontsize=11.2, fontweight="bold", va="top")
ax.plot([0.008, 0.992], [0.951, 0.951], color="#333", lw=1.3)

top, step = 0.925, 0.0565
for i, (field, value, before, after) in enumerate(rows):
    y = top - i * step
    kind, note = CONSEQ[(field, value)]
    bad = kind == "bad"
    ax.add_patch(Rectangle((0.008, y - 0.020), 0.984, 0.046,
                           color=("#fdecea" if bad else "#eef7f0"), zorder=0))
    put(ax, 0.012, y, f"`{field}`", fontsize=10.3, family="monospace", va="center")
    put(ax, 0.175, y, value, fontsize=10.6, family="monospace", va="center", fontweight="bold")
    put(ax, 0.275, y, before, fontsize=10.1, va="center",
        color=(RED if bad else GRN), fontweight="bold" if bad else "normal")
    put(ax, 0.375, y, note, fontsize=10.0, va="center", color=(RED if bad else GREY))
    put(ax, 0.845, y, after, fontsize=10.1, va="center", color=GRN,
        fontweight="bold" if bad else "normal")
    if i == 8:
        ax.plot([0.008, 0.992], [y - 0.028, y - 0.028], color="#999", lw=0.9, ls=":")

ax2 = fig.add_subplot(gs[1]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
ax2.add_patch(Rectangle((0.008, 0.02), 0.984, 0.96, fill=False, ec="#333", lw=1.2))
put(ax2, 0.022, 0.84, "ownership of the rule", fontsize=11.2, fontweight="bold", va="center")
put(ax2, 0.022, 0.58,
    f"local `if spec.steps <= 0` copies in strands_robots/training/:   {A['local_copies']}  ->  {B['local_copies']}\n"
    f"validate() call sites of the one shared gate:                    {A['gate_call_sites']}  ->  {B['gate_call_sites']}\n"
    f"the second factor (`global_batch_size`) checked by any backend:   no  ->  yes",
    fontsize=10.2, family="monospace", va="center", color="#111")
put(ax2, 0.556, 0.84, "no regression", fontsize=11.2, fontweight="bold", va="center")
put(ax2, 0.556, 0.58,
    "unusable values a backend cannot honor:   12 of 16 let through  ->  0\n"
    "usable run sizes (10000 / 32) still validate clean:      2 of 2\n"
    "RL specs, which read neither field, report nothing:  unchanged",
    fontsize=10.2, family="monospace", va="center", color="#111")

for a, y in placed:
    lo, hi = a.get_ylim(); pad = 0.04 * (hi - lo)
    assert lo - pad <= y <= hi + pad, (y, lo, hi)

out = pathlib.Path("/tmp/run_size_domain.png")
fig.savefig(out, dpi=118, bbox_inches="tight", pad_inches=0.3, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", out, im.shape, f"{out.stat().st_size/1024:.0f} KB")
