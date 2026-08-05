import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.load(open("/tmp/before.json"))
B = json.load(open("/tmp/after.json"))
assert A["tree"] != B["tree"], "before/after came from the same tree"
assert (A["divergences"], B["divergences"], A["cells"]) == (22, 0, 30), (A["divergences"], B["divergences"])
assert A["rows"]["3"] == B["rows"]["3"], "the usable-count control must be identical on both trees"
assert A["rows"]["None"] == B["rows"]["None"], "the unset control must be identical on both trees"
USABLE = set(A["usable"])
COLS = [("validate", "LerobotTrainer.validate()"), ("reserved", "what the run does"), ("tool", "lerobot_train tool")]

def ok(label, col, val):
    usable = label in USABLE
    if col == "validate":
        return val == ("no problem" if usable else "reports a problem")
    if col == "reserved":
        if label == "3":
            return val.startswith("3 episode")
        if label == "None":
            return val == "no split, no eval pass"
        return val == "refused before the run starts"
    return val == ("accepted" if usable else "refused (names the field)")

GREEN, RED, GREY = "#1b7f3b", "#b3261e", "#5f6368"
fig = plt.figure(figsize=(17.6, 7.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.115], hspace=0.10, wspace=0.055,
                      left=0.006, right=0.994, top=0.885, bottom=0.035)
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

XS = [0.012, 0.108, 0.415, 0.735]
for j, (data, title) in enumerate(((A, "main"), (B, "this change"))):
    ax = fig.add_subplot(gs[0, j]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="#c9ccd1", lw=1.3,
                               transform=ax.transAxes, clip_on=False))
    put(ax, 0.5, 1.055, title, ha="center", va="bottom", fontsize=15, fontweight="bold",
        transform=ax.transAxes)
    put(ax, 0.5, 0.972, f"contract divergences: {data['divergences']} of {data['cells']} cells",
        ha="center", va="center", fontsize=10.5,
        color=RED if data["divergences"] else GREEN, fontweight="bold")
    for x, head in zip(XS, ["val_episodes"] + [c[1] for c in COLS]):
        put(ax, x, 0.905, head, ha="left", va="center", fontsize=9.6, fontweight="bold", color="#202124")
    ax.plot([0.008, 0.992], [0.877, 0.877], color="#9aa0a6", lw=1.0, clip_on=False)
    y, step = 0.818, 0.0805
    for label in data["labels"]:
        row = data["rows"][label]
        if label in USABLE:
            ax.add_patch(plt.Rectangle((0.006, y - 0.033), 0.988, 0.066,
                                       color="#eef2f6", zorder=0, clip_on=False))
        put(ax, XS[0], y, label, ha="left", va="center", fontsize=10.4,
            family="monospace", fontweight="bold",
            color=GREY if label in USABLE else "#202124")
        for x, (key, _) in zip(XS[1:], COLS):
            val = row[key]
            good = ok(label, key, val)
            put(ax, x, y, val, ha="left", va="center", fontsize=9.3,
                color=GREEN if good else RED, fontweight="normal" if good else "bold")
        y -= step
    put(ax, 0.012, 0.033, "shaded = a value the contract honors (a positive integer count, or None)",
        ha="left", va="center", fontsize=8.4, color=GREY, style="italic")

foot = fig.add_subplot(gs[1, :]); foot.set_xlim(0, 1); foot.set_ylim(0, 1); foot.axis("off")
foot.add_patch(plt.Rectangle((0, 0), 1, 1, color="#f6f7f9", zorder=0))
put(foot, 0.012, 0.66,
    "val_episodes is converted into lerobot's real-valued dataset.eval_split, and lerobot holds out ceil(episodes_in_task * eval_split) - "
    "so a comparison is wrong at both ends.",
    ha="left", va="center", fontsize=9.6, color="#202124")
put(foot, 0.012, 0.26,
    "0 / -5 / nan produced no split and no eval cadence at all (the run trained on all 10 episodes and logged no validation loss, reported as launchable); "
    "True reserved 1, 2.7 reserved 3, 0.5 reserved 0 while still evaluating; '5' and [5] raised TypeError out of a validate() documented to return problems.",
    ha="left", va="center", fontsize=9.6, color="#202124")
fig.suptitle("val_episodes: one shared positive-count domain for both writers of lerobot's eval_split   "
             "(10-episode single-task dataset)", fontsize=12.4, y=0.985, color="#202124")

for ax, y in placed:
    lo, hi = ax.get_ylim()
    assert lo - 0.03 <= y <= hi + 0.07, f"text at y={y} outside {ax.get_ylim()}"

out = pathlib.Path("/tmp/art/val_episodes_domain.png")
fig.savefig(out, dpi=125, bbox_inches="tight", pad_inches=0.28, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out} {im.shape[1]}x{im.shape[0]}  divergences {A['divergences']} -> {B['divergences']}")
