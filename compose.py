import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

B = json.load(open("/tmp/before.json")); A = json.load(open("/tmp/after.json"))
assert B["tree"] != A["tree"], "before/after came from the same tree"

EMITTING = ["record", "replay", "dagger"]
SPELLINGS = ["true(bool)", "false(bool)", '"false"', '"off"', "None", "[]", "0", "1"]
USABLE = {"true(bool)": "true", "false(bool)": "false"}

def honors(mode, spelling, cell):
    kind, val = cell[0], cell[1]
    if mode == "teleoperate":                       # CLI refuses the flag: emit nothing, refuse nothing
        return kind == "accepted-inert"
    if spelling in USABLE:
        return kind == "emitted" and val == USABLE[spelling]
    return kind == "refused" and val is True        # refused, and the message names play_sounds

MODES = EMITTING + ["teleoperate"]
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes)
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

GREEN, RED, GREY = "#1a7f37", "#b3261e", "#57606a"
fig = plt.figure(figsize=(15.6, 10.2), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[2.55, 1.15, 1.05], hspace=0.30, wspace=0.10)

n_bad = {"main": 0, "pr": 0}
for col, (facts, label, key) in enumerate(
    [(B, "main  (586109c)", "main"), (A, "this change", "pr")]
):
    ax = fig.add_subplot(gs[0, col]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    put(ax, 0.5, 1.045, label, ha="center", fontsize=13, fontweight="bold")
    put(ax, 0.5, 0.985, "what build_lerobot_command does with each play_sounds spelling",
        ha="center", fontsize=9.2, color=GREY, style="italic")
    x0, w = 0.20, 0.195
    for j, mode in enumerate(MODES):
        put(ax, x0 + w * j + w / 2, 0.915, mode, ha="center", fontsize=9.4, fontweight="bold")
    top, floor = 0.855, 0.055
    step = (top - floor) / len(SPELLINGS)
    assert step > 0.045, step
    for i, sp in enumerate(SPELLINGS):
        y = top - step * i
        put(ax, 0.185, y, sp, ha="right", va="center", fontsize=9.4, family="monospace")
        for j, mode in enumerate(MODES):
            cell = facts["grid"][mode][sp]
            ok = honors(mode, sp, cell)
            if not ok:
                n_bad[key] += 1
            if cell[0] == "emitted":
                txt = f"--play_sounds {cell[1]}"
            elif cell[0] == "refused":
                txt = "refused"
            else:
                txt = "no flag emitted"
            ax.add_patch(Rectangle((x0 + w * j + 0.006, y - step * 0.40), w - 0.012, step * 0.80,
                                   transform=ax.transAxes, facecolor=GREEN if ok else RED,
                                   alpha=0.16, edgecolor=GREEN if ok else RED, lw=0.9))
            put(ax, x0 + w * j + w / 2, y, txt, ha="center", va="center", fontsize=7.6,
                family="monospace", color=GREEN if ok else RED)
    final_y = top - step * (len(SPELLINGS) - 1)
    assert final_y > 0.04, final_y

# --- round trip through lerobot's own CLI ---------------------------------
axr = fig.add_subplot(gs[1, :]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 0.93, "Round trip: what lerobot's own CLI parses out of the argv we emit",
    fontsize=11.5, fontweight="bold")
rows = [("mode", "requested", "main runs with", "this change runs with")]
for mode in ("record", "replay"):
    for want in (True, False):
        k = f"{mode}:{want}"
        rows.append((mode, str(want), str(B["roundtrip"][k]), str(A["roundtrip"][k])))
cols = [0.02, 0.16, 0.31, 0.52]
top, floor = 0.74, 0.06
step = (top - floor) / len(rows)
assert step > 0.09, step
for i, r in enumerate(rows):
    y = top - step * i
    head = i == 0
    for cx, cell in zip(cols, r):
        bad = (not head) and r[0] in ("record", "replay") and cols.index(cx) == 2 and cell != r[1]
        put(axr, cx, y, cell, fontsize=9.6, family="monospace" if not head else None,
            fontweight="bold" if head or bad else None, color=RED if bad else ("black" if head else GREY))
assert top - step * (len(rows) - 1) > 0.04

# --- the wire trace -------------------------------------------------------
axw = fig.add_subplot(gs[2, :]); axw.axis("off"); axw.set_xlim(0, 1); axw.set_ylim(0, 1)
put(axw, 0.0, 0.93, "The argv tail for  record  with  play_sounds=False  (detached subprocess; "
                    "nothing reports back to the caller)", fontsize=11.5, fontweight="bold")
def tail(facts):
    a = facts["modes"]["record"]["argv_off"]
    return " ".join(a[-6:])
lines = [("main", tail(B), RED), ("this change", tail(A), GREEN)]
top, floor = 0.62, 0.10
step = (top - floor) / max(len(lines), 1)
for i, (lab, txt, col) in enumerate(lines):
    y = top - step * i
    put(axw, 0.02, y, f"{lab:12}", fontsize=9.6, fontweight="bold")
    put(axw, 0.16, y, "... " + txt, fontsize=9.0, family="monospace", color=col)
put(axw, 0.02, 0.10,
    f"divergences from what each mode can honor: main {n_bad['main']} of 32 cells  ->  this change {n_bad['pr']} of 32",
    fontsize=10.4, fontweight="bold")

for ax, y, is_axes in placed:
    if is_axes:
        assert -0.03 <= y <= 1.07, (y, ax)
assert (n_bad["main"], n_bad["pr"]) == (24, 0), n_bad
assert B["roundtrip"]["record:False"] is True and A["roundtrip"]["record:False"] is False
assert B["modes"]["record"]["identical_argv"] and not A["modes"]["record"]["identical_argv"]
assert A["modes"]["teleoperate"]["true_token"] is None

out = pathlib.Path("/tmp/play_sounds.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image
im = np.array(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print("OK", out, im.shape, "divergences", n_bad)
