import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

A = json.load(open("/tmp/facts_pre1952.json"))     # upstream/main
B = json.load(open("/tmp/facts_robots-src.json"))  # this change
assert A["tree"] != B["tree"], "before/after measured the same tree"

COLS = [("server", "PolicyServer(port=)", "bind_ok"),
        ("cli", "--port  (CLI)", "bind_ok"),
        ("client", "RemotePolicy(port=)", "dial_ok")]

def wrong(row, key, okkey):
    accepted = row[key] == "accepted"
    return accepted != row[okkey]

def count_wrong(d):
    return sum(wrong(r, k, ok) for r in d["rows"] for k, _, ok in COLS)

W_A, W_B = count_wrong(A), count_wrong(B)
CELLS = len(A["rows"]) * len(COLS)
assert (CELLS, W_A, W_B) == (39, 20, 0), (CELLS, W_A, W_B)
assert isinstance(A["ephemeral_bound_port"], int) and A["ephemeral_bound_port"] > 0
assert isinstance(B["ephemeral_bound_port"], int) and B["ephemeral_bound_port"] > 0

GREEN, RED = "#1b7f4b", "#b3261e"
fig = plt.figure(figsize=(15.0, 8.4), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[8.2, 1.5], hspace=0.30, wspace=0.10,
                      left=0.035, right=0.982, top=0.885, bottom=0.045)
placed = []

def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, **kw)

fig.suptitle("One `port` parameter, three surfaces, three domains -> two domains, one per role",
             fontsize=15.5, fontweight="bold", y=0.972)
put(fig, 0.5, 0.930,
    "A server BINDS a port, so it may ask the kernel for an ephemeral one (`0`).  A client DIALS one, so it cannot.\n"
    "Green = the surface's verdict matches the domain its role actually has.  Measured by constructing each object.",
    ha="center", va="center", fontsize=10.4, color="#333333", transform=fig.transFigure)
placed.pop()

for col, (d, title) in enumerate([(A, "main @ 1bf78706"), (B, "this change")]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n_wrong = count_wrong(d)
    ax.set_title(f"{title}\n{n_wrong} of {CELLS} cells disagree with the role's domain",
                 fontsize=12.5, fontweight="bold",
                 color=RED if n_wrong else GREEN, pad=11)
    xs = [0.155, 0.415, 0.675, 0.935]
    put(ax, 0.075, 0.955, "port =", ha="center", va="center", fontsize=10.2, fontweight="bold")
    for (key, label, _), x in zip(COLS, xs[1:]):
        put(ax, x - 0.13, 0.955, label, ha="center", va="center", fontsize=9.9, fontweight="bold")
    top, step = 0.900, 0.0665
    for i, r in enumerate(d["rows"]):
        y = top - i * step
        put(ax, 0.075, y, r["label"], ha="center", va="center", fontsize=10.0,
            family="monospace", fontweight="bold")
        for (key, _, okkey), x in zip(COLS, xs[1:]):
            bad = wrong(r, key, okkey)
            accepted = r[key] == "accepted"
            ax.add_patch(Rectangle((x - 0.255, y - 0.028), 0.250, 0.056,
                                   facecolor=RED if bad else GREEN, alpha=0.20,
                                   edgecolor=RED if bad else GREEN, lw=1.0))
            txt = "accepted" if accepted else "refused"
            if bad:
                txt += "  <-- " + ("must refuse" if accepted else "is valid")
            put(ax, x - 0.130, y, txt, ha="center", va="center", fontsize=8.9,
                family="monospace", color=RED if bad else GREEN,
                fontweight="bold" if bad else "normal")

for col, (d, title) in enumerate([(A, "main @ 1bf78706"), (B, "this change")]):
    ax = fig.add_subplot(gs[1, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.05), 0.98, 0.90, facecolor="#f4f4f4",
                           edgecolor="#bbbbbb", lw=1.0))
    put(ax, 0.03, 0.76, "the documented ephemeral bind, end to end", fontsize=10.0,
        fontweight="bold", va="center")
    put(ax, 0.03, 0.47,
        f"PolicyServer(port=0).start()  ->  .port == {d['ephemeral_bound_port']}   (OS-assigned, read back)",
        fontsize=9.6, family="monospace", va="center", color=GREEN)
    put(ax, 0.03, 0.20,
        "unchanged by this PR - it is the one behaviour accepting 0 exists for",
        fontsize=9.2, va="center", color="#555555", style="italic")

for ax, y in placed:
    lo, hi = ax.get_ylim()
    assert lo - 0.03 <= y <= hi + 0.06, (y, lo, hi)

out = "/tmp/artifact_1952.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in [("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])]:
    n = int((np.abs(band - 255).sum(axis=2) > 20).sum())
    assert n == 0, (name, n)
print(f"OK {Image.open(out).size}  before={W_A}/{CELLS}  after={W_B}/{CELLS}")
