import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = json.load(open("/tmp/facts_main.json"))   # upstream/main
B = json.load(open("/tmp/facts_branch.json")) # this change

# ---- self-audit: every claim the figure makes ----------------------------
assert A["tree"] != B["tree"], "both probes resolved to the same tree"
assert A["out_of_domain"]["status"] == "returned"
assert A["out_of_domain"]["py_reproducible"] is True
assert A["out_of_domain"]["np_reproducible"] is False, "main's NumPy stream must be the un-seeded half"
assert B["out_of_domain"]["status"].startswith("ValueError: reseed_client_rngs: seed must be")
for t in (A, B):
    assert t["in_domain"]["py_reproducible"] and t["in_domain"]["np_reproducible"], "in-domain seed must stay reproducible on both trees"
n_partial_main = sum(1 for v in A["sweep"].values() if (v["py"] == "reseeded") != (v["np"] == "reseeded"))
n_partial_branch = sum(1 for v in B["sweep"].values() if (v["py"] == "reseeded") != (v["np"] == "reseeded"))
assert (n_partial_main, n_partial_branch) == (6, 0), (n_partial_main, n_partial_branch)

placed = []
def put(ax, x, y, s, **kw):
    # Record which coordinate system the y is in: an axes-fraction placement is
    # not comparable against the axes' data ylim.
    placed.append((ax, y, "axes" if kw.get("transform") is not None else "data"))
    return ax.text(x, y, s, **kw)

GREEN, RED, GREY, INK = "#1a7f37", "#b42318", "#8b949e", "#1f2328"
fig = plt.figure(figsize=(15.0, 10.4), dpi=125)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.92], hspace=0.52, wspace=0.20,
                      left=0.055, right=0.975, top=0.885, bottom=0.045)

fig.suptitle("A seed the client reseed cannot apply left the process half seeded", fontsize=16.5, fontweight="bold", y=0.968)
fig.text(0.5, 0.925,
         "reseed_client_rngs is what Gr00tPolicy.reset / Cosmos3Policy.reset call, so it is how a rollout makes an episode reproducible.\n"
         "Two episodes seeded identically: a reproducible stream draws the same numbers.  Seed = 2**32, one past the legacy NumPy global RNG's range.",
         ha="center", fontsize=10.4, color="#4a5568")

def stream_panel(ax, ep1, ep2, title, ok, note):
    x = np.arange(1, len(ep1) + 1)
    ax.plot(x, ep1, "-o", color=INK, ms=9, lw=1.8, label="episode 1", zorder=3)
    ax.plot(x, ep2, "--x", color=(GREEN if ok else RED), ms=11, mew=2.6, lw=1.8, label="episode 2", zorder=4)
    if not ok:
        ax.fill_between(x, ep1, ep2, color=RED, alpha=0.17, zorder=1)
    ax.set_ylim(-0.06, 1.10); ax.set_xlim(0.6, len(ep1) + 0.4)
    ax.set_xticks(x); ax.set_xlabel("draw #", fontsize=9.5)
    ax.set_ylabel("value", fontsize=9.5); ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_title(title, fontsize=11.2, fontweight="bold", color=INK, pad=8)
    for sp in ax.spines.values(): sp.set_color(GREEN if ok else RED); sp.set_linewidth(2.2)
    put(ax, 0.5, 1.015, note, transform=ax.transAxes, ha="center", fontsize=10.2,
        fontweight="bold", color=(GREEN if ok else RED))
    ax.legend(fontsize=8.6, loc="lower right", framealpha=0.94)

od = A["out_of_domain"]
stream_panel(fig.add_subplot(gs[0, 0]), od["py"][0], od["py"][1],
             "main - Python random, seed 2**32", True, "reproducible: the two episodes coincide")
stream_panel(fig.add_subplot(gs[0, 1]), od["np"][0], od["np"][1],
             "main - NumPy global RNG, same call, same seed", False, "NOT reproducible: NumPy was never reseeded")

# row 1 left: the refusal
axr = fig.add_subplot(gs[1, 0]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
for sp in axr.spines.values(): sp.set_visible(False)
axr.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, fill=True, facecolor="#f6f8fa",
                            edgecolor=GREEN, lw=2.2, transform=axr.transAxes, zorder=0))
put(axr, 0.5, 0.90, "this change - seed 2**32", ha="center", fontsize=11.2, fontweight="bold", color=INK)
put(axr, 0.5, 0.775, "refused before the first applier runs", ha="center", fontsize=10.2, fontweight="bold", color=GREEN)
msg = B["out_of_domain"]["status"].replace("ValueError: ", "")
wrapped, line = [], ""
for w in msg.split():
    if len(line) + len(w) + 1 > 52: wrapped.append(line); line = w
    else: line = (line + " " + w).strip()
wrapped.append(line)
put(axr, 0.5, 0.62, "ValueError", ha="center", fontsize=10.0, family="monospace", color=RED, fontweight="bold")
for i, ln in enumerate(wrapped):
    put(axr, 0.5, 0.53 - i * 0.082, ln, ha="center", fontsize=9.3, family="monospace", color=INK)
put(axr, 0.5, 0.10, "neither RNG moved  -  nothing half applied", ha="center", fontsize=9.6, style="italic", color="#4a5568")

idb = B["in_domain"]
stream_panel(fig.add_subplot(gs[1, 1]), idb["np"][0], idb["np"][1],
             "this change - NumPy global RNG, seed 4242 (in domain)", True,
             "unchanged: an accepted seed still reseeds every RNG")

# row 2: sweep table
axt = fig.add_subplot(gs[2, :]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
rows = list(A["sweep"].keys())
cols = [(0.085, "seed"), (0.315, "main: raises?"), (0.455, "main: Python"), (0.595, "main: NumPy"),
        (0.745, "this change"), (0.905, "this change: RNGs")]
put(axt, 0.5, 0.955, "Measured verdicts - every seed a provider's reset() may hand this helper",
    ha="center", fontsize=11.4, fontweight="bold", color=INK)
for x, lbl in cols:
    put(axt, x, 0.845, lbl, ha="center", fontsize=9.4, fontweight="bold", color="#4a5568")
axt.plot([0.02, 0.98], [0.805, 0.805], color="#d0d7de", lw=1.2)
step = 0.062
for i, s in enumerate(rows):
    y = 0.735 - i * step
    a, b = A["sweep"][s], B["sweep"][s]
    partial = (a["py"] == "reseeded") != (a["np"] == "reseeded")
    if partial:
        axt.add_patch(plt.Rectangle((0.02, y - 0.024), 0.96, 0.050, facecolor=RED, alpha=0.085,
                                    edgecolor="none", transform=axt.transData, zorder=0))
    put(axt, cols[0][0], y, s, ha="center", fontsize=9.2, family="monospace", color=INK)
    put(axt, cols[1][0], y, a["raised"] or "no", ha="center", fontsize=9.2, family="monospace",
        color=(GREY if a["raised"] is None else GREEN))
    put(axt, cols[2][0], y, a["py"], ha="center", fontsize=9.2, family="monospace",
        color=(RED if partial and a["py"] == "reseeded" else GREY))
    put(axt, cols[3][0], y, a["np"], ha="center", fontsize=9.2, family="monospace",
        color=(RED if partial else GREY))
    put(axt, cols[4][0], y, b["raised"] or "no", ha="center", fontsize=9.2, family="monospace",
        color=(GREEN if b["raised"] else GREY))
    both = "untouched" if b["py"] == "untouched" else "all reseeded"
    put(axt, cols[5][0], y, both, ha="center", fontsize=9.2, family="monospace",
        color=(GREEN if b["raised"] else GREY))
y0 = 0.735 - len(rows) * step - 0.012
axt.plot([0.02, 0.98], [y0, y0], color="#d0d7de", lw=1.2)
put(axt, 0.5, y0 - 0.055,
    f"half-seeded outcomes:  main {n_partial_main} of {len(rows)}  ->  this change {n_partial_branch} of {len(rows)}"
    "      (shaded rows: Python random reseeded, NumPy not, reset() returned)",
    ha="center", fontsize=9.9, fontweight="bold", color=INK)

for ax, y, frame in placed:
    if frame == "axes":
        assert -0.02 <= y <= 1.06, f"axes-fraction text at y={y} outside the panel"
    else:
        lo, hi = ax.get_ylim()
        assert lo <= y <= hi, f"data text at y={y} outside {ax.get_ylim()}"

out = pathlib.Path("/tmp/reseed_seed_domain.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(matplotlib.image.imread(out) * 255).astype(int)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print("OK", out, im.shape, f"{out.stat().st_size/1024:.0f} KB")
