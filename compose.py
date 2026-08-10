import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

A = json.load(open("/tmp/art_main.json"))   # upstream/main
B = json.load(open("/tmp/art_branch.json")) # this change
assert A["tree"] != B["tree"], "both dumps came from the same tree"
CAP = A["cap"]; assert B["cap"] == CAP
assert len(A["tlost"]) > 0 and len(B["tlost"]) == 0
assert A["tool_save"] == "success" and A["tool_load"] == "error"
assert B["tool_save"] == "error"
WIN_LO, WIN_HI = min(A["tlost"]), max(A["tlost"])

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes)
    placed.append(y); ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 9.6), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.06],
                      hspace=0.42, wspace=0.16, left=0.055, right=0.975, top=0.885, bottom=0.055)
fig.suptitle("harness_memory: a summary inside the documented 64 KiB budget saved, then could never be read back",
             fontsize=15.5, fontweight="bold", y=0.965)
fig.text(0.5, 0.925, "save_trace measured the caller's payload; load_trace measured that payload plus the provenance block it writes beside it. "
         "Every number below is measured on both trees by one script.", ha="center", fontsize=10.2, style="italic", color="#333333")

# ---- row 1: the sweep -----------------------------------------------------
ax = fig.add_subplot(gs[0, :])
for dump, colour, lo in ((A, "#c0392b", 0), (B, "#1e8449", 1)):
    label = "upstream/main" if lo == 0 else "this change"
    xs_s = [r["size"] for r in dump["tsweep"] if r["saved"]]
    xs_l = [r["size"] for r in dump["tsweep"] if r["loadable"]]
    base = 0.55 if lo == 0 else 0.05
    ax.scatter(xs_s, [base + 0.16] * len(xs_s), s=13, marker="s", color=colour, label=f"{label}: save accepted")
    ax.scatter(xs_l, [base] * len(xs_l), s=13, marker="o", facecolors="none", edgecolors=colour,
               label=f"{label}: load accepted")
    ax.text(WIN_LO - 250, base + 0.08, label, fontsize=10.5, fontweight="bold", color=colour, va="center", ha="right")
ax.axvspan(WIN_LO, WIN_HI, color="#c0392b", alpha=0.11)
ax.axvline(CAP, color="#555555", ls="--", lw=1.1)
ax.annotate(f"the {WIN_HI - WIN_LO + 1}-byte window where\nmain accepts a save and refuses the load",
            xy=((WIN_LO + WIN_HI) / 2, 0.90), xytext=((WIN_LO + WIN_HI) / 2 - 90, 1.02),
            fontsize=9.6, color="#c0392b", ha="center",
            arrowprops={"arrowstyle": "->", "color": "#c0392b", "lw": 1.1})
ax.text(CAP + 4, 0.42, f"_MAX_SUMMARY_BYTES = {CAP}", fontsize=9.2, color="#555555", rotation=90, va="center")
ax.set_ylim(-0.12, 1.16); ax.set_yticks([])
ax.set_xlabel("size of the summary the caller hands in (bytes, json.dumps sort_keys=True)", fontsize=10.3)
ax.set_title("Swept through the agent tool: on main the two bands separate, so a saved summary is not a loadable one",
             fontsize=11.4, pad=8)
ax.legend(loc="lower left", fontsize=8.6, ncol=2, framealpha=0.95)
ax.grid(axis="x", alpha=0.25)

# ---- row 2 left: the remedy ----------------------------------------------
axr = fig.add_subplot(gs[1, 0]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 1.055, "Following the remedy the load failure names, exactly", fontsize=11.6, fontweight="bold")
put(axr, 0.0, 0.965, '"delete it with delete_trace and re-save"', fontsize=9.4, style="italic", color="#555555")
rows = [("upstream/main", A["remedy"], "#c0392b"), ("this change", B["remedy"], "#1e8449")]
y = 0.87
for label, lines, colour in rows:
    put(axr, 0.0, y, label, fontsize=10.4, fontweight="bold", color=colour)
    y -= 0.075
    for line in lines:
        put(axr, 0.035, y, ("- " + line)[:96], fontsize=9.3, family="monospace", color="#222222")
        y -= 0.068
    y -= 0.045
put(axr, 0.0, y, "The same summary reproduces the same unreadable file, so the\ninstruction cannot be carried out. After the change the refusal\narrives at the save, where the caller can act on it.",
    fontsize=9.3, color="#333333")

# ---- row 2 right: the invariant + envelope -------------------------------
axi = fig.add_subplot(gs[1, 1]); axi.axis("off"); axi.set_xlim(0, 1); axi.set_ylim(0, 1)
put(axi, 0.0, 1.055, "Bytes measured at each end, by payload shape", fontsize=11.6, fontweight="bold")
put(axi, 0.0, 0.965, "The save side must measure what the load side will recompute.", fontsize=9.4, style="italic", color="#555555")
hdr_y = 0.885
for x, t in ((0.0, "payload"), (0.30, "main: save / load"), (0.66, "this change: save / load")):
    put(axi, x, hdr_y, t, fontsize=9.5, fontweight="bold")
labels = list(A["invariant"])
n_rows = len(labels) + 4
step = (hdr_y - 0.16) / n_rows
assert step > 0.030, step
y = hdr_y - step
for label in labels:
    a, b = A["invariant"][label], B["invariant"][label]
    def fmt(d):
        s, l = d["on_save"], d["on_load"]
        return f"{'-' if s is None else s} / {'-' if l is None else l}", (s is not None and s == l)
    ta, oka = fmt(a); tb, okb = fmt(b)
    put(axi, 0.0, y, label, fontsize=9.3, family="monospace")
    put(axi, 0.30, y, ta + ("  ok" if oka else "  not measured"), fontsize=9.3, family="monospace",
        color="#1e8449" if oka else "#c0392b")
    put(axi, 0.66, y, tb + ("  ok" if okb else "  mismatch"), fontsize=9.3, family="monospace",
        color="#1e8449" if okb else "#c0392b")
    y -= step
y -= step * 0.4
put(axi, 0.0, y, "agent tool, summary at the budget", fontsize=9.3, family="monospace"); y -= step
put(axi, 0.035, y, f"save_trace -> {A['tool_save']}", fontsize=9.3, family="monospace", color="#c0392b")
put(axi, 0.5, y, f"save_trace -> {B['tool_save']}", fontsize=9.3, family="monospace", color="#1e8449"); y -= step
put(axi, 0.035, y, f"load_trace -> {A['tool_load']}", fontsize=9.3, family="monospace", color="#c0392b")
put(axi, 0.5, y, f"load_trace -> {B['tool_load']}", fontsize=9.3, family="monospace", color="#1e8449"); y -= step * 1.15
put(axi, 0.0, y, f"Saved-but-unreadable sizes over the swept range:  main {len(A['tlost'])}  ->  this change {len(B['tlost'])}",
    fontsize=9.6, fontweight="bold", color="#1e8449")
assert y > 0.02, y
assert all(-0.04 <= v <= 1.10 for v in placed), [v for v in placed if not -0.04 <= v <= 1.10]

out = pathlib.Path("/tmp/harness_memory_summary_budget.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.3, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out} {im.shape[1]}x{im.shape[0]}  window={WIN_LO}-{WIN_HI} ({WIN_HI-WIN_LO+1}B)  lost {len(A['tlost'])} -> {len(B['tlost'])}")
