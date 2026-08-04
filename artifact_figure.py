"""Compose the robot_mesh numeric-option figure from the two measured trees."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.loads(Path("/tmp/before.json").read_text())   # upstream/main
B = json.loads(Path("/tmp/after.json").read_text())    # this change
assert A["tree"] != B["tree"], "both probes resolved to the same tree"

ROWS = list(A["matrix"])
WIRE_COLS = ["duration", "policy_port"]
TOOL_COLS = ["timeout", "limit"]

GREEN, RED, AMBER = "#1b7f3b", "#b02020", "#8a6d00"
placed: list[tuple[plt.Axes, float]] = []


def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(14.4, 9.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.32, 1.0], hspace=0.30, wspace=0.16)

# ── top: the verdict matrix ────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.5, 1.045, "robot_mesh: four numeric options, two owners",
    ha="center", va="bottom", fontsize=16, fontweight="bold")
put(ax, 0.5, 0.985,
    "A value unusable for every one of the four options, through the real tool in both trees.\n"
    "duration and policy_port ride inside the command body validate_command inspects; timeout and limit never enter one.",
    ha="center", va="top", fontsize=9.6, color="#333333")

x_val, xs = 0.055, [0.20, 0.34, 0.545, 0.665, 0.845]
hdr = 0.845
for x, lab in zip(xs, ["duration", "policy_port", "timeout", "limit", "timeout      limit"], strict=True):
    put(ax, x, hdr, lab, ha="center", va="center", fontsize=10.5, fontweight="bold", family="monospace")
put(ax, 0.27, hdr + 0.058, "bounded by validate_command", ha="center", va="center",
    fontsize=10, style="italic", color="#333333")
put(ax, 0.605, hdr + 0.058, "owned by the tool  —  on main", ha="center", va="center",
    fontsize=10, style="italic", color=RED)
put(ax, 0.845, hdr + 0.058, "owned by the tool  —  this change", ha="center", va="center",
    fontsize=10, style="italic", color=GREEN)
ax.plot([0.455, 0.455], [0.06, 0.90], color="#999999", lw=1.1)
ax.plot([0.745, 0.745], [0.06, 0.90], color="#999999", lw=1.1)
put(ax, x_val, hdr, "value", ha="left", va="center", fontsize=10.5, fontweight="bold")

step = 0.145
for i, key in enumerate(ROWS):
    y = hdr - 0.10 - i * step
    put(ax, x_val, y, key, ha="left", va="center", fontsize=11, family="monospace")
    cells = [(xs[0], A["matrix"][key]["duration"][0]), (xs[1], A["matrix"][key]["policy_port"][0]),
             (xs[2], A["matrix"][key]["timeout"][0]), (xs[3], A["matrix"][key]["limit"][0]),
             (xs[4] - 0.055, B["matrix"][key]["timeout"][0]), (xs[4] + 0.055, B["matrix"][key]["limit"][0])]
    for x, verdict in cells:
        ok = verdict == "bounded"
        col = GREEN if ok else (AMBER if verdict == "raised" else RED)
        ax.add_patch(plt.Rectangle((x - 0.048, y - 0.052), 0.096, 0.104,
                                   facecolor=col, alpha=1.0, edgecolor="none"))
        label = {"bounded": "refused", "accepted": "ACCEPTED", "raised": "RAISED"}[verdict]
        put(ax, x, y, label, ha="center", va="center", fontsize=8.6, color="white",
            fontweight="bold", family="monospace")

nb = sum(1 for k in ROWS for c in TOOL_COLS if A["matrix"][k][c][0] != "bounded")
na = sum(1 for k in ROWS for c in TOOL_COLS if B["matrix"][k][c][0] != "bounded")
wb = sum(1 for k in ROWS for c in WIRE_COLS if A["matrix"][k][c][0] != "bounded")
cells_total = len(ROWS) * 2
assert (wb, nb, na, cells_total) == (0, 10, 0, 10), (wb, nb, na, cells_total)
put(ax, 0.5, 0.028,
    f"unbounded cells — validate_command's two options {wb}/{cells_total} · "
    f"the tool's two on main {nb}/{cells_total} · with this change {na}/{cells_total}",
    ha="center", va="center", fontsize=11, fontweight="bold")

# ── bottom left: the stop-path cap ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
put(ax2, 0.0, 1.02, "action=\"stop\": the budget that reached the transport",
    ha="left", va="bottom", fontsize=12.5, fontweight="bold")
put(ax2, 0.0, 0.955,
    "stop caps the wait at 5s so it cannot hang — min(timeout, 5.0).\n"
    "A cap is not a domain: min(nan, 5.0) is nan.",
    ha="left", va="top", fontsize=9.3, color="#333333")

put(ax2, 0.02, 0.79, "requested", ha="left", va="center", fontsize=9.8, fontweight="bold", family="monospace")
put(ax2, 0.42, 0.79, "on main", ha="center", va="center", fontsize=9.8, fontweight="bold", family="monospace")
put(ax2, 0.78, 0.79, "this change", ha="center", va="center", fontsize=9.8, fontweight="bold", family="monospace")
for i, key in enumerate(A["stop_wire"]):
    y = 0.70 - i * 0.135
    bw = A["stop_wire"][key]["on_wire"]
    aw = B["stop_wire"][key]["on_wire"]
    put(ax2, 0.02, y, key, ha="left", va="center", fontsize=10.4, family="monospace")
    ok_b = bw == ["5.0"] or (bw and bw[0] == key)
    put(ax2, 0.42, y, bw[0] if bw else "—", ha="center", va="center", fontsize=10.4,
        family="monospace", color=GREEN if ok_b else RED,
        fontweight="normal" if ok_b else "bold")
    put(ax2, 0.78, y, aw[0] if aw else "refused, nothing sent", ha="center", va="center",
        fontsize=10.4, family="monospace", color=GREEN)
assert A["stop_wire"]["nan"]["on_wire"] == ["nan"], A["stop_wire"]["nan"]
assert B["stop_wire"]["nan"]["on_wire"] == [], B["stop_wire"]["nan"]
assert A["stop_wire"]["30.0"]["on_wire"] == B["stop_wire"]["30.0"]["on_wire"] == ["5.0"]
put(ax2, 0.02, 0.035,
    "The cap itself is unchanged: 30.0 still reaches the transport as 5.0.",
    ha="left", va="center", fontsize=9.4, style="italic", color="#333333")

# ── bottom right: the inbox cap ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
labels = list(A["inbox_cap"])
before = [A["inbox_cap"][k]["returned"] for k in labels]
after = [B["inbox_cap"][k]["returned"] for k in labels]
assert before == [50, 5, 120, 120, 120], before
assert after == [50, 5, -1, -1, -1], after

y = np.arange(len(labels))[::-1]
ax3.barh(y + 0.19, [v if v > 0 else 0 for v in before], height=0.36,
         color=[GREEN if v in (50, 5) else RED for v in before], label="on main")
ax3.barh(y - 0.19, [v if v > 0 else 0 for v in after], height=0.36,
         color=[GREEN if v > 0 else "#dddddd" for v in after], label="this change")
for i, (k, bv, av) in enumerate(zip(labels, before, after, strict=True)):
    yy = y[i]
    if bv > 0:
        ax3.text(bv + 2, yy + 0.19, f"{bv} of 120", va="center", fontsize=9,
                 color=GREEN if bv in (50, 5) else RED, fontweight="bold" if bv == 120 else "normal")
    ax3.text(2 if av <= 0 else av + 2, yy - 0.19,
             "refused" if av <= 0 else f"{av} of 120", va="center", fontsize=9, color=GREEN)
ax3.set_yticks(y); ax3.set_yticklabels([f"limit={k}" for k in labels], family="monospace", fontsize=10)
ax3.set_xlim(0, 152); ax3.set_xlabel("messages pulled into the agent's context", fontsize=9.6)
ax3.set_title("action=\"inbox\": the cap over a 120-message buffer",
              fontsize=12.5, fontweight="bold", loc="left", pad=26)
ax3.text(0.0, 1.045,
         "limit is a slice index. A non-positive or nan cap selected the WHOLE buffer —\n"
         "the opposite of a cap, on the action that reads another peer's stream.",
         transform=ax3.transAxes, ha="left", va="bottom", fontsize=9.3, color="#333333")
ax3.legend(loc="lower right", fontsize=9, frameon=False)
for side in ("top", "right"):
    ax3.spines[side].set_visible(False)

out = Path("/tmp/robot_mesh_numeric_options.png")
fig.savefig(out, dpi=125, bbox_inches="tight", pad_inches=0.34, facecolor="white")
plt.close(fig)

# ── self-audit ─────────────────────────────────────────────────────────────
for a_, yv in placed:
    lo, hi = a_.get_ylim()
    pad = 0.09 * (hi - lo)
    assert lo - pad <= yv <= hi + pad, f"text at y={yv} outside {a_.get_ylim()}"
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=-1) > 20).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("figure OK", im.shape, out.stat().st_size, "bytes")
