import json, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.load(open("/tmp/facts_ctrl-main.json"))   # upstream/main
B = json.load(open("/tmp/facts_robots-mine.json")) # this branch
assert A["tree"] != B["tree"], "before/after came from the same tree"

VERS = ["1.0.0", "1.4.1", "1.5.0"]
# --- self-audit: every claim the figure makes, re-derived from the dumps ---
for v in ("1.0.0", "1.4.1"):
    assert A["rows"][v]["cli_buckets_help_rc"] == 2
    assert A["rows"][v]["gate_refused"] is False
    assert "No such command 'buckets'" in A["rows"][v]["message"]
    assert B["rows"][v]["gate_refused"] is True
    assert "huggingface_hub>=1.5" in B["rows"][v]["message"]
    assert "No such command" not in B["rows"][v]["message"]
assert A["rows"]["1.5.0"]["cli_buckets_help_rc"] == 0
assert A["rows"]["1.5.0"]["status"] == B["rows"]["1.5.0"]["status"] == "success"
assert A["rows"]["1.5.0"]["message"] == B["rows"]["1.5.0"]["message"], "the capable path must be untouched"

RED, GREEN, GREY = "#b3261e", "#1b6b3a", "#5f6368"
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.2, 9.1))
gs = fig.add_gridspec(3, 1, height_ratios=[0.30, 1.34, 0.40], hspace=0.20)

# ---------- header ----------
ax0 = fig.add_subplot(gs[0]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
put(ax0, 0.5, 0.86, "The huggingface_hub floor bucket sync declares vs the release that ships the `hf` bucket CLI",
    ha="center", va="top", fontsize=16, fontweight="bold")
put(ax0, 0.5, 0.42,
    "`sync_dataset_to_bucket` runs `hf buckets create` and `hf sync`. Both first ship in huggingface_hub 1.5.0.\n"
    "Rows below are measured: the `hf` CLI is the genuine binary from a shadow venv of each release; only the version string the gate reads is emulated.",
    ha="center", va="top", fontsize=10.6, color=GREY)

# ---------- the matrix ----------
ax = fig.add_subplot(gs[1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
COLS = [0.008, 0.115, 0.285, 0.640]
HDRS = ["huggingface_hub", "real `hf buckets\n--help` exit code", "on main  (floor: >=1.0)", "on this change  (floor: >=1.5)"]
for x, h in zip(COLS, HDRS):
    put(ax, x, 0.975, h, ha="left", va="top", fontsize=11.2, fontweight="bold")
ax.plot([0.004, 0.996], [0.905, 0.905], color="#202124", lw=1.5, transform=ax.transAxes)

def wrap(s, w=52):
    return "\n".join(textwrap.fill(p, w) for p in s.strip().splitlines() if p.strip())

y = 0.855
for v in VERS:
    a, b = A["rows"][v], B["rows"][v]
    ships = a["cli_buckets_help_rc"] == 0
    band = "#e8f5e9" if ships else "#fdecea"
    ax.add_patch(plt.Rectangle((0.004, y - 0.245), 0.992, 0.250, transform=ax.transAxes,
                               facecolor=band, edgecolor="none", zorder=0))
    put(ax, COLS[0], y, v, ha="left", va="top", fontsize=13, fontweight="bold", family="monospace")
    put(ax, COLS[0], y - 0.075,
        "ships the\nsubcommands" if ships else "no `buckets`\nsubcommand",
        ha="left", va="top", fontsize=9.2, color=GREEN if ships else RED)
    put(ax, COLS[1], y, f"rc = {a['cli_buckets_help_rc']}", ha="left", va="top",
        fontsize=12, family="monospace", color=GREEN if ships else RED, fontweight="bold")
    put(ax, COLS[1], y - 0.070, "(subcommand\n exists)" if ships else '("No such\n command")',
        ha="left", va="top", fontsize=8.8, color=GREY)
    for x, side, other in ((COLS[2], a, "main"), (COLS[3], b, "branch")):
        actionable = ("huggingface_hub>=1.5" in side["message"]) or side["status"] == "success"
        colour = GREEN if actionable else RED
        put(ax, x, y, f"gate: {'REFUSED' if side['gate_refused'] else 'accepted'}   ->   status={side['status']}",
            ha="left", va="top", fontsize=10, family="monospace", color=colour, fontweight="bold")
        put(ax, x, y - 0.058, wrap(side["message"], 50), ha="left", va="top",
            fontsize=8.9, family="monospace", color="#202124")
    y -= 0.262

# ---------- footer ----------
ax2 = fig.add_subplot(gs[2]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.008, 0.90,
    "On main a caller on 1.0-1.4.x satisfies the documented `>=1.0` minimum, passes the gate, and receives the raw CLI usage noise\n"
    "the gate exists to replace - and the remedy it was given (`pip install -U 'huggingface_hub>=1.0'`) resolves that same CLI.",
    ha="left", va="top", fontsize=10.4, color=RED)
put(ax2, 0.008, 0.40,
    "The capable release is untouched: on 1.5.0 both trees synced the dataset for real, to a byte-identical bucket URI\n"
    f"({A['rows']['1.5.0']['message']}).  A fresh install resolves 1.26.0 either way; the change is to what a constrained environment may resolve.",
    ha="left", va="top", fontsize=10.4, color=GREEN)

for ax_, yv in placed:
    lo, hi = ax_.get_ylim()
    assert lo - 0.05 <= yv <= hi + 0.08, (yv, lo, hi)

out = "/tmp/bucket_cli_floor.png"
fig.savefig(out, dpi=125, bbox_inches="tight", pad_inches=0.32, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", Image.open(out).size, "| rows audited:", VERS)
