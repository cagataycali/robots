import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

B = json.load(open("/tmp/before.json"))
A = json.load(open("/tmp/after.json"))
assert B["tree"] != A["tree"], "both probes resolved to the same tree"

# --- facts the figure claims, re-derived from the two runs ---
bp = {c["requested"]: c for c in B["positions"]}
ap = {c["requested"]: c for c in A["positions"]}
assert bp[70000]["wire"] == 4464 and bp[70000]["status"] == "success"
assert bp[-1]["wire"] == 65535 and bp[-1]["status"] == "success"
assert bp[65536]["wire"] == 0 and bp[5000]["wire"] == 5000
for good in (0, 2048, 4095):
    assert bp[good]["wire"] == good == ap[good]["wire"], f"honored case changed: {good}"
    assert bp[good]["status"] == ap[good]["status"] == "success"
for bad in (5000, 70000, -1, 65536):
    assert ap[bad]["status"] == "error" and ap[bad]["wire"] is None and ap[bad]["bytes"] == ""
BAD_MAIN = sum(1 for c in B["positions"] if c["status"] == "success" and not 0 <= c["requested"] <= 4095)
assert BAD_MAIN == 4

VALID_HI = 4095
FIELD_HI = 65535
placed = []

def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.2, 11.6), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.32, 1.0], hspace=0.30, wspace=0.11,
                      left=0.122, right=0.968, top=0.905, bottom=0.038)

fig.text(0.5, 0.968, "serial_tool: the servo position the caller asks for vs the one the wire carries",
         ha="center", fontsize=17, fontweight="bold")
fig.text(0.5, 0.940,
         "Measured over a real pty (a serial device) on both trees. Goal_Position is written as two masked bytes: "
         "value & 0xFF, (value >> 8) & 0xFF.",
         ha="center", fontsize=10.5, color="#333333")

REQ = [0, 2048, 4095, 5000, 70000, -1, 65536]
ypos = np.arange(len(REQ))[::-1]

for col, (label, data, tone) in enumerate([
    ("main: every request accepted", bp, "#c62828"),
    ("this change: only what the field holds", ap, "#2e7d32"),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.axvspan(0, VALID_HI, color="#2e7d32", alpha=0.10, zorder=0)
    ax.axvline(VALID_HI, color="#2e7d32", ls="--", lw=1.4, zorder=1)
    put(ax, VALID_HI + 900, len(REQ) - 0.32, "Goal_Position ceiling 4095", fontsize=9,
        color="#2e7d32", fontweight="bold")

    for y, req in zip(ypos, REQ):
        row = data[req]
        wire = row["wire"]
        ax.plot([0, FIELD_HI], [y, y], color="#e8e8e8", lw=0.9, zorder=1)
        if wire is None:
            ax.plot([300], [y], marker="X", ms=13, color="#2e7d32", zorder=4)
            put(ax, 2400, y, "REFUSED -- port never opened", fontsize=9.6,
                color="#2e7d32", va="center", fontweight="bold")
        else:
            inside = 0 <= req <= VALID_HI
            c = "#2e7d32" if inside else tone
            ax.plot([wire], [y], marker="o", ms=11, color=c, zorder=4)
            note = f"wire={wire}  ({wire / 4095 * 360:.0f} deg)"
            if not inside:
                note += "  <-- not asked for"
            put(ax, min(wire + 1600, 29000), y, note, fontsize=9.4, color=c, va="center",
                fontweight="bold" if not inside else "normal")

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"position={r}" for r in REQ], fontsize=10.0, fontfamily="monospace")
    ax.set_xlim(-2600, FIELD_HI + 2600)
    ax.set_ylim(-0.7, len(REQ) - 0.3)
    ax.set_xlabel("position written into the two-byte Goal_Position field (ticks)", fontsize=10)
    ax.set_title(label, fontsize=12.5, fontweight="bold", color=tone, pad=9)
    ax.grid(axis="x", alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# --- the option table ---
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.5, 0.985, "every numeric option reaching the bus, measured on the same pty",
    ha="center", fontsize=12.5, fontweight="bold")

cols = [0.012, 0.150, 0.238, 0.565, 0.655]
heads = ["call", "main", "what main did", "this change", "what this change reports"]
put(ax, cols[0], 0.905, heads[0], fontsize=10, fontweight="bold")
put(ax, cols[1], 0.905, heads[1], fontsize=10, fontweight="bold")
put(ax, cols[2], 0.905, heads[2], fontsize=10, fontweight="bold")
put(ax, cols[3], 0.905, heads[3], fontsize=10, fontweight="bold")
put(ax, cols[4], 0.905, heads[4], fontsize=10, fontweight="bold")
ax.plot([0.008, 0.992], [0.878, 0.878], color="#333333", lw=1.1)

WHY = {
    "motor_id=255": "frame ID byte becomes 0xFF, a third header copy",
    "motor_id=True": "silently addressed motor 1, printed 'Motor True'",
    "velocity=70000": "wire carried 4464, reported 70000",
    "read_bytes=4": "read 4 bytes in 0.15s (the working case)",
    "read_bytes=0": "returned 0 bytes at once, like a timeout",
    "read_bytes=2.7": "leaked a bare float-index TypeError",
    "timeout=nan": "waited 0.00s of the budget, still 'success'",
    "timeout=inf": "leaked 'timestamp out of range for time_t'",
    "baudrate=2.7": "opened the port at 2 baud",
}
step = 0.0885
y = 0.828
for bo, ao in zip(B["options"], A["options"]):
    ok_main = bo["status"] == "success"
    benign = bo["label"] == "read_bytes=4"
    mc = "#2e7d32" if benign else ("#c62828" if ok_main else "#ef6c00")
    ac = "#2e7d32" if ao["status"] == "success" else "#1565c0"
    if not benign:
        ax.add_patch(plt.Rectangle((0.008, y - 0.030), 0.984, 0.070, color="#c62828",
                                   alpha=0.045, zorder=0))
    put(ax, cols[0], y, bo["label"], fontsize=10.2, fontfamily="monospace", va="center")
    put(ax, cols[1], y, bo["status"], fontsize=10.2, color=mc, va="center", fontweight="bold")
    put(ax, cols[2], y, WHY[bo["label"]], fontsize=9.9, color=mc, va="center")
    put(ax, cols[3], y, ao["status"], fontsize=10.2, color=ac, va="center", fontweight="bold")
    put(ax, cols[4], y, ao["text"][:62], fontsize=8.9, color=ac, va="center", fontfamily="monospace")
    y -= step

put(ax, 0.008, 0.058,
    "Only the options an action reads are checked: action=\"read\" ignores a bad motor_id, "
    "action=\"list_ports\" reads none of them,\nand an unset motor_id / position is still reported by "
    "the action's own \"required\" message. timeout=0 stays valid as pyserial's non-blocking poll.",
    fontsize=9.6, color="#333333", va="top")

outside = []
for _ax, _y in placed:
    lo, hi = _ax.get_ylim()
    pad = 0.03 * (hi - lo)
    if not lo - pad <= _y <= hi + pad:
        outside.append((_y, (lo, hi)))
assert not outside, f"text outside its axes: {outside}"
out = Path("/tmp/serial_register_domain.png")
fig.savefig(out, dpi=125, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
border = np.concatenate([im[:8].ravel(), im[-8:].ravel(), im[:, :8].ravel(), im[:, -8:].ravel()])
assert (border != 255).sum() == 0, f"content touches the border: {(border != 255).sum()} px"
print("OK", out, Image.open(out).size, "| main accepted", BAD_MAIN, "unusable positions; this change 0")
