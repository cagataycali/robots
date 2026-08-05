import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

A = json.load(open("/tmp/main.json")); B = json.load(open("/tmp/branch.json"))
assert A["tree"] != B["tree"], "both probes ran on the same tree"
assert A["api_valid_down"] == A["api_out_of_range"], "indistinguishability claim failed"

VALID = ["8000  (default)", "9001"]
UNUSABLE = [k for k in A["rows"] if k not in VALID]
assert len(UNUSABLE) == 6, len(UNUSABLE)
for k in UNUSABLE:
    assert A["rows"][k]["construct"] == "accepted", k
    assert B["rows"][k]["construct"] == "REFUSED", k
    assert A["rows"][k]["variant"] == "Zenoh (Wireless)", k
for k in VALID:
    assert B["rows"][k]["construct"] == "accepted", k
    assert A["rows"][k]["rest"] == B["rows"][k]["rest"], k   # byte-identical: no regression

GREEN, RED, DARK, GREY = "#1b7f3b", "#b3261e", "#202124", "#5f6368"
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.4, 9.5), dpi=125)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 1, height_ratios=[3.05, 1.0], hspace=0.13,
                      left=0.012, right=0.988, top=0.895, bottom=0.035)

fig.text(0.5, 0.965, "ReachyMiniDriver(api_port=...)  -  what the port reaches", ha="center",
         fontsize=19, fontweight="bold", color=DARK)
fig.text(0.5, 0.925,
         "The port is interpolated verbatim into the daemon REST URL and the Lite WebSocket target. Nothing downstream refuses it.",
         ha="center", fontsize=11.5, color=GREY)

# ---- main table -----------------------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
COLS = [(0.012, "api_port"), (0.135, "REST target built on main"),
        (0.470, "ws:// target built on main"), (0.700, "variant auto-detected"), (0.880, "this change")]
top, step = 0.925, 0.0985
for x, h in COLS:
    put(ax, x, top + 0.045, h, fontsize=11.3, fontweight="bold", color=DARK)
ax.plot([0.008, 0.992], [top + 0.022, top + 0.022], color=DARK, lw=1.3)

order = VALID + UNUSABLE
for i, key in enumerate(order):
    y = top - i * step
    ra, rb = A["rows"][key], B["rows"][key]
    good = key in VALID
    if not good:
        ax.add_patch(Rectangle((0.008, y - 0.030), 0.984, 0.075, transform=ax.transAxes,
                               facecolor=RED, alpha=0.055, zorder=0, lw=0))
    put(ax, COLS[0][0], y, key, fontsize=11.6, fontweight="bold", family="monospace",
        color=DARK if good else RED)
    put(ax, COLS[1][0], y, ra["rest"].replace("/api/daemon/status", "/api/daemon/status"),
        fontsize=9.9, family="monospace", color=GREY if good else RED)
    put(ax, COLS[2][0], y, ra["ws"], fontsize=9.9, family="monospace", color=GREY if good else RED)
    put(ax, COLS[3][0], y, ra["variant"], fontsize=10.2, color=GREY if good else RED)
    if good:
        put(ax, COLS[4][0], y, "accepted", fontsize=10.6, fontweight="bold", color=GREEN)
        put(ax, COLS[4][0], y - 0.030, "unchanged, byte for byte", fontsize=8.7, color=GREY)
    else:
        put(ax, COLS[4][0], y, "REFUSED", fontsize=10.6, fontweight="bold", color=GREEN)
        put(ax, COLS[4][0], y - 0.030, "at construction", fontsize=8.7, color=GREY)
    if not good:
        put(ax, COLS[3][0], y - 0.030, "connect() logged success", fontsize=8.5, style="italic", color=RED)

# ---- why the variant flips + indistinguishability -------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.plot([0.008, 0.992], [0.97, 0.97], color=DARK, lw=1.0)
put(ax2, 0.012, 0.86, "Why an unusable port silently became a Wireless robot", fontsize=12.2,
    fontweight="bold", color=DARK)
put(ax2, 0.012, 0.70,
    "reachy_transport.api() reports every failure as a result, not an exception:", fontsize=10.4, color=GREY)
put(ax2, 0.030, 0.555, f"api(host, 8000, ...)   daemon down      ->  {A['api_valid_down']}",
    fontsize=9.9, family="monospace", color=DARK)
put(ax2, 0.030, 0.425, f"api(host, 99999, ...)  port out of range ->  {A['api_out_of_range']}",
    fontsize=9.9, family="monospace", color=RED)
put(ax2, 0.030, 0.295, "byte-identical: the caller is sent to debug the daemon", fontsize=9.7,
    style="italic", fontweight="bold", color=RED)
put(ax2, 0.030, 0.145,
    'connect():  is_lite = not status.get("wireless_version", True)   ->  False   ->  Zenoh link chosen',
    fontsize=9.9, family="monospace", color=DARK)

put(ax2, 0.560, 0.86, "Scope", fontsize=12.2, fontweight="bold", color=DARK)
for j, line in enumerate([
    "The port now goes through the shared utils.tcp_port_error domain,",
    "before any base-class state is allocated - the same domain the mesh",
    "bridges, the agent tools and the five dialing policy providers use.",
    "",
    "The fail-safe that treats a genuinely unreachable daemon as Wireless",
    "is unchanged and pinned: a daemon that is down is not a caller mistake.",
    "",
    "No policy, simulation, rendering, recording or asset behaviour changes.",
]):
    put(ax2, 0.560, 0.70 - j * 0.093, line, fontsize=10.1,
        color=DARK if line.startswith("The fail-safe") or line.startswith("No policy") else GREY)

for a, y in placed:
    lo, hi = a.get_ylim()
    assert lo - 0.06 <= y <= hi + 0.06, f"text at y={y} outside {a.get_ylim()}"

out = pathlib.Path("/tmp/api_port_domain.png")
fig.savefig(out, dpi=125, facecolor="white", bbox_inches="tight", pad_inches=0.3)
plt.close(fig)

im = np.array(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", out, im.shape)
