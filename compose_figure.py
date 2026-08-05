"""Compose the measured before/after figure for the VERA port domain."""
from __future__ import annotations
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

B = json.load(open("/tmp/before.json"))
A = json.load(open("/tmp/after.json"))
assert B["tree"] != A["tree"], "before/after came from the same tree"

# --- measured facts this figure claims -----------------------------------
b = {r["value"]: r for r in B["rows"]}
a = {r["value"]: r for r in A["rows"]}
assert b["2.7"]["verdict"] == "accepted" and b["2.7"]["client_port"] == "2" and b["2.7"]["argv"] == "2.7"
assert b["2.7"]["server_uri"].endswith(":2.7") and b["2.7"]["distinct"] == 2
assert b["True"]["client_port"] == "1" and b["True"]["argv"] == "True"
assert b["True"]["server_uri"] == "ws://127.0.0.1:True"
for v in ("0", "-1", "70000", "'8820'"):
    assert b[v]["verdict"] == "accepted", v
assert b["nan"]["verdict"] == "refused" and "convert float NaN" in b["nan"]["msg"]
assert b["8820"]["verdict"] == a["8820"]["verdict"] == "accepted"
assert b["8820"]["client_port"] == a["8820"]["client_port"] == "8820"
assert b["8820"]["argv"] == a["8820"]["argv"] == "8820"
for v in ("2.7", "True", "0", "-1", "70000", "nan", "'8820'"):
    assert a[v]["verdict"] == "refused", v
    assert "expected 1-65535" in a[v]["msg"], v
assert B["vis"]["0"] == A["vis"]["0"] == "viewer disabled"
assert B["vis"]["8821"] == A["vis"]["8821"] == "8821"
n_div = sum(1 for r in B["rows"] if r["verdict"] == "accepted" and r["distinct"] > 1)
n_bad = sum(1 for r in B["rows"] if r["verdict"] == "accepted" and r["value"] != "8820")
assert (n_div, n_bad) == (2, 6), (n_div, n_bad)

ORDER = ["8820", "2.7", "True", "0", "-1", "70000", "nan", "'8820'"]
NOTE = {
    "8820":   "usable port - the control",
    "2.7":    "THREE destinations, TWO ports",
    "True":   "THREE destinations, TWO ports",
    "0":      "no port to dial or bind",
    "-1":     "no port to dial or bind",
    "70000":  "outside the 16-bit port space",
    "nan":    "bare coercion error, names nothing",
    "'8820'": "not an int; works by coincidence",
}
GREEN, RED, AMBER, GREY = "#1b7f3b", "#b3161d", "#a8620a", "#3c3c3c"

placed: list[tuple] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(17.4, 8.9))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 2, height_ratios=[8.0, 1.35], hspace=0.20, wspace=0.055,
                      left=0.012, right=0.988, top=0.885, bottom=0.045)

COLS = [0.008, 0.135, 0.315, 0.545, 0.735]
HDRS = ["server_port=", "client dials", "config.server_uri", "server argv", "verdict"]
TOP, STEP = 0.885, 0.098

for panel, (data, title, sub) in enumerate([
    (b, "main @ 62e375da", "one value, three consumers, three coercions"),
    (a, "this change", "one domain, checked once on the config"),
]):
    ax = fig.add_subplot(gs[0, panel]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="#bbbbbb", lw=1.1,
                               transform=ax.transAxes))
    put(ax, 0.008, 0.975, title, fontsize=15, fontweight="bold", va="top")
    put(ax, 0.008, 0.930, sub, fontsize=10.5, style="italic", color="#555555", va="top")
    for x, h in zip(COLS, HDRS):
        put(ax, x, TOP, h, fontsize=9.6, fontweight="bold", color="#333333", va="top")
    ax.plot([0.005, 0.995], [TOP - 0.022] * 2, color="#999999", lw=0.9,
            transform=ax.transAxes, clip_on=False)

    for i, key in enumerate(ORDER):
        y = TOP - 0.045 - i * STEP
        r = data[key]
        if r["verdict"] == "refused":
            colour = GREEN if "expected 1-65535" in r.get("msg", "") else RED
            band = "#eaf5ee" if colour == GREEN else "#fdeceb"
        elif r["distinct"] > 1:
            colour, band = RED, "#fdeceb"
        elif key == "8820":
            colour, band = GREY, "#f4f4f4"
        else:
            colour, band = AMBER, "#fdf3e3"
        ax.add_patch(plt.Rectangle((0.004, y - 0.052), 0.992, 0.086, fc=band,
                                   ec="none", transform=ax.transAxes, zorder=0))
        put(ax, COLS[0], y, key, fontsize=11.5, family="monospace",
            fontweight="bold", color=colour, va="center")
        if r["verdict"] == "accepted":
            put(ax, COLS[1], y, f":{r['client_port']}", fontsize=11.5,
                family="monospace", color=colour, va="center")
            put(ax, COLS[2], y, r["server_uri"], fontsize=10.2,
                family="monospace", color=colour, va="center")
            put(ax, COLS[3], y, f"--port {r['argv']}", fontsize=10.2,
                family="monospace", color=colour, va="center")
            tag = "accepted" if r["distinct"] == 1 else f"accepted ({r['distinct']} ports)"
            put(ax, COLS[4], y, tag, fontsize=10.2, fontweight="bold",
                color=colour, va="center")
        else:
            put(ax, COLS[1], y, "-- nothing built --", fontsize=10.2,
                family="monospace", color=colour, va="center")
            short = r["msg"].replace("VeraConfig: ", "")
            put(ax, COLS[2], y, short, fontsize=9.6, family="monospace",
                color=colour, va="center")
            put(ax, COLS[4], y, "refused" if colour == GREEN else f"raised {r['exc']}",
                fontsize=10.2, fontweight="bold", color=colour, va="center")
        put(ax, COLS[0], y - 0.036, NOTE[key], fontsize=8.6, style="italic",
            color="#6a6a6a", va="center")

# --- no-regression band --------------------------------------------------
for panel, (data, src) in enumerate([(B, "main @ 62e375da"), (A, "this change")]):
    ax = fig.add_subplot(gs[1, panel]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#f0f6fb", ec="#9fc3dd", lw=1.0,
                               transform=ax.transAxes))
    put(ax, 0.012, 0.83, f"vis_port: the documented zero is a mode selector, not a port   ({src})",
        fontsize=10.4, fontweight="bold", color="#164a70", va="center")
    for j, k in enumerate(["8821", "0", "-1", "True"]):
        y = 0.52 - j * 0.155
        val = data["vis"][k]
        c = GREY if not val.startswith("refused") else GREEN
        put(ax, 0.030, y, f"vis_port={k:<6} ->  {val}", fontsize=10.0,
            family="monospace", color=c, va="center")

fig.suptitle(
    "VeraConfig.server_port: measured at every consumer, before and after   "
    "(offline - no server, no socket, no vera package)",
    fontsize=13.6, fontweight="bold", y=0.968)

out = pathlib.Path("/tmp/vera_port_domain.png")
fig.savefig(out, dpi=115, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

# --- self-audit ---------------------------------------------------------
bad = [(str(ax), y) for ax, y in placed if not (-0.02 <= y <= 1.06)]
assert not bad, f"text outside axes: {bad}"
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  size={im.shape[1]}x{im.shape[0]}  divergent_rows={n_div}  unusable_accepted={n_bad}")
