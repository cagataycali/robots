import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.load(open("/tmp/before.json"))
B = json.load(open("/tmp/after.json"))
assert A["tree"] != B["tree"], "both probes read the same tree"

SP = list(A["rows"])
def disagreements(d):
    n = 0
    for s in SP:
        arg, env = d["rows"][s]["argument"], d["rows"][s]["environment"]
        if arg["posture"] in ("secure", "insecure") and arg["posture"] != env["posture"]:
            n += 1
    return n
DIS_A, DIS_B = disagreements(A), disagreements(B)
assert (DIS_A, DIS_B) == (6, 0), (DIS_A, DIS_B)
# every honored boolean must be byte-identical across trees (the no-regression half)
for k in ["True", "False", "None"]:
    assert A["entry"][k] == B["entry"][k], k
assert A["entry"]["'false'"]["outcome"] == "started"
assert A["entry"]["'false'"]["runtime_setting"] == "'false'"
assert A["entry"]["'false'"]["insecure_warning"] is True
assert B["entry"]["'false'"]["outcome"] == "refused"
assert B["entry"]["'false'"]["insecure_warning"] is False
assert A["numpy"]["np.True_"]["real_bool"] is False and B["numpy"]["np.True_"]["real_bool"] is True

RED, GRN, AMB, GREY = "#c0392b", "#1e8449", "#b9770e", "#7f8c8d"
placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 10.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.32, 0.78, 0.62], hspace=0.30,
                      left=0.035, right=0.975, top=0.925, bottom=0.035)

fig.suptitle("allow_insecure: one setting, two sources - the argument is now checked, not returned as given",
             fontsize=15.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.951, "resolve_allow_insecure(explicit, env_value)  -  measured on strands-labs/robots "
         f"main ({A['tree'].rsplit('/',1)[-1]}) and on this change", ha="center", fontsize=10, color="#555")

# ---------- row 1: the posture matrix ----------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.5, 1.045, "What each source resolves the same spelling to", ha="center",
    fontsize=12.5, fontweight="bold", transform=ax.transAxes)

COLS = [("as the ARGUMENT\n(documented higher precedence)", 0.255), ("as DEVICE_CONNECT_\nALLOW_INSECURE", 0.435)]
GROUPS = [("main", 0.13, A), ("this change", 0.60, B)]
TOP, LAST = 0.845, 0.075
STEP = (TOP - LAST) / (len(SP) - 1)
assert STEP > 0.045, STEP

for gname, gx, data in GROUPS:
    put(ax, gx + 0.175, 0.985, gname, ha="center", fontsize=12, fontweight="bold",
        transform=ax.transAxes, color="#222")
    for label, dx in COLS:
        put(ax, gx + dx - 0.10, 0.905, label, ha="center", va="center", fontsize=8.4,
            color="#444", transform=ax.transAxes)
    y = TOP
    for s in SP:
        put(ax, gx - 0.005, y, f"{s!r}", ha="right", va="center", fontsize=10,
            family="monospace", transform=ax.transAxes)
        arg = data["rows"][s]["argument"]["posture"]
        env = data["rows"][s]["environment"]["posture"]
        disagree = arg in ("secure", "insecure") and arg != env
        for text, dx in ((arg, COLS[0][1]), (env, COLS[1][1])):
            if text == "refused":
                fc, tc, lbl = "#eaf4ea", GRN, "refused"
            elif text == "insecure":
                fc, tc, lbl = ("#fdecea", RED, "INSECURE")
            else:
                fc, tc, lbl = "#eef3f8", "#1f4e79", "secure"
            ax.add_patch(plt.Rectangle((gx + dx - 0.093, y - 0.031), 0.186, 0.062,
                                       transform=ax.transAxes, facecolor=fc,
                                       edgecolor=RED if (disagree and dx == COLS[0][1]) else "#bbb",
                                       linewidth=2.0 if (disagree and dx == COLS[0][1]) else 0.8, zorder=1))
            put(ax, gx + dx, y, lbl, ha="center", va="center", fontsize=9.6, color=tc,
                fontweight="bold" if lbl != "secure" else "normal", transform=ax.transAxes, zorder=2)
        if disagree:
            put(ax, gx + 0.545, y, "<- disagree", ha="left", va="center", fontsize=8.6,
                color=RED, fontweight="bold", transform=ax.transAxes)
        y -= STEP
    put(ax, gx + 0.175, LAST - 0.075,
        f"sources disagree on {disagreements(data)} of {len(SP)} spellings",
        ha="center", fontsize=11, fontweight="bold",
        color=RED if disagreements(data) else GRN, transform=ax.transAxes)

# ---------- row 2: the entrypoint ledger ----------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.5, 1.10, "init_device_connect(robot, allow_insecure=...) - end to end, env var unset",
    ha="center", fontsize=12.5, fontweight="bold", transform=ax2.transAxes)
HEAD = ["argument", "main: outcome", "main: runtime setting", "main: INSECURE warning",
        "this change: outcome", "warning"]
XS = [0.055, 0.20, 0.375, 0.575, 0.755, 0.925]
for h, x in zip(HEAD, XS):
    put(ax2, x, 0.885, h, ha="center", fontsize=8.9, color="#444", fontweight="bold",
        transform=ax2.transAxes)
keys = ["'false'", "True", "False", "None"]
TOP2, LAST2 = 0.70, 0.16
STEP2 = (TOP2 - LAST2) / (len(keys) - 1)
assert STEP2 > 0.10, STEP2
y = TOP2
for k in keys:
    ea, eb = A["entry"][k], B["entry"][k]
    bad = k == "'false'"
    if bad:
        ax2.add_patch(plt.Rectangle((0.015, y - 0.085), 0.97, 0.17, transform=ax2.transAxes,
                                    facecolor="#fdf2f0", edgecolor=RED, linewidth=1.4, zorder=0))
    cells = [k, ea["outcome"], ea["runtime_setting"], "FIRED" if ea["insecure_warning"] else "-",
             eb["outcome"], "FIRED" if eb["insecure_warning"] else "-"]
    for i, (c, x) in enumerate(zip(cells, XS)):
        col = "#222"
        if i in (1, 2, 3) and bad:
            col = RED
        if i == 4:
            col = GRN if eb["outcome"] == "refused" else "#222"
        put(ax2, x, y, c, ha="center", va="center", fontsize=9.6, family="monospace",
            color=col, fontweight="bold" if col != "#222" else "normal", transform=ax2.transAxes, zorder=2)
    y -= STEP2
put(ax2, 0.5, 0.02, "The three boolean rows are byte-identical on both trees - nothing that worked changes.",
    ha="center", fontsize=9.6, color=GREY, style="italic", transform=ax2.transAxes)

# ---------- row 3: why checked rather than parsed ----------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.5, 1.02, "Why the argument is checked rather than parsed with the same vocabulary",
    ha="center", fontsize=12.5, fontweight="bold", transform=ax3.transAxes)
lines = [
    ("The environment vocabulary is ('true', '1', 'yes'). Parsing the argument with it would move which "
     "spellings invert, not remove the inversion:", GREY),
    ("   \"on\", \"enabled\", \"y\"  ->  resolve to SECURE while reading as an opt-in", RED),
    ("A numpy boolean - what a caller's own comparison produces - is now normalized: "
     f"np.array(False) returned {A['numpy']['np.array(False)']['returned']} "
     f"(real bool: {A['numpy']['np.array(False)']['real_bool']})  ->  "
     f"{B['numpy']['np.array(False)']['returned']} (real bool: {B['numpy']['np.array(False)']['real_bool']})", "#1f4e79"),
    ("No policy, simulation, rendering, recording or asset behaviour changes; every figure above is a measurement.", GREY),
]
TOP3, LAST3 = 0.78, 0.14
STEP3 = (TOP3 - LAST3) / (len(lines) - 1)
assert STEP3 > 0.14, STEP3
y = TOP3
for text, col in lines:
    put(ax3, 0.02, y, text, ha="left", va="center", fontsize=10.2, color=col,
        family="monospace" if col == RED else None, transform=ax3.transAxes)
    y -= STEP3

for a, yy, axes_coords in placed:
    if axes_coords:
        assert -0.12 <= yy <= 1.12, (yy, "axes-fraction text out of frame")
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

out = pathlib.Path("/tmp/allow_insecure_domain.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nonwhite == 0, (name, nonwhite)
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  disagreements {DIS_A} -> {DIS_B}  border clean")
