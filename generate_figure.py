import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

A = json.loads(open("/tmp/facts_main.json").read())      # upstream/main
B = json.loads(open("/tmp/facts_branch.json").read())     # this change
assert A["tree"] != B["tree"], "before/after must come from different trees"

CASES = ["0.02 (default)", "0", "-1", "nan", "True", "inf"]
COL = {"0.02 (default)": "#1b7f3b", "0": "#c0392b", "-1": "#d35400",
       "nan": "#8e44ad", "True": "#2471a3", "inf": "#7f1d1d"}
REQUESTED = 0.02
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 10.4), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.24, 1.0], hspace=0.30, wspace=0.10,
                      left=0.062, right=0.976, top=0.905, bottom=0.055)

# ---- row 1: what each period does to the command thread -----------------
ax = fig.add_subplot(gs[0, :])
for label in CASES:
    st = np.array(A["traces"][label]["stamps"], dtype=float)
    n_total = A["traces"][label]["n"]
    if len(st) == 0: continue
    ax.step(st, np.arange(1, len(st) + 1), where="post", lw=2.0, color=COL[label],
            label=f"poll_period = {label}   ->  {n_total:,} polls")
    if label == "inf":
        ax.plot(st[0], 1, marker="X", ms=13, color=COL[label], zorder=6, mew=2, mec="white")
ref_t = np.arange(0, 0.2501, REQUESTED)
ax.plot(ref_t, np.arange(1, len(ref_t) + 1), ls=":", lw=1.7, color="#555",
        label=f"the requested cadence ({REQUESTED} s = 50 Hz)")
ax.set_yscale("symlog", linthresh=10)
ax.set_xlim(-0.004, 0.258); ax.set_ylim(0, 6e5)
ax.set_xlabel("seconds since the command loop started", fontsize=10.5)
ax.set_ylabel("cumulative inbound command polls", fontsize=10.5)
ax.set_title("The loop period is the whole cadence of the command thread\n"
             "real HardwareRtpsBridge._poll_loop, 0.25 s window, identical on both trees",
             fontsize=12.6, fontweight="bold", pad=11)
ax.grid(alpha=0.26, ls=":"); ax.legend(loc="center right", fontsize=9.4, framealpha=0.95)
put(ax, 0.128, 1.1e5, "0, -1 and nan return from Event.wait immediately:\n"
                      "the thread spins ~20,000x the requested rate, unbounded",
    fontsize=10.2, color="#c0392b", fontweight="bold", ha="center",
    bbox=dict(fc="#fdecea", ec="#c0392b", lw=1.1, boxstyle="round,pad=0.42"))
put(ax, 0.020, 2.4, "inf raises OverflowError out of wait:\nthe loop thread dies after one poll",
    fontsize=9.6, color="#7f1d1d", fontweight="bold", ha="left",
    bbox=dict(fc="#fdf2f2", ec="#7f1d1d", lw=1.0, boxstyle="round,pad=0.34"))

# ---- row 2: constructor verdicts, before | after ------------------------
def table(gspec, facts, title, tint):
    a = fig.add_subplot(gspec); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.add_patch(Rectangle((0, 0), 1, 1, fc=tint, ec="#999", lw=1.2, transform=a.transAxes, zorder=0))
    put(a, 0.5, 0.955, title, fontsize=12.0, fontweight="bold", ha="center")
    put(a, 0.055, 0.862, "period", fontsize=9.9, fontweight="bold")
    put(a, 0.315, 0.862, "spin_period", fontsize=9.9, fontweight="bold")
    put(a, 0.560, 0.862, "poll_period", fontsize=9.9, fontweight="bold")
    put(a, 0.800, 0.862, "the thread it starts", fontsize=9.9, fontweight="bold")
    a.plot([0.04, 0.96], [0.838, 0.838], color="#888", lw=1.0, transform=a.transAxes)
    top, step = 0.775, 0.1085
    for i, label in enumerate(CASES):
        y = top - i * step
        v = facts["verdicts"][label]
        ok = label == "0.02 (default)"
        put(a, 0.055, y, label, fontsize=10.1, family="monospace",
            fontweight="bold" if ok else "normal")
        for x, key in ((0.315, "ros"), (0.560, "rtps")):
            kind = v[key][0]
            good = (kind == "accepted") if ok else (kind == "refused")
            put(a, x, y, "accepted" if kind == "accepted" else kind,
                fontsize=10.1, family="monospace", fontweight="bold",
                color="#1b7f3b" if good else "#c0392b")
        n = facts["traces"][label]["n"]
        if ok:
            note, c = f"{n} polls at 50 Hz", "#1b7f3b"
        elif facts["verdicts"][label]["rtps"][0] == "refused":
            note, c = "never started", "#1b7f3b"
        elif facts["traces"][label]["died"]:
            note, c = "died after 1 poll", "#c0392b"
        elif n < 5:
            note, c = f"{n} poll in 0.25 s (1 Hz)", "#c0392b"
        else:
            note, c = f"{n:,} polls: busy-spin", "#c0392b"
        put(a, 0.800, y, note, fontsize=9.5, color=c, family="monospace")
    return a

table(gs[1, 0], A, "upstream/main", "#fdf4f3")
table(gs[1, 1], B, "this change", "#f2faf4")

fig.suptitle("A hardware bridge loop period that cannot pace a thread is refused rather than run",
             fontsize=14.2, fontweight="bold", y=0.972)
fig.text(0.5, 0.936, "HardwareRosBridge.spin_period / HardwareRtpsBridge.poll_period  -  guarded ahead of each "
         "transport probe, so the verdict is identical with and without the [ros2] extra",
         fontsize=10.3, ha="center", color="#444")

# ---- self-audit ---------------------------------------------------------
for a, y in placed:
    lo, hi = a.get_ylim()
    assert lo - 0.06 * (hi - lo) <= y <= hi + 0.08 * (hi - lo), f"text at y={y} outside {a.get_ylim()}"
for label in ("0", "-1", "nan"):
    assert A["traces"][label]["n"] > 100_000, label
    assert A["verdicts"][label]["rtps"][0] == "accepted"
    assert B["verdicts"][label]["rtps"][0] == "refused"
    assert B["verdicts"][label]["ros"][0] == "refused"
assert A["traces"]["inf"]["died"] == "OverflowError"
assert A["traces"]["True"]["n"] == 1
for tree in (A, B):
    assert tree["verdicts"]["0.02 (default)"]["ros"][0] == "accepted"
    assert tree["verdicts"]["0.02 (default)"]["rtps"][0] == "accepted"
n_acc_main = sum(1 for c in CASES if c != "0.02 (default)" and A["verdicts"][c]["rtps"][0] == "accepted")
n_acc_br = sum(1 for c in CASES if c != "0.02 (default)" and B["verdicts"][c]["rtps"][0] == "accepted")
assert (n_acc_main, n_acc_br) == (5, 0), (n_acc_main, n_acc_br)

fig.savefig("/tmp/artifact_1978.png", bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(Image.open("/tmp/artifact_1978.png").convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print(f"OK  {im.shape[1]}x{im.shape[0]}  main accepted {n_acc_main}/5 unusable, branch {n_acc_br}/5")
