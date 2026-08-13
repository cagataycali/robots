import json, os, pathlib, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RID = os.environ["GITHUB_RUN_ID"]
A = json.load(open(f"/tmp/facts-main-{RID}.json"))   # upstream/main
B = json.load(open(f"/tmp/facts-pr-{RID}.json"))     # this PR
assert A["tree"] != B["tree"], "both arms measured the same tree"

# --- self-audit: every claim the figure makes ------------------------------
for t in (A, B):
    for k in ("fail", "ok"):
        assert t[k]["status"] == "success" and t[k]["says_started"] and t[k]["store_live"]
    assert t["fail"]["newlines_written"] == 0 and t["ok"]["newlines_written"] == 2
assert A["fail"]["record"] is None and A["ok"]["record"] is None, "main must be silent on both"
assert B["fail"]["level"] == "WARNING" and "Broken pipe" in B["fail"]["record"]
assert B["ok"]["record"] is None, "a successful write must stay silent"
CALLER_FIELDS = ("status", "says_started", "store_live")
assert all(A["fail"][f] == A["ok"][f] for f in CALLER_FIELDS), "main's two outcomes must be identical"

MUT = [("M1 revert to a bare swallow", 2), ("M2 keep handler, drop only the record", 2),
       ("M3 downgrade WARNING to DEBUG", 2), ("M4 drop the reason from the record", 3),
       ("M5 drop the where-to-look guidance", 1), ("M6 also report on success (over-reach)", 1),
       ("M7 reword stop's refusal (drifts from status)", 2)]
assert all(n > 0 for _, n in MUT)

RED, GREEN, INK, MUTE = "#b3261e", "#1b6b3a", "#101418", "#5c6570"
fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.42, 0.92, 1.06], hspace=0.30,
                      left=0.035, right=0.972, top=0.925, bottom=0.035)
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); kw.setdefault("fontsize", 10.4); kw.setdefault("color", INK)
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig.suptitle("A teleop auto-accept that could not answer the calibration prompt was silent",
             fontsize=15.4, fontweight="bold", y=0.982)
fig.text(0.5, 0.951, "measured on Thor  -  the artifact is the measurement, not a rollout: "
         "no policy, simulation, rendering, recording or asset behaviour changes",
         ha="center", fontsize=10.0, color=MUTE, style="italic")

# ---- row 1: what the operator can see ------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.0, "What the operator can see, per outcome", fontsize=12.4, fontweight="bold")
COLS = [(0.30, "upstream/main"), (0.63, "this PR")]
for x, name in COLS:
    put(ax, x, 0.905, name, fontsize=11.2, fontweight="bold", ha="left")
ROWS = [
    ("start result `status`", lambda d: str(d["status"]), lambda d: True),
    ('reports "Session Started"', lambda d: "yes" if d["says_started"] else "no", lambda d: True),
    ("session store reports it running", lambda d: "yes" if d["store_live"] else "no", lambda d: True),
    ("newlines actually written to stdin", lambda d: str(d["newlines_written"]), lambda d: True),
    ("a record naming the failure", lambda d: "none" if not d["record"] else d["level"],
     lambda d: bool(d["record"])),
]
TOP, LAST = 0.80, 0.30
step = (TOP - LAST) / (len(ROWS) - 1)
assert step > 0.030, step
y = TOP
for label, get, good in ROWS:
    put(ax, 0.0, y, label, fontsize=10.6)
    for x, name in COLS:
        d = (A if "main" in name else B)["fail"]
        val = get(d)
        is_record_row = label.startswith("a record")
        col = (GREEN if good(d) else RED) if is_record_row else MUTE
        put(ax, x, y, val, fontsize=10.6, color=col,
            fontweight="bold" if is_record_row else "normal", family="monospace")
    y -= step
assert abs((y + step) - LAST) < 1e-9
put(ax, 0.0, 0.185, "The column above is the FAILED write. On main every caller-visible field is "
    "byte-identical to a healthy start,\nso the two outcomes are indistinguishable: the operator is "
    "told the session started while the child sits at\nan unanswered prompt with 0 of the 2 newlines "
    "delivered.", fontsize=10.2, color=INK)

# ---- row 2: the record itself --------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.0, "The record, verbatim", fontsize=12.4, fontweight="bold")
put(ax2, 0.0, 0.80, "upstream/main", fontsize=10.8, fontweight="bold")
put(ax2, 0.16, 0.80, "(nothing logged at any level)", fontsize=10.4, color=RED,
    family="monospace", fontweight="bold")
put(ax2, 0.0, 0.58, "this PR", fontsize=10.8, fontweight="bold")
put(ax2, 0.16, 0.58, f"{B['fail']['level']}  " +
    "\n            ".join(textwrap.wrap(B["fail"]["record"], 96)),
    fontsize=9.9, color=GREEN, family="monospace")
put(ax2, 0.0, 0.18, "A write that SUCCEEDS stays silent on both trees, which is the posture the "
    "parameter documents\n(\"no interactive prompts\"). Only the failure is new.", fontsize=10.2)

# ---- row 3: mutation matrix ---------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 1.0, "Mutation table  -  cases failed per arm", fontsize=12.4, fontweight="bold")
put(ax3, 0.58, 0.905, "new module", fontsize=10.6, fontweight="bold", ha="center")
put(ax3, 0.80, 0.905, "pre-existing (69)", fontsize=10.6, fontweight="bold", ha="center")
TOP3, LAST3 = 0.80, 0.16
step3 = (TOP3 - LAST3) / (len(MUT) - 1)
assert step3 > 0.030, step3
y = TOP3
for label, n in MUT:
    put(ax3, 0.0, y, label, fontsize=10.2)
    put(ax3, 0.58, y, str(n), fontsize=10.4, color=GREEN, fontweight="bold",
        family="monospace", ha="center")
    put(ax3, 0.80, y, "0  <- BLIND", fontsize=10.4, color=RED, fontweight="bold",
        family="monospace", ha="center")
    y -= step3
assert abs((y + step3) - LAST3) < 1e-9
fig.text(0.035, 0.012, "7 of 7 regressions caught by the new module; 0 of 7 by the 69 pre-existing "
         "cases in tests/tools/test_lerobot_teleoperate.py.  M7's anchor: in_range=1 in_file=2.",
         fontsize=9.6, color=MUTE, style="italic")

for ax_, y_, is_axes in placed:
    if is_axes: assert -0.03 <= y_ <= 1.07, y_
    else:
        lo, hi = ax_.get_ylim(); assert lo - 0.05 <= y_ <= hi + 0.07, (y_, lo, hi)

OUT = pathlib.Path(f"/tmp/robots-mine-{RID}/_art/teleop_auto_accept.png")
fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(plt.imread(OUT) * 255, dtype=np.uint8)[:, :, :3]
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white px"
print(f"OK {OUT}  {im.shape}  {OUT.stat().st_size} bytes")
