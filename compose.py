import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

RUN = os.environ["GITHUB_RUN_ID"]
A = json.load(open(f"/tmp/probe-main-{RUN}.json"))   # upstream/main
B = json.load(open(f"/tmp/probe-pr-{RUN}.json"))     # this PR
M = json.load(open(f"/tmp/mutate-{RUN}.json"))

# ---- audit every claim before drawing -------------------------------------
def key(r): return (r["fn"], r["exc"])
a = {key(r): r for r in A["rows"]}
b = {key(r): r for r in B["rows"]}
assert set(a) == set(b) and len(a) == 8
TRANSPORT = {"ZError", "OSError"}
PROGRAMMER = {"TypeError", "AttributeError"}
for k, r in a.items():
    assert r["raised"] is None, f"main should swallow everything: {k}"
    assert r["close_called"] and r["session_ref_dropped"]
n_silent_main = sum(1 for k, r in a.items() if not r["records"])
n_misreport_main = sum(1 for k, r in a.items()
                       if any("session closed" in m for _, m in r["records"]))
assert n_silent_main == 4 and n_misreport_main == 4, (n_silent_main, n_misreport_main)
for k, r in b.items():
    if k[1] in TRANSPORT:
        assert r["raised"] is None and r["records"], f"transport fault must be recorded: {k}"
        assert not any("session closed" in m for _, m in r["records"])
    else:
        assert r["raised"] and r["raised"].startswith(k[1]), f"programmer error must propagate: {k}"
assert A["healthy"]["records"] == B["healthy"]["records"] == [["INFO", "Zenoh mesh session closed"]]
assert M["caught_new"] == 6 and M["caught_old"] == 0 and M["n"] == 6
print("audit OK:", f"main silent={n_silent_main}/8 misreported={n_misreport_main}/8")

RED, GREEN, AMBER, INK = "#c0392b", "#1e8449", "#b9770e", "#1b2631"
fig = plt.figure(figsize=(16.2, 11.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 0.92, 0.30], hspace=0.30,
                      left=0.035, right=0.978, top=0.925, bottom=0.035)
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); kw.setdefault("fontsize", 10.4); kw.setdefault("color", INK)
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)

fig.suptitle("A Zenoh session close that failed was reported as one that succeeded",
             fontsize=17.5, fontweight="bold", color=INK, y=0.982)
fig.text(0.5, 0.949,
         "strands_robots/mesh/session.py -- the two _SESSION.close() calls in the module that defines "
         "zenoh_error_types(), whose docstring names close and excludes programmer errors",
         ha="center", fontsize=11.2, color="#5d6d7e")

# ---- row 1: consequence matrix -------------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.055, "What an operator is told when _SESSION.close() raises",
    fontsize=13.2, fontweight="bold", transform=ax.transAxes)
cols = [0.0, 0.145, 0.285, 0.635]
put(ax, cols[0], 0.985, "teardown", fontweight="bold", fontsize=10.6)
put(ax, cols[1], 0.985, "close raises", fontweight="bold", fontsize=10.6)
put(ax, cols[2], 0.985, "upstream/main", fontweight="bold", fontsize=10.6, color=RED)
put(ax, cols[3], 0.985, "this PR", fontweight="bold", fontsize=10.6, color=GREEN)
order = [("release_session", e) for e in ("ZError", "OSError", "TypeError", "AttributeError")] + \
        [("_atexit_cleanup", e) for e in ("ZError", "OSError", "TypeError", "AttributeError")]
TOP, LAST = 0.905, 0.145
step = (TOP - LAST) / (len(order) - 1)
assert step > 0.030, step
def render(r):
    if r["raised"]:
        return f"propagates -- {r['raised'].split(':')[0]}", AMBER
    if not r["records"]:
        return "(nothing logged at all)", RED
    lvl, msg = r["records"][0]
    return f"{lvl}  {msg[:58]}", (RED if "session closed" in msg else GREEN)
y = TOP
for i, k in enumerate(order):
    if i % 4 == 0:
        ax.add_patch(Rectangle((-0.004, y - step * 3.62), 1.008, step * 4.0,
                               transform=ax.transData, facecolor="#f4f6f7", edgecolor="none", zorder=0))
    put(ax, cols[0], y, k[0] + "()" if i % 4 == 0 else "", fontsize=10.2, fontweight="bold", family="monospace")
    put(ax, cols[1], y, k[1], fontsize=10.2, family="monospace",
        color=(INK if k[1] in TRANSPORT else "#7d3c98"))
    for c, src in ((cols[2], a[k]), (cols[3], b[k])):
        txt, col = render(src)
        put(ax, c, y, txt, fontsize=9.5, family="monospace", color=col)
    y -= step
assert abs((y + step) - LAST) < 1e-9
put(ax, cols[0], 0.075,
    "purple = a bug, not a transport fault: zenoh_error_types()'s docstring excludes it so it "
    "\"surfaces loudly instead of being swallowed by a best-effort cleanup path\"",
    fontsize=9.6, color="#5d6d7e")
put(ax, cols[0], 0.028,
    f"main: {n_misreport_main}/8 reported a clean close, {n_silent_main}/8 said nothing -- and all 8 dropped the "
    "session reference, so nothing could retry.   healthy close, both trees: "
    f"{A['healthy']['records'][0][0]} {A['healthy']['records'][0][1]!r} (unchanged)",
    fontsize=9.8, color=INK)

# ---- row 2: mutation matrix ----------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.075, "Would a regression be caught? (mutate the fix, run both test sets)",
    fontsize=13.2, fontweight="bold", transform=ax2.transAxes)
put(ax2, 0.0, 0.965, "plausible regression", fontweight="bold", fontsize=10.6)
put(ax2, 0.615, 0.965, "this PR", fontweight="bold", fontsize=10.6, color=GREEN)
put(ax2, 0.755, 0.965, "pre-existing tests", fontweight="bold", fontsize=10.6, color=RED)
rows = M["rows"]
T2, L2 = 0.865, 0.155
s2 = (T2 - L2) / (len(rows) - 1)
assert s2 > 0.030, s2
y = T2
for label, (fa, _), (fb, _) in rows:
    ctrl = not label.startswith("M")
    put(ax2, 0.0, y, label, fontsize=10.0, family="monospace", color=("#5d6d7e" if ctrl else INK))
    put(ax2, 0.615, y, f"{fa} failed", fontsize=10.0, family="monospace",
        color=("#5d6d7e" if ctrl else GREEN))
    put(ax2, 0.755, y, f"{fb} failed" + ("" if ctrl else "   <- BLIND"), fontsize=10.0,
        family="monospace", color=("#5d6d7e" if ctrl else RED))
    y -= s2
assert abs((y + s2) - L2) < 1e-9
put(ax2, 0.0, 0.075,
    f"caught by this PR: {M['caught_new']} of {M['n']}    caught by the pre-existing tests: "
    f"{M['caught_old']} of {M['n']}    "
    "(M2/M5 keep the narrow catch and drop only the record -- a structural pin cannot see that)",
    fontsize=9.8, color=INK)

# ---- row 3: gate ---------------------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(Rectangle((-0.004, 0.02), 1.008, 0.96, facecolor="#eaf2f8", edgecolor="#aed6f1"))
put(ax3, 0.012, 0.86, "Gate", fontweight="bold", fontsize=11.4)
put(ax3, 0.012, 0.60,
    "MUJOCO_GL=egl pytest tests: 28463 passed / 257 skipped / 0 failed (618s)  ==  28448 on main + 15 new cases\n"
    "ruff check + ruff format --check + mypy: clean (14 examples/isaac_gs errors byte-identical to the pristine base)\n"
    "mesh/session.py: 6 missing lines -> 0. No policy, simulation, rendering, recording or asset behaviour changes:\n"
    "this is transport teardown reporting, so the artifact above is the measurement rather than a rollout.",
    fontsize=10.2, family="monospace")

for ax_, yv, is_axes in placed:
    lo, hi = ax_.get_ylim()
    assert (-0.03 <= yv <= 1.10) if is_axes else (lo - 0.05 <= yv <= hi + 0.07), (yv, is_axes)

out = pathlib.Path("_art/session_close_report.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.array(Image.open(out).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert bad == 0, f"{side} border has {bad} non-white px"
print(f"WROTE {out}  {im.shape[1]}x{im.shape[0]}  border clean")
