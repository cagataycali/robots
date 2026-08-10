"""Compose the measured figure for the Reachy transport degradation-branch coverage."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

F = json.load(open("/tmp/art_facts.json"))
OUT = pathlib.Path("/tmp/art_degradation_branches.png")

# ---- assert every number we are about to render ------------------------------
assert F["cov_before"] == {"stmts": 152, "missing": 6, "pct": 96.1,
                           "missing_lines": [29, 30, 230, 231, 286, 287]}, F["cov_before"]
assert F["cov_after"]["missing"] == 0 and F["cov_after"]["pct"] == 100.0, F["cov_after"]
assert F["imu"]["after_malformed"] == [] and F["imu"]["after_good"] == [{"accel": [0, 0, 9.8]}]
assert F["ws_readable"]["additional_headers"] == {"Authorization": "Bearer secret-token"}
assert F["ws_readable"]["extra_headers"] is None
assert F["ws_unreadable"]["extra_headers"] == {"Authorization": "Bearer secret-token"}
assert F["ws_unreadable"]["additional_headers"] is None
assert F["host"] == {"resolvable": "10.1.2.3", "unresolvable": "reachy-mini.local"}
MUT = {m["id"]: m for m in F["mutations"]}
assert len(MUT) == 5
n_caught_new = sum(1 for m in MUT.values() if m["new"]["failed"])
n_caught_old = sum(1 for m in MUT.values() if m["old"]["failed"])
assert (n_caught_new, n_caught_old) == (5, 1), (n_caught_new, n_caught_old)

GREEN, RED, GREY, INK = "#1b7f3b", "#b3261e", "#5f6368", "#111111"
BAND_R, BAND_G = "#fdecea", "#e9f6ee"

placed = []
def put(ax, x, y, s, axes_coords=True, **kw):
    kw.setdefault("fontsize", 9.6)
    kw.setdefault("va", "center")
    if axes_coords:
        kw["transform"] = ax.transAxes
        placed.append((ax, y, True))
    else:
        placed.append((ax, y, False))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 10.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[4.5, 4.1, 1.05], hspace=0.30,
                      left=0.028, right=0.985, top=0.925, bottom=0.035)

fig.suptitle("Reachy transport: the three degradation branches that keep a hardware link usable",
             fontsize=15.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.951,
         "strands_robots/device_connect/reachy_transport.py  --  every cell below is measured, "
         "not asserted from the source",
         ha="center", fontsize=10.2, color=GREY, style="italic")

# ================= ROW 1: the three branches =================================
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.045, "1.  What each branch protects, and whether anything drove it",
    fontsize=12.4, fontweight="bold")

COLS = [0.005, 0.175, 0.470, 0.590, 0.700]
HEAD = ["branch", "what it protects", "driven before", "driven now", "measured outcome on that path"]
for x, h in zip(COLS, HEAD):
    put(ax, x, 0.905, h, fontsize=9.9, fontweight="bold", color=GREY)
ax.plot([0.0, 1.0], [0.868, 0.868], color=GREY, lw=0.9, transform=ax.transAxes, clip_on=False)

rows1 = [
    ("ZenohLink.start\n  -> _on_imu wrapper",
     "a malformed IMU frame is dropped so the\nsubscription stays alive (its byte-identical\njoints twin was the only one driven)",
     "joints only", "yes",
     f"malformed frame -> forwarded {F['imu']['after_malformed']}\n"
     f"then a good frame -> {json.dumps(F['imu']['after_good'][0])}\n"
     "i.e. dropped AND still delivering"),
    ("resolve_host",
     "an unresolvable hostname is passed through\nrather than refused (mDNS .local names the\nstdlib resolver may not answer for)",
     "resolve path\nonly", "yes",
     f"resolvable   -> {F['host']['resolvable']!r}\n"
     f"unresolvable -> {F['host']['unresolvable']!r}\n"
     "i.e. the caller gets a host it can try"),
    ("WebSocketLink.start\n  -> header keyword",
     "the bearer credential survives a connect that\ninspect.signature cannot read, by falling back\nto the legacy extra_headers keyword",
     "no", "yes",
     "signature readable   -> additional_headers=Bearer ...\n"
     "signature unreadable -> extra_headers=Bearer ...\n"
     "i.e. the token is not lost on the fallback"),
]
TOP1, LAST1 = 0.760, 0.130
STEP1 = (TOP1 - LAST1) / (len(rows1) - 1)
assert STEP1 > 0.20, STEP1
for i, (name, prot, before, now, outcome) in enumerate(rows1):
    y = TOP1 - i * STEP1
    ax.add_patch(Rectangle((0.0, y - STEP1 * 0.44), 1.0, STEP1 * 0.88, transform=ax.transAxes,
                           facecolor=BAND_G if i % 2 == 0 else "#f6f8fa", edgecolor="none", zorder=0))
    put(ax, COLS[0], y, name, fontsize=9.7, fontweight="bold", family="monospace")
    put(ax, COLS[1], y, prot, fontsize=9.3)
    put(ax, COLS[2], y, before, fontsize=9.6, color=RED, fontweight="bold")
    put(ax, COLS[3], y, now, fontsize=9.6, color=GREEN, fontweight="bold")
    put(ax, COLS[4], y, outcome, fontsize=8.9, family="monospace", color=INK)
assert abs((TOP1 - (len(rows1) - 1) * STEP1) - LAST1) < 1e-9

# ================= ROW 2: mutation matrix ====================================
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.055,
    "2.  Mutation matrix -- a plausible regression on each branch, against both arms",
    fontsize=12.4, fontweight="bold")
put(ax2, 0.0, 0.945,
    "The pre-existing arm is the 16 tests this file already had. A mutation it does not catch is a "
    "regression the suite would have shipped.",
    fontsize=9.5, color=GREY, style="italic")

C2 = [0.005, 0.055, 0.560, 0.760]
for x, h in zip(C2, ["", "mutation applied to the production branch", "new 6 tests", "pre-existing 16 tests"]):
    put(ax2, x, 0.855, h, fontsize=9.9, fontweight="bold", color=GREY)
ax2.plot([0.0, 1.0], [0.818, 0.818], color=GREY, lw=0.9, transform=ax2.transAxes, clip_on=False)

TOP2, LAST2 = 0.715, 0.115
STEP2 = (TOP2 - LAST2) / (len(F["mutations"]) - 1)
assert STEP2 > 0.10, STEP2
for i, m in enumerate(F["mutations"]):
    y = TOP2 - i * STEP2
    blind = not m["old"]["failed"]
    ax2.add_patch(Rectangle((0.0, y - STEP2 * 0.42), 1.0, STEP2 * 0.84, transform=ax2.transAxes,
                            facecolor=BAND_R if blind else "#f6f8fa", edgecolor="none", zorder=0))
    put(ax2, C2[0], y, m["id"], fontsize=9.8, fontweight="bold", family="monospace")
    put(ax2, C2[1], y, m["label"], fontsize=9.5)
    put(ax2, C2[2], y, f"caught  ({m['new']['n_failed']} failed)",
        fontsize=9.5, color=GREEN, fontweight="bold", family="monospace")
    if blind:
        put(ax2, C2[3], y, "BLIND  (all 16 passed)",
            fontsize=9.5, color=RED, fontweight="bold", family="monospace")
    else:
        put(ax2, C2[3], y, f"caught  ({m['old']['n_failed']} failed)",
            fontsize=9.5, color=GREY, family="monospace")
assert abs((TOP2 - (len(F["mutations"]) - 1) * STEP2) - LAST2) < 1e-9
put(ax2, C2[1], 0.022,
    f"{n_caught_new} of 5 caught by the new tests   |   "
    f"{5 - n_caught_old} of 5 invisible to the suite as it stood   "
    "(M2, a copy-paste slip, is honestly caught by the existing forwarding test)",
    fontsize=9.7, fontweight="bold")

# ================= ROW 3: coverage + gate ====================================
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
b, a = F["cov_before"], F["cov_after"]
lines3 = [
    f"coverage of reachy_transport.py (full suite):  {b['stmts']} stmts, "
    f"{b['missing']} missing, {b['pct']}%   ->   {a['missing']} missing, {a['pct']}%"
    f"      uncovered lines {b['missing_lines']} -> []",
    "gate: 27539 passed / 257 skipped / 0 failed   |   ruff check + format clean   |   "
    "mypy 0 errors outside examples/   |   419 repo-wide guard tests pass",
    "tests only plus two production docstrings; the docstring-stripped AST digest of the module is "
    "unchanged, so no executable line moved.",
]
TOP3, LAST3 = 0.80, 0.16
STEP3 = (TOP3 - LAST3) / (len(lines3) - 1)
assert STEP3 > 0.20, STEP3
for i, s in enumerate(lines3):
    y = TOP3 - i * STEP3
    put(ax3, 0.005, y, s, fontsize=9.9, family="monospace" if i < 2 else None,
        color=INK if i < 2 else GREY, style=None if i < 2 else "italic")
assert abs((TOP3 - (len(lines3) - 1) * STEP3) - LAST3) < 1e-9

# ---- layout guards ----------------------------------------------------------
for axx, y, is_axes in placed:
    lo, hi = (-0.03, 1.10) if is_axes else axx.get_ylim()
    assert lo <= y <= hi, f"text at y={y} outside {lo}..{hi}"
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(matplotlib.image.imread(OUT))[:, :, :3]
im8 = (im * 255).astype(int) if im.max() <= 1.0 else im.astype(int)
for name, band in (("top", im8[:8]), ("bottom", im8[-8:]), ("left", im8[:, :8]), ("right", im8[:, -8:])):
    nw = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print(f"OK {OUT}  size={im8.shape[1]}x{im8.shape[0]}  texts={len(placed)}  border clean")
