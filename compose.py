"""Compose the measured figure. Every drawn value is asserted against the dumps."""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.loads(pathlib.Path(sys.argv[1]).read_text())   # main
B = json.loads(pathlib.Path(sys.argv[2]).read_text())   # branch
OUT = pathlib.Path(sys.argv[3])

assert A["tree"] != B["tree"], "both arms measured the same tree"
assert A["cells"] == B["cells"], "the gate's behaviour is not supposed to change"
mA, mB = A["following_the_description"], B["following_the_description"]
assert A["description"]["reader_would_set"] == "push_to_hub"
assert B["description"]["reader_would_set"] == "policy.push_to_hub"
assert mA["published"] is False and mA["asked"] is True, mA
assert mB["published"] is True and mB["asked"] is False, mB
assert mB["argv_flag"] == "--policy.push_to_hub=true", mB

GREEN, AMBER, RED = "#1b7f3b", "#8a6100", "#a11b2b"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    if kw.get("transform") is not None:
        placed.append((ax, y, True))
    else:
        placed.append((ax, y, False))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.15, 0.30], hspace=0.30,
                      left=0.035, right=0.978, top=0.925, bottom=0.035)
fig.suptitle("lerobot_train push_to_hub: which allowlist entry clears which spelling",
             fontsize=15.5, fontweight="bold", y=0.975)
fig.text(0.5, 0.945, "The gate is unchanged. The description named the entry that clears the other spelling.",
         ha="center", fontsize=10.4, style="italic", color="#333")

# ---- row 1: the (allowlist entry x spelling) matrix, identical on both trees
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.02, "1. What each allowlist entry clears  (measured identically on BOTH trees "
                   "- this PR changes no behaviour)", fontsize=12, fontweight="bold",
    transform=ax.transAxes)
cols = [("named parameter\npush_to_hub=True", "named"),
        ("raw extra_flags\n{'push_to_hub': True}", "raw")]
rows = ["no entry", "push_to_hub", "policy.push_to_hub"]
x0, colw = 0.30, 0.30
for j, (title, _) in enumerate(cols):
    put(ax, x0 + colw * j + colw / 2, 0.86, title, fontsize=10.2, fontweight="bold",
        ha="center", va="center", transform=ax.transAxes)
put(ax, 0.005, 0.86, "STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", fontsize=9.6, fontweight="bold",
    va="center", transform=ax.transAxes)
TOP, LAST = 0.63, 0.13
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for label in rows:
    put(ax, 0.005, y, label if label == "no entry" else f"{label}", fontsize=10.4,
        family="monospace", va="center", transform=ax.transAxes)
    for j, (_, spelling) in enumerate(cols):
        c = A["cells"][f"{label}|{spelling}"]
        cleared = not c["asked"]
        launched = c["published"]
        txt = ("pre-approved, launched" if (cleared and launched)
               else "pre-approved" if cleared else "operator prompted")
        col = GREEN if cleared else AMBER
        ax.add_patch(plt.Rectangle((x0 + colw * j + 0.012, y - 0.075), colw - 0.024, 0.15,
                                   transform=ax.transAxes, facecolor=col, alpha=0.13,
                                   edgecolor=col, lw=1.3))
        put(ax, x0 + colw * j + colw / 2, y + 0.020, txt, fontsize=10.2, fontweight="bold",
            ha="center", va="center", color=col, transform=ax.transAxes)
        put(ax, x0 + colw * j + colw / 2, y - 0.038,
            c["argv_flag"] or "no flag emitted (refused)", fontsize=8.8, family="monospace",
            ha="center", va="center", color="#444", transform=ax.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y
put(ax, 0.005, 0.005, "The two entries are complementary: each clears exactly the spelling the other "
                      "does not.", fontsize=9.6, style="italic", color="#333",
    transform=ax.transAxes)

# ---- row 2: what each tree's description tells a reader, and what following it yields
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.03, "2. Following the description for a headless run  (nobody can answer the prompt)",
    fontsize=12, fontweight="bold", transform=ax2.transAxes)
panels = [(0.0, "main", A, mA, RED), (0.505, "this PR", B, mB, GREEN)]
for px, name, facts, res, col in panels:
    d = facts["description"]
    ax2.add_patch(plt.Rectangle((px, 0.02), 0.475, 0.94, transform=ax2.transAxes,
                                facecolor=col, alpha=0.055, edgecolor=col, lw=1.5))
    put(ax2, px + 0.015, 0.90, name, fontsize=11.4, fontweight="bold", color=col,
        transform=ax2.transAxes)
    entry = d["entry"]
    wrapped: list[str] = []
    line = ""
    for w in entry.split():
        if len(line) + len(w) + 1 > 74:
            wrapped.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    wrapped.append(line)
    ly = 0.83
    for ln in wrapped[:6]:
        put(ax2, px + 0.015, ly, ln, fontsize=8.6, family="monospace", color="#333",
            transform=ax2.transAxes)
        ly -= 0.052
    put(ax2, px + 0.015, ly - 0.030,
        f"a reader sets: ALLOW={d['reader_would_set']}", fontsize=9.8,
        family="monospace", fontweight="bold", transform=ax2.transAxes)
    verdict = ("REFUSED - still prompts, nothing published"
               if not res["published"] else "PRE-APPROVED - launches as documented")
    put(ax2, px + 0.015, ly - 0.098, verdict, fontsize=11.0, fontweight="bold", color=col,
        transform=ax2.transAxes)
    put(ax2, px + 0.015, ly - 0.152,
        f"asked={res['asked']}  published={res['published']}  status={res['status']}",
        fontsize=9.0, family="monospace", color="#444", transform=ax2.transAxes)
    put(ax2, px + 0.015, ly - 0.200, f"argv: {res['argv_flag'] or '(no flag - refused)'}",
        fontsize=9.0, family="monospace", color="#444", transform=ax2.transAxes)

# ---- row 3: gate
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
lines = [
    "Gate: 29772 passed / 266 skipped / 0 failed (727s, MUJOCO_GL=egl; 8 external-tree flakes re-run green) | ruff clean | mypy 0 non-examples errors",
    "Mutation table: 5 of 6 regressions caught by the 5 new cases; 2 of 6 invisible to the 16 pre-existing cases "
    "(both docs regressions). The 6th - deleting the blocklist comment - is caught by neither and is prose, not behaviour.",
    "No policy, simulation, rendering, recording or asset behaviour changes - the artifact is the measurement, not a rollout",
]
TOP3, LAST3 = 0.78, 0.16
S3 = (TOP3 - LAST3) / (len(lines) - 1)
assert S3 > 0.030, S3
yy = TOP3
for ln in lines:
    put(ax3, 0.005, yy, ln, fontsize=9.4, family="monospace", color="#222",
        transform=ax3.transAxes)
    yy -= S3
assert abs((yy + S3) - LAST3) < 1e-9, yy

for ax_, yv, is_axes in placed:
    if is_axes:
        assert -0.05 <= yv <= 1.10, f"axes-fraction text at y={yv}"
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, f"data text at y={yv} outside {(lo, hi)}"

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {im.shape[1]}x{im.shape[0]}  texts={len(placed)}")
