import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUN = os.environ["GITHUB_RUN_ID"]
F = json.load(open(f"/tmp/art-{RUN}.json"))
OUT = pathlib.Path(f"_art/quantile-norm-probe-coverage.png")

# ---- derive + assert every claim the figure makes
probes = F["probes"]
hole = [p for p in probes if p["missing"]]
assert len(probes) == 8, len(probes)
assert len(hole) == 1 and hole[0]["name"] == "_policy_uses_quantile_norm", hole
assert hole[0]["missing"] == [235, 236, 238], hole
assert F["lines"]["before"] == [235, 236, 238], F["lines"]
assert F["lines"]["after"] == [], F["lines"]
muts = [m for m in F["mutations"] if m["tag"] != "--"]
ctrl = next(m for m in F["mutations"] if m["tag"] == "--")
assert len(muts) == 4
assert all(m["new"]["failed"] > 0 for m in muts), "a mutation the new tests miss"
assert all(m["base"]["failed"] == 0 for m in muts), "a mutation the base suite catches"
assert ctrl["new"]["failed"] == 0 and ctrl["base"]["failed"] == 0
cons = {(c["state"], c["ptype"]): c for c in F["consequence"]}
assert cons[("live registry (control)", "molmoact2")]["warned"] is True
assert cons[("unreadable default_factory", "molmoact2")]["warned"] is True
assert cons[("unreadable default_factory", "act")]["warned"] is False
assert cons[("no normalization_mapping", "molmoact2")]["warned"] is False

GREEN, RED, GREY, INK = "#1b7f4b", "#b3261e", "#5b6470", "#101317"
placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 9.3); kw.setdefault("color", INK); kw.setdefault("va", "center")
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.0, 12.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.92, 0.80], hspace=0.16,
                      left=0.035, right=0.975, top=0.925, bottom=0.045)
fig.suptitle("The one registry probe with an uncovered branch, and what its fallback is for",
             fontsize=15.5, fontweight="bold", y=0.972, color=INK)
fig.text(0.5, 0.945, "strands_robots/training/lerobot.py  -  tests only; no library behaviour changes",
         ha="center", fontsize=10.2, color=GREY, style="italic")

# ---------- row 1: the eight sibling registry probes
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965, "1.  Eight probes read lerobot's live registry with a documented static fallback. Seven bodies were fully covered.",
    fontsize=11.6, fontweight="bold")
TOP, LAST = 0.845, 0.135
step = (TOP - LAST) / (len(probes) - 1)
assert step > 0.030, step
put(ax, 0.035, TOP + 0.075, "registry probe", fontsize=9.4, fontweight="bold", color=GREY)
put(ax, 0.50, TOP + 0.075, "reads", fontsize=9.4, fontweight="bold", color=GREY)
put(ax, 0.735, TOP + 0.075, "uncovered lines (main)", fontsize=9.4, fontweight="bold", color=GREY)
y = TOP
for p in probes:
    is_hole = bool(p["missing"])
    col = RED if is_hole else GREEN
    if is_hole:
        ax.add_patch(plt.Rectangle((0.022, y - 0.052), 0.956, 0.104,
                                   facecolor=RED, alpha=0.09, edgecolor=RED, lw=1.1, zorder=0))
    put(ax, 0.035, y, p["name"], fontsize=9.6, family="monospace",
        fontweight="bold" if is_hole else "normal")
    reads = "a field's DEFAULT (default_factory)" if is_hole else "a field NAME / the registry keys"
    put(ax, 0.50, y, reads, fontsize=9.2, color=INK if is_hole else GREY)
    label = f"{p['missing']}  ({len(p['missing'])} of {p['n']})" if is_hole else "none"
    put(ax, 0.735, y, label, fontsize=9.4, family="monospace", color=col,
        fontweight="bold" if is_hole else "normal")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(ax, 0.035, 0.045,
    "Calling the default is what creates a third outcome: the type is present, yet its answer cannot be read.",
    fontsize=9.5, color=GREY, style="italic")

# ---------- row 2: mutation matrix
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.955, "2.  Four plausible regressions in those branches - every one invisible to the suite as it stood.",
    fontsize=11.6, fontweight="bold")
rows = F["mutations"]
T2, L2 = 0.775, 0.145
s2 = (T2 - L2) / (len(rows) - 1)
assert s2 > 0.030, s2
put(ax2, 0.035, T2 + 0.10, "mutation of _policy_uses_quantile_norm", fontsize=9.4, fontweight="bold", color=GREY)
put(ax2, 0.575, T2 + 0.10, "this PR", fontsize=9.4, fontweight="bold", color=GREY)
put(ax2, 0.755, T2 + 0.10, "pre-existing tests", fontsize=9.4, fontweight="bold", color=GREY)
y = T2
for m in rows:
    is_ctrl = m["tag"] == "--"
    if not is_ctrl:
        ax2.add_patch(plt.Rectangle((0.735, y - 0.050), 0.243, 0.100,
                                    facecolor=RED, alpha=0.10, edgecolor="none", zorder=0))
    put(ax2, 0.035, y, f"{m['tag']:>3s}  {m['label']}", fontsize=9.6,
        family="monospace", color=GREY if is_ctrl else INK)
    nf, bf = m["new"]["failed"], m["base"]["failed"]
    put(ax2, 0.575, y, f"{nf} failed / {m['new']['passed']} passed", fontsize=9.4,
        family="monospace", color=GREEN if (nf > 0) != is_ctrl else GREY)
    tag = f"{bf} failed / {m['base']['passed']} passed" + ("" if is_ctrl else "   <- BLIND")
    put(ax2, 0.755, y, tag, fontsize=9.4, family="monospace", color=GREY if is_ctrl else RED)
    y -= s2
assert abs((y + s2) - L2) < 1e-9
put(ax2, 0.035, 0.048,
    "M3's anchor occurs twice in the module, so each mutation is scoped to the function by AST line range.",
    fontsize=9.5, color=GREY, style="italic")

# ---------- row 3: the consequence
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 0.945, "3.  Why the two branches must stay different: what a caller is told about a dataset with no q01/q99.",
    fontsize=11.6, fontweight="bold")
T3, L3 = 0.700, 0.230
c = F["consequence"]
s3 = (T3 - L3) / (len(c) - 1)
assert s3 > 0.030, s3
put(ax3, 0.035, T3 + 0.115, "registry state", fontsize=9.4, fontweight="bold", color=GREY)
put(ax3, 0.400, T3 + 0.115, "policy_type", fontsize=9.4, fontweight="bold", color=GREY)
put(ax3, 0.560, T3 + 0.115, "probe answers", fontsize=9.4, fontweight="bold", color=GREY)
put(ax3, 0.730, T3 + 0.115, "validate() warns?", fontsize=9.4, fontweight="bold", color=GREY)
y = T3
for r in c:
    put(ax3, 0.035, y, r["state"], fontsize=9.5)
    put(ax3, 0.400, y, r["ptype"], fontsize=9.5, family="monospace")
    put(ax3, 0.560, y, str(r["probe"]), fontsize=9.5, family="monospace",
        color=GREEN if r["probe"] else GREY)
    put(ax3, 0.730, y, "yes - mis-normalization flagged" if r["warned"] else "no",
        fontsize=9.5, color=GREEN if r["warned"] else GREY, fontweight="bold" if r["warned"] else "normal")
    y -= s3
assert abs((y + s3) - L3) < 1e-9
put(ax3, 0.035, 0.118,
    "Row 2 is the fallback doing its job: with the answer unreadable, the static set keeps the warning firing.",
    fontsize=9.5, color=GREY, style="italic")
put(ax3, 0.035, 0.035,
    f"Gate: lines {F['lines']['before']} uncovered -> {F['lines']['after'] or 'none'};  "
    "ruff + ruff format + mypy clean;  MUJOCO_GL=egl full suite green.",
    fontsize=9.4, family="monospace", color=GREY)

for ax_, y_, is_axes in placed:
    lo, hi = ax_.get_ylim()
    if is_axes:
        assert -0.03 <= y_ <= 1.10, y_
    else:
        assert lo - 0.03 * (hi - lo) <= y_ <= hi + 0.10 * (hi - lo), (y_, lo, hi)

fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.30)
plt.close(fig)

im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {Image.open(OUT).size}  {OUT.stat().st_size // 1024} KiB")
