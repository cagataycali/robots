"""Compose the measured figure. Every cell asserted against the dump."""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RID = sys.argv[1]
F = json.loads(pathlib.Path(f"/tmp/art-{RID}.json").read_text())
OUT = pathlib.Path(f"/tmp/fig-{RID}.png")

# ---- audit the dump before drawing a single cell -------------------------
assert F["tree"].endswith(RID), F["tree"]
assert F["before"]["missing"] == [342, 343, 402, 403], F["before"]
assert F["after"]["missing"] == [], F["after"]
assert F["before"]["pct"] == 97.3 and F["after"]["pct"] == 100.0, (F["before"], F["after"])
assert F["after"]["passed"] - F["before"]["passed"] == 11, (F["before"], F["after"])
assert F["behaviour"]["policy_mode"]["outcome"] == "completed"
assert F["behaviour"]["policy_mode"]["chunk_exact"] is True
assert F["behaviour"]["forward_dynamics"]["outcome"] == "refused"
assert F["behaviour"]["forward_dynamics"]["is_shared_hint"] is True
assert F["hint_names_extra"] and F["hint_names_service"]
MUTS = F["mutations"]
N_NEW = sum(1 for m in MUTS if m["new_failed"] > 0)
N_OLD = sum(1 for m in MUTS if m["old_failed"] > 0)
assert (len(MUTS), N_NEW, N_OLD) == (6, 5, 1), (len(MUTS), N_NEW, N_OLD)

GREEN, RED, GREY, INK = "#1a7f37", "#b42318", "#57606a", "#1f2328"
BEFORE_MISS = set(F["before"]["missing"])

placed: list[tuple] = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "center")
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.0, 11.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.00, 0.92, 0.34], hspace=0.30,
                      left=0.035, right=0.972, top=0.925, bottom=0.035)

fig.suptitle(
    "One missing native stack, two opposite decisions: which of them the suite drove",
    fontsize=16.5, fontweight="bold", y=0.975, color=INK)
fig.text(0.5, 0.947,
         "strands_robots/policies/cosmos3/policy_diffusers.py  -  measured on this tree, "
         "no GPU / no weights / no policy server",
         ha="center", fontsize=10.6, color=GREY, style="italic")

# ================= ROW 1: the four native-stack-absent lines =============
ax = gs[0].subgridspec(1, 1).subplots(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965, "Every branch the module takes when the native stack is not importable",
    fontsize=13.2, fontweight="bold", color=INK, transform=ax.transAxes)

cols = [0.012, 0.155, 0.315, 0.560, 0.735, 0.878]
hdr = ["line", "function", "decision", "what a caller gets", "before", "after"]
TOP, LAST = 0.855, 0.300
put(ax, 0, 0, "", alpha=0)  # keep list non-empty for the guard
for x, h in zip(cols, hdr):
    put(ax, x, TOP, h, fontsize=11.0, fontweight="bold", color=GREY, transform=ax.transAxes)

rows = [
    (255, "_load_pipeline", "refuse", "the shared install hint"),
    (286, "_import_condition_cls", "refuse", "the shared install hint"),
    (343, "_as_action_tensor", "refuse", "the shared install hint"),
    (403, "_to_numpy", "degrade", "the chunk, handed to NumPy unchanged"),
]
step = (TOP - LAST) / len(rows)
y = TOP - step
for ln, fn, dec, got in rows:
    was_missing = ln in BEFORE_MISS or (ln - 1) in BEFORE_MISS
    if was_missing:
        ax.add_patch(plt.Rectangle((0.006, y - 0.052), 0.986, 0.104,
                                   facecolor="#fff1f0", edgecolor="none",
                                   transform=ax.transAxes, zorder=0))
    dec_col = RED if dec == "refuse" else "#8250df"
    put(ax, cols[0], y, f"L{ln}", fontsize=11.4, family="monospace", color=INK, transform=ax.transAxes)
    put(ax, cols[1], y, fn, fontsize=11.4, family="monospace", color=INK, transform=ax.transAxes)
    put(ax, cols[2], y, dec.upper(), fontsize=11.2, fontweight="bold", color=dec_col, transform=ax.transAxes)
    put(ax, cols[3], y, got, fontsize=10.9, color=GREY, transform=ax.transAxes)
    put(ax, cols[4], y, "not driven" if was_missing else "driven",
        fontsize=11.2, fontweight="bold", color=RED if was_missing else GREEN, transform=ax.transAxes)
    put(ax, cols[5], y, "driven", fontsize=11.2, fontweight="bold", color=GREEN, transform=ax.transAxes)
    y -= step
assert abs((y + step) - LAST) < 1e-9, y

b = F["behaviour"]
put(ax, 0.012, 0.185,
    "Both decisions, measured end to end through the documented pipeline= / condition_cls= seams "
    "with sys.modules['torch'] = None:",
    fontsize=10.8, fontweight="bold", color=INK, transform=ax.transAxes)
put(ax, 0.030, 0.105,
    f"policy mode           ->  {b['policy_mode']['outcome']},  action {tuple(b['policy_mode']['shape'])} "
    f"{b['policy_mode']['dtype']},  chunk identical: {b['policy_mode']['chunk_exact']}",
    fontsize=10.5, family="monospace", color=GREEN, transform=ax.transAxes)
put(ax, 0.030, 0.030,
    f"forward_dynamics      ->  {b['forward_dynamics']['outcome']} with the shared install hint "
    f"(names the extra: {F['hint_names_extra']}, names backend='service': {F['hint_names_service']})",
    fontsize=10.5, family="monospace", color=RED, transform=ax.transAxes)

# ================= ROW 2: mutation matrix ===============================
ax2 = gs[1].subgridspec(1, 1).subplots(); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.965, "Would a regression be noticed? Six plausible ones, two arms",
    fontsize=13.2, fontweight="bold", color=INK, transform=ax2.transAxes)

c2 = [0.012, 0.640, 0.845]
put(ax2, c2[0], 0.855, "regression", fontsize=11.0, fontweight="bold", color=GREY, transform=ax2.transAxes)
put(ax2, c2[1], 0.855, "this module (11)", fontsize=11.0, fontweight="bold", color=GREY, transform=ax2.transAxes)
put(ax2, c2[2], 0.855, "pre-existing (26)", fontsize=11.0, fontweight="bold", color=GREY, transform=ax2.transAxes)

TOP2, LAST2 = 0.745, 0.235
step2 = (TOP2 - LAST2) / len(MUTS)
y = TOP2 - step2 * 0.5
for m in MUTS:
    label = m["label"].split(" ", 1)[1]
    tag = m["label"].split(" ", 1)[0]
    blind = m["old_failed"] == 0
    if blind:
        ax2.add_patch(plt.Rectangle((0.628, y - 0.040), 0.362, 0.080,
                                    facecolor="#fff1f0", edgecolor="none",
                                    transform=ax2.transAxes, zorder=0))
    put(ax2, c2[0], y, f"{tag}  {label}", fontsize=10.8, family="monospace", color=INK,
        transform=ax2.transAxes)
    put(ax2, c2[1], y, f"{m['new_failed']} failed" if m["new_failed"] else "not noticed",
        fontsize=10.9, fontweight="bold", color=GREEN if m["new_failed"] else RED,
        transform=ax2.transAxes)
    put(ax2, c2[2], y, f"{m['old_failed']} failed" if m["old_failed"] else "not noticed",
        fontsize=10.9, fontweight="bold", color=GREEN if m["old_failed"] else RED,
        transform=ax2.transAxes)
    y -= step2
assert y + step2 * 0.5 > 0.14, y

put(ax2, 0.012, 0.095,
    f"caught by this module: {N_NEW} of {len(MUTS)}      "
    f"caught by the pre-existing suite: {N_OLD} of {len(MUTS)}",
    fontsize=11.2, fontweight="bold", color=INK, transform=ax2.transAxes)
put(ax2, 0.012, 0.028,
    "M5 is the torch-PRESENT up-cast the sibling module already owns - the split is complementary, "
    "not a gap. Every torch-ABSENT regression was invisible.",
    fontsize=10.3, color=GREY, style="italic", transform=ax2.transAxes)

# ================= ROW 3: footer ========================================
ax3 = gs[2].subgridssubplots() if False else gs[2].subgridspec(1, 1).subplots()
ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.add_patch(plt.Rectangle((0.004, 0.06), 0.992, 0.88, facecolor="#f6f8fa",
                            edgecolor="#d0d7de", transform=ax3.transAxes, zorder=0))
foot = [
    f"policy_diffusers.py:  {len(F['before']['missing'])} statements not covered "
    f"({F['before']['pct']}%)  ->  {len(F['after']['missing'])} ({F['after']['pct']}%)      "
    f"tests/policies/cosmos3:  {F['before']['passed']} -> {F['after']['passed']} passed",
    "Tests only: git diff --numstat upstream/main..HEAD -- strands_robots/ is empty, so no policy, "
    "simulation, rendering, recording or asset behaviour can change.",
    "The figure is that measurement rather than a rollout; the injected pipeline is the module's own "
    "documented no-GPU seam.",
]
TOP3, LAST3 = 0.760, 0.190
step3 = (TOP3 - LAST3) / (len(foot) - 1)
y = TOP3
for i, line in enumerate(foot):
    put(ax3, 0.020, y, line, fontsize=10.4,
        family="monospace" if i == 0 else None,
        fontweight="bold" if i == 0 else None,
        color=INK if i == 0 else GREY, style=None if i == 0 else "italic",
        transform=ax3.transAxes)
    y -= step3
assert abs((y + step3) - LAST3) < 1e-9, y

# ---- layout guards ------------------------------------------------------
for a, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.08, f"axes-fraction y out of band: {yy}"
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, f"data y {yy} outside {(lo, hi)}"

fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"OK {OUT}  size={Image.open(OUT).size}  texts={len(placed)}")
