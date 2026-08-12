import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ART = pathlib.Path("_art")
F = json.load(open(ART / "facts.json"))
assert F["tree"].startswith("/tmp/robots-mine"), F["tree"]
S, C = F["short"], F["conv"]
M = F["mutations"]

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("va", "top"); kw.setdefault("fontsize", 9.3)
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.4, 11.0), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.32, 1.0], hspace=0.16, wspace=0.05,
                      left=0.022, right=0.978, top=0.905, bottom=0.028)

fig.suptitle("move_to: the not-reached envelope is a retryable report, not a dead end",
             fontsize=16.5, fontweight="bold", y=0.972)
fig.text(0.5, 0.936,
         "MuJoCo headless (MUJOCO_GL=egl), the motion-primitive suite's inline arm. "
         "Tests only in this change: no production line moves, so all three frames are main's behaviour.",
         ha="center", fontsize=10.4, style="italic", color="#333")

PANELS = [
    ("01_home.png", "1. at rest", "no move_to issued yet", "#444"),
    ("02_not_reached.png", "2. move_to(..., tol=0.02, max_steps=2)",
     f'status=error   reached=False   steps={S["steps"]}\n'
     f'position_error_m={S["position_error_m"]:.4f}   ik_residual_m={S["ik_residual_m"]:.4f}\n'
     '"The servo may need more steps"  <- the IK solved it', "#b23"),
    ("03_converged.png", "3. the identical call, max_steps=400",
     f'status=success   reached=True   steps={C["steps"]}\n'
     f'position_error_m={C["position_error_m"]:.4f}   (tol 0.02)\n'
     "the advice in panel 2, followed", "#1a7a3c"),
]
for col, (fn, title, cap, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(np.asarray(Image.open(ART / fn).convert("RGB")))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.4)
    ax.set_title(title, fontsize=11.4, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(cap, fontsize=9.4, family="monospace", color="#222", labelpad=8)

# ---- lower left: what the agent reads -------------------------------------
axl = fig.add_subplot(gs[1, 0:2]); axl.axis("off")
axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.985, "What the refusal hands back (json details block)",
    transform=axl.transAxes, fontsize=12.2, fontweight="bold")

ROWS = [
    ("field", "panel 2  (max_steps=2)", "panel 3  (max_steps=400)", True),
    ("status", "error", "success", False),
    ("reached", "False", "True", False),
    ("steps", str(S["steps"]), str(C["steps"]), False),
    ("position_error_m", f'{S["position_error_m"]:.4f}', f'{C["position_error_m"]:.4f}', False),
    ("ik_residual_m", f'{S["ik_residual_m"]:.4f}', f'{C["ik_residual_m"]:.4f}', False),
    ("frame / frame_type", f'{S["frame"]} / {S["frame_type"]}', f'{C["frame"]} / {C["frame_type"]}', False),
    ("payload key count", str(len([k for k in S if k not in ("status", "text")])),
     str(len([k for k in C if k not in ("status", "text")])), False),
]
TOP, LAST = 0.905, 0.400
step = (TOP - LAST) / (len(ROWS) - 1)
assert step > 0.030, step
y = TOP
for name, a, b, hdr in ROWS:
    w = "bold" if hdr else "normal"
    put(axl, 0.005, y, name, transform=axl.transAxes, family="monospace", fontweight=w, fontsize=9.9)
    put(axl, 0.335, y, a, transform=axl.transAxes, family="monospace", fontweight=w, fontsize=9.9,
        color="#b23" if not hdr else "#000")
    put(axl, 0.660, y, b, transform=axl.transAxes, family="monospace", fontweight=w, fontsize=9.9,
        color="#1a7a3c" if not hdr else "#000")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

NOTES = [
    "Both halves carry the SAME payload keys, so one reader shape serves success and failure.",
    "The two residuals stay separate fields: a small ik_residual_m beside a large position_error_m",
    "says the pose is solvable and only the budget was short - the distinction a retry depends on.",
    "Distinct from the pre-flight unreachable refusal, which fires before any tick and reports",
    "no position_error_m at all; that branch already had a test, this one had none.",
]
NTOP, NLAST = 0.330, 0.045
nstep = (NTOP - NLAST) / (len(NOTES) - 1)
assert nstep > 0.030, nstep
y = NTOP
for line in NOTES:
    put(axl, 0.005, y, line, transform=axl.transAxes, fontsize=9.5, color="#333")
    y -= nstep
assert abs((y + nstep) - NLAST) < 1e-9, (y, NLAST)

# ---- lower right: coverage + mutation ------------------------------------
axr = fig.add_subplot(gs[1, 2]); axr.axis("off")
axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 0.985, "Measurement", transform=axr.transAxes, fontsize=12.2, fontweight="bold")

cov = F["coverage"]
n_caught = sum(1 for r in M["rows"] if r["new_failed"] > 0)
n_blind = sum(1 for r in M["rows"] if r["old_failed"] == 0)
assert (n_caught, len(M["rows"])) == (7, 7), (n_caught, len(M["rows"]))
assert n_blind == 5, n_blind
LINES = [
    ("motion_primitives_base.py", True, "#000"),
    (f'  {cov["stmts"]} statements   {cov["before_pct"]}% -> {cov["after_pct"]}%', False, "#1a7a3c"),
    (f'  missing {cov["before_missing"]} -> {cov["after_missing"]}', False, "#1a7a3c"),
    ("", False, "#000"),
    ("Mutation table (7 regressions)", True, "#000"),
    (f'  caught by the new cases:      {n_caught} of 7', False, "#1a7a3c"),
    (f'  invisible to the 232 existing: {n_blind} of 7', False, "#b23"),
    ("  the 2 the existing suite sees are", False, "#555"),
    ("  the ones that also break success.", False, "#555"),
    ("", False, "#000"),
    ("Suite", True, "#000"),
    (f'  base    {F["suite"]["base"]} passed', False, "#333"),
    (f'  branch  {F["suite"]["branch"]} passed  (+{F["suite"]["new_cases"]})', False, "#1a7a3c"),
    ("  0 skipped-to-failed, 0 production lines", False, "#333"),
]
RTOP, RLAST = 0.905, 0.055
rstep = (RTOP - RLAST) / (len(LINES) - 1)
assert rstep > 0.030, rstep
y = RTOP
for txt, bold, colour in LINES:
    if txt:
        put(axr, 0.005, y, txt, transform=axr.transAxes, family="monospace", fontsize=9.5,
            fontweight="bold" if bold else "normal", color=colour)
    y -= rstep
assert abs((y + rstep) - RLAST) < 1e-9, (y, RLAST)

for ax, yv, is_axes in placed:
    if is_axes:
        assert -0.03 <= yv <= 1.07, (yv, "axes-fraction out of band")
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, (yv, lo, hi)

out = ART / "move_to_not_reached_envelope.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB"), dtype=np.int16)
h, w, _ = im.shape
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {w}x{h}  clean border")
print(f"diffs: home-vs-2step {F['diff_home_short']*100:.2f}%  2step-vs-converged {F['diff_short_conv']*100:.2f}%  "
      f"home-vs-converged {F['diff_home_conv']*100:.2f}%")
