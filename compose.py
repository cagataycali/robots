from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

A = json.load(open("/tmp/art_main.json")); B = json.load(open("/tmp/art_branch.json"))
assert A["tree"] != B["tree"], "before/after came from one tree"
assert A["divergence"] == B["divergence"], "the recursion must be untouched"
DIV = A["divergence"]; H = A["horizons"]

# --- claims asserted before anything is drawn ---
flipped = [l for l in A["probe"] if A["verdicts"][l]["outcome"] != B["verdicts"][l]["outcome"]]
assert len(flipped) == 10, len(flipped)
assert all(A["verdicts"][l]["outcome"] == "accepted" for l in flipped)
assert all(B["verdicts"][l]["outcome"] == "refused" for l in flipped)
assert DIV["1.5"][1] > 200 and DIV["1.5"][-1] > 1e16
assert DIV["1000000.0"][-1] == float("inf")
assert DIV["-2.0"][-1] > 1e27 and DIV["-0.5"][-1] == 1.0
assert A["run"]["w_sum"] == B["run"]["w_sum"] and A["run"]["w_absmax"] == B["run"]["w_absmax"]
assert A["run"]["status"] == B["run"]["status"] == "success"
assert A["sac_quiet"] == B["sac_quiet"] == {"1.5": [], "nan": []}

placed: list[tuple] = []
def put(ax, x, y, s, coords="axes", **kw):
    kw.setdefault("transform", ax.transAxes if coords == "axes" else ax.transData)
    placed.append((ax, y, coords)); return ax.text(x, y, s, **kw)

GREEN, RED, INK, GREY = "#137333", "#b3261e", "#16191d", "#5f6368"
fig = plt.figure(figsize=(15.6, 11.0), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.06, 1.0], width_ratios=[1.30, 1.0],
                      hspace=0.30, wspace=0.16, left=0.062, right=0.982, top=0.905, bottom=0.048)

fig.suptitle("PPO: the advantage trace decays by gamma * lam, so bounding gamma alone does not bound it",
             fontsize=15.5, fontweight="bold", color=INK, y=0.973)
fig.text(0.5, 0.936, "gamma is fixed at 0.99 throughout - inside the closed interval [0, 1] its own preflight "
                     "accepts - so every divergence below is reachable on a spec that gate passes",
         ha="center", fontsize=10.6, color=GREY, style="italic")

# ---- row 1: the divergence (identical on both trees) ----
ax = fig.add_subplot(gs[0, :])
STYLE = {  # lam -> (colour, dash, accepted?)
    "0.0": (GREEN, "-", True), "0.95": (GREEN, "-", True), "1.0": (GREEN, "-", True),
    "1.5": (RED, "--", False), "2.0": (RED, "--", False), "1000000.0": (RED, ":", False),
    "-0.5": ("#8a6d00", "-.", False), "-2.0": (RED, "-.", False),
}
LABEL = {"0.0": "lam = 0    (TD(0), one-step)", "0.95": "lam = 0.95 (the default)",
         "1.0": "lam = 1    (Monte-Carlo)", "1.5": "lam = 1.5   -> decay 1.485",
         "2.0": "lam = 2     -> decay 1.980", "1000000.0": "lam = 1e6   -> decay 990000",
         "-0.5": "lam = -0.5  -> trace collapses", "-2.0": "lam = -2    -> decay |-1.980|"}
for l, (c, ls, ok) in STYLE.items():
    ys = [v if np.isfinite(v) else 1e33 for v in DIV[l]]
    ax.plot(H, ys, ls, color=c, lw=3.0 if ok else 2.3, marker="o" if ok else "^",
            ms=7 if ok else 6, label=LABEL[l], alpha=1.0 if ok else 0.92, zorder=4 if ok else 3)
ax.axhspan(0.5, 1e3, color=GREEN, alpha=0.055, zorder=0)
put(ax, 0.014, 0.09, "the accepted domain stays bounded in the horizon", color=GREEN,
    fontsize=10.4, fontweight="bold")
ax.set_yscale("log"); ax.set_ylim(0.4, 4e33)
ax.set_xlabel("rollout horizon T (steps)", fontsize=11.6)
ax.set_ylabel("largest |advantage| out of compute_gae\n(rollout of unit rewards)", fontsize=11.2)
ax.set_title("Measured on this backend's own compute_gae - byte-identical on both trees, the recursion is untouched",
             fontsize=12.0, color=INK, pad=9)
ax.grid(alpha=0.30, which="both", ls=":"); ax.set_xticks(H); ax.set_xticklabels([str(h) for h in H])
ax.legend(loc="upper left", fontsize=9.6, ncol=2, framealpha=0.94)
ax.annotate(f"{DIV['1.5'][-1]:.2g}", xy=(H[-1], DIV["1.5"][-1]), xytext=(-58, 14),
            textcoords="offset points", color=RED, fontsize=10.4, fontweight="bold")
ax.annotate("inf", xy=(H[1], 1e33), xytext=(-6, -26), textcoords="offset points",
            color=RED, fontsize=10.4, fontweight="bold")

# ---- row 2 left: verdict table ----
axv = fig.add_subplot(gs[1, 0]); axv.axis("off"); axv.set_xlim(0, 1); axv.set_ylim(0, 1)
axv.set_title("What PpoTrainer.validate() reports for spec.lam", fontsize=12.2, color=INK, pad=8, loc="left")
put(axv, 0.005, 0.945, "lam", fontsize=10.4, fontweight="bold", color=GREY)
put(axv, 0.175, 0.945, "main", fontsize=10.4, fontweight="bold", color=GREY)
put(axv, 0.320, 0.945, "this change", fontsize=10.4, fontweight="bold", color=GREY)
put(axv, 0.545, 0.945, "consequence on main", fontsize=10.4, fontweight="bold", color=GREY)
CONS = {"0.0": "honored - TD(0)", "0.95": "honored - the default", "1.0": "honored - Monte-Carlo",
        "1.5": "advantage 6.3e+16 by T=96", "2.0": "advantage 3.1e+28 by T=96",
        "-0.5": "trace stops accumulating", "-2.0": "advantage 1.0e+28 by T=96",
        "1000000.0": "advantage overflows to inf", "True": "silent lam of 1 (wrong estimator)",
        "nan": "every advantage non-finite", "inf": "every advantage non-finite",
        "'0.95'": "TypeError from the recursion", "None": "TypeError from the recursion"}
TOP, FLOOR, PAD = 0.905, 0.055, 0.006
rows = A["probe"]; step = (TOP - FLOOR - PAD * len(rows)) / len(rows)
assert step > 0.030, step
y = TOP
for l in rows:
    a, b = A["verdicts"][l]["outcome"], B["verdicts"][l]["outcome"]
    changed = a != b
    if changed:
        axv.add_patch(plt.Rectangle((0.0, y - step - 0.002), 1.0, step + 0.004,
                                    transform=axv.transAxes, color=RED, alpha=0.055, zorder=0))
    yc = y - step * 0.62
    put(axv, 0.005, yc, l, fontsize=10.3, family="monospace", color=INK, fontweight="bold")
    put(axv, 0.175, yc, a, fontsize=10.3, color=RED if changed else GREEN,
        fontweight="bold" if changed else "normal")
    put(axv, 0.320, yc, b, fontsize=10.3, color=GREEN, fontweight="bold" if changed else "normal")
    put(axv, 0.545, yc, CONS[l], fontsize=9.9, color=GREY)
    y -= step + PAD
assert y > 0.040, y
put(axv, 0.005, 0.014, f"{len(flipped)} of {len(rows)} probed values change from accepted to refused; "
    "the 3 usable ones are untouched", fontsize=10.0, color=INK, fontweight="bold")

# ---- row 2 right: no-regression ledger ----
axr = fig.add_subplot(gs[1, 1]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
axr.set_title("A run whose lam the trace can decay by is untouched", fontsize=12.2, color=INK, pad=8, loc="left")
r = [("real PPO run (so100 reach, 3 iters, seed 0)", "main", "this change"),
     ("validate() problems", str(A["run"]["validate_problems"]), str(B["run"]["validate_problems"])),
     ("spec.lam", f"{A['run']['lam']}", f"{B['run']['lam']}"),
     ("run status", A["run"]["status"], B["run"]["status"]),
     ("checkpoint parameters", f"{A['run']['n_params']:,}", f"{B['run']['n_params']:,}"),
     ("max |parameter|", A["run"]["w_absmax"], B["run"]["w_absmax"]),
     ("sum of parameters", A["run"]["w_sum"], B["run"]["w_sum"]),
     ("fast_sac problems about lam=1.5", "none", "none"),
     ("fast_sac problems about lam=nan", "none", "none")]
y2 = 0.925
for i, (k, a, b) in enumerate(r):
    bold = i == 0
    put(axr, 0.005, y2, k, fontsize=10.4 if bold else 10.0,
        fontweight="bold" if bold else "normal", color=GREY if bold else INK)
    put(axr, 0.545, y2, a, fontsize=10.2 if bold else 9.6, family="monospace" if not bold else None,
        fontweight="bold" if bold else "normal", color=GREY if bold else INK)
    put(axr, 0.795, y2, b, fontsize=10.2 if bold else 9.6, family="monospace" if not bold else None,
        fontweight="bold" if bold else "normal", color=GREY if bold else GREEN)
    y2 -= 0.070 if bold else 0.062
    if bold: axr.axhline(y2 + 0.030, xmin=0.0, xmax=1.0, color=GREY, lw=0.7, alpha=0.5)
assert y2 > 0.20, y2
axr.add_patch(plt.Rectangle((0.0, 0.075), 1.0, 0.145, transform=axr.transAxes,
                            color=GREEN, alpha=0.075, zorder=0))
put(axr, 0.018, 0.170, "Bit-identical to 16 digits over 34,715 parameters:", fontsize=10.5,
    fontweight="bold", color=GREEN)
put(axr, 0.018, 0.118, "the gate reports, it does not change what a usable spec trains.",
    fontsize=10.2, color=INK)
put(axr, 0.005, 0.022, "FastSAC bootstraps a target-Q rather than a trace, so it never reads lam\n"
    "and must not report on it - which is why this is a separate gate.", fontsize=9.7,
    color=GREY, style="italic")

# ---- layout guards ----
for ax_, yy, coords in placed:
    if coords == "axes":
        assert -0.03 <= yy <= 1.07, f"axes-fraction y={yy} outside the panel"
    else:
        lo, hi = ax_.get_ylim(); assert lo <= yy <= hi, f"data y={yy} outside {(lo, hi)}"

out = pathlib.Path("/tmp/gae_lambda_domain.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.array(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  {out.stat().st_size/1024:.0f} KB")
