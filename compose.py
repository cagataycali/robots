"""Compose the RTC re-anchoring degradation figure from the two measured trees."""
import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

RUN = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/art-main-{RUN}.json").read_text())     # upstream/main
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RUN}.json").read_text())   # this PR
assert A["tree"] != B["tree"], "both arms measured the same tree"

ROWS_A = {r["label"]: r for r in A["rows"]}
ROWS_B = {r["label"]: r for r in B["rows"]}
LABELS = [r["label"] for r in A["rows"]]
DEGRADED = ["no postprocessor", "postprocessor yields a dict"]

# --- measured invariants asserted before anything is drawn -----------------
assert ROWS_A["healthy (postprocessor converts)"]["is_reanchored"]
assert ROWS_B["healthy (postprocessor converts)"]["is_reanchored"]
for lab in DEGRADED:
    assert ROWS_A[lab]["is_stale"] and ROWS_B[lab]["is_stale"], lab
    assert ROWS_A[lab]["n_warnings"] == 0, f"main already reported {lab}"
    assert ROWS_B[lab]["n_warnings"] == 1, f"PR did not report {lab} exactly once"
    assert ROWS_A[lab]["info_enabled"] and ROWS_B[lab]["info_enabled"], lab
BENIGN = "absolute-action policy (benign)"
assert ROWS_A[BENIGN]["n_warnings"] == 0 and ROWS_B[BENIGN]["n_warnings"] == 0
N_SILENT_MAIN = sum(1 for lab in DEGRADED if ROWS_A[lab]["n_warnings"] == 0)
N_SILENT_PR = sum(1 for lab in DEGRADED if ROWS_B[lab]["n_warnings"] == 0)
assert (N_SILENT_MAIN, N_SILENT_PR) == (2, 0)

healthy = ROWS_B["healthy (postprocessor converts)"]
stale = np.array(healthy["stale_model_space"], dtype=float).ravel()
reanch = np.array(healthy["expected_reanchored"], dtype=float).ravel()
fed_healthy = np.array(healthy["prefix"], dtype=float).ravel()
assert np.allclose(fed_healthy, reanch, atol=1e-4)
fed_degraded = np.array(ROWS_B[DEGRADED[0]]["prefix"], dtype=float).ravel()
assert np.allclose(fed_degraded, stale, atol=1e-4)
SHIFT = A["state_shift"]
frame_err = float(np.abs(reanch - stale).max())

GREEN, RED, GREY, BLUE = "#1a7f37", "#b3261e", "#57606a", "#0b62d6"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 12.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.0, 0.78], hspace=0.42,
                      left=0.055, right=0.975, top=0.925, bottom=0.035)

fig.suptitle(
    "A relative-action RTC leftover that cannot be re-anchored: what the model is fed, and what the operator is told",
    fontsize=14.5, fontweight="bold", y=0.972,
)
fig.text(0.5, 0.945,
         "strands_robots/policies/lerobot_local -- measured on upstream/main and on this branch. "
         "Every value below is read from the two runs' JSON dumps.",
         ha="center", fontsize=10, color=GREY)

# ---- row 1: the prefix actually handed to the denoiser --------------------
ax = fig.add_subplot(gs[0])
x = np.arange(len(stale))
ax.fill_between(x, stale, reanch, step="mid", color=RED, alpha=0.13,
                label=f"frame error carried into the next chunk (max {frame_err:.0f})")
ax.step(x, reanch, where="mid", color=GREEN, lw=2.4,
        label="re-anchored prefix -- healthy path (both trees, unchanged)")
ax.step(x, stale, where="mid", color=RED, lw=2.4, ls="--",
        label="stale model-space prefix -- both degraded fallbacks (both trees)")
ax.set_xticks(x)
ax.set_xticklabels([f"t{t}\nj{d}" for t in range(len(stale) // 4) for d in range(4)], fontsize=7.5)
ax.set_xlabel("unexecuted chunk tail: timestep x action dimension", fontsize=10)
ax.set_ylabel("prefix value fed as\nprev_chunk_left_over", fontsize=10)
ax.set_title("The robot state moved by " + str(SHIFT) + " between the two chunks, so the two prefixes differ by exactly that shift",
             fontsize=11, pad=8)
ax.grid(alpha=0.25, axis="y")
ax.legend(loc="upper left", fontsize=9.2, framealpha=0.95)

# ---- row 2: report matrix ------------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.045, "Was the outcome reported?  A degradation must be; the benign case must not.",
    transform=ax2.transAxes, fontsize=11.5, fontweight="bold")
COLS = [(0.315, "prefix fed"), (0.545, "upstream/main"), (0.775, "this PR")]
put(ax2, 0.0, 0.935, "fallback", transform=ax2.transAxes, fontsize=10, fontweight="bold", color=GREY)
for cx, name in COLS:
    put(ax2, cx, 0.935, name, transform=ax2.transAxes, fontsize=10, fontweight="bold",
        color=GREY, ha="center")

TOP, LAST = 0.80, 0.30
STEP = (TOP - LAST) / (len(LABELS) - 1)
assert STEP > 0.030, STEP
for i, lab in enumerate(LABELS):
    y = TOP - i * STEP
    degraded = lab in DEGRADED
    put(ax2, 0.0, y, lab, transform=ax2.transAxes, fontsize=10.2, va="center")
    put(ax2, 0.315, y, "STALE" if ROWS_B[lab]["is_stale"] else "re-anchored",
        transform=ax2.transAxes, fontsize=10, va="center", ha="center",
        color=RED if ROWS_B[lab]["is_stale"] else GREEN,
        fontweight="bold" if ROWS_B[lab]["is_stale"] else "normal")
    for cx, rows in ((0.545, ROWS_A), (0.775, ROWS_B)):
        n = rows[lab]["n_warnings"]
        ok = (n == 1) if degraded else (n == 0)
        txt = "silent" if n == 0 else f"reported x{n}"
        ax2.add_patch(Rectangle((cx - 0.088, y - 0.045), 0.176, 0.09,
                                transform=ax2.transAxes,
                                facecolor=(GREEN if ok else RED), alpha=0.16, lw=0))
        put(ax2, cx, y, txt, transform=ax2.transAxes, fontsize=10, va="center", ha="center",
            color=(GREEN if ok else RED), fontweight="bold")
    if degraded:
        put(ax2, 0.995, y, "degradation", transform=ax2.transAxes, fontsize=8.6, va="center",
            ha="right", color=GREY, style="italic")
assert abs((TOP - (len(LABELS) - 1) * STEP) - LAST) < 1e-9

put(ax2, 0.0, 0.155,
    f"Degradations left silent:  upstream/main {N_SILENT_MAIN} of {len(DEGRADED)}"
    f"    ->    this PR {N_SILENT_PR} of {len(DEGRADED)}",
    transform=ax2.transAxes, fontsize=11.5, fontweight="bold", color=BLUE)
put(ax2, 0.0, 0.055,
    "Both trees log \"re-anchoring enabled\" at load time for every row above, so on main the "
    "INFO line is the only signal and it is wrong.",
    transform=ax2.transAxes, fontsize=9.6, color=GREY)

# ---- row 3: the report, verbatim ----------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 0.99, "What this PR emits, once per policy (measured, verbatim):",
    transform=ax3.transAxes, fontsize=11.5, fontweight="bold")
lines: list[tuple[str, str]] = []
for lab in DEGRADED:
    msg = ROWS_B[lab]["warning"]
    assert "STALE coordinate frame" in msg and "chunk-seam prefix" in msg, lab
    head, _, tail = msg.partition("(")
    cause, _, rest = tail.partition(")")
    lines.append((lab, head.strip()))
    lines.append(("", f"({cause})" + rest.rstrip()))
T3, L3 = 0.80, 0.30
S3 = (T3 - L3) / (len(lines) - 1)
assert S3 > 0.030, S3
for i, (lab, text) in enumerate(lines):
    y = T3 - i * S3
    if lab:
        put(ax3, 0.0, y, lab, transform=ax3.transAxes, fontsize=9.4, va="center",
            color=RED, fontweight="bold")
    put(ax3, 0.235, y, text, transform=ax3.transAxes, fontsize=9.0, va="center",
        family="monospace", color="#111111")
put(ax3, 0.0, 0.135,
    "The benign absolute-action fallback stays silent, and a postprocessor that raises stays fatal "
    "-- both pinned as scope boundaries.",
    transform=ax3.transAxes, fontsize=9.6, color=GREY)
put(ax3, 0.0, 0.035,
    "Wording mirrors _resolve_rtc_rebase_steps, which already warns for this same consequence when "
    "LeRobot's re-anchor helper is missing.",
    transform=ax3.transAxes, fontsize=9.6, color=GREY)

for ax_obj, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.07, (y, "axes-fraction out of band")
    else:
        lo, hi = ax_obj.get_ylim()
        assert lo - 0.05 * (hi - lo) <= y <= hi + 0.10 * (hi - lo), (y, lo, hi)

OUT = pathlib.Path("_art/rtc_reanchor_degradation.png")
fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.3, facecolor="white")
plt.close(fig)

im = np.asarray(matplotlib.image.imread(OUT) * 255).astype(int)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"OK {OUT}  {im.shape[1]}x{im.shape[0]}  silent {N_SILENT_MAIN}->{N_SILENT_PR}  frame_err={frame_err:.0f}")
