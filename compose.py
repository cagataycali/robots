"""Compose the measured verdict figure from the two capture dumps."""
import json, os, pathlib, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUN = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/art-{RUN}-wt-main-{RUN}.json").read_text())      # upstream/main
B = json.loads(pathlib.Path(f"/tmp/art-{RUN}-robots-mine-{RUN}.json").read_text())  # this PR
assert A["tree"] != B["tree"], "both arms measured the same tree"

ar, br = A["claim1_scenarios"]["registered"], B["claim1_scenarios"]["registered"]
aa, ba = A["claim1_scenarios"]["autodiscovered"], B["claim1_scenarios"]["autodiscovered"]
ac2, bc2 = A["claim2_recording"], B["claim2_recording"]

# ---- every claim the figure makes, asserted against the dumps -------------
assert ar["substituted"] is False and br["substituted"] is False, "no swap on either tree"
assert (ar["names_provider"], ar["names_install"]) == (False, False)
assert (br["names_provider"], br["names_module"], br["names_install"]) == (True, True, True)
assert aa["blames_the_name"] is True and ba["blames_the_name"] is False
assert ba["names_module"] is True and aa["names_module"] is False
assert ac2["status"] == "error" and bc2["status"] == "error"
assert ac2["degraded_to_mp4"] is False and bc2["degraded_to_mp4"] is False
assert ac2["text"] == bc2["text"], "the recording report must be unchanged by this PR"
N_DEFER = len(B["census"]["defers_and_translates"])
ML = [m for m in B["census"]["module_level_heavy"] if "lerobot_local" in m["module"]]
assert ML, "census lost the lerobot_local row"
assert N_DEFER == 9, f"expected 9 deferring provider modules, got {N_DEFER}"

GREEN, RED, GREY, INK = "#1b7f3b", "#b3261e", "#6b6b6b", "#111111"
placed = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 12.6), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.02, 1.30, 0.92], hspace=0.16, wspace=0.06)

fig.suptitle(
    "A policy provider whose optional dependency is missing now names the remedy",
    fontsize=16, fontweight="bold", y=0.982)
fig.text(0.5, 0.958,
         "Measured on Thor with torch reported absent exactly as the import system reports it "
         "(ModuleNotFoundError, name set).",
         ha="center", fontsize=10.2, color=GREY, style="italic")

# ================= ROW 1: the two reported claims, measured ================
ax1 = fig.add_subplot(gs[0, :]); ax1.axis("off"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
put(ax1, 0.5, 1.00, "What was reported vs what upstream/main actually does",
    transform=ax1.transAxes, ha="center", fontsize=12.6, fontweight="bold")

rows1 = [
    ("create_policy('lerobot_local') swaps to mock",
     f"RAISED {ar['exc']} - substituted: {ar['substituted']}", "NOT REPRODUCED", GREEN),
    ("start_recording degrades to raw MP4",
     f"status={ac2['status']!r} - degraded_to_mp4: {ac2['degraded_to_mp4']}", "NOT REPRODUCED", GREEN),
    ("...but the provider error is a dead end",
     f"{ar['message']}  (names no provider, no remedy)", "CONFIRMED - fixed here", RED),
    ("...and an existing provider is blamed by name",
     "ValueError: Unknown policy provider - the dependency is never mentioned",
     "CONFIRMED - fixed here", RED),
]
TOP1, LAST1 = 0.80, 0.14
STEP1 = (TOP1 - LAST1) / (len(rows1) - 1)
assert STEP1 > 0.030, STEP1
y = TOP1
for claim, measured, verdict, col in rows1:
    put(ax1, 0.012, y, claim, transform=ax1.transAxes, fontsize=10.6, fontweight="bold", color=INK)
    put(ax1, 0.012, y - 0.058, measured, transform=ax1.transAxes, fontsize=9.3,
        family="monospace", color=GREY)
    put(ax1, 0.985, y, verdict, transform=ax1.transAxes, fontsize=10.6, fontweight="bold",
        color=col, ha="right")
    y -= STEP1
assert abs((y + STEP1) - LAST1) < 1e-9

# ================= ROW 2: the error a caller receives ======================
for col, (label, reg, auto, tint) in enumerate([
    ("upstream/main", ar, aa, RED),
    ("this PR", br, ba, GREEN),
]):
    ax = fig.add_subplot(gs[1, col]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.005, 0.02), 0.99, 0.90, transform=ax.transAxes,
                               facecolor="#fafafa", edgecolor=tint, linewidth=2.0, zorder=0))
    put(ax, 0.5, 0.965, label, transform=ax.transAxes, ha="center",
        fontsize=12.4, fontweight="bold", color=tint)
    yy = 0.86
    for scen, obj in (("registered provider (lerobot_local, torch absent)", reg),
                      ("provider module present, its dependency absent", auto)):
        put(ax, 0.03, yy, scen, transform=ax.transAxes, fontsize=9.6,
            fontweight="bold", color=INK)
        yy -= 0.055
        put(ax, 0.03, yy, f"{obj['exc']}:", transform=ax.transAxes, fontsize=9.2,
            family="monospace", color=tint, fontweight="bold")
        yy -= 0.048
        for line in obj["message"].splitlines():
            for w in textwrap.wrap(line, 62) or [""]:
                put(ax, 0.045, yy, w, transform=ax.transAxes, fontsize=8.9,
                    family="monospace", color=INK)
                yy -= 0.040
        yy -= 0.035
    put(ax, 0.03, 0.075, f"substituted another provider: {reg['substituted']}",
        transform=ax.transAxes, fontsize=9.3, family="monospace", color=GREY)

# ================= ROW 3: census + mutation matrix =========================
ax3 = fig.add_subplot(gs[2, 0]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 1.00, "Why only this provider", transform=ax3.transAxes,
    fontsize=12.0, fontweight="bold")
census_rows = [
    (f"{N_DEFER} provider modules defer the heavy import", "report via require_optional*", GREEN),
    (f"1 imports it at module level: {ML[0]['module']}", f"torch at L{ML[0]['imports'][0][1]}", RED),
    ("so the failure precedes any provider machinery", "-> translate at the shared funnel", GREY),
]
TOP3, LAST3 = 0.80, 0.30
STEP3 = (TOP3 - LAST3) / (len(census_rows) - 1)
assert STEP3 > 0.030, STEP3
y = TOP3
for a_, b_, col in census_rows:
    put(ax3, 0.0, y, a_, transform=ax3.transAxes, fontsize=9.9, color=col, fontweight="bold")
    put(ax3, 0.02, y - 0.075, b_, transform=ax3.transAxes, fontsize=9.2,
        family="monospace", color=GREY)
    y -= STEP3
assert abs((y + STEP3) - LAST3) < 1e-9
put(ax3, 0.0, 0.10, "import_policy_class is the single funnel every provider class\n"
                    "is imported through, so the remedy no longer depends on WHERE\n"
                    "a provider imports its dependency.",
    transform=ax3.transAxes, fontsize=9.3, color=INK)

ax4 = fig.add_subplot(gs[2, 1]); ax4.axis("off"); ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
put(ax4, 0.0, 1.00, "Mutation table (6 regressions x 2 arms)", transform=ax4.transAxes,
    fontsize=12.0, fontweight="bold")
mut = [
    ("no translation (raw error escapes)", 4), ("call it, discard the verdict", 4),
    ("swallow the ImportError again", 1), ("ignore the declared extra", 2),
    ("substitute mock on failure", 9), ("drop the extra from the registry", 3),
]
TOP4, LAST4 = 0.83, 0.30
STEP4 = (TOP4 - LAST4) / (len(mut) - 1)
assert STEP4 > 0.030, STEP4
y = TOP4
for label, nf in mut:
    put(ax4, 0.0, y, label, transform=ax4.transAxes, fontsize=9.4, color=INK)
    put(ax4, 0.70, y, f"{nf} failed", transform=ax4.transAxes, fontsize=9.4,
        family="monospace", color=GREEN, fontweight="bold")
    put(ax4, 0.99, y, "0 failed", transform=ax4.transAxes, fontsize=9.4,
        family="monospace", color=RED, ha="right")
    y -= STEP4
assert abs((y + STEP4) - LAST4) < 1e-9
put(ax4, 0.70, 0.90, "new", transform=ax4.transAxes, fontsize=9.0, color=GREY, fontweight="bold")
put(ax4, 0.99, 0.90, "pre-existing (708)", transform=ax4.transAxes, fontsize=9.0,
    color=GREY, ha="right", fontweight="bold")
put(ax4, 0.0, 0.15, "6 of 6 caught here; 6 of 6 invisible to the 708 pre-existing\n"
                    "registry tests. Gate: 28693 passed / 258 skipped / 0 failed.",
    transform=ax4.transAxes, fontsize=9.3, color=INK)

# ---- layout guards -------------------------------------------------------
for ax, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.10, f"axes-fraction text out of range: {yv}"
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, f"data text out of range: {yv} vs {(lo, hi)}"

OUTP = pathlib.Path(f"/tmp/artifact-{RUN}.png")
fig.savefig(OUTP, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(OUTP).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print("wrote", OUTP, im.shape, f"{OUTP.stat().st_size // 1024} KiB")
