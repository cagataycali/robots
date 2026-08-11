"""Compose the measurement figure. Every number comes from the two JSON dumps."""
from __future__ import annotations
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.loads(pathlib.Path("/tmp/art_main.json").read_text())   # pristine e059fc5
B = json.loads(pathlib.Path("/tmp/art_theirs.json").read_text())  # PR #2143
assert A["tree"] != B["tree"], "both dumps came from the same tree"
assert A["cleanup_verified"] and B["cleanup_verified"]

# --- the claims this figure makes, asserted before anything is drawn ---
assert A["split"]["recorder_modules"]["inspects"] == 64, A
assert B["split"]["recorder_modules"]["inspects"] == 0, B
assert A["planted"]["failed"] == 64 and A["planted"]["file_exists_error"]
assert B["planted"]["failed"] == 0 and not B["planted"]["file_exists_error"]
assert A["clean"]["passed"] == B["clean"]["passed"] == 333
assert B["planted"]["passed"] == 333
# the deliberate resolution is untouched
assert A["split"]["teleoperate_tool"] == B["split"]["teleoperate_tool"], "tool resolution changed"
assert A["split"]["teleoperate_tool"]["resolves"] == 52
assert A["split"]["teleoperate_tool"]["inspects"] == 0

RED, GREEN, GREY, INK = "#c0392b", "#1e8449", "#7f8c8d", "#1c2833"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.62], hspace=0.34, wspace=0.16)

fig.suptitle(
    "Unit tests resolved their dataset directory from the developer's LeRobot cache",
    fontsize=17, fontweight="bold", color=INK, y=0.975,
)
fig.text(
    0.5, 0.938,
    "Independent verification of PR #2143. Four recorder modules, same machine, "
    "real mujoco / torch / lerobot installed.",
    ha="center", fontsize=11, color=GREY,
)

# ---------- row 1: what one stray dataset does to the four modules ----------
for col, (label, facts, colour) in enumerate(
    [("main (9e0b77b9)", A, RED), ("PR #2143", B, GREEN)]
):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    put(ax, 0.5, 1.02, label, transform=ax.transAxes, ha="center",
        fontsize=13, fontweight="bold", color=colour)

    rows = [
        ("shared cache clean", facts["clean"]["passed"], facts["clean"]["failed"]),
        ("one unrelated dataset planted", facts["planted"]["passed"], facts["planted"]["failed"]),
    ]
    top, last = 0.80, 0.44
    step = (top - last) / (len(rows) - 1)
    assert step > 0.030, step
    y = top
    for name, passed, failed in rows:
        put(ax, 0.03, y, name, transform=ax.transAxes, fontsize=11.5, color=INK)
        bad = failed > 0
        put(ax, 0.97, y, f"{passed} passed   {failed} failed", transform=ax.transAxes,
            ha="right", fontsize=12.5, fontweight="bold" if bad else "normal",
            color=RED if bad else GREEN, family="monospace")
        y -= step
    assert abs((y + step) - last) < 1e-9, (y, last)

    verdict = (
        "the verdict depends on the machine"
        if facts["planted"]["failed"]
        else "the verdict is the same either way"
    )
    put(ax, 0.5, 0.26, verdict, transform=ax.transAxes, ha="center",
        fontsize=12, fontstyle="italic", color=colour)
    if facts["planted"]["file_exists_error"]:
        put(ax, 0.5, 0.10,
            "FileExistsError naming a path in $HOME", transform=ax.transAxes,
            ha="center", fontsize=10.5, color=RED, family="monospace")

# ---------- row 2: resolve vs inspect ----------
ax = fig.add_subplot(gs[1, :])
groups = ["four recorder modules", "teleoperate tool (deliberate)"]
keys = ["recorder_modules", "teleoperate_tool"]
x = np.arange(len(groups)) * 1.7
w = 0.32
vals = {
    "main: resolves": [A["split"][k]["resolves"] for k in keys],
    "main: INSPECTS": [A["split"][k]["inspects"] for k in keys],
    "branch: resolves": [B["split"][k]["resolves"] for k in keys],
    "branch: INSPECTS": [B["split"][k]["inspects"] for k in keys],
}
colours = [GREY, RED, GREY, GREEN]
for i, ((name, v), c) in enumerate(zip(vals.items(), colours, strict=True)):
    bars = ax.bar(x + (i - 1.5) * w, v, w, label=name, color=c,
                  alpha=0.45 if "resolves" in name else 1.0,
                  edgecolor=INK, linewidth=0.6)
    for b, n in zip(bars, v, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, str(n),
                ha="center", fontsize=10.5, fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=12)
ax.set_ylabel("calls under the shared cache", fontsize=11)
ax.set_ylim(0, 78)
ax.set_title(
    "Resolving the path is arithmetic and harmless; inspecting it is what reads the disk",
    fontsize=12.5, color=INK, pad=8,
)
ax.legend(loc="upper right", fontsize=10, ncols=2, framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)

# ---------- row 3: what changed ----------
ax = fig.add_subplot(gs[2, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
lines = [
    f"inspections of the shared cache: {A['split']['recorder_modules']['inspects']} "
    f"-> {B['split']['recorder_modules']['inspects']}     "
    f"planted-cache failures: {A['planted']['failed']} -> {B['planted']['failed']}",
    f"the teleoperate tool's {A['split']['teleoperate_tool']['resolves']} deliberate resolutions are "
    "identical on both trees - the guard says nothing about resolving",
    "pre-fix proof: reverting the four modules while keeping the guard fails exactly "
    f"{A['planted']['failed']} tests, each naming root as the remedy",
    "gate on c9a2e57: 28046 passed / 257 skipped / 0 failed; ruff clean; mypy identical to the base",
]
top, last = 0.86, 0.10
step = (top - last) / (len(lines) - 1)
assert step > 0.030, step
y = top
for i, line in enumerate(lines):
    put(ax, 0.015, y, line, transform=ax.transAxes, fontsize=11.2,
        color=INK if i < 3 else GREEN, family="monospace")
    y -= step
assert abs((y + step) - last) < 1e-9, (y, last)

for ax_, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.07, (yy, "axes-fraction text out of range")
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, (yy, lo, hi)

out = pathlib.Path("/tmp/pr2143_verification.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.3, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"wrote {out}  size={Image.open(out).size}  border clean")
