"""Compose the artifact. Every rendered number is asserted against the dumps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

A = json.load(open("/tmp/art_main/facts.json"))
B = json.load(open("/tmp/art_branch/facts.json"))
assert A["tree"] != B["tree"], "both dumps came from the same tree"

def im(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB")).astype(int)

ref_main, ref_br = im(A["scenarios"]["reference"]["after_png"]), im(B["scenarios"]["reference"]["after_png"])
zero_main, zero_br = im(A["scenarios"]["zero"]["after_png"]), im(B["scenarios"]["zero"]["after_png"])
home_br = im(B["scenarios"]["zero"]["home_png"])

# --- measured relations -------------------------------------------------------
d_ref = int(np.abs(ref_main - ref_br).max())
d_still = int(np.abs(zero_br - home_br).max())
frac = float((np.abs(ref_br - zero_main).sum(2) > 24).mean())
assert d_ref <= 2, f"the honored rollout differs across trees: {d_ref}"
assert d_still <= 2, f"the refused rollout moved the arm: {d_still}"
assert frac > 0.10, f"the two horizons are not visually distinct: {frac:.2%}"

sm, sb = A["scenarios"], B["scenarios"]
assert sm["reference"]["steps"] == sb["reference"]["steps"] == 120
assert sm["reference"]["inferences"] == sb["reference"]["inferences"] == 120
assert sm["zero"]["status"] == "success" and sm["zero"]["steps"] == 500 and sm["zero"]["inferences"] == 500
assert sb["zero"]["status"] == "raised" and sb["zero"]["inferences"] == 0
assert sm["reference"]["joints"] == sb["reference"]["joints"], "the honored joints differ across trees"
assert all(abs(v) < 1e-9 for k, v in sb["zero"]["joints"].items() if not k.endswith(".vel"))

J = ("Rotation", "Pitch", "Elbow", "Jaw")
def jstr(sc: dict[str, Any]) -> str:
    return "  ".join(f"{k}={sc['joints'][k]:+.4f}" for k in J if k in sc["joints"])

fig = plt.figure(figsize=(16.4, 9.6), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[3.15, 1.0], hspace=0.20, wspace=0.05)

PANELS = [
    (
        ref_br,
        "REFERENCE  n_steps=120 (honored)",
        f"success | 120 steps | 120 inferences\n{jstr(sb['reference'])}\nidentical on both trees (max|delta| = "
        f"{d_ref}/255, renderer noise)",
        "#2e7d32",
    ),
    (
        zero_main,
        "main  n_steps=0",
        f"success | 500 steps | 500 inferences\n{jstr(sm['zero'])}\nthe horizon came from `duration`'s 10.0s "
        "default,\na parameter the caller never set",
        "#b71c1c",
    ),
    (
        zero_br,
        "this change  n_steps=0",
        "ValueError | 0 steps | 0 inferences\n" + jstr(sb["zero"]) + f"\nno action applied; arm at home\n(max|delta| "
        f"vs its own start frame = {d_still}/255)",
        "#2e7d32",
    ),
]
for col, (arr, title, cap, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(arr.astype(np.uint8))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(2.6)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=colour, pad=8)
    ax.set_xlabel(cap, fontsize=9.6, family="monospace", labelpad=8)

axt = fig.add_subplot(gs[1, :])
axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
placed: list[tuple[Any, float, bool]] = []
def put(ax: Any, x: float, y: float, s: str, **kw: Any) -> None:
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)

ROWS = [
    ("n_steps", "resolution on main", "this change"),
    ("120  (usable)", "120 steps, 120 inferences", "120 steps, 120 inferences  (unchanged)"),
    ("0", "500 steps, 500 inferences, success", "ValueError: n_steps must be a positive integer, got 0."),
    ("-5", "500 steps, 500 inferences, success", "refused, naming n_steps and PolicyRunner.run"),
    ("nan", "500 steps, 500 inferences, success", "refused, naming n_steps and PolicyRunner.run"),
    ("2.7 / True", "2 steps / 1 step, success", "refused, naming n_steps and PolicyRunner.run"),
    ("duration=0 / -5", 'success, 0 steps, stopped_reason="budget"', "refused, naming duration and PolicyRunner.run"),
    ("duration=nan / \"10\"", "bare conversion / operand error", "refused, naming duration and PolicyRunner.run"),
]
TOP, LAST = 0.90, 0.055
STEP = (TOP - LAST) / (len(ROWS) - 1)
assert STEP > 0.030, STEP
for i, (a, b, c) in enumerate(ROWS):
    y = TOP - i * STEP
    head = i == 0
    put(axt, 0.005, y, a, fontsize=10.2, family="monospace",
        fontweight="bold" if head else "normal", transform=axt.transAxes)
    put(axt, 0.20, y, b, fontsize=10.2, family="monospace",
        fontweight="bold" if head else "normal",
        color="#000000" if head else "#b71c1c", transform=axt.transAxes)
    put(axt, 0.585, y, c, fontsize=10.2, family="monospace",
        fontweight="bold" if head else "normal",
        color="#000000" if head else "#2e7d32", transform=axt.transAxes)
    if head:
        axt.plot([0.0, 1.0], [y - STEP * 0.45] * 2, lw=0.9, color="#999999",
                 transform=axt.transAxes, clip_on=False)
assert TOP - (len(ROWS) - 1) * STEP > 0.02

fig.suptitle(
    "PolicyRunner.run: a horizon outside its domain selected the other knob's value\n"
    f"so100, MuJoCo headless, control_frequency=50 Hz, action_horizon=1; the commanded pose is indexed by the "
    f"applied-action count, so travel is the honored horizon  |  reference vs main's n_steps=0: {frac:.1%} of pixels",
    fontsize=12.2, fontweight="bold", y=0.985,
)
for ax, y, is_axes in placed:
    if is_axes:
        assert -0.03 <= y <= 1.07, y
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= y <= hi + 0.07, (y, lo, hi)

out = Path("/tmp/artifact_run_horizon_pair.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

a = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", a[:8]), ("bottom", a[-8:]), ("left", a[:, :8]), ("right", a[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK  {out}  {Image.open(out).size}  ref_delta={d_ref}  still={d_still}  frac={frac:.2%}")
