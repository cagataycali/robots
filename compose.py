"""Compose the horizon-domain figure from the two measured runs."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

A = json.loads(pathlib.Path("/tmp/art_main/facts.json").read_text())      # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch/facts.json").read_text())    # this change
assert A["tree"] != B["tree"], (A["tree"], B["tree"])

def img(root, name):
    return np.asarray(Image.open(pathlib.Path(root) / name).convert("RGB")).astype(int)

ref_a, ref_b = img("/tmp/art_main", "reference_after.png"), img("/tmp/art_branch", "reference_after.png")
main_bool = img("/tmp/art_main", "bool_after.png")
br_bool_before = img("/tmp/art_branch", "bool_before.png")
br_bool = img("/tmp/art_branch", "bool_after.png")

# --- self-audit: every claim the figure makes, measured -----------------------
ref_delta = int(np.abs(ref_a - ref_b).max())
assert ref_delta <= 2, f"reference render differs across trees by {ref_delta}"
assert A["reference"]["travel"] == B["reference"]["travel"], "reference rollout diverged"
assert A["reference"]["steps"] == B["reference"]["steps"] == 120
assert A["bool_horizon"]["status"] == "success" and A["bool_horizon"]["steps"] == 1
assert A["fractional_horizon"]["status"] == "success" and A["fractional_horizon"]["steps"] == 2
assert B["bool_horizon"]["status"] == "error" and B["bool_horizon"]["policy_calls"] == 0
assert B["fractional_horizon"]["status"] == "error" and B["fractional_horizon"]["policy_calls"] == 0
still = int(np.abs(br_bool - br_bool_before).max())
assert still <= 2, f"the refused rollout moved the arm ({still})"
diff_frac = float((np.abs(main_bool - ref_a).sum(axis=2) > 24).mean())
assert diff_frac > 0.10, f"panels not legibly different ({diff_frac:.2%})"
print(f"audit OK: ref delta={ref_delta}  refused-still={still}  main-vs-reference diff={diff_frac:.2%}")

placed: list[tuple[object, float, bool]] = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.16, wspace=0.05)

panels = [
    (ref_a, "A. The horizon the caller asked for", f"n_steps=120  ->  120 steps, travel {A['reference']['travel']} rad\nidentical on both trees (max pixel delta {ref_delta}/255)", "#1a7f37"),
    (main_bool, "B. main: n_steps=True", f"status=\"success\"  ->  {A['bool_horizon']['steps']} step, travel {A['bool_horizon']['travel']} rad\na horizon the caller never asked for", "#b42318"),
    (br_bool, "C. this change: n_steps=True", f"status=\"error\"  ->  0 policy queries, travel {B['bool_horizon']['travel']} rad\nthe arm never moved (max pixel delta {still}/255 vs its own start)", "#1a7f37"),
]
for col, (im, title, caption, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(im.astype(np.uint8))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(caption, fontsize=9.4, labelpad=7, linespacing=1.45)

ax = fig.add_subplot(gs[1, :])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.5, 1.055, "The same step-count horizon, on run_policy and on the identically-named eval_policy budget",
    ha="center", fontsize=13, fontweight="bold", transform=ax.transAxes)

rows = [
    ("run_policy(n_steps=)", "requested", "main", "this change"),
    ("120  (a whole number of steps)", "120 steps", f"success, {A['reference']['steps']} steps, travel {A['reference']['travel']}", f"success, {B['reference']['steps']} steps, travel {B['reference']['travel']}"),
    ("True  (a flag wired to the horizon)", "-", f"success, {A['bool_horizon']['steps']} step, travel {A['bool_horizon']['travel']}", "error: n_steps must be a positive integer, got True."),
    ("2.7  (a fractional horizon)", "-", f"success, {A['fractional_horizon']['steps']} steps, travel {A['fractional_horizon']['travel']}", "error: n_steps must be a positive integer, got 2.7."),
    ("max_steps=0  (the legacy alias)", "-", "error: n_steps must be > 0  <- names a parameter never passed", "error: max_steps must be a positive integer, got 0."),
    ("eval_policy(max_steps=2.7)  [reference]", "-", "error: max_steps must be a positive integer", "error: max_steps must be a positive integer"),
]
xs = [0.012, 0.268, 0.372, 0.700]
# Derive the pitch from the row count so the footer always has room: the last
# row lands on LAST and the footer sits below it.
TOP, LAST, FOOTER = 0.930, 0.330, 0.205
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.045, STEP
y = TOP
for i, row in enumerate(rows):
    header = i == 0
    if not header and i % 2 == 0:
        ax.add_patch(plt.Rectangle((0.004, y - 0.052), 0.992, 0.072, color="#f4f6f8", zorder=0))
    for x, cell in zip(xs, row, strict=True):
        colour = "#111111"
        weight = "bold" if header else "normal"
        if not header and x == xs[2] and "success" in cell:
            colour = "#b42318"
        if not header and x == xs[2] and "names a parameter" in cell:
            colour = "#b42318"
        if not header and x == xs[3] and "error" in cell:
            colour = "#1a7f37"
        if not header and x == xs[3] and "success" in cell:
            colour = "#1a7f37"
        put(ax, x, y, cell, fontsize=9.5, family="monospace", color=colour,
            fontweight=weight, va="center", transform=ax.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y + STEP, LAST)
assert FOOTER > 0.02, FOOTER
put(ax, 0.012, FOOTER,
    "Both entry points now resolve the horizon through the shared positive-count domain, so one parameter name\n"
    "carries one contract. Every usable horizon is honoured exactly as before -- panel A is bit-identical on both trees.",
    fontsize=9.6, color="#333333", va="top", transform=ax.transAxes, linespacing=1.5)

for ax_, y_, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y_ <= 1.07, y_
    else:
        lo, hi = ax_.get_ylim()
        assert min(lo, hi) - 0.05 <= y_ <= max(lo, hi) + 0.07, y_

out = pathlib.Path("/tmp/art_branch/horizon_domain.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.3, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white pixels"
print("figure:", out, Image.open(out).size)
