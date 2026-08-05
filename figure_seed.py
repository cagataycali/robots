"""Compose the seed-domain artifact from two measured runs (main vs this change)."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

A = json.loads(pathlib.Path("/tmp/before.json").read_text())  # main
B = json.loads(pathlib.Path("/tmp/after.json").read_text())  # this change
assert A["tree"] != B["tree"], "before/after came from the same tree"

BACKENDS = ["lerobot", "cosmos3", "fast_sac", "ppo"]
# What each value IS, and therefore what a correct verdict looks like.
UNUSABLE = ["-1", "-5", "True", "2.7", "3.0", "nan", "inf", "'42'", "[7]"]
USABLE = ["0", "42", "None"]
LABELS = UNUSABLE + USABLE
WANT = {**{k: "refused" for k in UNUSABLE}, **{k: "accepted" for k in USABLE}}

GOOD, BAD = "#1b7f3b", "#b3261e"
RED_BG, GRN_BG = "#fdecea", "#eaf6ee"


def div(run: dict) -> int:
    """Cells whose verdict is not the one the appliers can honor."""
    return sum(1 for v in LABELS for b in BACKENDS if run["rows"][v][b] != WANT[v])


CELLS = len(LABELS) * len(BACKENDS)
D_A, D_B = div(A), div(B)
assert (CELLS, D_A, D_B) == (48, 36, 0), (CELLS, D_A, D_B)

placed: list[tuple[object, float]] = []


def put(ax, x, y, s, **kw):  # noqa: ANN001, ANN201, D103
    placed.append((ax, y))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 9.4))
gs = fig.add_gridspec(2, 2, height_ratios=[2.55, 1.0], hspace=0.20, wspace=0.11)

fig.suptitle(
    "TrainSpec.seed reached four different appliers with no domain of its own",
    fontsize=16.5,
    fontweight="bold",
    y=0.982,
)
fig.text(
    0.5,
    0.948,
    "Verdict from each backend's own validate(); a cell is red when it is not the verdict the appliers can honor. "
    "Measured on main vs this change, same script, same probe set.",
    ha="center",
    fontsize=10.4,
    color="#333333",
)

for col, (run, title) in enumerate(((A, "main (62e375da)"), (B, "this change"))):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    d = div(run)
    put(
        ax,
        0.5,
        1.035,
        f"{title}   -   {d} of {CELLS} cells not the honorable verdict",
        ha="center",
        fontsize=12.6,
        fontweight="bold",
        color=BAD if d else GOOD,
    )
    xs = [0.245, 0.425, 0.605, 0.815]
    put(ax, 0.115, 0.945, "seed=", ha="center", fontsize=10.4, fontweight="bold", color="#444444")
    for x, b in zip(xs, BACKENDS, strict=True):
        put(ax, x, 0.945, b, ha="center", fontsize=10.4, fontweight="bold", color="#444444")
    y = 0.888
    for v in LABELS:
        band = RED_BG if any(run["rows"][v][b] != WANT[v] for b in BACKENDS) else GRN_BG
        ax.add_patch(mpatches.Rectangle((0.02, y - 0.026), 0.96, 0.052, facecolor=band, edgecolor="none", zorder=0))
        put(ax, 0.115, y, v, ha="center", fontsize=11.0, family="monospace", fontweight="bold")
        for x, b in zip(xs, BACKENDS, strict=True):
            got = run["rows"][v][b]
            ok = got == WANT[v]
            put(
                ax,
                x,
                y,
                got,
                ha="center",
                fontsize=10.2,
                family="monospace",
                color=GOOD if ok else BAD,
                fontweight="normal" if ok else "bold",
            )
        if v == UNUSABLE[-1]:
            ax.plot([0.02, 0.98], [y - 0.030, y - 0.030], color="#999999", lw=0.9, ls=":")
        y -= 0.0655
    put(
        ax,
        0.5,
        0.135,
        "above the dotted line: no applier can honor the value    below: every applier honors it",
        ha="center",
        fontsize=9.3,
        style="italic",
        color="#555555",
    )

# --- consequence: torch's modulo makes two distinct seeds one stream -----------
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
put(
    ax,
    0.5,
    1.06,
    "Why 'accepted' was the wrong verdict: torch.manual_seed reduces mod 2**64",
    ha="center",
    fontsize=11.6,
    fontweight="bold",
)
put(ax, 0.055, 0.86, "seed asked for", fontsize=9.8, fontweight="bold", color="#444444")
put(ax, 0.315, 0.86, "seed actually used", fontsize=9.8, fontweight="bold", color="#444444")
put(ax, 0.635, 0.86, "first 4 draws of both", fontsize=9.8, fontweight="bold", color="#444444")
y = 0.68
for supplied, info in A["collide"].items():
    assert info["identical"], supplied
    put(ax, 0.055, y, supplied, fontsize=10.6, family="monospace", fontweight="bold")
    put(ax, 0.315, y, str(info["actual"]), fontsize=10.6, family="monospace", color=BAD, fontweight="bold")
    put(ax, 0.635, y, str(info["draws"][:3])[:-1] + ", ...]", fontsize=9.6, family="monospace")
    y -= 0.165
put(
    ax,
    0.5,
    0.045,
    "identical streams: the run IS reproducible - under a seed nobody asked for, that another caller could have named",
    ha="center",
    fontsize=9.2,
    style="italic",
    color=BAD,
)

# --- control: a usable seed is untouched, byte for byte -----------------------
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
put(ax, 0.5, 1.06, "Unchanged: what a usable seed puts on the wire", ha="center", fontsize=11.6, fontweight="bold")
assert A["control"] == B["control"], (A["control"], B["control"])
rows = [
    ("seed=42 -> Cosmos Hydra override", A["control"]["cosmos_override"][0]),
    ("seed=42 -> LeRobot argv token", A["control"]["lerobot_argv"][0]),
    ("seed=0 / 42 / None on all 4 backends", "accepted, both trees"),
    ("backends that never read the field", "mock, groot: report nothing"),
]
y = 0.80
for label, value in rows:
    put(ax, 0.05, y, label, fontsize=9.9, color="#333333")
    put(ax, 0.60, y, value, fontsize=9.9, family="monospace", color=GOOD, fontweight="bold")
    y -= 0.185
put(
    ax,
    0.5,
    0.055,
    "byte-identical on both trees - the change adds a refusal, it does not alter an honored run",
    ha="center",
    fontsize=9.2,
    style="italic",
    color="#555555",
)

for ax_obj, yy in placed:
    lo, hi = ax_obj.get_ylim()
    pad = 0.09 * (hi - lo)
    assert lo - pad <= yy <= hi + pad, f"text at y={yy} outside {ax_obj.get_ylim()}"

out = pathlib.Path("/tmp/seed_domain.png")
fig.savefig(out, dpi=118, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 20).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out} {Image.open(out).size} main={D_A}/{CELLS} branch={D_B}/{CELLS}")
