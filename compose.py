"""Compose the measured verdict figure from the two captured trees."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

A = json.loads(pathlib.Path("/tmp/census/before.json").read_text())  # upstream/main
B = json.loads(pathlib.Path("/tmp/census/after.json").read_text())  # this change

assert A["tree"] != B["tree"], "before/after came from the same tree"

FLAGS = ["dataset_push_to_hub", "dataset_video", "display_data"]
OPT_OUTS = ["false", "no", "off", "0"]

# --- derive every number the figure states -------------------------------
n_cells = len(FLAGS) * len(OPT_OUTS)
inverted_before = sum(
    1 for f in FLAGS for v in OPT_OUTS if A["rows"][f][v]["outcome"] == "emitted"
)
inverted_after = sum(
    1 for f in FLAGS for v in OPT_OUTS if B["rows"][f][v]["outcome"] == "emitted"
)
assert (n_cells, inverted_before, inverted_after) == (12, 12, 0), (
    n_cells,
    inverted_before,
    inverted_after,
)
# the honored argv must be byte-identical across trees
assert A["honored"] == B["honored"], "the honored path changed"
assert "--dataset.push_to_hub" in A["honored"]["opt_out_false"]
i = A["honored"]["opt_out_false"].index("--dataset.push_to_hub")
assert A["honored"]["opt_out_false"][i + 1] == "false"
j = A["honored"]["opt_in_true"].index("--dataset.push_to_hub")
assert A["honored"]["opt_in_true"][j + 1] == "true"
# the agent schema descriptions improved
assert A["schema"]["auto_accept_calibration"].startswith("Parameter ")
assert not B["schema"]["auto_accept_calibration"].startswith("Parameter ")

RED, GREEN, GREY = "#c0392b", "#1e8449", "#5d6d7e"
placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.6, 10.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.32, 1.0, 0.62], hspace=0.30)

fig.suptitle(
    "A posture flag read for truthiness selects the branch it names the opposite of",
    fontsize=16.5,
    fontweight="bold",
    y=0.975,
)
fig.text(
    0.5,
    0.941,
    "build_lerobot_command, action='start' with a dataset (record mode) - what each opt-out spelling put on the lerobot argv",
    ha="center",
    fontsize=11,
    color="#333333",
)

# ---------------- row 1: the verdict grid --------------------------------
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
put(ax, 0.5, 1.02, "the caller wrote an opt-out", ha="center", fontsize=12.5,
    fontweight="bold", transform=ax.transAxes)

col_x = [0.335, 0.485, 0.635, 0.785]
put(ax, 0.012, 0.905, "flag", fontsize=11, fontweight="bold", transform=ax.transAxes)
for cx, v in zip(col_x, OPT_OUTS, strict=True):
    put(ax, cx, 0.905, f'"{v}"', ha="center", fontsize=11.5, fontweight="bold",
        family="monospace", transform=ax.transAxes)

TOP, LAST = 0.735, 0.115
rows_all = [(f, tree) for f in FLAGS for tree in ("main", "this change")]
STEP = (TOP - LAST) / (len(rows_all) - 1)
assert STEP > 0.045, STEP

for k, (flag, tree) in enumerate(rows_all):
    y = TOP - k * STEP
    src = A if tree == "main" else B
    label = f"{flag}" if tree == "main" else ""
    if label:
        put(ax, 0.012, y + 0.030, label, fontsize=11.4, fontweight="bold",
            family="monospace", transform=ax.transAxes)
    put(ax, 0.030, y, f"{tree:12s}", fontsize=10.2, color="#222222",
        family="monospace", transform=ax.transAxes)
    for cx, v in zip(col_x, OPT_OUTS, strict=True):
        cell = src["rows"][flag][v]
        emitted = cell["outcome"] == "emitted"
        colour = RED if emitted else GREEN
        text = cell["detail"] if emitted else "refused"
        ax.add_patch(
            plt.Rectangle(
                (cx - 0.070, y - 0.017), 0.140, 0.036,
                transform=ax.transAxes, facecolor=colour, alpha=1.0, zorder=1,
            )
        )
        put(ax, cx, y, text, ha="center", va="center", fontsize=8.6, color="white",
            family="monospace", zorder=2, transform=ax.transAxes)

put(ax, 0.5, 0.038,
    f"opt-out spellings that selected the opposite posture:   main {inverted_before} of {n_cells}"
    f"      this change {inverted_after} of {n_cells}",
    ha="center", fontsize=12.2, fontweight="bold", transform=ax.transAxes)

# ---------------- row 2: no regression -----------------------------------
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis("off")
put(ax2, 0.5, 1.03, "a real boolean is untouched - the argv is byte-identical on both trees",
    ha="center", fontsize=12.5, fontweight="bold", transform=ax2.transAxes)

lines = [
    ("dataset_push_to_hub=False", "--dataset.push_to_hub false", "the explicit opt-out still emitted"),
    ("dataset_push_to_hub=True", "--dataset.push_to_hub true", "the opt-in still uploads"),
]
TOP2, LAST2 = 0.78, 0.50
STEP2 = (TOP2 - LAST2) / (len(lines) - 1)
for k, (call, argv, note) in enumerate(lines):
    y = TOP2 - k * STEP2
    put(ax2, 0.030, y, f"{call:30s}", fontsize=10.8, family="monospace",
        transform=ax2.transAxes)
    put(ax2, 0.330, y, f"{argv:32s}", fontsize=10.8, family="monospace",
        color=GREEN, fontweight="bold", transform=ax2.transAxes)
    put(ax2, 0.640, y, note, fontsize=10.4, color=GREY, transform=ax2.transAxes)

put(ax2, 0.030, 0.33,
    "the two undocumented flags also stop reaching the model as a placeholder:",
    fontsize=11.2, fontweight="bold", transform=ax2.transAxes)
put(ax2, 0.030, 0.20,
    f'main         auto_accept_calibration: "{A["schema"]["auto_accept_calibration"]}"',
    fontsize=9.6, family="monospace", color=RED, transform=ax2.transAxes)
put(ax2, 0.030, 0.07,
    f'this change  auto_accept_calibration: "{B["schema"]["auto_accept_calibration"][:66]}..."',
    fontsize=9.6, family="monospace", color=GREEN, transform=ax2.transAxes)

# ---------------- row 3: why it went unreported --------------------------
ax3 = fig.add_subplot(gs[2])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis("off")
ax3.add_patch(
    plt.Rectangle((0.012, 0.06), 0.976, 0.86, transform=ax3.transAxes,
                  facecolor="#f4f6f7", edgecolor="#bdc3c7")
)
notes = [
    "Nothing reported any of it: the argv goes to a subprocess launched with start_new_session=True, so the call",
    "returns status=\"success\" with a pid and the posture it chose is never read back. On main the refusal now",
    "precedes the session record and the subprocess; the flags a mode does not emit are still not checked, so",
    "replay accepts all four spellings and play_sounds - which no mode emits - is excluded by that same rule.",
]
TOP3, LAST3 = 0.76, 0.20
STEP3 = (TOP3 - LAST3) / (len(notes) - 1)
for k, line in enumerate(notes):
    put(ax3, 0.030, TOP3 - k * STEP3, line, fontsize=10.3, color="#212f3d",
        transform=ax3.transAxes)

# --------------- layout guards -------------------------------------------
for axis, y, axes_coords in placed:
    if axes_coords:
        assert -0.04 <= y <= 1.07, f"axes-fraction y out of band: {y}"
    else:
        lo, hi = axis.get_ylim()
        assert lo - 0.05 <= y <= hi + 0.07, f"data y {y} outside {(lo, hi)}"

out = pathlib.Path("/tmp/census/flag_domain.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nonwhite = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print("OK", out, Image.open(out).size)
