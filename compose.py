"""Compose the before/after figure from the two measured dumps."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

A = json.loads(pathlib.Path("/tmp/art_main.json").read_text())   # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch.json").read_text()) # this branch
assert A["tree"] != B["tree"], "both dumps came from the same tree"
assert A["totals"] == {"tools": 16, "placeholder": 13, "phantom": 1}, A["totals"]
assert B["totals"] == {"tools": 16, "placeholder": 0, "phantom": 0}, B["totals"]
assert sorted(A["params"]) == sorted(B["params"])
assert all(v["placeholder"] for v in A["params"].values()), "every measured param must be a placeholder on main"
assert not any(v["placeholder"] for v in B["params"].values())
assert A["description"]["host RCE"] is False and B["description"]["host RCE"] is True

ORDER = [
    "gr00t_inference.hf_repo", "gr00t_inference.hf_subfolder", "gr00t_inference.hf_local_dir",
    "gr00t_inference.hf_token", "gr00t_inference.lifecycle", "gr00t_inference.remove_volumes",
    "gr00t_inference.force",
    "lerobot_teleoperate.policy_path", "lerobot_teleoperate.dagger_input_device",
    "lerobot_teleoperate.dagger_num_episodes",
    "train_policy.lora_r", "train_policy.lora_alpha", "train_policy.lora_target_modules",
]
assert sorted(ORDER) == sorted(A["params"]), "row order must cover exactly the measured params"

RED, GREEN, INK, MUTED = "#b3261e", "#1b5e20", "#111111", "#5f6368"

def wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return lines[:3]

fig = plt.figure(figsize=(16.6, 11.4), dpi=124)
gs = fig.add_gridspec(2, 1, height_ratios=[8.4, 1.6], hspace=0.10,
                      left=0.012, right=0.988, top=0.925, bottom=0.028)
fig.suptitle(
    "What the model is told about each parameter it may pass",
    fontsize=17.5, fontweight="bold", y=0.977,
)
fig.text(0.5, 0.947,
         "A @tool schema is derived from the docstring; the decorator substitutes "
         '"Parameter <name>" for anything absent from Args:.',
         ha="center", fontsize=10.6, color=MUTED)

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.pop("axes_coords", True)
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, transform=ax.transAxes if axes_coords else ax.transData, **kw)

ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
X_P, X_A, X_B = 0.008, 0.215, 0.575
put(ax, X_P, 0.978, "parameter", fontsize=11, fontweight="bold", color=INK)
put(ax, X_A, 0.978, "on main: what the model receives", fontsize=11, fontweight="bold", color=RED)
put(ax, X_B, 0.978, "with this change", fontsize=11, fontweight="bold", color=GREEN)
ax.plot([0.004, 0.996], [0.962, 0.962], lw=1.2, color=INK, transform=ax.transAxes)

TOP, LAST = 0.905, 0.045
STEP = (TOP - LAST) / (len(ORDER) - 1)
assert STEP > 0.030, STEP
for i, key in enumerate(ORDER):
    y = TOP - i * STEP
    ax.add_patch(Rectangle((0.004, y - STEP * 0.42), 0.992, STEP * 0.90,
                           transform=ax.transAxes, facecolor=RED, alpha=0.055,
                           edgecolor="none", zorder=0))
    put(ax, X_P, y, key, fontsize=9.9, family="monospace", color=INK, va="center")
    put(ax, X_A, y, f'"{A["params"][key]["text"]}"', fontsize=9.9, family="monospace",
        color=RED, va="center")
    lines = wrap(B["params"][key]["text"], 66)
    put(ax, X_B, y, "\n".join(lines), fontsize=8.5, color=GREEN, va="center", linespacing=1.25)
assert abs((TOP - (len(ORDER) - 1) * STEP) - LAST) < 1e-9

axf = fig.add_subplot(gs[1]); axf.axis("off"); axf.set_xlim(0, 1); axf.set_ylim(0, 1)
axf.add_patch(Rectangle((0.004, 0.02), 0.992, 0.96, transform=axf.transAxes,
                        facecolor="#f4f4f4", edgecolor="#d0d0d0", zorder=0))
rows = [
    ("across all 16 bound agent tools",
     f'placeholder parameters {A["totals"]["placeholder"]}   '
     f'docstring entries naming no parameter {A["totals"]["phantom"]}',
     f'placeholder parameters {B["totals"]["placeholder"]}   '
     f'docstring entries naming no parameter {B["totals"]["phantom"]}'),
    ("gr00t_inference tool description",
     'the "host RCE" operator-config note: absent',
     'the "host RCE" operator-config note: present (moved ahead of Args:)'),
    ("why ten of the thirteen looked documented",
     "7 sat under a Container lifecycle args header; 3 shared one a / b / c: entry",
     "each has its own Args: entry, so each reaches the schema"),
]
F_TOP, F_LAST = 0.80, 0.20
F_STEP = (F_TOP - F_LAST) / (len(rows) - 1)
assert F_STEP > 0.030, F_STEP
for i, (label, before, after) in enumerate(rows):
    y = F_TOP - i * F_STEP
    put(axf, 0.012, y, label, fontsize=10.2, fontweight="bold", color=INK, va="center")
    put(axf, 0.245, y, before, fontsize=9.6, family="monospace", color=RED, va="center")
    put(axf, 0.615, y, after, fontsize=9.6, family="monospace", color=GREEN, va="center")

for a_, y, is_axes in placed:
    if is_axes:
        assert -0.04 <= y <= 1.06, (a_, y)
    else:
        lo, hi = a_.get_ylim()
        assert lo - 0.03 <= y <= hi + 0.03, (a_, y)

out = pathlib.Path("/tmp/agent_tool_parameter_descriptions.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
h, w, _ = im.shape
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {out}  {w}x{h}")
