"""Compose the WBC command-override figure from the two measured runs."""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

A = json.loads(pathlib.Path("/tmp/art_main/facts.json").read_text())      # upstream/main
B = json.loads(pathlib.Path("/tmp/art_branch/facts.json").read_text())    # this change
assert A["honored"]["tree"] != B["honored"]["tree"], "the before/after pair is not real"

za = np.load("/tmp/art_main/honored_z.npy"); zb = np.load("/tmp/art_branch/honored_z.npy")
assert np.array_equal(za, zb), "the honored rollout must be identical across trees"
ha = A["honored"]; hb = B["honored"]; na = A["nan_height"]; nb = B["nan_height"]
assert (ha["pelvis_z_end"], ha["n_steps"]) == (hb["pelvis_z_end"], hb["n_steps"]) == (0.7477, 200)
assert na["status"] == "error" and "unresolved keys" in na["text"]
assert nb["status"] == "error" and "height must be a finite number" in nb["text"]
assert na["ticks"] > 0 and nb["ticks"] > 0

img = np.asarray(Image.open("/tmp/art_main/honored.png").convert("RGB"))
imb = np.asarray(Image.open("/tmp/art_branch/honored.png").convert("RGB"))
delta = int(np.abs(img.astype(int) - imb.astype(int)).max())
assert delta <= 2, delta
assert float(((img.max(2).astype(int) - img.min(2)) > 45).mean()) > 0.10, "the G1 is not in frame"

placed: list[tuple[object, float]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

MONO = {"family": "monospace", "fontsize": 8.6, "va": "top"}
fig = plt.figure(figsize=(15.0, 10.4), dpi=124)
gs = gridspec.GridSpec(3, 2, height_ratios=[1.42, 0.86, 0.62], hspace=0.30, wspace=0.16)

# --- row 1 left: the honored rollout, real headless MuJoCo -------------------
ax = fig.add_subplot(gs[0, 0]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("A  Honored command: the balance policy holds station", fontsize=11.5, fontweight="bold")
ax.set_xlabel(
    f'run_policy(policy_provider="wbc", policy_kwargs={{"target_velocity": [0,0,0], "height": 0.74}})\n'
    f'{ha["n_steps"]} ticks @ 50 Hz | pelvis z {ha["pelvis_z_start"]:.4f} -> {ha["pelvis_z_end"]:.4f} m'
    f' | base drift {ha["base_x_travel"]:+.4f} m\n'
    f'byte-identical pelvis trace on both trees; render differs by max {delta}/255 (renderer noise)',
    fontsize=8.9, family="monospace")

# --- row 1 right: the trace, both trees overlaid -----------------------------
ax2 = fig.add_subplot(gs[0, 1])
t = np.arange(za.size) / 50.0
ax2.plot(t, za, lw=3.4, color="#9aa7b4", label="upstream/main")
ax2.plot(t, zb, lw=1.3, color="#1f77b4", ls="--", label="this change")
ax2.axhline(0.74, color="#2ca02c", lw=1.0, ls=":", label="commanded height 0.74 m")
ax2.set_xlabel("rollout time (s)", fontsize=9.5); ax2.set_ylabel("pelvis height (m)", fontsize=9.5)
ax2.set_title("Pelvis height under the honored command\n(the two traces coincide exactly)",
              fontsize=11.0, fontweight="bold")
ax2.legend(fontsize=8.6, loc="upper right"); ax2.grid(alpha=0.28)
ax2.set_ylim(0.70, 0.82)

# --- row 2: what a caller is told for the SAME unusable command --------------
for col, (label, facts, colour) in enumerate([
    ("B  upstream/main", na, "#c62828"),
    ("C  this change", nb, "#2e7d32"),
]):
    axt = fig.add_subplot(gs[1, col]); axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off")
    axt.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, transform=axt.transAxes,
                                facecolor=colour, alpha=0.055, zorder=0))
    put(axt, 0.02, 0.96, f'{label} - policy_kwargs={{"height": nan}}', fontweight="bold",
        fontsize=11.0, va="top", color=colour)
    body = facts["text"].replace("Policy failed: ", "")
    wrapped, line = [], ""
    for word in body.split():
        if len(line) + len(word) + 1 > 74:
            wrapped.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    wrapped.append(line)
    put(axt, 0.02, 0.80, "run_policy status: error\nmessage:", **MONO)
    put(axt, 0.02, 0.66, "\n".join(wrapped[:5]), **MONO)
    verdict = ("names the embodiment, not the command:\nthe caller is sent to debug 15 joint names"
               if col == 0 else "names the parameter that was supplied,\nbefore the network is queried")
    put(axt, 0.02, 0.20, verdict, fontsize=9.4, va="top", style="italic", color=colour)

# --- row 3: the measured ledger ---------------------------------------------
axl = fig.add_subplot(gs[2, :]); axl.set_xlim(0, 1); axl.set_ylim(0, 1); axl.axis("off")
rows = [
    ("command component", "source", "upstream/main", "this change"),
    ("height (overrides height_cmd)", "per-call kwarg", "nan reaches the observation block", "refused: finite number"),
    ("target_orientation (rpy_cmd)", "per-call kwarg", "nan/inf per component reaches it", "refused: per component"),
    ("gait_frequency (freq_cmd)", "per-call kwarg", "0/-x/nan -> ValueError naming GaitClock", "refused: > 0, names the kwarg"),
    ("gait_frequency (freq_cmd)", "constructor", "stored raw; True -> silent 1.0 Hz", "refused at construction"),
    ("gait_frequency (freq_cmd)", "config.freq_cmd", "finite only -> clock refuses 0 later", "refused: > 0 for the gait layout"),
    ("target_velocity", "per-call kwarg", "already validated", "unchanged"),
]
xs = [0.015, 0.255, 0.400, 0.715]
top, gap = 0.90, 0.128
for i, row in enumerate(rows):
    y = top - i * gap
    assert y > 0.02, f"table row {i} overflows the panel at y={y}"
    bold = "bold" if i == 0 else "normal"
    if i == 0:
        axl.plot([0.01, 0.99], [y - 0.045, y - 0.045], color="#444", lw=0.9,
                 transform=axl.transAxes, clip_on=False)
    for x, cell in zip(xs, row, strict=True):
        col = "#c62828" if (i and x == xs[2]) else ("#2e7d32" if (i and x == xs[3]) else "#111")
        put(axl, x, y, cell, fontsize=8.9, family="monospace", va="center", fontweight=bold, color=col)

fig.suptitle(
    "WBC: a per-call command override is validated on the domain the config enforces for the field it overrides\n"
    "Real Unitree G1, real GR00T-WholeBodyControl-Balance.onnx weights, headless MuJoCo",
    fontsize=13.0, fontweight="bold", y=0.985)

for ax_, y in placed:
    lo, hi = ax_.get_ylim()
    assert lo - 0.05 <= y <= hi + 0.07, f"text at y={y} escapes {(lo, hi)}"

out = pathlib.Path("/tmp/wbc_command_override_domains.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

fin = np.asarray(Image.open(out).convert("RGB"), int)
for name, band in (("top", fin[:8]), ("bottom", fin[-8:]), ("left", fin[:, :8]), ("right", fin[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white pixels"
print(f"OK {out} {Image.open(out).size} {out.stat().st_size // 1024} KB")
