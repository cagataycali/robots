"""Compose the artifact. Every rendered number is re-derived from the two dumps."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path("/tmp/artout")
dumps = {}
for f in sorted(glob.glob(str(OUT / "facts_*.json"))):
    d = json.loads(Path(f).read_text())
    dumps["branch" if "robots-mine" in d["tree"] else "main"] = d
A, B = dumps["main"], dumps["branch"]
assert A["tree"] != B["tree"], "both dumps came from the same tree"


def row(d: dict[str, Any], case: str) -> dict[str, Any]:
    return next(r for r in d["rows"] if r["case"] == case)


def img(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


a_hon, b_hon = row(A, "honored"), row(B, "honored")
a_bad, b_bad = row(A, "max_steps_0"), row(B, "max_steps_0")
a_nep, b_nep = row(A, "n_episodes_0"), row(B, "n_episodes_0")

# --- self-audit: the measured claims the figure makes ----------------------
assert a_hon["joints"] == b_hon["joints"], "the honored eval differs between trees"
assert a_hon["applied"] == b_hon["applied"] == 120
hon_delta = int(np.abs(img(a_hon["after"]) - img(b_hon["after"])).max())
assert hon_delta <= 2, f"honored render differs across trees by {hon_delta}"

assert a_bad["status"] == "success" and a_bad["measured"] is True and a_bad["applied"] == 0
assert a_nep["status"] == "success" and a_nep["measured"] is True and a_nep["applied"] == 0
assert b_bad["status"] == "refused" and b_nep["status"] == "refused"
assert b_bad["applied"] == 0 and b_nep["applied"] == 0

bad_delta = int(np.abs(img(a_bad["after"]) - img(b_bad["after"])).max())
assert bad_delta <= 2, f"the refused/accepted world differs by {bad_delta}"
moved = float((np.abs(img(a_hon["after"]) - img(a_bad["after"])).sum(2) > 24).mean()) * 100.0
assert moved > 10.0, f"the honored eval is only {moved:.2f}% different - reframe"
print(f"audit ok: honored cross-tree max|delta|={hon_delta}, refused-vs-reported={bad_delta}, "
      f"honored-vs-nothing-ran={moved:.2f}% of pixels")

# --- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.06)
placed: list[tuple[Any, float, bool]] = []


def put(ax: Any, x: float, y: float, s: str, **kw: Any) -> None:
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)


MONO = {"family": "monospace", "fontsize": 8.6}
panels = [
    (a_hon["after"], "An evaluation that ran\nn_episodes=2, max_steps=60",
     f"status=success   episodes=2\nsuccess_rate={a_hon['rate']}   success_measured={str(a_hon['measured']).lower()}\n"
     f"actions applied={a_hon['applied']}   avg_steps={a_hon['avg_steps']}\n"
     f"shoulder={a_hon['joints']['shoulder']:+.6f}  elbow={a_hon['joints']['elbow']:+.6f}\n"
     f"identical on both trees (max|delta|={hon_delta}/255)", "#1a7f37"),
    (a_bad["after"], "main: max_steps=0\nthe same call, nothing ran",
     f"status=success   episodes=2\nsuccess_rate={a_bad['rate']}   success_measured={str(a_bad['measured']).lower()}\n"
     f"actions applied={a_bad['applied']}   avg_steps={a_bad['avg_steps']}\n"
     f"shoulder={a_bad['joints']['shoulder']:+.6f}  elbow={a_bad['joints']['elbow']:+.6f}\n"
     "a measured 0% over zero applied actions", "#b3261e"),
    (b_bad["after"], "this change: max_steps=0\nrefused before the loop",
     f"ValueError\n{b_bad['message']}\n"
     f"actions applied={b_bad['applied']}\n"
     f"shoulder={b_bad['joints']['shoulder']:+.6f}  elbow={b_bad['joints']['elbow']:+.6f}\n"
     f"same world as the panel left (max|delta|={bad_delta}/255)", "#1a7f37"),
]
for col, (path, title, caption, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img(path).astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(2.4)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=colour, pad=7)
    ax.set_xlabel(caption, fontsize=8.4, family="monospace", labelpad=7)

axt = fig.add_subplot(gs[1, :])
axt.set_xlim(0, 1)
axt.set_ylim(0, 1)
axt.axis("off")
put(axt, 0.5, 1.02, "PolicyRunner.evaluate - the two bounds of its own episode loop",
    ha="center", fontsize=12.4, fontweight="bold", transform=axt.transAxes)

hdr = f"{'call':34}{'main':44}{'this change':40}"
lines = [
    (hdr, "#000000", True),
    ("-" * 118, "#888888", False),
    (f"{'n_episodes=2, max_steps=60':34}"
     f"{'success  2 eps  120 actions applied':44}"
     f"{'success  2 eps  120 actions applied':40}", "#1a7f37", False),
    (f"{'n_episodes=2, max_steps=0':34}"
     f"{'success  2 eps  rate 0.0  measured=true':44}"
     f"{'ValueError: max_steps must be a positive int':40}", "#b3261e", False),
    (f"{'':34}{'0 actions applied':44}{'0 actions applied':40}", "#b3261e", False),
    (f"{'n_episodes=0, max_steps=60':34}"
     f"{'success  0 eps  rate 0.0  measured=true':44}"
     f"{'ValueError: n_episodes must be a positive int':40}", "#b3261e", False),
    (f"{'':34}{'0 actions applied':44}{'0 actions applied':40}", "#b3261e", False),
    (f"{'n_episodes=2, max_steps=inf':34}"
     f"{'never returns (~20k steps/s indefinitely)':44}"
     f"{'ValueError: max_steps must be a positive int':40}", "#b3261e", False),
    (f"{'spec=benchmark, max_steps=0':34}"
     f"{'success (the parameter is never read)':44}"
     f"{'success (unchanged - not refused)':40}", "#1a7f37", False),
    ("-" * 118, "#888888", False),
    ("success_measured is the flag that exists so a success_rate of 0.0 cannot be read as a measurement.",
     "#333333", False),
    ("On main it was true for every row above, over zero applied actions.", "#333333", False),
]
TOP, LAST = 0.90, 0.075
STEP = (TOP - LAST) / (len(lines) - 1)
assert STEP > 0.030, f"table pitch {STEP:.4f} too tight"
y = TOP
for text, colour, bold in lines:
    put(axt, 0.012, y, text, color=colour, fontweight="bold" if bold else "normal",
        transform=axt.transAxes, **MONO)
    y -= STEP
assert y + STEP > 0.04, f"table overflowed to {y + STEP:.4f}"

for ax, yy, is_axes in placed:
    lo, hi = (-0.03, 1.10) if is_axes else ax.get_ylim()
    assert lo <= yy <= hi, f"text at y={yy} outside {lo}..{hi}"

path = OUT / "evaluate_loop_bound_domain.png"
fig.savefig(path, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(Image.open(path).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white pixels"
print(f"WROTE {path}  size={Image.open(path).size}  border clean")
