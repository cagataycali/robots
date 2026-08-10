import json, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

F = json.load(open("/tmp/art/facts.json"))
m, h, r = F["metrics"], F["honored"], F["refused"]
assert h["start"] == "success" and h["steps"] == 120
assert r["start"] == "error" and r["retry"] == "success"
assert m["home_vs_refused_changed_px"] == 0 and m["home_vs_honored_diff_frac"] > 0.10

home = iio.imread("/tmp/art/home.png"); hon = iio.imread("/tmp/art/honored.png"); ref = iio.imread("/tmp/art/refused.png")
fig = plt.figure(figsize=(15.2, 8.2), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.06)

panels = [
    (home, "1. At rest", "before any start_policy call", "#666666"),
    (hon, "2. Well-formed call - rollout runs",
     f"start_policy(n_steps=120) with no policy_kwargs\n{m['home_vs_honored_diff_frac']*100:.1f}% of pixels differ from panel 1", "#2f7d32"),
    (ref, "3. Unsplattable policy_kwargs - refused",
     f"start_policy(policy_kwargs=[...])\nidentical to panel 1: {m['home_vs_refused_changed_px']} of {m['total_px']:,} pixels changed", "#b3261e"),
]
for col, (img, title, sub, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)
    ax.set_xlabel(sub, fontsize=8.6, labelpad=7)
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(2.2)

placed = []
def put(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes); placed.append(y); ax.text(x, y, s, **kw)

axl = fig.add_subplot(gs[1, :2]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.95, "What the new case pins about the refusal", fontsize=11.5, fontweight="bold")
rows = [
    ("verdict", "error - naming policy_kwargs, its type and a correct example"),
    ("envelope", "the shared policy_mapping_error() message, verbatim"),
    ("workers submitted to the executor", "none - the guard sits above self._executor.submit"),
    ("robot marked policy_running", str(r["policy_running"])),
    ("world after the refusal", f"{m['home_vs_refused_changed_px']} pixels changed vs panel 1"),
    ("identical call retried afterwards", f"{r['retry']} - the per-robot slot was never consumed"),
]
TOP, LAST = 0.78, 0.07
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.045, step
y = TOP
for k, v in rows:
    put(axl, 0.0, y, k, fontsize=9.3, va="center")
    put(axl, 0.42, y, v, fontsize=9.3, va="center", family="monospace", color="#1a1a1a")
    y -= step
assert abs((y + step) - LAST) < 1e-9, y

axr = fig.add_subplot(gs[1, 2]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
put(axr, 0.0, 0.95, "Refusal text (verbatim)", fontsize=10.5, fontweight="bold")
put(axr, 0.0, 0.83, textwrap.fill(r["text"], 52), fontsize=6.8, va="top", family="monospace", color="#7a1f16")
assert all(-0.03 <= p <= 1.07 for p in placed), [p for p in placed if not (-0.03 <= p <= 1.07)]

fig.suptitle("start_policy refuses an unsplattable policy_kwargs before it submits - and that refusal was never returned by a test",
             fontsize=12.2, fontweight="bold", y=0.985)
out = "/tmp/art/policy_kwargs_refusal.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(iio.imread(out))[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border {n} non-white px"
print("COMPOSED", out, im.shape)
