"""Compose the measured artifact. Every number is read from the two JSON dumps."""
from __future__ import annotations
import json, os, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio

RUN = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/art-main-{RUN}.json").read_text())    # base
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RUN}.json").read_text())  # this PR
assert A["tree"] != B["tree"], (A["tree"], B["tree"])
assert A["supports_stats_domain"] is False and B["supports_stats_domain"] is True

def img(d, label):
    return iio.imread(d["rows"][label]["frame"])[:, :, :3].astype(int)

own_a, own_b = img(A, "own_domain"), img(B, "own_domain")
wrong = img(A, "wrong_domain_bare")

# --- self-audit -----------------------------------------------------------
d_own = int(np.abs(own_a - own_b).max())
assert d_own <= 2, f"honored row must be byte-comparable across trees, got {d_own}"
frac = float((np.abs(own_a - wrong).sum(2) > 24).mean())
assert frac > 0.10, f"defect panel must be legible, got {frac:.2%}"
t_own = A["rows"]["own_domain"]["travel_m"]
t_wrong = A["rows"]["wrong_domain_bare"]["travel_m"]
assert abs(t_own - B["rows"]["own_domain"]["travel_m"]) < 1e-9
over = (t_wrong - t_own) / t_own
assert B["rows"]["wrong_domain_bare"]["outcome"] == "ValueError"
assert B["rows"]["wrong_domain_declared"]["outcome"] == "ValueError"
assert A["rows"]["wrong_domain_bare"]["outcome"] == "decoded"
print(f"audit ok: honored delta={d_own}/255  defect diff={frac:.2%}  overshoot={over:+.1%}")

placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.pop("transform", None) is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, transform=ax.transAxes, **kw)

fig = plt.figure(figsize=(15.6, 10.4), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.13, wspace=0.05)

panels = [
    (own_a, f"what the caller asked for\numi quantiles -> {t_own:.4f} m of EE travel",
     "#1a7f37"),
    (wrong, f"main: another domain's quantiles, accepted\n-> {t_wrong:.4f} m ({over:+.1%}), nothing reported",
     "#b42318"),
    (own_b, "this PR: the wrong domain is refused\nthe honored decode is unchanged",
     "#1a7f37"),
]
for i, (im, cap, col) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(im.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2.4)
    ax.set_xlabel(cap, fontsize=10.5, color=col, labelpad=7)

# --- table ---------------------------------------------------------------
axt = fig.add_subplot(gs[1, :]); axt.axis("off")
axt.set_xlim(0, 1); axt.set_ylim(0, 1)
rows = [
    ("decode of one nvidia/Cosmos3-Edge umi action chunk (16 steps x 10 cols), Franka panda + mink IK",
     "", "", True),
    ("call", "main", "this PR", True),
    ("stats = umi's own quantiles, stats_domain='umi'",
     "TypeError: no such keyword", f"decoded, {t_own:.4f} m", False),
    ("stats = bridge_orig_lerobot, stats_domain declared",
     f"decoded, {t_wrong:.4f} m ({over:+.1%})", "refused, names both domains", False),
    ("stats = bridge_orig_lerobot, no domain declared",
     f"decoded, {t_wrong:.4f} m ({over:+.1%})", "refused, names stats_domain", False),
    ("no stats= at all (droid / bridge)", "bundled stats loaded", "unchanged", False),
    ("load_action_stats('umi')", "advises passing stats", "advises stats + stats_domain", False),
]
TOP, LAST = 0.90, 0.13
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
y = TOP
for label, m, b, hdr in rows:
    w = "bold" if hdr else "normal"
    if hdr and not m:
        put(axt, 0.012, y, label, transform=axt.transAxes, fontsize=11.5, fontweight="bold")
    else:
        put(axt, 0.012, y, label, transform=axt.transAxes, fontsize=10.2, fontweight=w, family="monospace")
        put(axt, 0.545, y, m, transform=axt.transAxes, fontsize=10.2, fontweight=w,
            color="#000" if hdr else "#b42318")
        put(axt, 0.795, y, b, transform=axt.transAxes, fontsize=10.2, fontweight=w,
            color="#000" if hdr else "#1a7f37")
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)

foot = (f"umi, droid_lerobot and bridge_orig_lerobot are all 10 action columns, so the width check cannot "
        f"separate them.  Honored render identical across trees (max |delta| = {d_own}/255); "
        f"defect panel differs on {frac:.1%} of pixels.  Cosmos3-Edge documents umi + av, the two domains "
        f"with no bundled quantiles.")
put(axt, 0.012, 0.035, foot, transform=axt.transAxes, fontsize=9.0, color="#444", wrap=True)

for ax, yy, axes_coords in placed:
    lo, hi = (-0.03, 1.08) if axes_coords else ax.get_ylim()
    assert lo <= yy <= hi, (yy, lo, hi)

fig.suptitle("Cosmos 3: explicit de-normalization stats must declare their domain",
             fontsize=14.5, fontweight="bold", y=0.975)
out = pathlib.Path(f"/tmp/artifact-{RUN}.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.3, facecolor="white")
plt.close(fig)

im = iio.imread(out)[:, :, :3].astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band - 255).sum(2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print(f"wrote {out}  {im.shape[1]}x{im.shape[0]}")
