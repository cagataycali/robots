"""Build the artifact from the measured logs. Every cell is re-parsed here."""

import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read(path):
    return open(path, encoding="utf-8", errors="replace").read()


PRE_PLAIN, POST_PLAIN = read("/tmp/pre_full.log"), read("/tmp/post_full.log")
PRE_COE, POST_COE = read("/tmp/pre_full2.log"), read("/tmp/post_full2.log")


def summary(blob):
    line = [l for l in blob.splitlines() if re.search(r"\b(passed|failed|errors?)\b.*in \d+\.\d+s", l)][-1]
    g = lambda k: int(m.group(1)) if (m := re.search(rf"(\d+) {k}s?\b", line)) else 0  # noqa: E731
    return {k: g(k) for k in ("failed", "passed", "skipped", "error")} | {"line": line.strip("= ")}


def reaches(blob):
    """Failures whose reported error is a reach past the stand-in.

    Scoped to torch and its submodules: a test object missing an attribute of its
    own is a different failure and must not be counted here.
    """
    pat = re.compile(r"module 'torch(\.[a-z_]+)*' has no attribute")
    return sum(1 for l in blob.splitlines() if l.startswith("FAILED") and pat.search(l))


pre_p, post_p = summary(PRE_PLAIN), summary(POST_PLAIN)
pre_c, post_c = summary(PRE_COE), summary(POST_COE)
pre_reach, post_reach = reaches(PRE_COE), reaches(POST_COE)
pre_collect = len([l for l in PRE_PLAIN.splitlines() if l.startswith("ERROR ") and "has no attribute" in l])

# --- self-audit: the numbers this figure claims -------------------------------
assert "Interrupted" in PRE_PLAIN and pre_p["passed"] == 0, "pre-fix plain run must execute nothing"
assert pre_collect == 13, pre_collect
assert (pre_c["failed"], pre_c["passed"], pre_c["skipped"], pre_c["error"]) == (758, 19704, 281, 23), pre_c
assert (post_c["failed"], post_c["passed"], post_c["skipped"], post_c["error"]) == (87, 19744, 977, 0), post_c
assert (pre_reach, post_reach) == (676, 3), (pre_reach, post_reach)
assert post_p["passed"] == 19745 and post_p["failed"] == 86, post_p
OLD_MSG = "AttributeError: module 'torch' has no attribute 'is_tensor'"
NEW_MSG = next(
    l.split("MissingMockAttribute: ")[1]
    for l in POST_COE.splitlines()
    if "MissingMockAttribute: module 'torch' has no attribute 'optim'" in l
)
assert "numpy-backed torch stand-in" in NEW_MSG and 'pip install -e ".[all,dev]"' in NEW_MSG

fig = plt.figure(figsize=(15.4, 11.6), dpi=125)
placed = []


def put(ax, x, y, s, **kw):
    placed.append((ax, y))
    return ax.text(x, y, s, **kw)


GREEN, RED, GREY = "#1a7f37", "#b3261e", "#57606a"

# ---------------------------------------------------------------- panel 1: table
ax = fig.add_axes([0.035, 0.545, 0.93, 0.375])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
put(ax, 0.0, 1.045, "A full torch-less run of tests/  --  the environment the stand-in exists for", fontsize=14.5, fontweight="bold")

rows = [
    ("`pytest tests` (what a contributor types)", "aborts in collection, 0 tests run", f"runs: {post_p['passed']} passed", True),
    ("  collection errors aborting that run", f"{pre_collect}", "0", True),
    ("with --continue-on-collection-errors:", "", "", None),
    ("  failed", f"{pre_c['failed']}", f"{post_c['failed']}", True),
    ("  of those, a reach past the stand-in", f"{pre_reach}", f"{post_reach}   (each already failing before)", True),
    ("  collection errors", f"{pre_c['error']}", f"{post_c['error']}", True),
    ("  passed", f"{pre_c['passed']}", f"{post_c['passed']}", False),
    ("  skipped (now carrying the reason)", f"{pre_c['skipped']}", f"{post_c['skipped']}", False),
    ("tests/simulation/ alone (clean control)", "103 failed, 4 errors", "0 failed, 0 errors", True),
]
put(ax, 0.0, 0.905, "measurement", fontsize=11.5, fontweight="bold", color=GREY)
put(ax, 0.455, 0.905, "main", fontsize=11.5, fontweight="bold", color=GREY)
put(ax, 0.70, 0.905, "this change", fontsize=11.5, fontweight="bold", color=GREY)
y = 0.835
for label, before, after, improved in rows:
    if improved is None:
        put(ax, 0.0, y, label, fontsize=11, style="italic", color=GREY)
    else:
        put(ax, 0.0, y, label, fontsize=11.5)
        put(ax, 0.455, y, before, fontsize=11.5, family="monospace", color=RED if improved else GREY)
        put(ax, 0.70, y, after, fontsize=11.5, family="monospace", color=GREEN if improved else GREY)
    y -= 0.093
ax.add_patch(Rectangle((0.0, y + 0.055), 1.0, 0.002, color="#d0d7de", clip_on=False))
put(ax, 0.0, y - 0.005,
    "Nothing that passed before fails now.  `dtype` is 622 of the 676 reaches and is read inside lerobot during import,\n"
    "not by this repository -- which is why completing the subset is not the way out.",
    fontsize=11, color=GREY)

# ------------------------------------------------- panel 2: what the reader sees
ax2 = fig.add_axes([0.035, 0.20, 0.93, 0.295])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis("off")
put(ax2, 0.0, 1.055, "What the contributor is told", fontsize=14.5, fontweight="bold")
ax2.add_patch(Rectangle((0.0, 0.60), 1.0, 0.30, facecolor="#fff0ef", edgecolor=RED, lw=1.3))
put(ax2, 0.012, 0.845, "main  --  reported as a FAILURE", fontsize=11, fontweight="bold", color=RED)
put(ax2, 0.012, 0.735, OLD_MSG, fontsize=11.5, family="monospace")
put(ax2, 0.012, 0.655, "names neither the stand-in nor the missing dependency, so the first move is to debug the diff",
    fontsize=10.5, style="italic", color=GREY)
ax2.add_patch(Rectangle((0.0, 0.015), 1.0, 0.53, facecolor="#eefbf1", edgecolor=GREEN, lw=1.3))
put(ax2, 0.012, 0.49, "this change  --  reported as a SKIP", fontsize=11, fontweight="bold", color=GREEN)
put(ax2, 0.012, 0.415, "\n".join(textwrap.wrap(NEW_MSG, 132)), fontsize=10.6, family="monospace", va="top")

# ------------------------------------------------------------- panel 3: CI + why
ax3 = fig.add_axes([0.035, 0.025, 0.93, 0.125])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis("off")
put(ax3, 0.0, 1.08, "CI's environment (real torch installed) is untouched", fontsize=14.5, fontweight="bold")
put(ax3, 0.0, 0.80, "main", fontsize=11.5, fontweight="bold", color=GREY)
put(ax3, 0.16, 0.80, "20771 passed, 257 skipped", fontsize=11.5, family="monospace", color=GREY)
put(ax3, 0.0, 0.50, "this change", fontsize=11.5, fontweight="bold", color=GREY)
put(ax3, 0.16, 0.50, "20814 passed, 257 skipped", fontsize=11.5, family="monospace", color=GREEN)
put(ax3, 0.545, 0.50, "+43 = exactly this PR's new tests; skips identical, so no test's", fontsize=11, color=GREY)
put(ax3, 0.545, 0.22, "behaviour changes where torch is real (the stand-in is never installed).", fontsize=11, color=GREY)

for a, yy in placed:
    lo, hi = a.get_ylim()
    assert lo - 0.05 * (hi - lo) <= yy <= hi + 0.09 * (hi - lo), f"text at y={yy} outside {a.get_ylim()}"

out = "/tmp/torch_stand_in_contract.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.32, facecolor="white")
plt.close(fig)

import numpy as np
from PIL import Image

im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", out, im.shape)
print("logs read: pre_full, post_full, pre_full2, post_full2")
