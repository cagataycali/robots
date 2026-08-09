from __future__ import annotations
import json, pathlib, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

A = json.loads(pathlib.Path("/tmp/art/main/facts.json").read_text())
B = json.loads(pathlib.Path("/tmp/art/branch/facts.json").read_text())
assert A["tree"] != B["tree"], "both dumps came from the same tree"

def row(f, label):
    return next(r for r in f["rows"] if r["label"] == label)

a_ok, b_ok = row(A, "declared_as_list"), row(B, "declared_as_list")
a_bad, b_bad = row(A, "declared_as_bare_string"), row(B, "declared_as_bare_string")

def img(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(int)

def delta(p, q):
    return int(np.abs(img(p) - img(q)).max())

def diff_frac(p, q):
    return float((np.abs(img(p) - img(q)).sum(2) > 12).mean())

# ---- self-audit -------------------------------------------------------------
d_ref = delta(a_ok["png_after"], b_ok["png_after"])
assert d_ref <= 2, f"honored evaluation differs across trees: max|delta|={d_ref}"
assert a_ok["eval_status"] == b_ok["eval_status"] == "success"
assert a_ok["declared"] == b_ok["declared"] == ["panda"]
assert a_ok["joints"] == b_ok["joints"], "honored rollout ended in a different pose"

d_none = delta(a_bad["png_after"], b_bad["png_after"])
assert d_none <= 2, f"the two never-ran states differ: max|delta|={d_none}"
f_ran = diff_frac(a_ok["png_after"], a_bad["png_after"])
assert f_ran > 0.10, f"reference evaluation is not legible: {f_ran:.2%} differing"

assert a_bad["declared"] == ["p", "a", "n", "d", "a"], a_bad["declared"]
assert a_bad["construction"] == "accepted" and a_bad["eval_status"] == "error"
assert b_bad["construction"] == "REFUSED" and b_bad["eval_status"] == "(not reached)"
print(f"audit ok: ref delta={d_ref}  never-ran delta={d_none}  ran-vs-not={f_ran:.2%}")

# ---- figure -----------------------------------------------------------------
placed: list[tuple[object, float, bool]] = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)

MONO = dict(family="monospace", fontsize=8.4, va="top", linespacing=1.45)
fig = plt.figure(figsize=(16.2, 12.6), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.42, 0.78, 0.62], hspace=0.10, wspace=0.05)

fig.suptitle(
    "DeclarativeBenchmark: a robot name spelled without the list\n"
    "MuJoCo headless (MUJOCO_GL=egl), panda + MockPolicy, 120-step benchmark evaluation",
    fontsize=14, fontweight="bold", y=0.975)

panels = [
    (a_ok, "REFERENCE  supported_robots=['panda']\nevaluation ran (identical on both trees)", "#1a7f37"),
    (a_bad, "main  supported_robots='panda'\nconstructed, registered, evaluation REFUSED", "#b3261e"),
    (b_bad, "this change  supported_robots='panda'\nrefused at construction", "#1a7f37"),
]
for col, (r, title, colour) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(Image.open(r["png_after"]))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=10.2, color=colour, fontweight="bold", pad=7)
    moved = "arm moved: the 120 steps ran" if col == 0 else "arm at its home pose: nothing stepped"
    ax.set_xlabel(moved, fontsize=9.2, color=colour, labelpad=5)

reports = [
    (a_ok, "what a correctly declared benchmark does"),
    (a_bad, "what main reported"),
    (b_bad, "what this change reports"),
]
for col, (r, heading) in enumerate(reports):
    ax = fig.add_subplot(gs[1, col]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    put(ax, 0.02, 0.99, heading, transform=ax.transAxes, fontsize=9.6,
        fontweight="bold", va="top")
    body = [f"declared supported_robots : {r['declared']}",
            f"list_benchmarks()        : {r['list_benchmarks'] or '(n/a)'}",
            ""]
    body += textwrap.wrap(f"report: {r['message'] or r['eval_text']}", 74)[:6]
    put(ax, 0.02, 0.885, "\n".join(body), transform=ax.transAxes, **MONO)

ax = fig.add_subplot(gs[2, :]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
lines = [
    f"supported_robots='panda'   main: constructed, declared {a_bad['declared']}  (5 names, none of them 'panda')",
    f"                           this change: {b_bad['construction']} at construction, never registered",
    f"the benchmark's own default_robot='panda'   main: outside its own declared set -> evaluation error",
    "",
    f"honored path unchanged     supported_robots=['panda'] -> eval {a_ok['eval_status']} on both trees;"
    f" end pose identical; render max|delta| = {d_ref}/255",
    f"physics untouched          the two 'nothing ran' states agree to max|delta| = {d_none}/255;"
    f" the reference rollout differs from them on {f_ran:.1%} of pixels",
    f"trees                      main={A['tree']}   branch={B['tree']}",
]
put(ax, 0.008, 0.94, "measured", transform=ax.transAxes, fontsize=9.6, fontweight="bold", va="top")
TOP, FLOOR = 0.80, 0.06
STEP = (TOP - FLOOR) / len(lines)   # derived, not guessed
assert STEP > 0.030, f"fact-table pitch {STEP:.3f} is too tight to read"
y = TOP
for ln in lines:
    put(ax, 0.008, y, ln, transform=ax.transAxes, **MONO)
    y -= STEP
assert y > 0.02, f"fact table overflowed to y={y:.3f}"

for ax_, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.07, f"axes-fraction text at y={yy}"
    else:
        lo, hi = ax_.get_ylim()
        assert min(lo, hi) - 0.05 <= yy <= max(lo, hi) + 0.07, f"data text at y={yy}"

out = pathlib.Path("/tmp/art/benchmark_supported_robots.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("WROTE", out, Image.open(out).size)
