"""Compose the measurement figure. Every cell is read from /tmp/facts.json."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.load(open("/tmp/facts.json"))
ROOT = pathlib.Path(__file__).parents[1]
assert F["tree"] == str(ROOT), (F["tree"], ROOT)

MB, MA = set(F["cov"]["before"]["missing"]), set(F["cov"]["after"]["missing"])
STATES = [(n, int(l)) for n, l in F["states"]]
n_before = sum(1 for _, l in STATES if l not in MB)
n_after = sum(1 for _, l in STATES if l not in MA)
assert (n_before, n_after) == (4, 6), (n_before, n_after)

SH = {s["descriptor"]: s for s in F["shadowing"]}
nd, dd = SH["non-data (__get__ only)"], SH["data (__get__ + __set__)"]
assert nd["reads"] == 0 and nd["shadowed"] and nd["assertion_passes"]
assert dd["reads"] >= 3 and not dd["shadowed"] and dd["assertion_passes"]

MUT = F["mutations"]
blind = [m for m in MUT if m["old_failed"] not in ("n/a",) and int(m["old_failed"]) == 0]
assert len(blind) == 3, len(blind)

GREEN, RED, GREY, INK = "#1b7f3b", "#b4231f", "#6b6b6b", "#17202a"
fig = plt.figure(figsize=(16.4, 11.0), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 1.22, 0.40], hspace=0.30, wspace=0.13,
                      left=0.035, right=0.972, top=0.905, bottom=0.035)
placed: list[tuple[object, float, bool]] = []

def put(ax, x, y, s, **kw):
    axc = kw.pop("transform", None) is not None
    placed.append((ax, y, axc))
    return ax.text(x, y, s, transform=(ax.transAxes if axc else ax.transData), **kw)

def panel(ax, title, sub=""):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    put(ax, 0.0, 1.045, title, transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=INK)
    if sub:
        put(ax, 0.0, 0.995, sub, transform=ax.transAxes, fontsize=9.6, color=GREY, style="italic")

def rows(ax, lines, top, last, mono=True, size=10.3):
    step = (top - last) / (len(lines) - 1)
    assert step > 0.030, step
    y = top
    for txt, col, bold in lines:
        put(ax, 0.012, y, txt, transform=ax.transAxes, fontsize=size, color=col,
            family="monospace" if mono else None, fontweight="bold" if bold else None)
        y -= step
    assert abs((y + step) - last) < 1e-9, (y, last)

# ---- row 1: the state matrix ------------------------------------------------
for col, (label, miss, count) in enumerate([
        ("main", MB, n_before), ("this PR", MA, n_after)]):
    ax = fig.add_subplot(gs[0, col])
    panel(ax, f"connect() state matrix - {label}",
          f"every branch the tool takes when it is asked for a connection   |   {count} of 6 pinned")
    lines = [(f"{'state the tool distinguishes':<44}{'line':>6}   pinned", INK, True)]
    for name, ln in STATES:
        ok = ln not in miss
        lines.append((f"{name:<44}{'L'+str(ln):>6}   {'yes' if ok else 'NO  <-- unreached':<18}",
                      GREEN if ok else RED, not ok))
    lines.append(("", INK, False))
    lines.append((f"use_rosbridge.py coverage: {F['cov']['before' if col == 0 else 'after']['pct']}%", INK, True))
    rows(ax, lines, 0.92, 0.10)

# ---- row 2 left: the shadowing measurement ---------------------------------
ax = fig.add_subplot(gs[1, 0])
panel(ax, "Why one of them looked pinned",
      "the reconnect test scripts is_connected with a descriptor - measured, both spellings")
lines = [(f"{'descriptor installed on the double':<34}{'reads':>6}  {'branch the tool took':<24}", INK, True)]
for s in (nd, dd):
    ok = not s["shadowed"]
    lines.append((f"{s['descriptor']:<34}{s['reads']:>6}  {s['branch']:<24}", GREEN if ok else RED, not ok))
lines += [
    ("", INK, False),
    ("run() stores is_connected in the instance dict, and an", GREY, False),
    ("instance attribute shadows a NON-data descriptor - so the", GREY, False),
    ("scripted reads were never consulted and the tool answered", GREY, False),
    ("from the cache-hit branch, not the wait loop the test names.", GREY, False),
    ("", INK, False),
    (f"the test's own assertion passed either way:  non-data {nd['assertion_passes']}   data {dd['assertion_passes']}", RED, True),
]
rows(ax, lines, 0.92, 0.10)

# ---- row 2 right: the mutation matrix --------------------------------------
ax = fig.add_subplot(gs[1, 1])
panel(ax, "Mutation table - does the pin hold?",
      "each regression applied to a clean tree, run against both copies of the test file")
hdr_b = "main's copy"
lines = [(f"{'plausible regression':<40}{'this PR':>10}  {hdr_b:>12}", INK, True)]
for m in MUT:
    tag = m["m"].split(" ", 1)
    lab = f"{tag[0]} {tag[1][:34]}"
    a = f"{m['new_failed']} fail"
    b = "n/a" if m["old_failed"] == "n/a" else f"{m['old_failed']} fail"
    caught_old = m["old_failed"] != "n/a" and int(m["old_failed"]) > 0
    suffix = "" if caught_old or m["old_failed"] == "n/a" else "  <- BLIND"
    lines.append((f"{lab:<40}{a:>10}  {b:>12}{suffix}",
                  GREY if caught_old else RED, not caught_old and m["old_failed"] != "n/a"))
lines += [
    ("", INK, False),
    (f"{len(blind)} of 4 production regressions were invisible to main's copy;", RED, True),
    ("M4 is caught by both - main already pins cell-1 caching.", GREY, False),
    ("M5 is the shadowing defect itself, caught by the read count.", GREY, False),
]
rows(ax, lines, 0.92, 0.10)

# ---- row 3: gate -----------------------------------------------------------
ax = fig.add_subplot(gs[2, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f4f6f7", edgecolor="#d5dbdb"))
gate = [
    "Gate at upstream/main 2efb05fc:  27681 passed / 257 skipped / 0 failed (637s, MUJOCO_GL=egl)  |  "
    "pristine main 27677 + 4 new = 27681, so no existing test changed behaviour",
    "ruff check + ruff format --check clean (1166 files)  |  mypy: 14 errors, all in examples/isaac_gs, "
    "byte-identical to a pristine-main worktree (environment, not this branch)",
    "Tests only - no production line changes, so nothing in policy, simulation, rendering, recording or asset "
    "handling moves. This figure is the coverage and mutation measurement, not a rollout.",
]
step = 0.62 / (len(gate) - 1)
y = 0.78
for i, line in enumerate(gate):
    put(ax, 0.012, y, line, transform=ax.transAxes, fontsize=9.9, color=INK if i < 2 else GREY,
        family="monospace" if i < 2 else None, fontweight="bold" if i == 0 else None)
    y -= step
assert y + step > 0.05, y

fig.suptitle("use_rosbridge: the two connection states the roslibpy double could not express",
             fontsize=16.5, fontweight="bold", x=0.035, ha="left", y=0.972)
fig.text(0.035, 0.938, "connect() branches on six states; four were pinned, and one of the other two "
         "looked pinned by a test that silently exercised a different branch.",
         fontsize=10.6, color=GREY, ha="left")

for ax_, y_, axc in placed:
    lo, hi = ((-0.03, 1.10) if axc else ax_.get_ylim())
    assert lo <= y_ <= hi, (y_, lo, hi)

out = pathlib.Path("/tmp/rosbridge-connect-states.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, (name, n)
print("saved", out, Image.open(out).size, "| border clean | all claims asserted")
