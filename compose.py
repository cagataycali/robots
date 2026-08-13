"""Compose the GL-probe latch artifact from the two measured JSON dumps."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = json.loads(pathlib.Path("/tmp/glart-main.json").read_text())
B = json.loads(pathlib.Path("/tmp/glart-branch.json").read_text())
assert A["tree"] != B["tree"], "both arms measured the same tree"
assert A["cleared_cache_constructions"] == 1, A
assert B["cleared_cache_constructions"] == 0, B
assert A["latch_symbol_present"] is False and B["latch_symbol_present"] is True

fa = np.load("/tmp/glart-frame-main.npy")
fb = np.load("/tmp/glart-frame-branch.npy")
delta = int(np.abs(fa.astype(int) - fb.astype(int)).max())
assert delta <= 2, f"the render moved across trees: {delta}"
assert B["render_saturated_frac"] > 0.10

placed: list[tuple[object, float, bool]] = []


def put(ax: object, x: float, y: float, s: str, **kw: object) -> None:
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    ax.text(x, y, s, **kw)  # type: ignore[attr-defined]


MONO = {"family": "monospace"}
fig = plt.figure(figsize=(15.4, 12.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.20, 0.78, 0.58], width_ratios=[1.02, 1.0],
                      hspace=0.20, wspace=0.10)

fig.suptitle(
    "tests/simulation/mujoco: the GL probe built its 1x1 renderer again after a cache_clear()",
    fontsize=15.5, fontweight="bold", y=0.972,
)
fig.text(0.5, 0.947,
         "A second construction on a host whose first attempt failed aborts the interpreter uncatchably, "
         "so it takes the rest of the session with it.",
         ha="center", fontsize=10.6, style="italic", color="#333")

# --- row 1 left: a real frame of the kind the gated tests verify ---------------
axr = fig.add_subplot(gs[0, 0])
axr.imshow(fb)
axr.set_xticks([]); axr.set_yticks([])
axr.set_title("A real headless offscreen render (MUJOCO_GL=egl)", fontsize=11.6, fontweight="bold")
axr.set_xlabel(
    f"the output every requires_gl-gated test verifies -- {B['render_saturated_frac'] * 100:.1f}% saturated,\n"
    f"crate settled at z={B['crate_settled_z']:.4f} m. Byte-comparable across both trees "
    f"(max|delta|={delta}/255): 0 production lines change.",
    fontsize=9.4,
)

# --- row 1 right: the run ledger ----------------------------------------------
axl = fig.add_subplot(gs[0, 1]); axl.axis("off")
axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.975, "Whole directory on a host with no usable GL", fontsize=12.2,
    fontweight="bold", transform=axl.transAxes)
put(axl, 0.0, 0.930,
    "MUJOCO_GL=glfw, no display -- 3122 tests collected", fontsize=9.6, style="italic",
    color="#444", transform=axl.transAxes)

rows = [
    ("", "on main", "with this change"),
    ("probe renderers built", "2", "1"),
    ("interpreter aborts", "1", "0"),
    ("progress reached", "39%", "91%"),
    ("tests never reached", "~1900", "0 (from this cause)"),
    ("first victim", "an unrelated", "-"),
    ("", "render test", ""),
]
TOP, LAST = 0.855, 0.475
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for label, m, p in rows:
    bold = label != "" and not label.startswith(" ")
    put(axl, 0.02, y, label, fontsize=10.0, fontweight="bold" if label == "" else "normal",
        transform=axl.transAxes)
    put(axl, 0.575, y, m, fontsize=10.0, color="#b02020" if label and m not in ("", "-") else "#333",
        fontweight="bold" if label == "" else "normal", **MONO, transform=axl.transAxes)
    put(axl, 0.795, y, p, fontsize=10.0, color="#1a7a1a" if label and p not in ("", "-") else "#333",
        fontweight="bold" if label == "" else "normal", **MONO, transform=axl.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y

axl.plot([0.02, 0.98], [0.885, 0.885], color="#999", lw=0.9, transform=axl.transAxes, clip_on=False)
axl.plot([0.02, 0.98], [0.815, 0.815], color="#ccc", lw=0.7, transform=axl.transAxes, clip_on=False)

put(axl, 0.0, 0.395, "The abort main takes, verbatim:", fontsize=10.6, fontweight="bold",
    transform=axl.transAxes)
trace = [
    "tests/.../test_load_scene_interaction.py ..........",
    "Fatal Python error: Aborted",
    "",
    "Current thread 0x... (most recent call first):",
    '  File ".../test_load_scene_interaction.py", line 350 in',
    "    test_load_scene_render_returns_real_geometry_immediately",
    "timeout: the monitored command dumped core",
]
TT, TL = 0.348, 0.075
TS = (TT - TL) / (len(trace) - 1)
assert TS > 0.030, TS
ty = TT
for line in trace:
    put(axl, 0.02, ty, line, fontsize=8.3, color="#7a1010", **MONO, transform=axl.transAxes)
    ty -= TS
assert abs((ty + TS) - TL) < 1e-9, ty
put(axl, 0.0, 0.018,
    "The probe's own test clears the cache; the abort lands in whichever\n"
    "later test calls gl_available() next -- here an unrelated render test.",
    fontsize=8.9, style="italic", color="#444", transform=axl.transAxes)

# --- row 2: construction table -------------------------------------------------
axc = fig.add_subplot(gs[1, :]); axc.axis("off")
axc.set_xlim(0, 1); axc.set_ylim(0, 1)
put(axc, 0.0, 0.955, "Renderers constructed by one gl_available() after cache_clear()  (measured in-process)",
    fontsize=12.0, fontweight="bold", transform=axc.transAxes)

crows = [
    ("tree", "latch symbol", "constructions", "verdict", None),
    (f"main   ({A['tree'].split('/')[-1]})", "absent", f"{A['cleared_cache_constructions']}",
     "the hardware is probed a second time", "#b02020"),
    (f"branch ({B['tree'].split('/')[-1]})", "present", f"{B['cleared_cache_constructions']}",
     "the latched answer is returned", "#1a7a1a"),
]
CT, CL = 0.760, 0.430
CS = (CT - CL) / (len(crows) - 1)
assert CS > 0.030, CS
cy = CT
for tree, latch, n, verdict, col in crows:
    hdr = col is None
    put(axc, 0.02, cy, tree, fontsize=9.9, fontweight="bold" if hdr else "normal", **MONO,
        transform=axc.transAxes)
    put(axc, 0.30, cy, latch, fontsize=9.9, fontweight="bold" if hdr else "normal", **MONO,
        transform=axc.transAxes)
    put(axc, 0.44, cy, n, fontsize=9.9, fontweight="bold", color=col or "#333", **MONO,
        transform=axc.transAxes)
    put(axc, 0.56, cy, verdict, fontsize=9.9, fontweight="bold" if hdr else "normal",
        color=col or "#333", transform=axc.transAxes)
    cy -= CS
assert abs((cy + CS) - CL) < 1e-9, cy
axc.plot([0.02, 0.98], [0.700, 0.700], color="#999", lw=0.9, transform=axc.transAxes, clip_on=False)

put(axc, 0.0, 0.330, "Pre-fix proof, public surface only  (tests/simulation/mujoco/_gl_probe.py reverted to main)",
    fontsize=11.0, fontweight="bold", transform=axc.transAxes)
proof = [
    ("on main's source", "E  AssertionError: the cleared cache re-ran the renderer construction: ['constructed']", "#b02020"),
    ("on this branch", "1 passed in 0.29s", "#1a7a1a"),
]
PT, PL = 0.235, 0.140
PS = (PT - PL) / (len(proof) - 1)
py_ = PT
for label, text, col in proof:
    put(axc, 0.02, py_, label, fontsize=9.6, transform=axc.transAxes)
    put(axc, 0.22, py_, text, fontsize=9.0, color=col, **MONO, transform=axc.transAxes)
    py_ -= PS
put(axc, 0.0, 0.035,
    "Whole suite on this branch: 29457 passed / 266 skipped / 0 failed (11m, MUJOCO_GL=egl); "
    "3 files changed, 0 lines under strands_robots/.",
    fontsize=9.4, style="italic", color="#444", transform=axc.transAxes)

# --- row 3: mutation matrix ----------------------------------------------------
axm = fig.add_subplot(gs[2, :]); axm.axis("off")
axm.set_xlim(0, 1); axm.set_ylim(0, 1)
put(axm, 0.0, 0.945, "Mutation matrix  (5 plausible regressions x 2 arms, tests/simulation/mujoco/_gl_probe.py)",
    fontsize=12.0, fontweight="bold", transform=axm.transAxes)

muts = [
    ("mutation", "new tests", "3 pre-existing", None),
    ("(unmutated control)", "0 bad / 6 pass", "0 bad / 3 pass", "#333"),
    ("M1  drop the early-out (the original defect)", "2 bad", "0 bad   <- BLIND", "#b02020"),
    ("M2  latch only on success (retry a graceful failure)", "1 bad", "0 bad   <- BLIND", "#b02020"),
    ("M3  the force-skip poisons the hardware latch", "1 bad", "0 bad   <- BLIND", "#b02020"),
    ("M4  probe before honouring the force-skip", "1 bad", "0 bad   <- BLIND", "#b02020"),
    ("M5  drop the global so the latch never persists", "1 bad", "1 bad", "#555"),
]
MT, ML = 0.830, 0.185
MS = (MT - ML) / (len(muts) - 1)
assert MS > 0.030, MS
my_ = MT
for label, new, old, col in muts:
    hdr = col is None
    put(axm, 0.02, my_, label, fontsize=9.7, fontweight="bold" if hdr else "normal", **MONO,
        transform=axm.transAxes)
    put(axm, 0.545, my_, new, fontsize=9.7, fontweight="bold" if not hdr else "bold",
        color="#1a7a1a" if not hdr and "bad" in new and new != "0 bad / 6 pass" else "#333",
        **MONO, transform=axm.transAxes)
    put(axm, 0.700, my_, old, fontsize=9.7, fontweight="bold" if hdr else "normal",
        color=col if not hdr else "#333", **MONO, transform=axm.transAxes)
    my_ -= MS
assert abs((my_ + MS) - ML) < 1e-9, my_
axm.plot([0.02, 0.98], [0.875, 0.875], color="#999", lw=0.9, transform=axm.transAxes, clip_on=False)

put(axm, 0.0, 0.075,
    "5 of 5 caught here; 4 of 5 invisible to the 3 pre-existing cases. M5 is the one both catch "
    "(the latch write becomes local, so the module fails at import).",
    fontsize=9.4, style="italic", color="#444", transform=axm.transAxes)
put(axm, 0.0, 0.010,
    "M4 was invisible until a child-interpreter test was added: with the latch already primed, "
    "the ordering is unobservable in-process.",
    fontsize=9.4, style="italic", color="#444", transform=axm.transAxes)

for ax, yy, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yy <= 1.08, f"text at y={yy}"
    else:
        lo, hi = ax.get_ylim()
        assert min(lo, hi) - 0.05 <= yy <= max(lo, hi) + 0.07, f"data-coord text at y={yy}"

OUT = pathlib.Path("/tmp/gl-probe-latch.png")
fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.array(__import__("PIL.Image", fromlist=["Image"]).open(OUT).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"wrote {OUT}  {im.shape[1]}x{im.shape[0]}  render delta across trees={delta}/255")
