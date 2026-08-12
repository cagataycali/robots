from __future__ import annotations
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

F = json.loads(pathlib.Path(f"/tmp/art-rosbridge-{sys.argv[1]}.json").read_text())
ROOT = pathlib.Path(__file__).resolve().parents[1]
assert F["tree"] == str(ROOT), F["tree"]
assert F["restored_identically"] is True

RED, GREEN, GREY, INK = "#c0392b", "#1e8449", "#7f8c8d", "#1c2833"
MONO = {"family": "monospace"}
placed: list[tuple] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

# ---- derived facts, asserted -------------------------------------------------
cb, ca = F["cov_before"], F["cov_after"]
assert cb["miss"] == 2 and ca["miss"] == 0, (cb, ca)
assert cb["pct"] == 99 and ca["pct"] == 100
assert ca["passed"] - cb["passed"] == 15, (cb["passed"], ca["passed"])
HOLES = set(cb["missing"])
assert len(HOLES) == 2
muts = F["mutations"]
blind = [m for m in muts if m["old_failed"] == 0]
caught_new = [m for m in muts if m["new_failed"] > 0]
assert len(caught_new) == len(muts) == 6, len(muts)
assert len(blind) == 5, [m["label"] for m in blind]
assert F["control"]["new_failed"] == 0 and F["control"]["old_failed"] == 0
# parity: every row must agree
assert all(p["topic_refused"] == p["service_refused"] for p in F["parity"]), F["parity"]
n_refused = sum(1 for p in F["parity"] if p["topic_refused"])
assert n_refused == 4 and len(F["parity"]) == 6
w = {t["label"]: t["steps"][0] for t in F["wire"]}
svc = w["a mistyped service name"]
inc = w["a publish missing its interface type"]
com = w["the same publish, complete"]
assert svc["clients_dialed"] == 0 and svc["advertised"] == []
assert inc["clients_dialed"] == 1 and inc["advertised"] == []
assert com["clients_dialed"] == 1 and com["advertised"] == [["/cmd_vel", True, True, 1]]
assert "invalid service name" in svc["text"] and "publish requires topic and type" in inc["text"]

fig = plt.figure(figsize=(15.6, 12.4), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 0.78, 0.92], hspace=0.20,
                      left=0.035, right=0.975, top=0.945, bottom=0.035)
fig.suptitle("use_rosbridge: the two refusals nothing reached", fontsize=17, fontweight="bold", y=0.982)
fig.text(0.5, 0.958, "every number below is read from one measurement dump; the production diff is docstring-only",
         ha="center", fontsize=10.5, style="italic", color=GREY)

# ---------- row 1: the refusal inventory -------------------------------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.045, "1  Every refusal `use_rosbridge` makes, and which the suite reached",
    transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
rows = F["refusals"]
TOP, LAST = 0.965, 0.045
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
put(ax, 0.055, TOP + 0.045, "line", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=GREY, **MONO)
put(ax, 0.115, TOP + 0.045, "before", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=GREY)
put(ax, 0.205, TOP + 0.045, "after", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=GREY)
put(ax, 0.275, TOP + 0.045, "refusal", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=GREY)
y = TOP
for r in rows:
    hole = r["line"] in HOLES
    if hole:
        ax.add_patch(plt.Rectangle((0.03, y - 0.017), 0.955, 0.037, transform=ax.transAxes,
                                   facecolor="#fdecea", edgecolor="none", zorder=0))
    put(ax, 0.055, y, f'{r["line"]:>4}', transform=ax.transAxes, fontsize=9.5, color=INK, **MONO)
    put(ax, 0.115, y, "NOT REACHED" if hole else "reached", transform=ax.transAxes,
        fontsize=9.5, color=RED if hole else GREEN, fontweight="bold" if hole else "normal")
    put(ax, 0.205, y, "reached", transform=ax.transAxes, fontsize=9.5, color=GREEN)
    txt = r["text"].replace("return _err(", "").rstrip(")")
    put(ax, 0.275, y, txt[:92], transform=ax.transAxes, fontsize=9.0, color=INK, **MONO)
    y -= step
assert abs((y + step) - LAST) < 1e-9, y
put(ax, 0.03, -0.055,
    f'tools/use_rosbridge.py  {cb["stmts"]} statements:  {cb["miss"]} missing / {cb["pct"]}%'
    f'  ->  {ca["miss"]} missing / {ca["pct"]}%     '
    f'({cb["passed"]} cases -> {ca["passed"]}, +{ca["passed"]-cb["passed"]})',
    transform=ax.transAxes, fontsize=10.5, color=INK, fontweight="bold", **MONO)
put(ax, 0.03, -0.100,
    "The two holes are the `service` name refusal - whose `topic` twin, checked against the same _NAME_RE one line above, was pinned -",
    transform=ax.transAxes, fontsize=9.5, color=GREY, style="italic")
put(ax, 0.03, -0.140,
    "and the `publish` required-argument refusal, the only action in that family that writes to the robot. Both siblings were pinned.",
    transform=ax.transAxes, fontsize=9.5, color=GREY, style="italic")

# ---------- row 2: the wire trace -------------------------------------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 1.10, "2  What reached the bridge  (recorded off the roslibpy double)",
    transform=ax2.transAxes, fontsize=13, fontweight="bold", color=INK)
lanes = [
    ("a mistyped service name", svc, "refused before the WebSocket is dialed"),
    ("a publish missing its type", inc, "dialed, but no publisher advertised"),
    ("the same publish, complete", com, "advertise -> publish -> unadvertise"),
]
LT, LL = 0.86, 0.10
lstep = (LT - LL) / (len(lanes) - 1)
for label, st, note in lanes:
    yy = LT - lanes.index((label, st, note)) * lstep
    ok = st["status"] == "success"
    put(ax2, 0.012, yy + 0.035, label, transform=ax2.transAxes, fontsize=10.5, fontweight="bold", color=INK)
    put(ax2, 0.012, yy - 0.045, note, transform=ax2.transAxes, fontsize=9.0, style="italic", color=GREY)
    ax2.add_patch(plt.Rectangle((0.30, yy - 0.055), 0.685, 0.115, transform=ax2.transAxes,
                                facecolor="#eafaf1" if ok else "#fdecea", edgecolor="#d5d8dc", zorder=0))
    put(ax2, 0.315, yy + 0.020,
        f'clients dialed: {st["clients_dialed"]}     publishers advertised: {len(st["advertised"])}'
        + (f'   -> {st["advertised"][0][0]} advertised={st["advertised"][0][1]} '
           f'unadvertised={st["advertised"][0][2]} published={st["advertised"][0][3]}'
           if st["advertised"] else ""),
        transform=ax2.transAxes, fontsize=9.3, color=INK, **MONO)
    put(ax2, 0.315, yy - 0.030, f'{st["status"]}: {st["text"][:78]}',
        transform=ax2.transAxes, fontsize=9.3, color=GREEN if ok else RED, **MONO)
put(ax2, 0.012, -0.05,
    f'One name rule for two parameters: over {len(F["parity"])} spellings, `topic` and `service` agree on every one '
    f'({n_refused} refused, {len(F["parity"])-n_refused} carried past the name check).',
    transform=ax2.transAxes, fontsize=9.5, color=GREY, style="italic")

# ---------- row 3: mutation matrix ------------------------------------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 1.075, "3  Would a regression be caught?", transform=ax3.transAxes,
    fontsize=13, fontweight="bold", color=INK)
put(ax3, 0.545, 0.985, "these 15 cases", transform=ax3.transAxes, fontsize=9.5, fontweight="bold", color=GREY)
put(ax3, 0.700, 0.985, "the 82 pre-existing", transform=ax3.transAxes, fontsize=9.5, fontweight="bold", color=GREY)
MT, ML = 0.905, 0.175
mstep = (MT - ML) / len(muts)
for i, m in enumerate(muts):
    yy = MT - i * mstep
    put(ax3, 0.012, yy, m["label"], transform=ax3.transAxes, fontsize=10.0, color=INK)
    put(ax3, 0.545, yy, f'{m["new_failed"]} failed', transform=ax3.transAxes, fontsize=10.0,
        color=GREEN, fontweight="bold", **MONO)
    is_blind = m["old_failed"] == 0
    put(ax3, 0.700, yy, f'{m["old_failed"]} failed' + ("   <- BLIND" if is_blind else "   also caught"),
        transform=ax3.transAxes, fontsize=10.0, color=RED if is_blind else GREY,
        fontweight="bold" if is_blind else "normal", **MONO)
c = F["control"]
put(ax3, 0.012, ML - 0.035, "(unmutated control)", transform=ax3.transAxes, fontsize=10.0, style="italic", color=GREY)
put(ax3, 0.545, ML - 0.035, f'{c["new_failed"]} failed', transform=ax3.transAxes, fontsize=10.0, color=GREY, **MONO)
put(ax3, 0.700, ML - 0.035, f'{c["old_failed"]} failed', transform=ax3.transAxes, fontsize=10.0, color=GREY, **MONO)
put(ax3, 0.012, 0.045,
    f'{len(caught_new)} of {len(muts)} caught here; {len(blind)} of {len(muts)} invisible to the suite as it stands. '
    "M3 is caught by both - applying the interface-type rule to a service name also refuses a valid one.",
    transform=ax3.transAxes, fontsize=9.5, color=GREY, style="italic")
put(ax3, 0.012, -0.005,
    "Tests only: no library behaviour changes. The docstring-stripped AST digest of use_rosbridge.py is 2cd51ef54807b823 before and after.",
    transform=ax3.transAxes, fontsize=9.5, color=INK, style="italic")

# ---- layout guard ----
for a, yv, is_axes in placed:
    if is_axes:
        assert -0.16 <= yv <= 1.12, f"text out of band at y={yv}"
    else:
        lo, hi = a.get_ylim(); assert lo - 0.05 <= yv <= hi + 0.07, f"data y={yv} vs {(lo,hi)}"

out = pathlib.Path("/tmp/rosbridge-refusals.png")
fig.savefig(out, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nz = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nz == 0, f"{name} border has {nz} non-white px"
print("wrote", out, im.shape)
print(f"holes closed: {sorted(HOLES)} | {cb['pct']}% -> {ca['pct']}% | blind {len(blind)}/{len(muts)}")
