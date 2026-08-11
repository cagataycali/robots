from __future__ import annotations
import json, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RID = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/art-main-{RID}.json").read_text())    # upstream/main
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RID}.json").read_text())  # this PR
assert A["tree"] != B["tree"], "both arms measured the same tree"

# --- claims asserted before anything is drawn ---
assert A["has_stop_tool"] is False and B["has_stop_tool"] is True
assert A["halt_tool"] is None and B["halt_tool"] == "stop_rover"
assert A["ros2_tools"] == ["drive_rover", "get_pose_rover"], A["ros2_tools"]
assert B["ros2_tools"] == ["drive_rover", "get_pose_rover", "stop_rover"], B["ros2_tools"]
# the public method behaves identically on both trees
assert A["stop_method_wire"] == B["stop_method_wire"] == [{"linear_x": 0.0, "count": 1}]
# the latching drive is identical on both trees
assert A["session"][0]["published"] == B["session"][0]["published"] == [{"linear_x": 0.5, "count": 1}]
# cross-transport: main 2 of 3, PR 3 of 3
a_stop = sum(1 for v in A["transports"].values() if v["stop_tool"])
b_stop = sum(1 for v in B["transports"].values() if v["stop_tool"])
assert (a_stop, b_stop) == (2, 3), (a_stop, b_stop)
assert all(v["stop_method"] for v in A["transports"].values()), "every bridge has stop()"

RED, GREEN, INK, MUTED = "#c0392b", "#1e8449", "#17202a", "#7f8c8d"
placed: list[tuple[object, float, bool]] = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 9.4), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], width_ratios=[1.32, 1.0],
                      hspace=0.30, wspace=0.16, left=0.045, right=0.972, top=0.885, bottom=0.055)

fig.suptitle("A latching ROS 2 velocity command with no halt in the agent's tool surface",
             fontsize=16.5, fontweight="bold", y=0.968)
fig.text(0.5, 0.925,
         "One agent session on RosBridgedRobot: drive, then halt. Recorded off the forwarded use_ros call "
         "(no ROS 2 present).",
         ha="center", fontsize=10.6, color=MUTED, style="italic")

# ---------- row 1: the wire trace ----------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
put(ax, 0.0, 0.955, "cmd_vel messages published during the session", fontsize=12.4,
    fontweight="bold", color=INK, transform=ax.transAxes)

lanes = [("upstream/main", A, RED, 0.70), ("this PR", B, GREEN, 0.28)]
for label, data, col, ybase in lanes:
    put(ax, 0.0, ybase + 0.135, label, fontsize=11.6, fontweight="bold", color=col,
        transform=ax.transAxes)
    put(ax, 0.0, ybase + 0.045, "tools: " + ", ".join(data["ros2_tools"]), fontsize=9.9,
        family="monospace", color=MUTED, transform=ax.transAxes)
    ax.plot([0.30, 0.985], [ybase, ybase], color="#d5d8dc", lw=1.4, zorder=1)
    xs = [0.40, 0.72]
    for (step, x) in zip(data["session"], xs, strict=True):
        pub = step["published"]
        if pub:
            v = pub[0]["linear_x"]
            mark = GREEN if v == 0.0 else "#2471a3"
            ax.plot([x], [ybase], marker="o", ms=13, color=mark, zorder=3)
            put(ax, x, ybase + 0.058, f"linear.x = {v}", fontsize=9.6, ha="center",
                family="monospace", color=INK, transform=ax.transAxes)
            put(ax, x, ybase - 0.075, step["step"], fontsize=9.3, ha="center",
                family="monospace", color=MUTED, transform=ax.transAxes)
        else:
            ax.plot([x], [ybase], marker="X", ms=15, color=RED, zorder=3)
            put(ax, x, ybase + 0.058, "nothing published", fontsize=9.6, ha="center",
                fontweight="bold", color=RED, transform=ax.transAxes)
            put(ax, x, ybase - 0.075, "no stop tool to call", fontsize=9.3, ha="center",
                family="monospace", color=RED, transform=ax.transAxes)
    if data["halt_tool"] is None:
        ax.add_patch(Rectangle((0.66, ybase - 0.105), 0.125, 0.215, transform=ax.transAxes,
                               facecolor=RED, alpha=0.09, edgecolor=RED, lw=1.0, zorder=0))
        put(ax, 0.80, ybase, "base still latched at 0.5 m/s", fontsize=10.2, va="center",
            fontweight="bold", color=RED, transform=ax.transAxes)
    else:
        put(ax, 0.80, ybase, "base halted", fontsize=10.2, va="center",
            fontweight="bold", color=GREEN, transform=ax.transAxes)

put(ax, 0.0, 0.045,
    "The drive is byte-identical on both trees (one message, linear.x = 0.5, no trailing zero): the change is "
    "only what the agent can reach next.",
    fontsize=9.7, color=MUTED, style="italic", transform=ax.transAxes)

# ---------- row 2 left: cross-transport table ----------
axt = fig.add_subplot(gs[1, 0])
axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off")
put(axt, 0.0, 0.965, "Every bridge carries a public stop() - only two advertised it",
    fontsize=12.2, fontweight="bold", color=INK, transform=axt.transAxes)
hdr_y, last_y = 0.845, 0.235
rows = list(A["transports"].items())
step_y = (hdr_y - last_y) / (len(rows) - 1)
assert step_y > 0.030, step_y
put(axt, 0.005, hdr_y + 0.075, "transport", fontsize=10.1, fontweight="bold",
    color=MUTED, transform=axt.transAxes)
put(axt, 0.455, hdr_y + 0.075, "stop() method", fontsize=10.1, fontweight="bold",
    color=MUTED, transform=axt.transAxes)
put(axt, 0.655, hdr_y + 0.075, "stop tool (main)", fontsize=10.1, fontweight="bold",
    color=MUTED, transform=axt.transAxes)
put(axt, 0.885, hdr_y + 0.075, "this PR", fontsize=10.1, fontweight="bold",
    color=MUTED, transform=axt.transAxes)
y = hdr_y
for label, av in rows:
    bv = B["transports"][label]
    put(axt, 0.005, y, label, fontsize=10.4, family="monospace", color=INK, transform=axt.transAxes)
    put(axt, 0.455, y, "yes", fontsize=10.4, family="monospace", color=GREEN, transform=axt.transAxes)
    put(axt, 0.655, y, "yes" if av["stop_tool"] else "MISSING", fontsize=10.4, fontweight="bold",
        family="monospace", color=GREEN if av["stop_tool"] else RED, transform=axt.transAxes)
    put(axt, 0.885, y, "yes" if bv["stop_tool"] else "MISSING", fontsize=10.4, fontweight="bold",
        family="monospace", color=GREEN if bv["stop_tool"] else RED, transform=axt.transAxes)
    if not av["stop_tool"]:
        axt.add_patch(Rectangle((-0.005, y - 0.038), 1.0, 0.098, transform=axt.transAxes,
                                facecolor=RED, alpha=0.075, edgecolor="none", zorder=0))
    y -= step_y
assert abs((y + step_y) - last_y) < 1e-9, (y, last_y)
put(axt, 0.005, 0.09, f"advertised halts: {a_stop} of 3  ->  {b_stop} of 3", fontsize=11.0,
    fontweight="bold", color=INK, transform=axt.transAxes)

# ---------- row 2 right: measured facts ----------
axf = fig.add_subplot(gs[1, 1])
axf.set_xlim(0, 1); axf.set_ylim(0, 1); axf.axis("off")
put(axf, 0.0, 0.965, "Measured", fontsize=12.2, fontweight="bold", color=INK, transform=axf.transAxes)
facts = [
    ("drive with no duration", "1 message, linear.x = 0.5 (latches)"),
    ("stop() on the wire", "1 message, linear.x = 0.0 - identical on both trees"),
    ("halt reachable via tools (main)", "no"),
    ("halt reachable via tools (PR)", "stop_rover() -> zero Twist"),
    ("tests failing before the fix", "6 (5 new + the exact-set pin)"),
    ("tool closures never invoked", "4 of 10 -> 0 of 10"),
]
ftop, flast = 0.845, 0.185
fstep = (ftop - flast) / (len(facts) - 1)
assert fstep > 0.030, fstep
fy = ftop
for k, v in facts:
    put(axf, 0.005, fy, k, fontsize=10.2, color=MUTED, transform=axf.transAxes)
    put(axf, 0.005, fy - 0.052, v, fontsize=10.4, family="monospace", color=INK,
        transform=axf.transAxes)
    fy -= fstep
assert abs((fy + fstep) - flast) < 1e-9, (fy, flast)

out = pathlib.Path("_art/ros2_stop_tool.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

# ---- layout audit ----
for ax_obj, yv, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= yv <= 1.10, f"axes-fraction y={yv} outside panel"
    else:
        lo, hi = ax_obj.get_ylim()
        assert lo - 0.05 <= yv <= hi + 0.07, f"data y={yv} outside {(lo, hi)}"
import numpy as np
im = np.asarray(matplotlib.image.imread(out) * 255, dtype=int)[:, :, :3]
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print("figure OK", out, im.shape)
