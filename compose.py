"""Compose the measured figure for the hardware ``policy_port`` domain."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

A = json.loads(pathlib.Path("/tmp/facts_main.json").read_text())
B = json.loads(pathlib.Path("/tmp/facts_branch.json").read_text())
assert A["tree"] != B["tree"], "both halves came from the same tree"

PORTS = list(A["rows"])
USABLE = {"5555"}

GREEN, RED, GREY = "#1b7f37", "#b3261e", "#5f6368"
BG_OK, BG_BAD = "#e8f5eb", "#fdeceb"

placed: list[tuple[object, float, bool]] = []


def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)


def honors(port: str, row: dict) -> bool:
    """Did this tree do what the contract asks for this port?"""
    if port in USABLE:
        return row["connects"] == 1 and not row["refused_for_port"]
    return row["start_status"] == "error" and row["connects"] == 0 and row["refused_for_port"]


def named(tree: dict, port: str) -> str:
    row = tree["rows"][port]
    blob = f"{row['start_text']} {row['final_error']}"
    if "invalid policy_port" in blob:
        return "start_task names policy_port"
    if "policy_port is required" in blob and row["start_status"] == "error":
        return "start_task: port required"
    if "policy_port is required" in blob:
        return "'port is required' (it was supplied)"
    if "Gr00tPolicy" in blob:
        return "Gr00tPolicy names the provider"
    if row["connects"] == 1 and not row["refused_for_port"]:
        return "reaches the policy server"
    return "-"


n_bad_main = sum(1 for p in PORTS if not honors(p, A["rows"][p]))
n_bad_branch = sum(1 for p in PORTS if not honors(p, B["rows"][p]))
assert n_bad_branch == 0, f"branch still diverges on {n_bad_branch} port(s)"
assert n_bad_main > 0, "nothing to fix"

fig = plt.figure(figsize=(15.6, 11.4), dpi=124)
gs = fig.add_gridspec(2, 2, height_ratios=[1.42, 1.0], hspace=0.16, wspace=0.10)

fig.suptitle(
    "A hardware task's policy_port is judged before the arm is connected",
    fontsize=17.5,
    fontweight="bold",
    y=0.975,
)
fig.text(
    0.5,
    0.941,
    "Robot.start_task(instruction, policy_port=..., policy_provider='groot')  -  in-memory arm, recording connect path",
    ha="center",
    fontsize=10.6,
    color=GREY,
)

# ---------------------------------------------------------------- verdict grid
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

COL = {"port": 0.055, "main": 0.335, "branch": 0.705}
TOP, LAST = 0.845, 0.075
STEP = (TOP - LAST) / (len(PORTS) - 1)
assert STEP > 0.045, STEP

put(ax, COL["port"], 0.945, "policy_port", fontsize=11.5, fontweight="bold", transform=ax.transAxes)
put(ax, COL["main"], 0.945, "main", fontsize=11.5, fontweight="bold", color=RED, transform=ax.transAxes)
put(ax, COL["branch"], 0.945, "this change", fontsize=11.5, fontweight="bold", color=GREEN, transform=ax.transAxes)
put(ax, COL["main"] + 0.085, 0.945, "(what start_task returned / was the arm connected)", fontsize=8.8, color=GREY, transform=ax.transAxes)
ax.plot([0.04, 0.965], [0.905, 0.905], color="#c9ccd1", lw=1.0, transform=ax.transAxes, clip_on=False)

y = TOP
for port in PORTS:
    for key, tree in (("main", A), ("branch", B)):
        row = tree["rows"][port]
        ok = honors(port, row)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (COL[key] - 0.016, y - 0.018),
                0.335,
                0.052,
                boxstyle="round,pad=0.004",
                transform=ax.transAxes,
                facecolor=BG_OK if ok else BG_BAD,
                edgecolor=GREEN if ok else RED,
                lw=1.1,
                zorder=0,
            )
        )
        verdict = "success  'Task started'" if row["start_status"] == "success" else "error"
        arm = "arm CONNECTED" if row["connects"] else "arm untouched"
        put(
            ax,
            COL[key],
            y + 0.013,
            f"{verdict}   |   {arm}",
            fontsize=9.6,
            color=RED if not ok else GREEN,
            fontweight="bold" if not ok else "normal",
            transform=ax.transAxes,
        )
        put(ax, COL[key], y - 0.011, named(tree, port), fontsize=8.5, color=GREY, transform=ax.transAxes)
    label = f"{port}" + ("   (usable)" if port in USABLE else "")
    put(
        ax,
        COL["port"],
        y,
        label,
        fontsize=11.0,
        family="monospace",
        fontweight="bold" if port not in USABLE else "normal",
        transform=ax.transAxes,
    )
    y -= STEP
assert y + STEP >= LAST - 1e-9

put(
    ax,
    0.055,
    0.012,
    f"ports whose outcome does not honor the contract:  main {n_bad_main} of {len(PORTS)}   ->   this change {n_bad_branch} of {len(PORTS)}",
    fontsize=10.8,
    fontweight="bold",
    transform=ax.transAxes,
)

# ------------------------------------------------------- bus-denial sequence
for idx, (key, tree, colour) in enumerate((("main", A, RED), ("branch", B, GREEN))):
    axd = fig.add_subplot(gs[1, idx])
    axd.set_xlim(0, 1)
    axd.set_ylim(0, 1)
    axd.axis("off")
    d = tree["denial"]
    title = "main" if key == "main" else "this change"
    put(axd, 0.02, 0.955, f"{title}: the arm has one command bus", fontsize=12.2, fontweight="bold", color=colour, transform=axd.transAxes)
    lines = [
        ("1. start_task(policy_port=99999)", None),
        (f"   -> {d['first_status']}: {d['first_text'].splitlines()[0]}", d["first_status"] == "error"),
        ("2. concurrently, a well-formed task:", None),
        ("   start_task(policy_port=5555)", None),
        (f"   -> {d['second_status']}: {d['second_text'].splitlines()[0]}", d["second_status"] == "success"),
    ]
    ly = 0.80
    for text, good in lines:
        col = GREY if good is None else (GREEN if good else RED)
        put(
            axd,
            0.03,
            ly,
            text,
            fontsize=10.1,
            family="monospace",
            color=col,
            fontweight="bold" if good is not None else "normal",
            transform=axd.transAxes,
        )
        ly -= 0.108
    verdict = (
        "The bad port took the bus for its whole bring-up\nwindow, so the task that could have run was\nturned away."
        if d["second_status"] != "success"
        else "The bad port is refused before the claim, so the\nbus is still free for the task that can run."
    )
    put(axd, 0.03, 0.135, verdict, fontsize=10.4, color=colour, transform=axd.transAxes)
    assert ly > 0.14

for ax_, yv, is_axes in placed:
    if is_axes:
        assert -0.03 <= yv <= 1.07, f"axes-fraction y out of band: {yv}"
    else:
        lo, hi = ax_.get_ylim()
        assert lo - 0.03 <= yv <= hi + 0.07, f"data y out of band: {yv} vs {(lo, hi)}"

out = pathlib.Path("/tmp/policy_port_domain.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = Image.open(out).convert("RGB")
import numpy as np

a = np.asarray(im)
for name, band in (("top", a[:8]), ("bottom", a[-8:]), ("left", a[:, :8]), ("right", a[:, -8:])):
    nonwhite = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nonwhite == 0, f"{name} border has {nonwhite} non-white px"
print(f"OK {out}  {im.size}  divergences main={n_bad_main} branch={n_bad_branch}")
