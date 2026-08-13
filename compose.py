"""Compose the artifact. Every drawn number is read from the two capture dumps."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(sys.argv[1])
A = json.loads((OUT / "facts_main.json").read_text())
B = json.loads((OUT / "facts_branch.json").read_text())
assert A["tree"] != B["tree"], "both arms measured the same tree"

# ---- assertions on what will be drawn -------------------------------------
fa, fb = A["failed_dispatch"], B["failed_dispatch"]
assert fa["names_the_asyncio_internal"] and not fa["names_the_cause"]
assert fb["names_the_cause"] and not fb["names_the_asyncio_internal"]
assert fa["leaked_coroutines"] == 1 and fb["leaked_coroutines"] == 0
assert fa["actions_commanded"] == 0 and fb["actions_commanded"] == 0
assert A["healthy_nested"] == B["healthy_nested"], "the honored rollout must be identical"
assert A["healthy_sync"] == B["healthy_sync"]
assert A["render"]["achieved"] == B["render"]["achieved"], "the delivered pose must be identical"
N_ACTIONS = B["healthy_nested"]["actions"]

im_a = np.asarray(Image.open(OUT / "pose_main.png").convert("RGB")).astype(int)
im_b = np.asarray(Image.open(OUT / "pose_branch.png").convert("RGB")).astype(int)
delta = int(np.abs(im_a - im_b).max())
changed = int((np.abs(im_a - im_b).sum(axis=2) > 8).sum())
assert delta <= 2, delta
sat = float(((im_b.max(axis=2) - im_b.min(axis=2)) > 45).mean())
assert sat > 0.05, sat

placed: list[tuple[Any, float, bool]] = []


def put(ax: Any, x: float, y: float, s: str, **kw: Any) -> None:
    placed.append((ax, y, kw.get("transform") is not None))
    ax.text(x, y, s, **kw)


fig = plt.figure(figsize=(15.4, 13.2), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 0.72, 0.62], hspace=0.20, wspace=0.06)

fig.suptitle(
    "A hardware rollout dispatch failure now reports its own cause\n"
    "strands_robots.hardware_robot.Robot._run_control_loop  -  measured on Thor, arms unplugged",
    fontsize=15.5,
    fontweight="bold",
    y=0.975,
)

# ---- row 0: the rollout the dispatch delivers (identical on both trees) ----
for col, (label, path) in enumerate((("main", "pose_main.png"), ("this PR", "pose_branch.png"))):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(Image.open(OUT / path))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label}: the honored rollout, delivered", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(
        f"stream(action=\"execute\") -> nested dispatch -> {N_ACTIONS} applied actions\n"
        "so101 in MuJoCo (headless), commanded with the rollout's own final targets",
        fontsize=9.6,
    )
fig.text(
    0.5,
    0.638,
    f"identical across the two trees:  max|delta| = {delta}/255,  "
    f"{changed} of {im_a.shape[0] * im_a.shape[1]} pixels differ above a threshold of 8  "
    "->  the rollout this change dispatches is untouched",
    ha="center",
    fontsize=10.8,
    style="italic",
    color="#20502a",
)

# ---- row 1: what the caller is told when the dispatch fails ----------------
panels = (
    ("main", fa, "#7a1f1f", "#fdecec"),
    ("this PR", fb, "#20502a", "#ecf7ee"),
)
for col, (label, facts, colour, bg) in enumerate(panels):
    ax = fig.add_subplot(gs[1, col])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=bg, edgecolor=colour, lw=1.6))
    put(
        ax,
        0.5,
        0.90,
        f"{label}: the pool cannot start a thread",
        ha="center",
        fontsize=12.5,
        fontweight="bold",
        color=colour,
        transform=ax.transAxes,
    )
    lines = [
        ("injected cause", facts["injected_cause"]),
        ("RuntimeError the caller gets", facts["raised_message"]),
        ("names the cause", "yes" if facts["names_the_cause"] else "NO"),
        ("names an asyncio internal", "YES" if facts["names_the_asyncio_internal"] else "no"),
        ("un-awaited coroutines leaked", str(facts["leaked_coroutines"])),
        ("actions commanded", str(facts["actions_commanded"])),
    ]
    TOP, LAST = 0.72, 0.13
    STEP = (TOP - LAST) / (len(lines) - 1)
    assert STEP > 0.030, STEP
    y = TOP
    for name, value in lines:
        put(ax, 0.045, y, f"{name}:", fontsize=10.2, va="center", transform=ax.transAxes)
        emphasis = name in {"names the cause", "names an asyncio internal", "un-awaited coroutines leaked"}
        put(
            ax,
            0.50,
            y,
            value,
            fontsize=10.2,
            va="center",
            family="monospace",
            fontweight="bold" if emphasis else "normal",
            color=colour if emphasis else "#222222",
            transform=ax.transAxes,
        )
        y -= STEP
    assert abs((y + STEP) - LAST) < 1e-9, y

# ---- row 2: the ledger ----------------------------------------------------
ax = fig.add_subplot(gs[2, :])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
put(
    ax,
    0.5,
    0.94,
    "Only the probe's own RuntimeError means \"no loop is running\"",
    ha="center",
    fontsize=12.6,
    fontweight="bold",
    transform=ax.transAxes,
)
rows = [
    "stream(action=\"execute\") is an async def calling _execute_task_sync, so the "
    "\"already on a loop\" branch is that surface's live path.",
    "except RuntimeError wrapped the loop probe AND the nested dispatch, so a dispatch "
    "RuntimeError landed in a handler whose own asyncio.run",
    "is invalid on exactly that branch: the cause was replaced by "
    "\"asyncio.run() cannot be called from a running event loop\" and the",
    "task_runner coroutine the handler built was left un-awaited.",
    f"Both branches still drive the rollout exactly once "
    f"(nested {B['healthy_nested']['actions']} actions, sync {B['healthy_sync']['actions']}), "
    "and a failed dispatch commands the arm 0 times.",
]
TOP, LAST = 0.74, 0.10
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
y = TOP
for text in rows:
    put(ax, 0.02, y, text, fontsize=10.0, va="center", transform=ax.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, y

for ax_, y_, is_axes in placed:
    if is_axes:
        assert -0.03 <= y_ <= 1.10, (y_, is_axes)

fig.savefig(OUT / "artifact.png", bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

img = np.asarray(Image.open(OUT / "artifact.png").convert("RGB"))
for name, band in (
    ("top", img[:8]),
    ("bottom", img[-8:]),
    ("left", img[:, :8]),
    ("right", img[:, -8:]),
):
    non_white = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert non_white == 0, f"{name} border has {non_white} non-white px"
print("artifact OK", Image.open(OUT / "artifact.png").size, "delta", delta, "changed", changed, "sat", round(sat, 4))
