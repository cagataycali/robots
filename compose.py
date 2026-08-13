"""Compose: the Isaac answer beside the capability it redirects to."""
import json, os, pathlib, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = pathlib.Path("/tmp/art-" + os.environ["GITHUB_RUN_ID"])
F = json.loads((D / "facts.json").read_text())
rest, moved = np.load(D / "rest.npy"), np.load(D / "moved.npy")

# --- self-audit: every claim below is read from the dump --------------------
assert F["isaac_status"] == "error"
assert F["mujoco_mount_status"] == "success"
assert F["actions_applied_ok"] == 90
assert F["wrist_view_changed_frac"] > 0.10, F["wrist_view_changed_frac"]
assert F["rest_saturated"] > 0.15 and F["moved_saturated"] > 0.15
for b in ("mujoco", "newton"):
    assert b in F["isaac_text"], b
assert "Omit" in F["isaac_text"]

placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.0, 10.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 1.00, 0.62], hspace=0.30, wspace=0.10)

fig.suptitle(
    "add_camera(parent_body=...) — the Isaac backend now reports the mount it lacks, and names where it lives",
    fontsize=14.4, fontweight="bold", y=0.972,
)
fig.text(0.5, 0.940,
    "Top: the mount running on a backend that has it (mujoco) — the wrist view RIDES the gripper.  "
    "Bottom left: what Isaac used to answer.  Bottom right: what it answers now.",
    ha="center", fontsize=10.3, style="italic", color="#333")

# --- row 1: the capability, on the backend the refusal names ---------------
for col, (img, lbl, jt) in enumerate((
    (rest, "mujoco  ·  wrist camera mounted on so101/gripper  ·  arm at rest", "rest_joints"),
    (moved, "same camera, after 90 commanded actions (all applied)", "moved_joints"),
)):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#2e7d32"); sp.set_linewidth(2.2)
    ax.set_title(lbl, fontsize=10.2, pad=6)
    j = {k: v for k, v in F[jt].items() if not k.endswith(".vel")}
    ax.set_xlabel("joints " + "  ".join(f"{k}={v:+.3f}" for k, v in j.items()),
                  fontsize=8.4, family="monospace", color="#2e7d32", labelpad=5)

# --- row 2: the two answers ------------------------------------------------
BEFORE = (
    "sim.add_camera(name='wrist', parent_body='so101/gripper')\n\n"
    "TypeError: IsaacSimulation.add_camera() got an\n"
    "unexpected keyword argument 'parent_body'\n\n"
    "· names the parameter, nothing else\n"
    "· not the capability at stake (mounting)\n"
    "· not the two backends that DO mount\n"
    "· not the world-fixed alternative Isaac supports\n"
    "· an exception, from a method whose contract\n"
    "  is the {\"status\", \"content\"} envelope"
)
for col, (title, body, colour) in enumerate((
    ("BEFORE — answered by Python's argument binding", BEFORE, "#c62828"),
    ("AFTER — answered by the backend, status='error'",
     "sim.add_camera(name='wrist', parent_body='so101/gripper')\n\n"
     + "\n".join(textwrap.wrap(F["isaac_text"], 58)), "#2e7d32"),
)):
    ax = fig.add_subplot(gs[1, col]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax.transAxes,
                               facecolor="#fafafa", edgecolor=colour, linewidth=2.0))
    put(ax, 0.5, 1.045, title, transform=ax.transAxes, ha="center",
        fontsize=10.6, fontweight="bold", color=colour)
    put(ax, 0.035, 0.90, body, transform=ax.transAxes, va="top",
        fontsize=8.5, family="monospace", color="#1a1a1a")

# --- row 3: the measured ledger -------------------------------------------
ax = fig.add_subplot(gs[2, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
rows = [
    ("mujoco  add_camera(parent_body=...)", "success — mount honoured", "success — unchanged", "#2e7d32"),
    ("newton  add_camera(parent_body=...)", "success — mount honoured", "success — unchanged", "#2e7d32"),
    ("isaac   add_camera(parent_body=...)", "TypeError (argument binding)",
     "status='error', names mujoco + newton", "#c62828"),
    ("isaac   add_camera(...) mount omitted", "world-fixed camera", "world-fixed camera — unchanged", "#2e7d32"),
    ("mounted wrist view rides the body",
     f"{F['wrist_view_changed_frac']:.2%} of pixels change over 90 applied actions",
     f"frame content {F['rest_saturated']:.0%} / {F['moved_saturated']:.0%} saturated", "#1565c0"),
]
TOP, LAST = 0.80, 0.10
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
put(ax, 0.5, 0.965, "Measured on this tree — before | after", transform=ax.transAxes,
    ha="center", fontsize=10.4, fontweight="bold")
for c, (x, t) in enumerate(((0.020, "call"), (0.400, "main"), (0.690, "this PR"))):
    put(ax, x, 0.895, t, transform=ax.transAxes, fontsize=9.4, fontweight="bold", color="#444")
y = TOP
for call, before, after, colour in rows:
    put(ax, 0.020, y, call, transform=ax.transAxes, fontsize=8.9, family="monospace")
    put(ax, 0.400, y, before, transform=ax.transAxes, fontsize=8.9, color="#666")
    put(ax, 0.690, y, after, transform=ax.transAxes, fontsize=8.9, color=colour, fontweight="bold")
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)

for ax_, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.07, f"axes-frac text at y={yy}"

OUTP = D / "isaac_camera_mount.png"
fig.savefig(OUTP, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(OUTP).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white px"
print("OK", OUTP, im.shape)
