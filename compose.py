"""Compose the artifact: what a caller sees when the removal's recompile is refused."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ART = pathlib.Path("/tmp/art-31396282703")
M = json.loads((ART / "facts_main.json").read_text())
B = json.loads((ART / "facts_branch.json").read_text())

# --- the two runs must come from different trees ---------------------------
assert M["tree"] != B["tree"], (M["tree"], B["tree"])

# --- every claim below is asserted against the two dumps ------------------
assert M["remove_status"] == "success" and "removed." in M["remove_text"]
assert B["remove_status"] == "error" and "still registered" in B["remove_text"]
assert M["after_remove_registry"] == ["default"] and M["after_remove_model_cams"] == ["default", "watch"]
assert B["after_remove_registry"] == ["default", "watch"] == B["after_remove_model_cams"]
assert B["after_remove_spec_cams"] == ["default", "watch"]
assert M["after_remove_spec_cams"] == ["default"]
assert M["after_remove_render_status"] == "success"          # the divergence
assert M["later_add_status"] == B["later_add_status"] == "success"
assert M["after_add_render_status"] == "error"               # vanished at the later add
assert B["after_add_render_status"] == "success"
assert M["after_add_model_cams"] == ["default"]
assert B["after_add_model_cams"] == ["default", "watch"]

ref_m = np.asarray(Image.open(ART / "main_A_reference.png").convert("RGB")).astype(int)
ref_b = np.asarray(Image.open(ART / "branch_A_reference.png").convert("RGB")).astype(int)
ref_delta = int(np.abs(ref_m - ref_b).max())
assert ref_delta <= 2, f"the reference view must match across trees, got {ref_delta}"

after_b = np.asarray(Image.open(ART / "branch_C_after_later_add.png").convert("RGB")).astype(int)
changed = float((np.abs(ref_b - after_b).sum(axis=2) > 24).mean())
assert changed > 0.03, f"the later add must be visible in the surviving view, got {changed:.2%}"
assert not (ART / "main_C_after_later_add.png").exists(), "main produced no view, by construction"

placed: list[tuple[object, float, bool]] = []

def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.6, 9.0), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.60], hspace=0.20, wspace=0.06,
                      left=0.022, right=0.978, top=0.905, bottom=0.035)

fig.suptitle(
    "remove_camera: a refused recompile reported as a completed removal",
    fontsize=16.5, fontweight="bold", y=0.975,
)
fig.text(0.5, 0.936,
         "Same scene, same sequence in both trees: remove_camera('watch') whose validating "
         "spec.recompile is refused, then one unrelated add_object('crate').",
         ha="center", fontsize=10.6, style="italic", color="#333333")

MONO = {"family": "monospace", "fontsize": 9.1}

# ---- Panel A: the reference view ----------------------------------------
axa = fig.add_subplot(gs[0, 0])
axa.imshow(Image.open(ART / "main_A_reference.png"))
axa.set_xticks([]); axa.set_yticks([])
axa.set_title("A  the view 'watch' gives\n(identical in both trees)", fontsize=11.4, fontweight="bold")
axa.set_xlabel(f"max|main - branch| = {ref_delta}/255 over the whole frame", fontsize=9.3, color="#2f6f3f")
for s in axa.spines.values():
    s.set_edgecolor("#555555"); s.set_linewidth(1.6)

# ---- Panel B: main, no view left ----------------------------------------
axb = fig.add_subplot(gs[0, 1])
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis("off")
axb.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#241a1a", edgecolor="#a33a3a", linewidth=2.4))
axb.set_title("B  main: render('watch') after the later add_object", fontsize=11.4, fontweight="bold",
              color="#a33a3a")
put(axb, 0.5, 0.74, "NO VIEW", ha="center", color="#ff8b8b", fontsize=21, fontweight="bold",
    transform=axb.transAxes)
put(axb, 0.5, 0.63, "status = \"error\"", ha="center", color="#ffbfbf", transform=axb.transAxes, **MONO)
put(axb, 0.5, 0.545, f'"{M["after_add_render_text"]}"', ha="center", color="#ffd7d7",
    transform=axb.transAxes, family="monospace", fontsize=8.4)
for i, line in enumerate([
    "remove_camera said:  status=\"success\"",
    "                     \"Camera 'watch' removed.\"",
    "",
    "...yet render('watch') kept working right after it.",
    "The delete sat in the spec and was applied later,",
    "by the add_object call that never mentioned it.",
]):
    put(axb, 0.055, 0.40 - i * 0.056, line, color="#e6d2d2", transform=axb.transAxes,
        family="monospace", fontsize=8.5)

# ---- Panel C: branch, the view survives ---------------------------------
axc = fig.add_subplot(gs[0, 2])
axc.imshow(Image.open(ART / "branch_C_after_later_add.png"))
axc.set_xticks([]); axc.set_yticks([])
axc.set_title("C  this change: the same render, after the same add", fontsize=11.4, fontweight="bold",
              color="#2f6f3f")
axc.set_xlabel(f"the removal was refused, so the camera is still there\n"
               f"(the new crate is visible: {changed:.1%} of the frame changed vs A)",
               fontsize=9.3, color="#2f6f3f")
for s in axc.spines.values():
    s.set_edgecolor("#2f6f3f"); s.set_linewidth(2.0)

# ---- the measured ledger ------------------------------------------------
axt = fig.add_subplot(gs[1, :])
axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off")
axt.set_title("measured, one script run in each tree", fontsize=11.6, fontweight="bold", loc="left")

rows = [
    ("what the caller was told",
     f'success  -  "{M["remove_text"]}"',
     f'error    -  "...not removed ... still registered ..."'),
    ("list_cameras() right after",
     f'{M["after_remove_registry"]}', f'{B["after_remove_registry"]}'),
    ("the compiled model right after",
     f'{M["after_remove_model_cams"]}   <- disagrees with the registry',
     f'{B["after_remove_model_cams"]}   <- agrees'),
    ("the live spec right after",
     f'{M["after_remove_spec_cams"]}   <- disagrees with the model',
     f'{B["after_remove_spec_cams"]}   <- agrees'),
    ("render('watch') right after",
     f'{M["after_remove_render_status"]}  <- resolves a camera list_cameras denies',
     f'{B["after_remove_render_status"]}  <- consistent with all three'),
    ("then one unrelated add_object",
     f'{M["later_add_status"]}', f'{B["later_add_status"]}'),
    ("the model after that add",
     f'{M["after_add_model_cams"]}   <- the camera vanished here',
     f'{B["after_add_model_cams"]}   <- nothing landed later'),
    ("render('watch') after that add",
     f'{M["after_add_render_status"]}', f'{B["after_add_render_status"]}'),
]

TOP, LAST = 0.855, 0.075
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
X_LABEL, X_MAIN, X_PR = 0.012, 0.275, 0.638
put(axt, X_LABEL, 0.945, "", transform=axt.transAxes)
for x, head, col in ((X_LABEL, "", "#000000"), (X_MAIN, "main", "#a33a3a"),
                     (X_PR, "this change", "#2f6f3f")):
    if head:
        put(axt, x, 0.945, head, transform=axt.transAxes, fontweight="bold", fontsize=10.6, color=col)

y = TOP
for label, left, right in rows:
    put(axt, X_LABEL, y, label, transform=axt.transAxes, fontsize=9.4)
    put(axt, X_MAIN, y, left, transform=axt.transAxes, color="#a33a3a", **MONO)
    put(axt, X_PR, y, right, transform=axt.transAxes, color="#2f6f3f", **MONO)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
axt.axhline(0.905, xmin=0.008, xmax=0.992, color="#bbbbbb", linewidth=0.9)

# ---- layout guard ------------------------------------------------------
for ax, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.07, f"axes-fraction y out of panel: {yy}"
    else:
        lo, hi = ax.get_ylim()
        assert lo - 0.05 <= yy <= hi + 0.07, f"data y {yy} outside {(lo, hi)}"

OUT = ART / "remove_camera_refused_recompile.png"
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(OUT).convert("RGB"))
print("figure:", im.shape, OUT)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    print(f"  border {name}: {n} non-white px")
    assert n == 0, f"{name} border not clean: {n}"
print(f"OK  ref_delta={ref_delta}  changed_vs_A={changed:.2%}  rows={len(rows)} step={STEP:.4f}")
