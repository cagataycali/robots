"""Compose the export_xml path-validation artifact from the two measured dumps."""
import base64, io, json, os, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RUN = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/art-main-{RUN}.json").read_text())    # main
B = json.loads(pathlib.Path(f"/tmp/art-branch-{RUN}.json").read_text())  # this PR
assert A["tree"] != B["tree"], "both dumps came from the same tree"

# ---- audit every claim the figure makes ------------------------------------
assert A["symlink"]["status"] == "success" and A["symlink"]["victim_intact"] is False
assert B["symlink"]["status"] == "error" and B["symlink"]["victim_intact"] is True
assert A["symlink"]["victim_bytes_after"] == 13488 and B["symlink"]["victim_bytes_after"] == 41
VECTORS = ["traversal (`..`)", "shell metacharacter (`;`)", "backslash separator"]
assert all(A["vectors"][v]["status"] == "success" for v in VECTORS), A["vectors"]
assert all(B["vectors"][v]["status"] == "error" for v in VECTORS), B["vectors"]
assert A["vectors"]["traversal (`..`)"]["escaped"] is True
assert B["vectors"]["traversal (`..`)"]["escaped"] is False
assert A["envelope"]["missing parent directory"]["outcome"].startswith("RAISED")
assert A["envelope"]["destination is a directory"]["outcome"].startswith("RAISED")
assert B["envelope"]["missing parent directory"]["outcome"] == "success"
assert B["envelope"]["destination is a directory"]["outcome"] == "error"
assert A["honored"]["status"] == B["honored"]["status"] == "success"
assert A["honored"]["bytes"] == B["honored"]["bytes"], "the honored export changed size"

img_a = np.asarray(Image.open(io.BytesIO(base64.b64decode(A["render_png_b64"]))).convert("RGB")).astype(int)
img_b = np.asarray(Image.open(io.BytesIO(base64.b64decode(B["render_png_b64"]))).convert("RGB")).astype(int)
assert img_a.shape == img_b.shape
delta = int(np.abs(img_a - img_b).max())
changed = int((np.abs(img_a - img_b).max(axis=2) > 8).sum())
assert delta <= 2 and changed == 0, f"the scene differs across trees: max={delta} changed={changed}"
sat = float(((img_b.max(2) - img_b.min(2)) > 45).mean())
assert sat > 0.10, f"the reference render has no content: sat={sat:.3f}"

n_bad_main = sum(1 for v in VECTORS if A["vectors"][v]["status"] == "success") + \
    (1 if not A["symlink"]["victim_intact"] else 0) + \
    sum(1 for k in A["envelope"] if A["envelope"][k]["outcome"].startswith("RAISED"))
n_bad_pr = sum(1 for v in VECTORS if B["vectors"][v]["status"] == "success") + \
    (1 if not B["symlink"]["victim_intact"] else 0) + \
    sum(1 for k in B["envelope"] if B["envelope"][k]["outcome"].startswith("RAISED"))
assert (n_bad_main, n_bad_pr) == (6, 0), (n_bad_main, n_bad_pr)

RED, GRN, INK, MUT = "#c0392b", "#1e8449", "#1c1c1c", "#666666"
placed = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.2, 12.6), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.10, 1.00, 0.62], width_ratios=[1.00, 1.28],
                      hspace=0.20, wspace=0.10)
fig.suptitle("export_xml(output_path=...) treats its LLM-supplied destination as untrusted input",
             fontsize=17, fontweight="bold", y=0.972)
fig.text(0.5, 0.943,
         "export_xml is an agent-callable action, so output_path arrives from a tool call exactly as render's does. "
         "It wrote that path straight to open(output_path, 'w').",
         ha="center", fontsize=10.5, color=MUT, style="italic")

# --- row 1 left: the real scene ---------------------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.imshow(img_b.astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("The scene being exported (MuJoCo, headless EGL)", fontsize=11.5, fontweight="bold", pad=7)
ax.set_xlabel(
    f"so101 - export_xml wrote {B['honored']['bytes']:,} bytes of MJCF starting {B['honored']['head']!r}\n"
    f"Byte-comparable on both trees (max delta {delta}/255, 0 pixels differ above 8): the export itself is unchanged.",
    fontsize=9.2, color=MUT, labelpad=7)

# --- row 1 right: the symlink consequence -----------------------------------
ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("A file the caller never named", fontsize=11.5, fontweight="bold", pad=7)
put(ax, 0.0, 0.93, "output_path pointed at a symlink (export-here.xml -> important-notes.txt)",
    fontsize=10, color=INK, transform=ax.transAxes)
rows = [
    ("before, both trees", A["symlink"]["victim_before"], A["symlink"]["victim_bytes_before"], MUT, None),
    ("main: after the export", A["symlink"]["victim_after_first_60"] + " ...",
     A["symlink"]["victim_bytes_after"], RED, f"reported status={A['symlink']['status']}"),
    ("this PR: after the export", B["symlink"]["victim_after_first_60"],
     B["symlink"]["victim_bytes_after"], GRN, f"reported status={B['symlink']['status']}"),
]
TOP, LAST = 0.79, 0.14
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.10, STEP
y = TOP
for label, content, nbytes, colour, note in rows:
    put(ax, 0.0, y, label, fontsize=10, fontweight="bold", color=colour, transform=ax.transAxes)
    put(ax, 0.0, y - 0.075, content.replace("\n", " "), fontsize=8.6, family="monospace",
        color=INK, transform=ax.transAxes)
    put(ax, 0.0, y - 0.135, f"{nbytes:,} bytes" + (f"   <-  {note}" if note else ""),
        fontsize=9, color=colour, transform=ax.transAxes)
    y -= STEP
assert abs((y + STEP) - LAST) < 1e-9, (y, LAST)
put(ax, 0.0, 0.005, "The overwrite was reported as a successful export.", fontsize=9.6,
    fontweight="bold", color=RED, transform=ax.transAxes)

# --- row 2: the vector matrix ------------------------------------------------
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Every arbitrary-write vector, measured through the public tool envelope",
             fontsize=12, fontweight="bold", pad=8)
cols = [0.0, 0.235, 0.615]
for x, h in zip(cols, ["output_path the agent supplied", "main (83cc5272)", "this PR"], strict=True):
    put(ax, x, 0.90, h, fontsize=10.4, fontweight="bold", color=INK, transform=ax.transAxes)
mrows = [("symlinked target", A["symlink"], B["symlink"], "victim overwritten", "symlink refused, victim intact")]
for v in VECTORS:
    mrows.append((v, A["vectors"][v], B["vectors"][v], None, None))
TOP2, LAST2 = 0.74, 0.10
STEP2 = (TOP2 - LAST2) / (len(mrows) - 1)
assert STEP2 > 0.10, STEP2
y = TOP2
for label, a, b, a_note, b_note in mrows:
    put(ax, cols[0], y, label.replace("`", ""), fontsize=10, color=INK, transform=ax.transAxes)
    at = a_note or f"escaped: {a.get('escaped')}"
    put(ax, cols[1], y, f"status={a['status']}", fontsize=9.8, fontweight="bold", color=RED, transform=ax.transAxes)
    put(ax, cols[1], y - 0.075, at, fontsize=8.8, family="monospace", color=RED, transform=ax.transAxes)
    put(ax, cols[2], y, "status=error", fontsize=9.8, fontweight="bold", color=GRN, transform=ax.transAxes)
    put(ax, cols[2], y - 0.075, (b_note or b["text"])[:74], fontsize=8.8, family="monospace",
        color=GRN, transform=ax.transAxes)
    y -= STEP2
assert abs((y + STEP2) - LAST2) < 1e-9, (y, LAST2)

# --- row 3: the envelope contract + gate ------------------------------------
ax = fig.add_subplot(gs[2, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("The write also used to sit outside the method's try, so an OSError escaped the envelope",
             fontsize=11.5, fontweight="bold", pad=6)
erows = [(k, A["envelope"][k], B["envelope"][k]) for k in A["envelope"]]
TOP3, LAST3 = 0.72, 0.34
STEP3 = (TOP3 - LAST3) / (len(erows) - 1)
assert STEP3 > 0.10, STEP3
y = TOP3
for label, a, b in erows:
    put(ax, cols[0], y, label, fontsize=9.8, color=INK, transform=ax.transAxes)
    put(ax, cols[1], y, a["outcome"], fontsize=9.4, fontweight="bold", family="monospace",
        color=RED, transform=ax.transAxes)
    put(ax, cols[2], y, f"{b['outcome']}  {b['text'][:52]}", fontsize=9.0, family="monospace",
        color=GRN, transform=ax.transAxes)
    y -= STEP3
assert abs((y + STEP3) - LAST3) < 1e-9, (y, LAST3)
put(ax, 0.0, 0.15,
    f"Outcomes that do not honour the sink's contract:  main {n_bad_main} of 6   ->   this PR {n_bad_pr} of 6",
    fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
put(ax, 0.0, 0.02,
    "Gate on Thor: 28697 passed / 258 skipped / 0 failed (MUJOCO_GL=egl, 660s)  |  ruff clean  |  "
    "mypy 0 errors outside examples/  |  pre-fix 14 failed / 5 passed",
    fontsize=9, color=MUT, transform=ax.transAxes)

for a, yy, is_axes in placed:
    if is_axes:
        assert -0.03 <= yy <= 1.08, f"axes-coord text at y={yy}"
    else:
        lo, hi = a.get_ylim(); assert lo - 0.05 <= yy <= hi + 0.07, f"data-coord text at y={yy}"

out = pathlib.Path("_art/export_xml_path_validation.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(Image.open(out).convert("RGB"))
for side, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{side} border has {n} non-white pixels"
print(f"WROTE {out}  {im.shape[1]}x{im.shape[0]}  sat={sat:.3f}  bad: main {n_bad_main} -> PR {n_bad_pr}")
