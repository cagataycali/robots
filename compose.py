import json, math, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v3 as iio

A = json.loads(pathlib.Path("/tmp/art_main/facts.json").read_text())        # main, normal
AO = json.loads(pathlib.Path("/tmp/art_main_O/facts.json").read_text())     # main, python -O
B = json.loads(pathlib.Path("/tmp/art_branch/facts.json").read_text())      # this change
assert A["tree"] != B["tree"], (A["tree"], B["tree"])
assert A["tree"] == AO["tree"] and A["optimized"] is False and AO["optimized"] is True

src = np.load("/tmp/art_frames.npy")[12]
req = list(iio.imiter(B["rows"]["8"]["path"]))[12]        # requested quality 8
sub = list(iio.imiter(A["rows"]["True"]["path"]))[12]     # what True silently produced on main

# --- self-audit: every claim the figure makes -------------------------------
assert A["rows"]["True"]["outcome"] == "encoded"
assert A["rows"]["True"]["md5"] == A["rows"]["1"]["md5"], "True must be byte-identical to quality=1"
assert B["rows"]["True"]["outcome"] == "ValueError"
assert AO["rows"]["0"]["outcome"] == "encoded" and AO["rows"]["-5"]["outcome"] == "encoded"
assert B["rows"]["np.int64(8)"]["md5"] == B["rows"]["8"]["md5"], "numpy real must match the plain int"
assert A["rows"]["np.int64(8)"]["outcome"] == "AssertionError"
PSNR_REQ, PSNR_SUB = A["rows"]["8"]["psnr"], A["rows"]["True"]["psnr"]
assert PSNR_REQ - PSNR_SUB > 5.0, (PSNR_REQ, PSNR_SUB)

# crop where the two decodes differ most (high-detail edges)
d = np.abs(req.astype(np.int16) - sub.astype(np.int16)).sum(2)
CW, CH = 176, 132
best, bxy = -1, (0, 0)
for y in range(0, d.shape[0] - CH, 22):
    for x in range(0, d.shape[1] - CW, 22):
        s = d[y:y + CH, x:x + CW].sum()
        if s > best:
            best, bxy = s, (x, y)
cx, cy = bxy
def crop(im):
    c = im[cy:cy + CH, cx:cx + CW]
    return np.repeat(np.repeat(c, 3, axis=0), 3, axis=1)
print(f"crop at ({cx},{cy}) {CW}x{CH}; diff mass {best}")

fig = plt.figure(figsize=(15.4, 11.6), dpi=124)
gs = fig.add_gridspec(3, 3, height_ratios=[1.02, 1.02, 1.30], hspace=0.30, wspace=0.07)
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig.suptitle("encode_clip: a quality the encoder cannot honor is now refused, not silently substituted",
             fontsize=15.5, fontweight="bold", y=0.983)
fig.text(0.5, 0.958, "24 real MuJoCo frames (640x480, headless EGL) through the shared clip encoder. "
         "Row 2 is the 3x zoom of the boxed region.", ha="center", fontsize=10.2, style="italic", color="#333")

PANELS = [
    (src, "Source render", "the frames handed to encode_clip", "#37474f"),
    (req, f"quality=8 (requested)  ~  {PSNR_REQ:.2f} dB", "encoded on both trees - unchanged by this PR", "#1b5e20"),
    (sub, f"quality=True on main  ~  {PSNR_SUB:.2f} dB", "silently encoded at quality=1, the lowest offered", "#b71c1c"),
]
for col, (im, title, sub_t, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=11.6, color=colour, fontweight="bold", pad=6)
    ax.set_xlabel(sub_t, fontsize=9.6, color="#444", labelpad=5)
    ax.add_patch(Rectangle((cx, cy), CW, CH, fill=False, edgecolor="#ffb300", lw=2.0))
    az = fig.add_subplot(gs[1, col]); az.imshow(crop(im)); az.set_xticks([]); az.set_yticks([])
    for sp in az.spines.values(): sp.set_edgecolor("#ffb300"); sp.set_linewidth(2.0)
    az.set_xlabel(f"3x zoom  ({CW}x{CH} px)", fontsize=9.2, color="#666", labelpad=4)

ROWS = [
    ("8   (default)",      "8"),
    ("1",                  "1"),
    ("True",               "True"),
    ("0",                  "0"),
    ("-5",                 "-5"),
    ("500",                "500"),
    ("nan",                "nan"),
    ("'8'",                "'8'"),
    ("np.int64(8)",        "np.int64(8)"),
]
def cell(row):
    if row["outcome"] == "encoded":
        return f"encoded  {row['psnr']:.2f} dB", "#1b5e20" if row["psnr"] > 40 else "#b71c1c"
    if row["outcome"] == "ValueError" and "quality must be" in row.get("message", ""):
        return "refused: quality must be...", "#1b5e20"
    return f"{row['outcome']} escapes", "#b71c1c"

ax = fig.add_subplot(gs[2, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 1.045, "Measured outcome per candidate quality  (same 24 frames, same call, three interpreters)",
    fontsize=12.4, fontweight="bold", transform=ax.transAxes)
COLS = [(0.0, "encode_clip(quality=...)"), (0.265, "main"), (0.505, "main, python -O"), (0.755, "this change")]
for x, h in COLS:
    put(ax, x, 0.945, h, fontsize=11.0, fontweight="bold", color="#263238", transform=ax.transAxes)
ax.plot([0, 1], [0.915, 0.915], color="#90a4ae", lw=1.2, transform=ax.transAxes, clip_on=False)
TOP, LAST = 0.855, 0.155
STEP = (TOP - LAST) / (len(ROWS) - 1)
assert STEP > 0.045, STEP
for i, (label, key) in enumerate(ROWS):
    y = TOP - i * STEP
    put(ax, 0.0, y, label, fontsize=10.6, family="monospace", color="#263238", transform=ax.transAxes)
    for x, facts in ((0.265, A), (0.505, AO), (0.755, B)):
        txt, colour = cell(facts["rows"][key])
        put(ax, x, y, txt, fontsize=10.1, family="monospace", color=colour, transform=ax.transAxes)
FOOT = LAST - STEP - 0.012
assert FOOT > 0.02, FOOT
put(ax, 0.0, FOOT,
    f"main: True encoded byte-identically to quality=1 (md5 {A['rows']['True']['md5']}) - a "
    f"{PSNR_REQ - PSNR_SUB:.2f} dB loss under status success.   Under -O the encoder's assert is stripped, so 0 and -5\n"
    f"encoded too ({AO['rows']['0']['psnr']:.2f} dB) and nan / '8' leaked raw arithmetic errors.   "
    f"np.int64(8) now encodes byte-identically to the plain int (md5 {B['rows']['8']['md5']}): the conversion is load-bearing.",
    fontsize=9.7, color="#37474f", transform=ax.transAxes, va="top")

for ax_, y, is_axes in placed:
    if is_axes:
        assert -0.05 <= y <= 1.08, (y, is_axes)
    else:
        lo, hi = ax_.get_ylim(); assert lo - 0.05 <= y <= hi + 0.08, (y, lo, hi)

out = pathlib.Path("/tmp/encode_clip_quality.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
from PIL import Image
im = np.asarray(Image.open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    nw = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert nw == 0, f"{name} border has {nw} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}  psnr {PSNR_REQ:.2f} vs {PSNR_SUB:.2f}")
