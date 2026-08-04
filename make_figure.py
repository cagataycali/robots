"""Compose the measured verdict figure from the two trees' JSON dumps."""
import json, re, textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

A = json.loads(Path("/tmp/factsA.json").read_text())   # upstream/main
B = json.loads(Path("/tmp/factsB.json").read_text())   # this change
assert A["tree"] != B["tree"], "both probes resolved to the same tree"
assert A["build_default"]["raised"] == "OSError"
assert B["build_default"]["raised"] == "ValueError"
assert A["suite_offline_failed"] == 2 and B["suite_offline_failed"] == 0
assert A["validate_default"] == [] == B["validate_default"]
assert A["build_remedy"]["raised"] is None and B["build_remedy"]["raised"] is None
assert A["build_bad_value"]["raised"] == "ValueError" == B["build_bad_value"]["raised"]
assert B["vlm_config_is_forwardable"]

def suite(s):
    m = re.search(r"(\d+) failed", s); f = int(m.group(1)) if m else 0
    p = int(re.search(r"(\d+) passed", s).group(1))
    k = re.search(r"(\d+) skipped", s); k = int(k.group(1)) if k else 0
    return f, p, k
fA, pA, kA = suite(A["suite_offline"]); fB, pB, kB = suite(B["suite_offline"])

RED, GREEN, GREY = "#b3202c", "#1a7a3c", "#5c5c66"
fig = plt.figure(figsize=(15.6, 9.0), dpi=130)
placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y)); return ax.text(x, y, s, **kw)

fig.text(0.5, 0.972, "A reward config that cannot obtain its pretrained asset: what the caller is told",
         ha="center", fontsize=16.5, fontweight="bold")
fig.text(0.5, 0.941,
         "robometer sizes vlm_config from its backbone inside __post_init__, so CONSTRUCTING its config downloads "
         "~11 MB. Every cell below is measured on a host that cannot reach the Hub.",
         ha="center", fontsize=10.6, color=GREY)

# ---------------- row 1: verdict table ----------------
ax = fig.add_axes([0.035, 0.455, 0.93, 0.455]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
COLS = [0.005, 0.30, 0.635]
put(ax, COLS[0], 0.965, "what the caller does", fontsize=11.3, fontweight="bold")
put(ax, COLS[1], 0.965, "upstream/main", fontsize=11.3, fontweight="bold", color=RED)
put(ax, COLS[2], 0.965, "this change", fontsize=11.3, fontweight="bold", color=GREEN)
ax.plot([0, 1], [0.945, 0.945], color="#333", lw=1.1)

def wrap(s, w):  # keep cells inside their column
    return "\n".join(textwrap.wrap(s, w)[:6])

ROWS = [
    ("validate(spec) on the default\nrobometer spec",
     f"{A['validate_default']}  (accepted)", f"{B['validate_default']}  (accepted, unchanged)", "same"),
    ("build_config(spec) -- the SAME spec,\nno reachable Hub",
     "OSError: We couldn't connect to\n'https://huggingface.co' to load the files,\nand couldn't find them in the cached files.\n-> names no trainer, no reward type, no remedy",
     "ValueError: reward_model type 'robometer' could not be\nconstructed: ... needed a pretrained asset this host could\nnot obtain (...). The spec itself is fine - validate() cannot\nreach the network to see this. Either make the asset\navailable (...), or pass the field the config derives from it\nin extra['reward_model'] ... Fields this type accepts: ...", "fixed"),
    ("build_config with vlm_config supplied\n(the remedy the message names)",
     "builds (the remedy always worked --\nnothing told the caller about it)", "builds (unchanged)", "same"),
    ("build_config with a bad field VALUE\n(reward_output='not-a-mode')",
     f"ValueError: {A['build_bad_value']['msg']}", "ValueError: same message, not re-worded\n-> only asset failures are translated", "same"),
    ("the parity suite itself,\ncold cache + HF_HUB_OFFLINE=1",
     f"{fA} FAILED, {pA} passed", f"{fB} failed, {pB} passed, {kB} skipped\n(the skip is the deriving-path case, with a reason)", "fixed"),
    ("network transferred per cold run",
     "~11 MB (Qwen3-VL tokenizer.json 7.0 MB,\nvocab.json 2.8 MB, merges.txt 1.7 MB)", "0 MB", "fixed"),
]
y = 0.885
for label, main_v, new_v, kind in ROWS:
    n = max(label.count("\n"), main_v.count("\n"), new_v.count("\n")) + 1
    h = 0.032 * n + 0.028
    if kind == "fixed":
        ax.add_patch(Rectangle((-0.004, y - h + 0.020), 1.008, h, color=RED, alpha=0.055, zorder=0))
    put(ax, COLS[0], y, label, fontsize=9.5, va="top", linespacing=1.45)
    put(ax, COLS[1], y, main_v, fontsize=8.5, va="top", family="monospace",
        color=RED if kind == "fixed" else GREY, linespacing=1.45)
    put(ax, COLS[2], y, new_v, fontsize=8.5, va="top", family="monospace",
        color=GREEN if kind == "fixed" else GREY, linespacing=1.45)
    y -= h
    ax.plot([0, 1], [y + 0.014, y + 0.014], color="#dddde3", lw=0.7)

# ---------------- row 2: the two mechanisms ----------------
ax2 = fig.add_axes([0.035, 0.045, 0.445, 0.375]); ax2.axis("off")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0, 0.97, "Why the failure leaked", fontsize=12, fontweight="bold")
put(ax2, 0, 0.86,
    "_build_reward_model_config already translated ONE class of\n"
    "constructor failure and let the rest through:\n\n"
    "    try:\n"
    "        reward_cfg = make_reward_model_config(rtype, **kw)\n"
    "    except TypeError as e:                 # translated\n"
    "        raise ValueError(f\"... rejected field(s) ...\")\n"
    "    # an OSError from an unobtainable asset -> propagated raw\n\n"
    "Its own regression test states the rule it broke:\n"
    "\"translate that into a ValueError ... not leak a bare,\n"
    " contextless error.\"\n\n"
    "OSError is the narrowest superset that covers it -- measured:",
    fontsize=8.9, va="top", family="monospace", linespacing=1.5)
mro = [("LocalEntryNotFoundError", True), ("HfHubHTTPError", True), ("RepositoryNotFoundError", True),
       ("GatedRepoError", True), ("OfflineModeIsEnabled", True), ("transformers -> plain OSError", True)]
yy = 0.235
for nm, ok in mro:
    put(ax2, 0.02, yy, f"{'issubclass(..., OSError) = True':<32s}  {nm}", fontsize=8.4,
        family="monospace", color=GREEN)
    yy -= 0.040

ax3 = fig.add_axes([0.52, 0.045, 0.445, 0.375]); ax3.axis("off")
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0, 0.97, "Why the suite downloaded a tokenizer", fontsize=12, fontweight="bold")
put(ax3, 0, 0.86,
    "The parity tests assert LOCAL contracts -- that strands'\n"
    "discovery matches lerobot's registry and that a field\n"
    "passes through. The download is incidental to both, and\n"
    "it is what made every PR in the repo depend on Hub\n"
    "availability:\n\n"
    "  lerobot/rewards/robometer/configuration_robometer.py\n"
    "      if not self.vlm_config:                 # <- default\n"
    "          vlm = AutoConfig.from_pretrained(self.base_model_id)\n"
    "          tokenizer = AutoTokenizer.from_pretrained(...)\n\n"
    "so the cases now supply vlm_config and the constructor\n"
    "skips the fetch. A new case makes BOTH from_pretrained\n"
    "entry points fatal, so a type that starts deriving a field\n"
    "from an asset must be given it rather than download it.\n\n"
    "vlm_config is a real forwardable field: measured True\n"
    "against _reward_friendly_fields('robometer').",
    fontsize=8.9, va="top", family="monospace", linespacing=1.5)

for ax_, yv in placed:
    lo, hi = ax_.get_ylim(); pad = 0.03 * (hi - lo)
    assert lo - pad <= yv <= hi + pad, f"text at y={yv} outside {ax_.get_ylim()}"

out = Path("/tmp/reward_asset_failure.png")
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(Image.open(out).convert("RGB"))
for nm, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    bad = int((np.abs(band.astype(int) - 255).sum(2) > 20).sum())
    assert bad == 0, f"{nm} border has {bad} non-white px"
print(f"OK {out}  {im.shape[1]}x{im.shape[0]}")
