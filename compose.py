import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

pre  = json.loads(pathlib.Path("/tmp/edge_load_diffusers_0.39.0.json").read_text())
post = json.loads(pathlib.Path("/tmp/edge_load.json").read_text())
ok   = json.loads(pathlib.Path("/tmp/art_healthy.json").read_text())
video = np.load("/tmp/art_video.npy")

# ---- audit every claim against the dumps -------------------------------------
assert pre["diffusers_version"] == "0.39.0" and post["diffusers_version"] == "0.39.0"
assert ok["diffusers"] == "0.40.0.dev0"
assert pre["load_ok"] is False and post["load_ok"] is False
assert pre["load_error"].startswith("NotImplementedError: Cannot copy out of meta tensor")
assert "diffusers" not in pre["load_error"] and "Cosmos3-Edge" not in pre["load_error"], \
    "the pre-fix error must name neither the library nor the checkpoint"
assert post["load_error"].startswith("RuntimeError:")
for token in ("nvidia/Cosmos3-Edge", "0.39.0", "112 tensor(s)", "randomly initialized",
              "git+https://github.com/huggingface/diffusers", "backend='service'"):
    assert token in post["load_error"], token
assert ok["unloaded_after_guard"] == 0 and ok["n_params_scanned"] == 745
assert tuple(ok["video_shape"]) == (17, 256, 256, 3) == video.shape
assert ok["action_is_none_for_forward_dynamics"] is True
assert video.dtype == np.float32 and 0.0 <= float(video.min()) and float(video.max()) <= 1.0
# the generated video must actually move (a still filmstrip would prove nothing)
d = float(np.abs(video[-1] - video[0]).mean())
assert d > 0.01, f"generated video is static (mean |last-first| = {d:.4f})"

fig = plt.figure(figsize=(15.6, 12.4), dpi=124)
gs = fig.add_gridspec(3, 5, height_ratios=[1.06, 1.16, 0.98], hspace=0.30, wspace=0.06)
fig.suptitle(
    "Cosmos 3 diffusers backend: report which tensors a too-old diffusers could not fill",
    fontsize=15.5, fontweight="bold", y=0.983,
)
fig.text(0.5, 0.958,
         "measured on Thor (NVIDIA Thor, torch 2.11.0+cu130) with nvidia/Cosmos3-Edge - "
         "every number below is read from the run's own JSON dump",
         ha="center", fontsize=9.6, style="italic", color="#333333")

# ---- row 1: the real rollout the guarded backend still produces ---------------
idx = [0, 4, 8, 12, 16]
for col, i in enumerate(idx):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(np.clip(video[i], 0.0, 1.0)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"t = {i / ok['fps']:.2f}s", fontsize=9)
    if col == 0:
        ax.set_ylabel("predicted\nworld video", fontsize=9.5, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor("#1b7f3b"); s.set_linewidth(2.0)
fig.text(0.5, 0.706,
         f"Non-vacuity, on real hardware: with the guard in place and diffusers "
         f"{ok['diffusers']}, Cosmos3-Edge still loads "
         f"({ok['load_seconds']}s, {ok['unloaded_after_guard']} of "
         f"{ok['n_params_scanned']} parameters unfilled) and runs forward dynamics "
         f"in {ok['infer_seconds']}s at {ok['peak_gpu_gb']} GB peak -> "
         f"{video.shape[0]} frames @ {ok['fps']} fps.",
         ha="center", fontsize=10.1, color="#1b5e20")

# ---- row 2: what the caller is told -----------------------------------------
axr = fig.add_subplot(gs[1, :]); axr.axis("off"); axr.set_xlim(0, 1); axr.set_ylim(0, 1)
placed = []
def put(x, y, s, **kw):
    placed.append(y); axr.text(x, y, s, transform=axr.transAxes, **kw)

put(0.5, 1.035, "What the caller is told when loading nvidia/Cosmos3-Edge",
    ha="center", fontsize=12.6, fontweight="bold")
rows = [
    ("diffusers 0.39.0\n(has the pipeline,\ntoo old for this\ncheckpoint)",
     "NotImplementedError: Cannot copy out of\nmeta tensor; no data! Please use\n"
     "torch.nn.Module.to_empty() instead of\ntorch.nn.Module.to() ...\n\n"
     "-> names no library, no version, no\n   checkpoint, no remedy",
     "RuntimeError: Cosmos 3 checkpoint\n'nvidia/Cosmos3-Edge' was not fully loaded\n"
     "by the installed diffusers (0.39.0):\n112 tensor(s) were left uninitialized on\n"
     "the meta device (e.g. transformer.layers.0.\nself_attn.norm_q.weight), so the pipeline\n"
     "would run on randomly initialized weights.\nInstall a diffusers that supports this\n"
     "checkpoint: uv pip install 'diffusers @\ngit+https://github.com/huggingface/diffusers'",
     False),
    ("diffusers 0.40.0.dev0\n(supports this\ncheckpoint)",
     "loads and runs\n(0 of 745 parameters unfilled)",
     "loads and runs\n(0 of 745 parameters unfilled)\n-> unchanged",
     True),
]
put(0.135, 0.93, "installed diffusers", ha="center", fontsize=10.3, fontweight="bold")
put(0.435, 0.93, "on main", ha="center", fontsize=10.3, fontweight="bold")
put(0.775, 0.93, "with this change", ha="center", fontsize=10.3, fontweight="bold")
tops = [0.86, 0.30]
for (label, before, after, healthy), top in zip(rows, tops, strict=True):
    face = "#e8f5e9" if healthy else "#ffebee"
    axr.add_patch(Rectangle((0.005, top - (0.235 if healthy else 0.545)), 0.99,
                            (0.245 if healthy else 0.555), transform=axr.transAxes,
                            facecolor=face, edgecolor="#bbbbbb", lw=0.8, zorder=0))
    put(0.135, top - 0.02, label, ha="center", va="top", fontsize=9.4, fontweight="bold")
    put(0.275, top - 0.02, before, ha="left", va="top", fontsize=8.1, family="monospace",
        color="#7f1d1d" if not healthy else "#1b5e20")
    put(0.585, top - 0.02, after, ha="left", va="top", fontsize=8.1, family="monospace",
        color="#1b5e20")
assert all(-0.03 <= y <= 1.07 for y in placed), f"row-2 text out of axes: {placed}"

# ---- row 3: packaging ledger -------------------------------------------------
axp = fig.add_subplot(gs[2, :]); axp.axis("off"); axp.set_xlim(0, 1); axp.set_ylim(0, 1)
pl = []
def putp(x, y, s, **kw):
    pl.append(y); axp.text(x, y, s, transform=axp.transAxes, **kw)

putp(0.5, 1.02, "Packaging: the floors now name the release that ships the pipeline",
     ha="center", fontsize=12.4, fontweight="bold")
wheel = [
    ("diffusers release", "Cosmos3OmniPipeline", "CosmosActionCondition"),
    ("0.36.0", "absent", "absent"),
    ("0.37.1", "absent", "absent"),
    ("0.38.0", "absent", "absent"),
    ("0.39.0", "present", "present"),
]
TOP, LAST = 0.86, 0.40
STEP = (TOP - LAST) / (len(wheel) - 1)
assert STEP > 0.030, STEP
for i, (a, b, c) in enumerate(wheel):
    y = TOP - i * STEP
    bold = "bold" if i == 0 else "normal"
    col = "#111111" if i == 0 else ("#1b5e20" if b == "present" else "#7f1d1d")
    putp(0.035, y, a, fontsize=9.3, fontweight=bold, family="monospace", color=col)
    putp(0.175, y, b, fontsize=9.3, fontweight=bold, color=col)
    putp(0.315, y, c, fontsize=9.3, fontweight=bold, color=col)
putp(0.035, LAST - 1.35 * STEP,
     "measured by scanning each released wheel (pip download --no-deps) for the symbol",
     fontsize=8.5, style="italic", color="#444444")

changes = [
    ("declaration", "on main", "with this change"),
    ("[cosmos3-diffusers] diffusers", ">=0.30", ">=0.39"),
    ("[tool.uv] override-dependencies", ">=0.38.0", ">=0.39"),
    ("uv.lock resolves diffusers", "0.38.0  (no pipeline)", "0.39.0"),
]
TOP2, LAST2 = 0.86, 0.46
STEP2 = (TOP2 - LAST2) / (len(changes) - 1)
assert STEP2 > 0.030, STEP2
for i, (a, b, c) in enumerate(changes):
    y = TOP2 - i * STEP2
    bold = "bold" if i == 0 else "normal"
    putp(0.520, y, a, fontsize=9.2, fontweight=bold, family="monospace")
    putp(0.800, y, b, fontsize=9.2, fontweight=bold, family="monospace",
         color="#111111" if i == 0 else "#7f1d1d")
    putp(0.915, y, c, fontsize=9.2, fontweight=bold, family="monospace",
         color="#111111" if i == 0 else "#1b5e20")
putp(0.520, LAST2 - 1.30 * STEP2,
     "a uv override REPLACES a requirement, so at >=0.38.0 it silently discarded the\n"
     "extra's floor and the committed lock pinned a diffusers with no pipeline at all",
     fontsize=8.5, style="italic", color="#444444")
assert all(-0.08 <= y <= 1.07 for y in pl), f"row-3 text out of axes: {pl}"

out = pathlib.Path("/tmp/cosmos3_checkpoint_mismatch.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(plt.imread(out))[:, :, :3]
b = 8
for name, band in (("top", im[:b]), ("bottom", im[-b:]), ("left", im[:, :b]), ("right", im[:, -b:])):
    n = int((np.abs(band - 1.0) > 0.02).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", out, im.shape, f"mean|last-first| video = {d:.4f}")
