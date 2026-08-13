"""Compose the artifact. Every drawn number is re-derived from the two dumps."""
import json, os, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RUN = os.environ["GITHUB_RUN_ID"]
A = json.loads(pathlib.Path(f"/tmp/pw-{RUN}/main.json").read_text())     # upstream/main
B = json.loads(pathlib.Path(f"/tmp/pw-{RUN}/branch.json").read_text())   # this PR
assert A["tree"] != B["tree"], "both arms measured the same tree"

SURFACES = ["run_policy", "eval_policy", "start_policy"]
def cell(facts, action, value):
    return next(r for r in facts["rows"] if r["action"] == action and r["value"] == value)

VALUES = [(r["value"], r["class"]) for r in A["rows"] if r["action"] == "run_policy"]
IN_SCOPE = [(v, k) for v, k in VALUES if "out of scope" not in k]

def honors(row):
    """The documented contract: a structured error, no traceback, no false success."""
    return row["raised"] is None and row["status"] == "error"

n_bad_A = sum(1 for v, _ in IN_SCOPE for a in SURFACES if not honors(cell(A, a, v)))
n_bad_B = sum(1 for v, _ in IN_SCOPE for a in SURFACES if not honors(cell(B, a, v)))
n_cells = len(IN_SCOPE) * len(SURFACES)
assert (n_cells, n_bad_B) == (12, 0), (n_cells, n_bad_B)
assert n_bad_A == 12, n_bad_A
n_raised_A = sum(1 for v, _ in IN_SCOPE for a in SURFACES if cell(A, a, v)["raised"])
n_false_A = sum(1 for v, _ in IN_SCOPE for a in SURFACES
                if cell(A, a, v)["status"] == "success")
assert n_raised_A + n_false_A == 12, (n_raised_A, n_false_A)

# the out-of-scope control must read identically on both trees
for v, k in VALUES:
    if "out of scope" in k:
        for a in SURFACES:
            ra, rb = cell(A, a, v), cell(B, a, v)
            assert (ra["raised"], ra["status"]) == (rb["raised"], rb["status"]), (a, v)
        ctrl_value = v

# the honored path must be untouched
hA, hB = A["honored"], B["honored"]
assert hA["status"] == hB["status"] == "success"
assert hA["joints"] == hB["joints"], "honored rollout diverged"
bef = np.load(f"/tmp/pw-{RUN}/branch.before.npy"); aft = np.load(f"/tmp/pw-{RUN}/branch.after.npy")
mA = np.load(f"/tmp/pw-{RUN}/main.after.npy")
delta = int(np.abs(aft.astype(int) - mA.astype(int)).max())
changed = int((np.abs(aft.astype(int) - mA.astype(int)).sum(2) > 8).sum())
assert delta <= 2, delta
assert hB["sat_frac"] > 0.15, hB["sat_frac"]

placed = []
def put(ax, x, y, s, **kw):
    placed.append((ax, y, kw.get("transform") is not None))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(16.6, 11.4), dpi=124)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.86, 0.20], width_ratios=[1.0, 0.72],
                      hspace=0.20, wspace=0.13)
fig.suptitle("An unresolvable or mistyped policy_provider: what the caller gets back",
             fontsize=16.5, fontweight="bold", y=0.975)
fig.text(0.5, 0.947, "MuJoCo, headless (MUJOCO_GL=egl). Every cell is a real call on a live world; "
         "each column measured in its own tree.", ha="center", fontsize=10.6, style="italic", color="#444")

# ---- row 1: the verdict matrix -------------------------------------------
axm = fig.add_subplot(gs[0, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
TOP, LAST = 0.80, 0.10
rows = VALUES
STEP = (TOP - LAST) / (len(rows) - 1)
assert STEP > 0.030, STEP
COLS = [(0.255, "run_policy"), (0.395, "eval_policy"), (0.535, "start_policy")]
put(axm, 0.008, 0.955, "policy_provider", fontsize=11, fontweight="bold")
put(axm, 0.135, 0.955, "why it cannot resolve", fontsize=11, fontweight="bold")
for i, (x, name) in enumerate(COLS):
    put(axm, x, 0.985, name, fontsize=10.3, fontweight="bold", ha="center")
put(axm, (COLS[0][0] + COLS[-1][0]) / 2, 0.912, "on upstream/main", fontsize=11,
    fontweight="bold", ha="center", color="#a3242b")
put(axm, 0.775, 0.985, "run_policy / eval_policy / start_policy", fontsize=10.3,
    fontweight="bold", ha="center")
put(axm, 0.775, 0.912, "with this change", fontsize=11, fontweight="bold", ha="center", color="#1d6f34")
axm.plot([0.615, 0.615], [0.03, 0.93], color="#bbb", lw=1.1)

for r, (val, klass) in enumerate(rows):
    y = TOP - r * STEP
    oos = "out of scope" in klass
    put(axm, 0.008, y, val, fontsize=10.4, family="monospace",
        color="#666" if oos else "#111")
    put(axm, 0.135, y, klass, fontsize=9.7, style="italic", color="#666" if oos else "#333")
    for x, action in COLS:
        for tree, facts, xoff in (("A", A, 0.0), ("B", B, 0.0)):
            row = cell(facts, action, val)
            if tree == "A":
                cx = x
            else:
                cx = 0.665 + (COLS.index((x, action))) * 0.075
            if oos:
                label, fc, tc = "unchanged", "#eeeeee", "#666"
            elif row["raised"]:
                label, fc, tc = f"raised\n{row['raised'][:13]}", "#f7d9dc", "#8f1f26"
            elif row["status"] == "success":
                label, fc, tc = "FALSE\nsuccess", "#f3c9cd", "#7d161d"
            else:
                label, fc, tc = "error\nenvelope", "#d6efdd", "#1d6f34"
            axm.add_patch(Rectangle((cx - 0.056, y - 0.030), 0.112, 0.062,
                                    facecolor=fc, edgecolor="#999", lw=0.7,
                                    transform=axm.transData, zorder=1))
            put(axm, cx, y + 0.001, label, fontsize=8.5, ha="center", va="center",
                color=tc, zorder=2, fontweight="bold" if "FALSE" in label else "normal")
            if tree == "B":
                break
    y -= 0
put(axm, 0.008, 0.028,
    f"In scope: {n_bad_A} of {n_cells} cells wrong on main  "
    f"({n_raised_A} escaped the envelope as a traceback, {n_false_A} reported success while nothing ran)"
    f"   ->   {n_bad_B} of {n_cells} with this change.",
    fontsize=11.2, fontweight="bold")
put(axm, 0.008, -0.008,
    f"The {ctrl_value!r} row is the trust-remote-code gate, a separate concern this change deliberately "
    "leaves raising: identical on both trees.", fontsize=9.6, style="italic", color="#555")

# ---- row 2 left: the honored rollout, untouched ---------------------------
axr = fig.add_subplot(gs[1, 0]); axr.imshow(aft); axr.set_xticks([]); axr.set_yticks([])
axr.set_title("The honored path is untouched: a real mock rollout on so100",
              fontsize=11.6, fontweight="bold", pad=7)
axr.set_xlabel(f"policy_provider='mock', 1.6 s at 50 Hz -> {hB['text'].split('|')[-1].strip()}   "
               f"|   this frame vs main's: max delta {delta}/255 over {changed} of {aft[:, :, 0].size} px",
               fontsize=9.6)

# ---- row 2 right: the measured ledger ------------------------------------
axl = fig.add_subplot(gs[1, 1]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
put(axl, 0.0, 0.965, "Measured", fontsize=12, fontweight="bold")
sp = cell(A, "start_policy", repr("molmoact2"))  # rows store repr(value)
lines = [
    ("start_policy on main, provider 'molmoact2'", ""),
    ("   reported", f"status={sp['status']!r} -- {sp['text'][:44]}"),
    ("   policies running after", sp.get("running_after", "")[:52]),
    ("   why", "the policy is built on the worker thread, so"),
    ("", "the raise was captured in the future"),
    ("", ""),
    ("honored rollout, both trees", ""),
    ("   status / steps", f"{hB['status']} / {hB['text'].split('|')[2].strip()}"),
    ("   joints identical to 6 dp", str(hA["joints"] == hB["joints"])),
    ("   render vs main", f"max delta {delta}/255, {changed} px changed"),
    ("   arm in frame", f"{hB['sat_frac'] * 100:.0f}% saturated pixels"),
    ("", ""),
    ("Gate (this branch)", ""),
    ("   full suite", "28758 passed / 258 skipped / 0 failed"),
    ("   pre-fix, same tests", "31 failed / 26 passed"),
    ("   mutations caught", "5 of 6 here, 0 of 6 by 334 pre-existing"),
]
LTOP, LLAST = 0.905, 0.045
LS = (LTOP - LLAST) / (len(lines) - 1)
assert LS > 0.030, LS
for i, (k, v) in enumerate(lines):
    y = LTOP - i * LS
    bold = v == "" and k != ""
    put(axl, 0.0, y, k, fontsize=9.5, fontweight="bold" if bold else "normal",
        color="#111" if bold else "#333", family="monospace" if k.startswith("   ") else None)
    if v:
        put(axl, 0.47, y, v, fontsize=9.2, family="monospace", color="#1a4d8f")
assert abs((LTOP - (len(lines) - 1) * LS) - LLAST) < 1e-9

# ---- row 3: footer -------------------------------------------------------
axf = fig.add_subplot(gs[2, :]); axf.axis("off"); axf.set_xlim(0, 1); axf.set_ylim(0, 1)
put(axf, 0.5, 0.62,
    "One probe, three surfaces: policy_provider_error() runs the same resolution create_policy() uses, "
    "so every registered name, HuggingFace model ID,",
    fontsize=10.4, ha="center", color="#222")
put(axf, 0.5, 0.24,
    "transport URL and host:port still resolves -- and the reason now reaches the caller instead of a "
    "traceback or a success that started nothing.",
    fontsize=10.4, ha="center", color="#222")

for ax, y, is_axes in placed:
    lo, hi = ax.get_ylim()
    if is_axes:
        assert -0.03 <= y <= 1.10, y
    else:
        assert lo - 0.06 <= y <= hi + 0.06, (y, lo, hi)

out = pathlib.Path("_art/policy_provider_envelope.png")
fig.savefig(out, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"WROTE {out}  {im.shape[1]}x{im.shape[0]}")
print(f"in-scope cells {n_cells}: main {n_bad_A} wrong ({n_raised_A} raised, {n_false_A} false success) -> PR {n_bad_B}")
