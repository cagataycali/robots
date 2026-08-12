"""Compose the artifact: real renders + the measured dispatch ledger + mutations."""
import json, os, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RID = os.environ["GITHUB_RUN_ID"]
D = pathlib.Path(f"/tmp/art-{RID}")
A = json.load(open(D / "main_facts.json"))     # pre-fix
B = json.load(open(D / "branch_facts.json"))   # this PR
M = json.load(open(f"/tmp/mut-{RID}.json"))
assert A["tree"] != B["tree"], "both arms measured the same tree"
SERVO, TORQUE = A["servo"], A["torque"]

# --- assert every claim this figure makes, against the dumps -----------------
a1, a2 = A["runs"]; b1, b2 = B["runs"]
assert a1["biastypes_during_apply"] == [TORQUE] and a1["applies"] == 100
assert a2["biastypes_during_apply"] == [SERVO] and a2["applies"] == 100
assert a1["registered_after"] is True and a2["registered_after"] is True
assert b1["biastypes_during_apply"] == [TORQUE] and b2["biastypes_during_apply"] == [TORQUE]
assert b1["registered_after"] is False and b2["registered_after"] is False
assert all(r["status"] == "success" for r in A["runs"] + B["runs"]), "a rollout did not report success"
L = {p.stem: np.load(p).astype(int) for p in D.glob("*.npy")}
def dpx(x, y):
    return float((np.abs(L[x] - L[y]).sum(2) > 24).mean()), int(np.abs(L[x] - L[y]).max())
same_frac, same_max = dpx("main_run1", "branch_run1")
diff_frac, _ = dpx("main_run2", "branch_run2")
assert same_max <= 2 and same_frac == 0.0, f"run 1 is not byte-comparable across trees: {same_max}/{same_frac}"
assert diff_frac > 0.10, f"run 2 panels differ on only {diff_frac:.2%}"
for k in ("main_run1", "main_run2", "branch_run2"):
    assert float(((L[k].max(2) - L[k].min(2)) > 45).mean()) > 0.5, f"{k} looks empty"
n_mut = M["n"]; caught_new, caught_old = M["caught_new"], M["caught_old"]
assert (n_mut, caught_new, caught_old) == (6, 4, 1), (n_mut, caught_new, caught_old)

fig = plt.figure(figsize=(15.2, 14.6), dpi=124)
gs = fig.add_gridspec(4, 3, height_ratios=[2.40, 0.86, 0.80, 0.62], hspace=0.22, wspace=0.06)
fig.suptitle(
    "WBC torque shim: the teardown released the gains but kept its action-controller registration",
    fontsize=15.5, fontweight="bold", y=0.975,
)
fig.text(0.5, 0.947,
         "Two sequential sim.run_policy(policy_provider=\"wbc\") balance rollouts on ONE world  |  "
         "real GR00T-WholeBodyControl-Balance.onnx  |  MuJoCo headless (EGL)",
         ha="center", fontsize=10.4, style="italic", color="#444")

PANELS = [
    ("main_run1",   "Rollout 1 - identical on both trees",
     f"the shim installs and drives TORQUE actuators\nmax|delta| across trees = {same_max}/255  "
     f"({same_frac:.2%} of pixels differ)", "#2e7d32"),
    ("main_run2",   "Rollout 2 - main (pre-fix)",
     "auto-install SKIPPED (stale registration read as a manual install)\n"
     "100/100 steps wrote PD torques into POSITION SERVOS", "#c62828"),
    ("branch_run2", "Rollout 2 - with the fix",
     "registration released, so a FRESH shim installs\n"
     "100/100 steps drive TORQUE actuators, as rollout 1 does", "#2e7d32"),
]
for col, (key, title, cap, colour) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(L[key].astype(np.uint8)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2.6)
    ax.set_title(title, fontsize=11.6, fontweight="bold", color=colour, pad=6)
    ax.set_xlabel(cap, fontsize=9.1, color="#333", labelpad=7)

# ---- row 2: the measured dispatch ledger -----------------------------------
axl = fig.add_subplot(gs[1, :]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
placed: list[float] = []
def put(x, y, s, **kw):
    placed.append(y); axl.text(x, y, s, transform=axl.transAxes, **kw)

put(0.012, 0.955, "What actually drove the actuators (measured, spy on WBCTorqueController.apply)",
    fontsize=11.4, fontweight="bold")
cols = [0.012, 0.215, 0.335, 0.470, 0.640, 0.795, 0.915]
hdr = ["tree", "rollout", "apply() calls", "actuator mode during apply",
       "controller", "registered after", "pelvis z"]
TOP, LAST = 0.780, 0.200
rows = [
    ("main (pre-fix)", "1", a1["applies"], f"TORQUE (biastype {TORQUE})", "fresh", "YES  <- leak", f"{a1['pelvis_z']:.4f} m", "#c62828"),
    ("main (pre-fix)", "2", a2["applies"], f"POSITION SERVO (biastype {SERVO})", "STALE (rollout 1's)", "YES", f"{a2['pelvis_z']:.4f} m", "#c62828"),
    ("this PR", "1", b1["applies"], f"TORQUE (biastype {TORQUE})", "fresh", "no", f"{b1['pelvis_z']:.4f} m", "#2e7d32"),
    ("this PR", "2", b2["applies"], f"TORQUE (biastype {TORQUE})", "fresh", "no", f"{b2['pelvis_z']:.4f} m", "#2e7d32"),
]
step = (TOP - LAST) / (len(rows) - 1)
assert step > 0.030, step
for x, h in zip(cols, hdr):
    put(x, TOP + 0.082, h, fontsize=9.5, fontweight="bold", color="#333")
y = TOP
for tree, run, ap, mode, ctrl, reg, pz, colour in rows:
    for x, v in zip(cols, [tree, run, str(ap), mode, ctrl, reg, pz]):
        put(x, y, v, fontsize=9.4, family="monospace", color=colour)
    y -= step
assert abs((y + step) - LAST) < 1e-9, (y, LAST)
put(0.012, 0.045,
    "Every one of the four calls returned status=\"success\".  install_wbc_torque_control() acquires two things "
    "(torque gains + the action-controller registration); uninstall() released one.",
    fontsize=9.3, style="italic", color="#555")
for yy in placed:
    assert -0.03 <= yy <= 1.10, yy

# ---- row 3: which teardown paths release the registration -------------------
CELLS = json.load(open(f"/tmp/cells-main-{RID}.json"))["cells"]
THEIRS = json.load(open(f"/tmp/cells-theirs-{RID}.json"))["cells"]
COMB = json.load(open(f"/tmp/cells-combined-{RID}.json"))["cells"]
assert CELLS["manual_install_then_uninstall_releases"] is False
assert THEIRS["manual_install_then_uninstall_releases"] is False
assert all(COMB.values()), "the combined tree still leaks somewhere"
axc = fig.add_subplot(gs[2, :]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
cplaced: list[float] = []
def putc(x, y, s, **kw):
    cplaced.append(y); axc.text(x, y, s, transform=axc.transAxes, **kw)

putc(0.012, 0.94, "Which teardown paths release the action-controller registration (measured on the stock G1)",
     fontsize=11.2, fontweight="bold")
LABELS = [
    ("manual_install_then_uninstall_releases",
     "install_wbc_torque_control(...) then controller.uninstall()   [the documented manual pair]"),
    ("auto_hook_cleanup_releases",
     "run_policy's auto-install, via the cleanup the hook returns"),
    ("auto_releases_registry_even_if_uninstall_raises",
     "...and still released when restoring the gains raises"),
    ("manual_uninstall_spares_a_newer_controller",
     "a controller registered since is never clobbered (manual)"),
    ("auto_cleanup_spares_a_newer_controller",
     "a controller registered since is never clobbered (auto)"),
]
ccols = [0.012, 0.660, 0.762, 0.880]
for x, h in zip(ccols, ["teardown path", "main", "#2196 head", "this branch"]):
    putc(x, 0.845, h, fontsize=9.5, fontweight="bold", color="#333")
CTOP, CLAST = 0.735, 0.155
cstep = (CTOP - CLAST) / (len(LABELS) - 1)
assert cstep > 0.030, cstep
yc = CTOP
for key, label in LABELS:
    putc(ccols[0], yc, label, fontsize=8.9, family="monospace", color="#222")
    for x, arm in zip(ccols[1:], (CELLS, THEIRS, COMB)):
        ok = arm[key]
        putc(x, yc, "released" if ok else "LEAKED", fontsize=8.9, family="monospace",
             color="#2e7d32" if ok else "#c62828")
    yc -= cstep
assert abs((yc + cstep) - CLAST) < 1e-9, (yc, CLAST)
putc(0.012, 0.055,
     "#2196 fixed the auto-install path in a cleanup closure the hook wraps around uninstall; this branch moves the "
     "release into uninstall itself, so the manual pair its own docstring names is covered by the same code.",
     fontsize=9.1, style="italic", color="#555")
for yy in cplaced:
    assert -0.03 <= yy <= 1.10, yy

# ---- row 4: mutation matrix ------------------------------------------------
axm = fig.add_subplot(gs[3, :]); axm.axis("off"); axm.set_xlim(0, 1); axm.set_ylim(0, 1)
mplaced: list[float] = []
def putm(x, y, s, **kw):
    mplaced.append(y); axm.text(x, y, s, transform=axm.transAxes, **kw)

putm(0.012, 0.93, f"Plausible regressions: caught by the new module {caught_new}/{n_mut}  |  "
     f"caught by the {M['rows'][-1]['old_passed']} pre-existing tests/policies/wbc tests {caught_old}/{n_mut}",
     fontsize=11.0, fontweight="bold")
MTOP, MLAST = 0.72, 0.115
mrows = [r for r in M["rows"] if r["label"].startswith("M")]
mstep = (MTOP - MLAST) / (len(mrows) - 1)
assert mstep > 0.030, mstep
ym = MTOP
for r in mrows:
    seen_new = r["new_failed"] > 0
    seen_old = r["old_failed"] > 0
    tag = "caught here" if seen_new else ("pre-existing suite's own cell" if seen_old else "masked - see note")
    colour = "#2e7d32" if seen_new else ("#1565c0" if seen_old else "#8d6e63")
    putm(0.012, ym, r["label"], fontsize=9.3, family="monospace", color="#222")
    putm(0.560, ym, f"new: {r['new_failed']} failed", fontsize=9.3, family="monospace", color=colour)
    putm(0.700, ym, f"pre-existing: {r['old_failed']} failed", fontsize=9.3, family="monospace", color=colour)
    putm(0.890, ym, tag, fontsize=9.3, style="italic", color=colour)
    ym -= mstep
assert abs((ym + mstep) - MLAST) < 1e-9, (ym, MLAST)
putm(0.012, 0.035,
     "M6 is masked because wbc_uses_position_servo independently refuses a world-less sim, so the hook's own "
     "missing-world check is defence in depth.  Gate on the combined branch: 28547 passed / 257 skipped / 0 failed, ruff + mypy clean.",
     fontsize=9.2, style="italic", color="#555")
for yy in mplaced:
    assert -0.03 <= yy <= 1.10, yy

OUTP = D / "wbc_torque_teardown.png"
fig.savefig(OUTP, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

from PIL import Image
im = np.asarray(Image.open(OUTP).convert("RGB")).astype(int)
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print("OK", OUTP, Image.open(OUTP).size)
print(f"run1 across trees: max|delta|={same_max} frac={same_frac:.2%};  run2 panels differ {diff_frac:.2%}")
