"""Compose the measured figure. Every cell is read from capture.py's dump."""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

F = json.load(open(sys.argv[1]))
OUT = pathlib.Path(sys.argv[2])
ROOT = pathlib.Path(__file__).resolve().parents[1]
assert F["tree"] == str(ROOT), (F["tree"], ROOT)

SEC = "strands_robots/mesh/security.py"
ACL = "strands_robots/mesh/_acl_config.py"
cov = F["coverage"]
# ---- assert the accounting the figure will state ----
assert cov[SEC]["main_missing"] == [732, 733, 880], cov[SEC]["main_missing"]
assert cov[SEC]["pr_missing"] == [732, 733], cov[SEC]["pr_missing"]
assert cov[ACL]["main_missing"] == [157, 159, 604, 607], cov[ACL]["main_missing"]
assert cov[ACL]["pr_missing"] == [604, 607], cov[ACL]["pr_missing"]
muts = F["mutations"]
assert len(muts) == 5, len(muts)
assert all(m["pr"]["failed"] > 0 for m in muts), "every mutation must be caught by this PR"
assert all(m["main"]["failed"] == 0 for m in muts), "the claim is that main catches none"
assert all(m["main"]["passed"] == 93 for m in muts)
assert F["clean"]["pr"] == {"failed": 0, "passed": 106}, F["clean"]["pr"]
assert F["clean"]["main"] == {"failed": 0, "passed": 93}, F["clean"]["main"]

GREEN, RED, GREY, INK = "#1b7f3b", "#b3261e", "#8a8a8a", "#101010"
placed = []

def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 11.6), dpi=124)
gs = fig.add_gridspec(3, 1, height_ratios=[1.24, 1.0, 1.02], hspace=0.16,
                      left=0.018, right=0.982, top=0.945, bottom=0.022)
fig.suptitle("Two second-line guards on the mesh wire-authorisation path, and what reached them",
             fontsize=16.5, fontweight="bold", y=0.982)
fig.text(0.5, 0.958, f"measured on {F['base'][:8]} -- tests and docstrings only, no production line changes",
         ha="center", fontsize=10.6, color=GREY, style="italic")

# ---------------- ROW 1: the two guards ----------------
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965, "1.  Each guard is the second of two checks, and only the first one ran",
    fontsize=13.2, fontweight="bold", color=INK, transform=ax.transAxes)

COLS = [0.0, 0.175, 0.395, 0.605, 0.795]
HEAD = ["guard (the 2nd check)", "the 1st check in front of it",
        "what defeats the 1st", "layer the suite reached\non main", "and with this PR"]
for x, h in zip(COLS, HEAD):
    put(ax, x, 0.865, h, fontsize=10.0, fontweight="bold", color=GREY, transform=ax.transAxes, va="top")
ax.plot([0, 1], [0.815, 0.815], color=GREY, lw=0.9, transform=ax.transAxes)

ROWS1 = [
    ("validate_command:880\ncontrol-char post-check\non policy_host",
     "is_safe_policy_host()\nallowlist membership",
     "a refactor of the allowlist\ncompare -- named in the\nguard's own comment",
     "the 1st only\n(message: \"not in allowlist\")",
     "both, separately\n(gate widened -> post-check\nrefuses all 6 byte classes)"),
    ("_load_acl_file:157-159\nO_NOFOLLOW / ELOOP\nrefusal",
     "path.is_symlink()\nstatic lstat check",
     "a symlink swapped between\nis_symlink() and os.open()\n-- a 2-syscall TOCTOU window",
     "the 1st only\n(and its assertion could\nnot tell them apart)",
     "both, separately\n(race modelled -> cause is\nOSError errno=ELOOP)"),
]
TOP1, LAST1 = 0.735, 0.245
STEP1 = (TOP1 - LAST1) / (len(ROWS1) - 1)
assert STEP1 > 0.30, STEP1
for i, row in enumerate(ROWS1):
    y = TOP1 - i * STEP1
    ax.add_patch(Rectangle((-0.004, y - 0.20), 1.008, 0.315, transform=ax.transAxes,
                           facecolor="#f5f7fa" if i % 2 == 0 else "white",
                           edgecolor="none", zorder=0))
    for j, (x, cell) in enumerate(zip(COLS, row)):
        col = INK
        if j == 3: col = RED
        if j == 4: col = GREEN
        put(ax, x, y, cell, fontsize=9.5, color=col, transform=ax.transAxes, va="top",
            family="monospace" if j == 0 else None,
            fontweight="bold" if j == 4 else None)
put(ax, 0.0, 0.055,
    "Each guard's own comment records why it is kept. Neither was reached by any test, so either could have been "
    "deleted with the suite green.",
    fontsize=10.2, color=GREY, style="italic", transform=ax.transAxes, va="top")

# ---------------- ROW 2: mutation matrix ----------------
ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
put(ax2, 0.0, 0.965, "2.  Mutation table -- each plausible regression, against both test sets",
    fontsize=13.2, fontweight="bold", color=INK, transform=ax2.transAxes)
C2 = [0.0, 0.055, 0.60, 0.80]
for x, h in zip(C2, ["", "regression applied to the source",
                     "this PR's tests\n(106)", "main's own tests\n(93)"]):
    put(ax2, x, 0.855, h, fontsize=10.0, fontweight="bold", color=GREY, transform=ax2.transAxes, va="top")
ax2.plot([0, 1], [0.775, 0.775], color=GREY, lw=0.9, transform=ax2.transAxes)
TOP2, LAST2 = 0.685, 0.175
STEP2 = (TOP2 - LAST2) / (len(muts) - 1)
assert STEP2 > 0.09, STEP2
for i, m in enumerate(muts):
    y = TOP2 - i * STEP2
    ax2.add_patch(Rectangle((-0.004, y - 0.052), 1.008, 0.104, transform=ax2.transAxes,
                            facecolor="#f5f7fa" if i % 2 == 0 else "white", edgecolor="none", zorder=0))
    put(ax2, C2[0], y, m["tag"], fontsize=10.2, fontweight="bold", color=GREY,
        transform=ax2.transAxes, va="center", family="monospace")
    put(ax2, C2[1], y, m["label"], fontsize=10.4, color=INK, transform=ax2.transAxes, va="center")
    put(ax2, C2[2], y, f"{m['pr']['failed']} failed  <- caught", fontsize=10.4, color=GREEN,
        fontweight="bold", transform=ax2.transAxes, va="center", family="monospace")
    put(ax2, C2[3], y, f"{m['main']['passed']} passed  <- BLIND", fontsize=10.4, color=RED,
        fontweight="bold", transform=ax2.transAxes, va="center", family="monospace")
put(ax2, 0.0, 0.085,
    f"5 of 5 caught here; 0 of 5 by main. Unmutated: this PR {F['clean']['pr']['passed']} passed, "
    f"main {F['clean']['main']['passed']} passed. Sources restored byte-identical after each row.",
    fontsize=10.2, color=GREY, style="italic", transform=ax2.transAxes, va="top")

# ---------------- ROW 3: coverage accounting ----------------
ax3 = fig.add_subplot(gs[2]); ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
put(ax3, 0.0, 0.965, "3.  Complete accounting of the layer -- every uncovered line in the two modules that decide wire authorisation",
    fontsize=13.2, fontweight="bold", color=INK, transform=ax3.transAxes)
n_sec_b, n_sec_a = len(cov[SEC]["main_missing"]), len(cov[SEC]["pr_missing"])
n_acl_b, n_acl_a = len(cov[ACL]["main_missing"]), len(cov[ACL]["pr_missing"])
ROWS3 = [
    (f"security.py",
     f"{n_sec_b} uncovered ({cov[SEC]['main_pct']:.1f}%)  ->  {n_sec_a} ({cov[SEC]['pr_pct']:.1f}%)",
     "L880 now covered.  L732-733 remain: _coerce_int's except is unreachable -- the guards\n"
     "above it reject every value int() could raise on, and int(10**400) is the identity."),
    (f"_acl_config.py",
     f"{n_acl_b} uncovered ({cov[ACL]['main_pct']:.1f}%)  ->  {n_acl_a} ({cov[ACL]['pr_pct']:.1f}%)",
     "L157-159 now covered.  L604/607 remain: type-narrowing returns on a thread-local whose\n"
     "only writer stores the mypy-pinned (dict|None, str|None) shape."),
]
TOP3, LAST3 = 0.815, 0.545
STEP3 = TOP3 - LAST3
for i, (a, b, c) in enumerate(ROWS3):
    y = TOP3 - i * STEP3
    put(ax3, 0.0, y, a, fontsize=10.6, fontweight="bold", color=INK,
        transform=ax3.transAxes, va="top", family="monospace")
    put(ax3, 0.145, y, b, fontsize=10.6, color=GREEN, fontweight="bold",
        transform=ax3.transAxes, va="top", family="monospace")
    put(ax3, 0.40, y, c, fontsize=9.6, color=GREY, transform=ax3.transAxes, va="top")
ax3.plot([0, 1], [0.435, 0.435], color=GREY, lw=0.9, transform=ax3.transAxes)
put(ax3, 0.0, 0.375, "Three assertions could not tell the two layers apart, and are now exact:",
    fontsize=11.0, fontweight="bold", color=INK, transform=ax3.transAxes, va="top")
FIXES = [
    "test_rejects_crlf_in_policy_host  --  accepted either layer's message, so it passed on the membership gate alone.",
    "test_acl_load_refuses_symlink  --  matched a lowercase \"symlink\" that pytest derives into tmp_path from the test's own",
    "        name, so it passed for any ValueError naming the path, including the ELOOP one (\"...symbolic links\").",
    "test_validate_command_finite_numerics  --  its module docstring credited _coerce_int's int(...) wrap for refusals the",
    "        explicit math.isfinite and range guards actually make.",
]
TOPF, STEPF = 0.290, 0.062
for i, s in enumerate(FIXES):
    y = TOPF - i * STEPF
    put(ax3, 0.008, y, s, fontsize=9.5, color=INK if not s.startswith(" ") else GREY,
        transform=ax3.transAxes, va="top", family="monospace")
assert TOPF - (len(FIXES) - 1) * STEPF > 0.015, TOPF - (len(FIXES) - 1) * STEPF

# ---------------- layout guards ----------------
for a, y, axes_coords in placed:
    if axes_coords:
        assert -0.03 <= y <= 1.07, f"axes-fraction y out of range: {y}"
    else:
        lo, hi = a.get_ylim()
        assert lo - 0.05 <= y <= hi + 0.07, f"data y {y} outside {(lo, hi)}"

fig.savefig(OUT, dpi=124, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)
im = np.asarray(Image.open(OUT).convert("RGB")).astype(int)
h, w, _ = im.shape
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band - 255).sum(axis=2) > 12).sum())
    assert n == 0, f"{name} border has {n} non-white px"
print(f"OK {OUT}  {w}x{h}  borders clean")
