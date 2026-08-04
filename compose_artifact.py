import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

A = pathlib.Path("/tmp/art")
before = json.loads((A / "before_facts.json").read_text())
after = json.loads((A / "after_facts.json").read_text())
tb = json.loads(pathlib.Path("/tmp/before.json").read_text())
ta = json.loads(pathlib.Path("/tmp/after.json").read_text())

# --- self-audit the measured facts ---------------------------------------
assert before["measured"]["verdict"] == "success" and after["measured"]["verdict"] == "error"
assert before["measured"]["joint_state"] == [0.1, -1.3, 0.3], before["measured"]["joint_state"]
assert after["measured"]["joint_state"] == before["home"]
assert after["diff_home_vs_result"] == 0.0, "the refusal must leave the arm byte-identical to home"
assert before["diff_intended_vs_result"] > 0.10, before["diff_intended_vs_result"]
rows = list(zip(tb["rows"], ta["rows"], strict=True))
n_accepted_before = sum(1 for b, _ in rows if not b["case"].startswith("CONTROL") and b["main"]["verdict"] != "error")
n_accepted_after = sum(1 for _, a in rows if not a["case"].startswith("CONTROL") and a["main"]["verdict"] != "error")
n_controls_ok = sum(1 for b, a in rows if b["case"].startswith("CONTROL")
                    and b["main"]["verdict"] == a["main"]["verdict"] == "success")
assert (n_accepted_before, n_accepted_after, n_controls_ok) == (11, 0, 2), (n_accepted_before, n_accepted_after, n_controls_ok)

imgs = {
    "intended": np.asarray(iio.imread(A / "before_intended.png")),
    "before": np.asarray(iio.imread(A / "before_result.png")),
    "after": np.asarray(iio.imread(A / "after_result.png")),
}
# the reference panel is produced by both trees; they must agree
ref_after = np.asarray(iio.imread(A / "after_intended.png"))
ref_delta = int(np.abs(imgs["intended"].astype(int) - ref_after.astype(int)).max())
assert ref_delta <= 2, f"the reference render differs across trees by {ref_delta}"

placed: list[float] = []
def put(ax, x, y, s, **kw):
    placed.append(y)
    return ax.text(x, y, s, transform=ax.transAxes, **kw)

fig = plt.figure(figsize=(15.6, 11.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0], hspace=0.30, wspace=0.06,
                      left=0.035, right=0.965, top=0.885, bottom=0.045)

fig.suptitle(
    "Isaac set_joint_positions({'shouldre': 1.15, 'elbow': -1.30}) - 'shouldre' is a typo for 'shoulder'",
    fontsize=14.5, fontweight="bold", y=0.975)
fig.text(0.5, 0.938,
         "Isaac cannot render on this host, so the joint state each tree actually left in the articulation is "
         "replayed onto a MuJoCo arm declaring the same joint names.\nThe pictures are the measured writes.",
         ha="center", fontsize=10.2, style="italic", color="#333333")

PANELS = [
    ("intended", "What the call asks for",
     f"shoulder={after['intended']['shoulder']}  elbow={after['intended']['elbow']}",
     "#1a1a1a", "#888888"),
    ("before", "main: status=\"success\"",
     f"shoulder={before['measured']['joint_state'][0]} (never written)  elbow={before['measured']['joint_state'][1]}\n"
     f"the typo was skipped silently - half the pose applied, reported as all of it\n"
     f"differs from the requested pose across {before['diff_intended_vs_result']:.1%} of the frame",
     "#a11212", "#a11212"),
    ("after", "this change: status=\"error\"",
     "\"unresolved 'positions' keys ['shouldre'] on robot 'arm'.\n Its joints are ['shoulder', 'elbow', 'wrist']\"\n"
     f"the write is refused, so the arm is byte-identical to before ({after['diff_home_vs_result']:.0%} of pixels differ)",
     "#12661f", "#12661f"),
]
for col, (key, title, caption, tc, ec) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(imgs[key])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(ec); sp.set_linewidth(2.4)
    ax.set_title(title, fontsize=12.6, fontweight="bold", color=tc, pad=7)
    ax.set_xlabel(caption, fontsize=9.3, color=tc, linespacing=1.5)

# ---- verdict table ------------------------------------------------------
ax = fig.add_subplot(gs[1, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
put(ax, 0.0, 0.965, "Every value an Isaac joint-state write was handed, measured on both trees "
                    "(same probe script, each run printing the tree it resolved)",
    fontsize=11.6, fontweight="bold")
COLS = [(0.005, "positions=", "left"), (0.415, "main", "left"), (0.555, "what main left", "left"),
        (0.775, "this change", "left")]
for x, h, ha in COLS:
    put(ax, x, 0.885, h, fontsize=10.3, fontweight="bold", ha=ha, color="#333333")
ax.plot([0, 1], [0.862, 0.862], color="#999999", lw=1.0, transform=ax.transAxes, clip_on=False)

def outcome(rec, home):
    if rec["verdict"] == "raised":
        return "raised past the envelope", "#a11212"
    if rec["verdict"] == "error":
        return "refused", "#12661f"
    if rec["nonfinite_written"]:
        return f"non-finite joint state {rec['joint_state']}", "#a11212"
    if rec["width"] != len(home):
        return f"articulation resized to width {rec['width']}", "#a11212"
    if not rec["changed"]:
        return "nothing written", "#a11212"
    return f"{rec['joint_state']}", "#a11212"

y = 0.828
for b, a in rows:
    case = b["case"]
    ctrl = case.startswith("CONTROL")
    label = case.replace("CONTROL ", "")
    bm, am = b["main"], a["main"]
    if ctrl:
        ax.plot([0, 1], [y + 0.036, y + 0.036], color="#999999", lw=1.0, ls=":",
                transform=ax.transAxes, clip_on=False)
    btxt, bcol = outcome(bm, b["main"]["joint_state"] if ctrl else tb["home"])
    if ctrl:
        bcol = "#12661f"; btxt = f"applied {bm['joint_state']}"
    put(ax, 0.005, y, label + ("   (a usable call)" if ctrl else ""), fontsize=9.5,
        family="monospace", color="#1a1a1a" if not ctrl else "#12661f")
    bv = "success" if bm["verdict"] == "success" else bm["verdict"]
    put(ax, 0.415, y, bv, fontsize=9.5, family="monospace",
        color="#12661f" if ctrl else "#a11212", fontweight="bold")
    put(ax, 0.555, y, btxt, fontsize=9.2, family="monospace", color=bcol)
    av = am["verdict"]
    put(ax, 0.775, y, "refused" if av == "error" else f"{av}, applied {am['joint_state']}",
        fontsize=9.5, family="monospace", color="#12661f", fontweight="bold")
    # the worker-thread amplification, where it applies
    if b["queued"]["pump_swallowed"]:
        y -= 0.040
        put(ax, 0.030, y, "\u21b3 from a worker thread: status=\"success\", and the pump swallowed "
                          f"{b['queued']['pump_swallowed'].split(':')[0]} - the failure had no channel",
            fontsize=8.7, style="italic", color="#a11212")
    y -= 0.052

put(ax, 0.005, y - 0.008,
    f"{n_accepted_before} of {n_accepted_before} unusable values accepted on main \u2192 {n_accepted_before} refused; "
    f"both usable calls unchanged. The value domain now lives on SimEngine, shared with the MuJoCo writers.",
    fontsize=10.4, fontweight="bold", color="#1a1a1a")

assert all(-0.02 <= t <= 1.0 for t in placed), [t for t in placed if not -0.02 <= t <= 1.0]
out = A / "isaac_joint_state_domain.png"
fig.savefig(out, dpi=115, bbox_inches="tight", pad_inches=0.30, facecolor="white")
plt.close(fig)

im = np.asarray(iio.imread(out))
b = 8
border = np.concatenate([im[:b].reshape(-1, im.shape[2]), im[-b:].reshape(-1, im.shape[2]),
                         im[:, :b].reshape(-1, im.shape[2]), im[:, -b:].reshape(-1, im.shape[2])])
nonwhite = int((border[:, :3] < 250).any(1).sum())
print("saved", out, im.shape, "non-white border px:", nonwhite, "size KB:", out.stat().st_size // 1024)
assert nonwhite == 0, nonwhite
