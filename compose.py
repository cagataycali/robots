import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = pathlib.Path("/tmp/artdata-" + __import__("os").environ["GITHUB_RUN_ID"])
A = json.loads((D / "facts_main.json").read_text())
B = json.loads((D / "facts_branch.json").read_text())
assert A["tree"] != B["tree"], "both arms measured the same tree"

watch_a = np.load(D / "watch_before_main.npy"); watch_b = np.load(D / "watch_before_branch.npy")
plate_a = np.load(D / "plate_after_main.npy"); plate_b = np.load(D / "plate_after_branch.npy")
d_rig = int(np.abs(watch_a.astype(int) - watch_b.astype(int)).max())
d_plate = int(np.abs(plate_a.astype(int) - plate_b.astype(int)).max())
print("max|watch_before main-branch| =", d_rig, "  max|plate_after main-branch| =", d_plate)
assert d_rig <= 2, d_rig
assert d_plate <= 2, d_plate

# every claim re-derived from the two dumps
assert A["remove_object"]["status"] == B["remove_object"]["status"] == "success"
assert A["after"]["ncam"] == B["after"]["ncam"] == 3
assert len(A["after"]["cameras"]) == 4 and len(B["after"]["cameras"]) == 3
assert "watch" in A["after"]["cameras"] and "watch" not in B["after"]["cameras"]
msg_a, msg_b = A["after"]["watch_render_error"], B["after"]["watch_render_error"]
assert "watch" in msg_a.split("Available:")[1] and "watch" not in msg_b.split("Available:")[1]
for cam in ("plate_cam", "fixed"):
    assert A["after"]["verdicts"][cam]["resolvable"] and B["after"]["verdicts"][cam]["resolvable"], cam
    assert B["after"]["verdicts"][cam]["listed"], cam

placed: list[tuple] = []
def put(ax, x, y, s, **kw):
    axes_coords = kw.get("transform") is not None
    placed.append((ax, y, axes_coords))
    return ax.text(x, y, s, **kw)

fig = plt.figure(figsize=(15.4, 10.4), dpi=124)
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.20, wspace=0.06,
                      left=0.020, right=0.980, top=0.905, bottom=0.030)
fig.suptitle("remove_object drops the cameras mounted on the body it removes", fontsize=17, fontweight="bold", y=0.972)
fig.text(0.5, 0.935, "MuJoCo, headless (MUJOCO_GL=egl). add_camera(parent_body='crate') makes the camera a child of that body, "
         "so deleting the body deletes the camera at recompile.", ha="center", fontsize=10.4, style="italic")

for col, (img, title, sub) in enumerate((
    (watch_b, "1. through 'watch', mounted on the crate", f"the view the camera gave  (identical on both trees, max|delta| = {d_rig}/255)"),
    (plate_a, "2. 'plate_cam' after remove_object, main", "a camera on another body: unaffected"),
    (plate_b, "3. 'plate_cam' after remove_object, this PR", f"byte-identical to panel 2 (max|delta| = {d_plate}/255)"),
)):
    ax = fig.add_subplot(gs[0, col]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11.2, fontweight="bold", pad=5)
    ax.set_xlabel(sub, fontsize=9.4)
    for s in ax.spines.values():
        s.set_edgecolor("#2e7d32" if col else "#37474f"); s.set_linewidth(2.0)

axt = fig.add_subplot(gs[1, :]); axt.axis("off"); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
rows = []
for tag, F in (("main", A), ("this PR", B)):
    aft = F["after"]
    rows.append(("hdr", f"{tag}:  remove_object('crate') -> {F['remove_object']['status']}   \"{F['remove_object']['text']}\""))
    rows.append(("row", f"list_cameras() = {aft['cameras']}      compiled model ncam = {aft['ncam']}"))
    agree = len(aft["cameras"]) == aft["ncam"]
    rows.append(("ok" if agree else "bad",
                 f"registry entries {len(aft['cameras'])} vs model cameras {aft['ncam']}: "
                 + ("agree" if agree else "DISAGREE - one entry names a camera the model does not have")))
    rows.append(("ok" if "watch" not in aft["watch_render_error"].split("Available:")[1] else "bad",
                 f"render(camera_name='watch') -> {aft['watch_render_error']}"))
    if "watch" in aft["watch_render_error"].split("Available:")[1]:
        rows.append(("bad", "        ...the sentence offers 'watch' as an available alternative to itself"))
    rows.append(("row", "'plate_cam' (on another body) resolvable: "
                 f"{aft['verdicts']['plate_cam']['resolvable']}      'fixed' (world-fixed) resolvable: {aft['verdicts']['fixed']['resolvable']}"))
    rows.append(("gap", ""))
rows.pop()
TOP, FLOOR = 0.955, 0.045
STEP = (TOP - FLOOR) / len(rows)
assert STEP > 0.030, STEP
COL = {"hdr": "#0d47a1", "ok": "#1b5e20", "bad": "#b71c1c", "row": "#263238", "gap": "#263238"}
y = TOP
for kind, text in rows:
    if kind != "gap":
        put(axt, 0.012, y, text, fontsize=10.6 if kind == "hdr" else 10.0, family="monospace",
            color=COL[kind], fontweight="bold" if kind in ("hdr", "bad") else "normal",
            va="top", transform=axt.transAxes)
    y -= STEP
assert y > 0.010, y

for ax, yv, axes_coords in placed:
    lo, hi = (-0.03, 1.07) if axes_coords else ax.get_ylim()
    assert lo <= yv <= hi, (yv, lo, hi)

p = pathlib.Path("/tmp/artdata-" + __import__("os").environ["GITHUB_RUN_ID"]) / "remove_object_camera_cascade.png"
fig.savefig(p, bbox_inches="tight", pad_inches=0.30, facecolor="white")
im = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB"))
for name, band in (("top", im[:8]), ("bottom", im[-8:]), ("left", im[:, :8]), ("right", im[:, -8:])):
    n = int((np.abs(band.astype(int) - 255).sum(2) > 12).sum())
    print(f"  border {name}: {n} non-white px"); assert n == 0, name
print("SAVED", p, im.shape)
