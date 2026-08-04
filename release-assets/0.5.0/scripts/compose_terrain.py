import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A = "/tmp/relnotes/assets"
fr = np.load(f"{A}/terrain_frames.npz")
facts = {(f["terrain"], f["difficulty"]): f for f in json.load(open(f"{A}/terrain_facts.json"))}

def font(sz, bold=False):
    p = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try: return ImageFont.truetype(p, sz)
    except Exception: return ImageFont.load_default()

F_T, F_B, F_M = font(19, True), font(14), font(13)
W, H = 560, 420
CAP = 52
PAD = 10

def panel(key, title, sub):
    im = Image.new("RGB", (W, H + CAP), "white")
    im.paste(Image.fromarray(fr[key]), (0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, H, W, H + CAP], fill=(246, 247, 249))
    d.line([0, H, W, H], fill=(200, 204, 210))
    d.text((10, H + 6), title, font=F_T, fill=(17, 24, 39))
    d.text((10, H + 30), sub, font=F_M, fill=(75, 85, 99))
    return im

# Row 1: the four terrain kinds.
row1 = []
for k, label in (("rough", "rough"), ("stairs", "stairs"), ("pyramid", "pyramid"), ("slope", "slope")):
    f = facts[(k, 1.0)]
    row1.append(panel(k, f'terrain="{label}"',
                      f'ground at origin {f["ground_z_at_origin"]:.3f} m  |  go2 base {f["base_z"]:.4f} m'))

# Row 2: the difficulty curriculum on stairs (3 panels) + a facts card.
row2 = []
for d_ in (1.0, 2.5, 4.0):
    f = facts[("stairs", d_)]
    clear = f["base_z"] - f["ground_z_at_origin"]
    row2.append(panel(f"stairs_d{d_}", f'difficulty={d_}',
                      f'ground {f["ground_z_at_origin"]:.2f} m  |  base {f["base_z"]:.4f} m  |  clearance {clear:.4f} m'))

card = Image.new("RGB", (W, H + CAP), "white")
dc = ImageDraw.Draw(card)
dc.rectangle([0, 0, W - 1, H + CAP - 1], outline=(200, 204, 210), width=1)
dc.text((16, 14), "Measured on this run", font=font(18, True), fill=(17, 24, 39))
lines = [
    ("create_world(terrain=..., difficulty=...)", (17, 24, 39), font(13, True)),
    ("", None, None),
    ("A floating base is seated on the LOCAL terrain", (55, 65, 81), F_B),
    ("surface, not at an absolute z:", (55, 65, 81), F_B),
    ("", None, None),
    ("  difficulty   ground_z    base_z    clearance", (17, 24, 39), font(12, True)),
]
for d_ in (1.0, 2.5, 4.0):
    f = facts[("stairs", d_)]
    c = f["base_z"] - f["ground_z_at_origin"]
    lines.append((f"     {d_:<9}  {f['ground_z_at_origin']:.2f} m     {f['base_z']:.4f}    {c:.4f} m",
                  (55, 65, 81), font(12)))
lines += [
    ("", None, None),
    ("Standing clearance is constant to 4 dp while the", (5, 105, 60), F_B),
    ("ground under it quadruples. Absolute z would have", (5, 105, 60), F_B),
    ("buried or floated the robot at every level.", (5, 105, 60), F_B),
    ("", None, None),
    ("get_ground_height(x, y) reports the same surface", (75, 85, 99), F_M),
    ("the seating used - so a policy can query it too.", (75, 85, 99), F_M),
]
y = 48
for txt, col, fo in lines:
    if txt: dc.text((16, y), txt, font=fo, fill=col)
    y += 21 if txt else 9

def hstack(ims):
    w = sum(i.width for i in ims) + PAD * (len(ims) - 1)
    out = Image.new("RGB", (w, ims[0].height), "white")
    x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + PAD
    return out

r1, r2 = hstack(row1), hstack(row2 + [card])
TOP = 64
fig = Image.new("RGB", (max(r1.width, r2.width) + 2 * PAD, TOP + r1.height + PAD + r2.height + PAD), "white")
d = ImageDraw.Draw(fig)
d.text((PAD + 2, 14), "Locomotion ground: 4 terrain kinds + a difficulty curriculum",
       font=font(26, True), fill=(17, 24, 39))
d.text((PAD + 2, 44), "MuJoCo headless (MUJOCO_GL=egl) on NVIDIA Jetson AGX Thor - unitree go2, 12 actuators, settled 120 steps",
       font=font(13), fill=(107, 114, 128))
fig.paste(r1, (PAD, TOP))
fig.paste(r2, (PAD, TOP + r1.height + PAD))
fig.save(f"{A}/feat_terrain.png")

# ---- self-audit -------------------------------------------------------------
a = np.asarray(fig)
border = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3),
                         a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
nonwhite = int((np.abs(border.astype(int) - 255).sum(1) > 12).sum())
# every kind must render distinct ground
keys = ["rough", "stairs", "pyramid", "slope"]
pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
diffs = [float((np.abs(fr[keys[i]].astype(int) - fr[keys[j]].astype(int)).sum(2) > 30).mean()) for i, j in pairs]
# clearance invariance
cl = [round(facts[("stairs", d_)]["base_z"] - facts[("stairs", d_)]["ground_z_at_origin"], 4) for d_ in (1.0, 2.5, 4.0)]
print("size", fig.size, "| border nonwhite", nonwhite)
print("pairwise terrain differing-pixel fraction:", [round(x, 4) for x in diffs])
print("clearances:", cl)
assert nonwhite == 0, nonwhite
assert min(diffs) > 0.02, diffs
assert len(set(cl)) == 1, cl
print("AUDIT OK")
