import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A = "/tmp/relnotes/assets"
def font(sz, bold=False):
    p = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try: return ImageFont.truetype(p, sz)
    except Exception: return ImageFont.load_default()
def mono(sz, bold=False):
    p = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    try: return ImageFont.truetype(p, sz)
    except Exception: return ImageFont.load_default()

RED, GREEN, INK, GREY = (185, 28, 28), (5, 122, 61), (17, 24, 39), (107, 114, 128)
PAD = 12

def cell(img, title, tcol, lines, capw=None):
    w = capw or img.shape[1]
    caph = 26 + 20 * len(lines)
    im = Image.new("RGB", (w, img.shape[0] + caph), "white")
    im.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(im)
    y0 = img.shape[0]
    d.rectangle([0, y0, w, y0 + caph], fill=(247, 248, 250))
    d.line([0, y0, w, y0], fill=(203, 207, 213))
    d.text((10, y0 + 5), title, font=font(17, True), fill=tcol)
    for i, (t, c) in enumerate(lines):
        d.text((10, y0 + 27 + 20 * i), t, font=mono(13), fill=c)
    return im

def hstack(ims, pad=PAD):
    w = sum(i.width for i in ims) + pad * (len(ims) - 1)
    h = max(i.height for i in ims)
    out = Image.new("RGB", (w, h), "white"); x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + pad
    return out
def vstack(ims, pad=PAD):
    h = sum(i.height for i in ims) + pad * (len(ims) - 1)
    out = Image.new("RGB", (max(i.width for i in ims), h), "white"); y = 0
    for i in ims:
        out.paste(i, (0, y)); y += i.height + pad
    return out

def titled(body, title, sub, tsz=26):
    TOP = 66
    fig = Image.new("RGB", (body.width + 2 * PAD, TOP + body.height + PAD), "white")
    d = ImageDraw.Draw(fig)
    d.text((PAD + 2, 14), title, font=font(tsz, True), fill=INK)
    d.text((PAD + 2, 45), sub, font=font(13), fill=GREY)
    fig.paste(body, (PAD, TOP))
    return fig

def audit(fig, name, checks):
    a = np.asarray(fig)
    b = np.concatenate([a[:6].reshape(-1,3), a[-6:].reshape(-1,3),
                        a[:,:6].reshape(-1,3), a[:,-6:].reshape(-1,3)])
    nw = int((np.abs(b.astype(int)-255).sum(1) > 12).sum())
    print(f"{name}: size={fig.size} border_nonwhite={nw}")
    assert nw == 0, nw
    for label, ok, val in checks:
        print(f"  {label}: {val}")
        assert ok, (label, val)
    print(f"  {name} AUDIT OK")

# ============================ FIGURE 1: gripper (#1652) ======================
gb = np.load(f"{A}/gripper_before.npz"); ga = np.load(f"{A}/gripper_after.npz")
fb = {f["action_key"]: f for f in json.load(open(f"{A}/gripper_before.json"))}
fa = {f["action_key"]: f for f in json.load(open(f"{A}/gripper_after.json"))}

def gcell(fr, facts, key, tree):
    f = facts[key]
    g = f["gap_after_m"]
    openish = g > 0.03
    col = GREEN if openish else RED
    verdict = "fingers OPEN" if openish else "fingers SHUT"
    return cell(fr[key], f'send_action({{"{key}": 1.0}})', col, [
        (f'status  = "{f["status"]}"', INK),
        (f"gap     = {g*1000:8.3f} mm   {verdict}", col),
        (f"{'  <- what robot_action_keys() returns' if key.startswith('actuator') else '  (joint spelling; no policy sees this)'}",
         GREY),
    ])

row_b = hstack([gcell(gb, fb, "actuator8", "before"), gcell(gb, fb, "finger_joint1", "before")])
row_a = hstack([gcell(ga, fa, "actuator8", "after"), gcell(ga, fa, "finger_joint1", "after")])

def band(text, col, w):
    im = Image.new("RGB", (w, 34), "white"); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 33], fill=(254, 242, 242) if col == RED else (240, 253, 244))
    d.text((10, 7), text, font=font(17, True), fill=col)
    return im

body1 = vstack([
    band("BEFORE  -  strands-robots at #1652^   (the same command, two spellings, 255x apart)", RED, row_b.width),
    row_b,
    band("AFTER   -  one shared write path: both spellings honor the command identically", GREEN, row_a.width),
    row_a,
])
fig1 = titled(body1,
    "Fixed: no policy could open a tendon-driven gripper  (#1652)",
    "MuJoCo headless on Jetson AGX Thor - franka panda, actuator8 is a tendon drive with ctrlrange [0, 255]. "
    "1.0 is the normalized full-scale 'open' a policy emits.")
fig1.save(f"{A}/fix_gripper.png")
audit(fig1, "fix_gripper", [
    ("pre-fix actuator spelling shut", fb["actuator8"]["gap_after_m"] < 0.001, fb["actuator8"]["gap_after_m"]),
    ("pre-fix joint spelling open", fb["finger_joint1"]["gap_after_m"] > 0.07, fb["finger_joint1"]["gap_after_m"]),
    ("post-fix both agree", abs(fa["actuator8"]["gap_after_m"] - fa["finger_joint1"]["gap_after_m"]) < 1e-6,
     (fa["actuator8"]["gap_after_m"], fa["finger_joint1"]["gap_after_m"])),
    ("pre-fix panels visually differ (the jaw is in frame)",
     float((np.abs(gb["actuator8"].astype(int)-gb["finger_joint1"].astype(int)).sum(2) > 30).mean()) > 0.02,
     round(float((np.abs(gb["actuator8"].astype(int)-gb["finger_joint1"].astype(int)).sum(2) > 30).mean()), 4)),
    ("post-fix panels agree to renderer noise (that IS the fix)",
     float((np.abs(ga["actuator8"].astype(int)-ga["finger_joint1"].astype(int)).sum(2) > 30).mean()) == 0.0
     and int(np.abs(ga["actuator8"].astype(int)-ga["finger_joint1"].astype(int)).max()) <= 2,
     (round(float((np.abs(ga["actuator8"].astype(int)-ga["finger_joint1"].astype(int)).sum(2) > 30).mean()), 6),
      int(np.abs(ga["actuator8"].astype(int)-ga["finger_joint1"].astype(int)).max()))),
])

# ============================ FIGURE 2: scene rewind (#1763) ==================
rb = np.load(f"{A}/rewind_before.npz"); ra = np.load(f"{A}/rewind_after.npz")
Fb = json.load(open(f"{A}/rewind_before.json")); Fa = json.load(open(f"{A}/rewind_after.json"))

def rcell(fr, F, phase, tree):
    p = F[phase]
    j1 = p["joints"]["joint1"]; j4 = p["joints"]["joint4"]
    if phase == "before":
        title, col = "scene as built (arm parked, crate settled)", INK
    else:
        rewound = abs(j1) < 1e-3
        title = "after add_robot(name='b')"
        col = RED if rewound else GREEN
    lines = [
        (f'joint1 = {j1:+.4f} rad    joint4 = {j4:+.4f} rad', col if phase == "after" else INK),
        (f'crate  z = {p["crate_z"]:.4f} m', col if phase == "after" else INK),
    ]
    if phase == "after":
        rewound = abs(j1) < 1e-3
        lines.append(("  ALL joints -> 0, crate back at spawn" if rewound
                      else "  pose and contact state preserved", col))
    else:
        lines.append(("  crate resting on the pedestal", GREY))
    return cell(fr[phase], title, col, lines)

row_pre = hstack([rcell(rb, Fb, "before", "pre"), rcell(rb, Fb, "after", "pre")])
row_post = hstack([rcell(ra, Fa, "before", "post"), rcell(ra, Fa, "after", "post")])
body2 = vstack([
    band(f"BEFORE  -  at #1763^   add_robot returned \"{Fb['add_robot_status']}\" and rewound the whole world", RED, row_pre.width),
    row_pre,
    band(f"AFTER   -  add_robot returned \"{Fa['add_robot_status']}\" and left the scene exactly as it was", GREEN, row_post.width),
    row_post,
])
fig2 = titled(body2,
    "Fixed: adding a robot no longer rewinds the scene it joins  (#1763)",
    "MuJoCo headless on Jetson AGX Thor - a parked panda, a 0.5 kg crate settled on a pedestal, "
    "then one ordinary incremental edit: add a second robot.")
fig2.save(f"{A}/fix_rewind.png")
audit(fig2, "fix_rewind", [
    ("pre-fix arm rewound to zero", abs(Fb["after"]["joints"]["joint1"]) < 1e-3, Fb["after"]["joints"]["joint1"]),
    ("pre-fix crate teleported to spawn", Fb["after"]["crate_z"] > 0.6, Fb["after"]["crate_z"]),
    ("post-fix pose preserved",
     abs(Fa["after"]["joints"]["joint1"] - Fa["before"]["joints"]["joint1"]) < 1e-6,
     (Fa["before"]["joints"]["joint1"], Fa["after"]["joints"]["joint1"])),
    ("post-fix crate preserved", abs(Fa["after"]["crate_z"] - Fa["before"]["crate_z"]) < 1e-6,
     (Fa["before"]["crate_z"], Fa["after"]["crate_z"])),
    ("the two 'before' renders match across trees (same rig)",
     int(np.abs(rb["before"].astype(int)-ra["before"].astype(int)).max()) <= 2,
     int(np.abs(rb["before"].astype(int)-ra["before"].astype(int)).max())),
    ("pre-fix edit visibly changed the scene",
     float((np.abs(rb["before"].astype(int)-rb["after"].astype(int)).sum(2) > 30).mean()) > 0.05,
     round(float((np.abs(rb["before"].astype(int)-rb["after"].astype(int)).sum(2) > 30).mean()), 4)),
])
print("\nDONE")
