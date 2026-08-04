import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
A = "/tmp/relnotes/assets"
z = np.load(f"{A}/grounding.npz"); F = json.load(open(f"{A}/grounding.json"))
def font(s,b=False):
    p=("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    try: return ImageFont.truetype(p,s)
    except Exception: return ImageFont.load_default()
def mono(s,b=False):
    p=("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    try: return ImageFont.truetype(p,s)
    except Exception: return ImageFont.load_default()
INK,GREY,GREEN,BLUE=(17,24,39),(107,114,128),(5,122,61),(29,78,216)
PAD=12
u,v = F["pixel"]

# Panel 1: the frame the agent saw, with the pixel it named.
p1 = Image.fromarray(z["frame"]).copy(); d = ImageDraw.Draw(p1)
for r,w in ((26,3),(17,2)): d.ellipse([u-r,v-r,u+r,v+r], outline=(255,215,0), width=w)
d.line([u-40,v,u-8,v], fill=(255,215,0), width=3); d.line([u+8,v,u+40,v], fill=(255,215,0), width=3)
d.line([u,v-40,u,v-8], fill=(255,215,0), width=3); d.line([u,v+8,u,v+40], fill=(255,215,0), width=3)
lbl=f"pixel ({u}, {v})"
tw=d.textlength(lbl,font=mono(15,True))
d.rectangle([u+32,v-52,u+42+tw,v-26], fill=(0,0,0)); d.text((u+37,v-49), lbl, font=mono(15,True), fill=(255,215,0))

# Panel 2: where the arm went.
p2 = Image.fromarray(z["frame_at"])

def cap(im, title, rows):
    caph = 26+20*len(rows)
    out = Image.new("RGB",(im.width, im.height+caph),"white"); out.paste(im,(0,0))
    dd = ImageDraw.Draw(out); y0=im.height
    dd.rectangle([0,y0,out.width,y0+caph], fill=(247,248,250)); dd.line([0,y0,out.width,y0], fill=(203,207,213))
    dd.text((10,y0+5), title, font=font(17,True), fill=INK)
    for i,(t,c) in enumerate(rows): dd.text((10,y0+27+20*i), t, font=mono(13), fill=c)
    return out

c1 = cap(p1, "1.  the agent names a pixel", [
    ('sim.render(camera_name="eye")', INK),
    (f'  -> chose pixel ({u}, {v})', BLUE),
    ("  no joint targets, no IK written by hand", GREY)])
c2 = cap(p2, "3.  move_to() put the jaw there", [
    (f'reached={F["reached"]}  ik_residual={F["ik_residual_m"]*1000:.2f} mm', GREEN),
    (f'jaw over cube, xy error {F["pad_over_cube_xy_error_mm"]:.1f} mm', GREEN),
    ("  analytic mink IK, shipped as a primitive", GREY)])

wp, tr = F["world_point"], F["cube_truth"]
card = Image.new("RGB",(470, c1.height),"white"); dc=ImageDraw.Draw(card)
dc.rectangle([0,0,card.width-1,card.height-1], outline=(203,207,213), width=1)
dc.text((16,14), "2.  get_world_point(camera, pixels)", font=font(17,True), fill=INK)
rows=[
 ("", None),
 ('get_world_point(camera_name="eye",', INK),
 (f'                pixels=[[{u}, {v}]])', INK),
 ("", None),
 (f'  -> [{wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f}]  m', BLUE),
 ("", None),
 ("cube ground truth (get_body_state):", GREY),
 (f'     [{tr[0]:.4f}, {tr[1]:.4f}, {tr[2]:.4f}]  m', GREY),
 (f'  cube top face z = {tr[2]+0.015:.4f} m', GREY),
 ("", None),
 (f'xy grounding error : {F["grounding_xy_error_mm"]:.2f} mm', GREEN),
 (f'z  vs top face     : {abs(wp[2]-(tr[2]+0.015))*1000:.2f} mm', GREEN),
 ("", None),
 ("The pixel resolves to the point on the", GREY),
 ("visible surface - which is what an agent", GREY),
 ("looking at a frame actually means.", GREY),
]
y=44
for t,c in rows:
    if t: dc.text((16,y), t, font=mono(13) if t.startswith(("get_","  ->","     ","xy ","z  ","  cube","                ")) else font(13), fill=c)
    y += 20 if t else 9

body = Image.new("RGB",(c1.width+card.width+c2.width+2*PAD, c1.height),"white")
body.paste(c1,(0,0)); body.paste(card,(c1.width+PAD,0)); body.paste(c2,(c1.width+card.width+2*PAD,0))
TOPH=68
fig = Image.new("RGB",(body.width+2*PAD, TOPH+body.height+PAD),"white")
d = ImageDraw.Draw(fig)
d.text((PAD+2,14), "New: an agent can point at a pixel and the arm goes there  (#1649 + #1654)",
       font=font(25,True), fill=INK)
d.text((PAD+2,45), "get_world_point() grounds a pixel to a world coordinate; move_to() / set_gripper() / rotate_wrist() are "
                   "analytic primitives backed by shared mink IK. MuJoCo headless on Jetson AGX Thor.",
       font=font(13), fill=GREY)
fig.paste(body,(PAD,TOPH)); fig.save(f"{A}/feat_grounding.png")

a=np.asarray(fig)
bd=np.concatenate([a[:6].reshape(-1,3),a[-6:].reshape(-1,3),a[:,:6].reshape(-1,3),a[:,-6:].reshape(-1,3)])
nw=int((np.abs(bd.astype(int)-255).sum(1)>12).sum())
moved=float((np.abs(z["frame"].astype(int)-z["frame_at"].astype(int)).sum(2)>30).mean())
print("size",fig.size,"border_nonwhite",nw)
print("grounding xy err mm:",F["grounding_xy_error_mm"],"| z vs top face mm:",round(abs(wp[2]-(tr[2]+0.015))*1000,2))
print("arm visibly moved between panels:",round(moved,4))
assert nw==0, nw
assert F["grounding_xy_error_mm"] < 15, F
assert moved > 0.05, moved
assert F["reached"] is True
print("AUDIT OK")
