import mujoco as mj, numpy as np
from PIL import Image, ImageDraw
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
s=MuJoCoSimEngine(); s.create_world()
print(s.add_object("asked", shape="ellipsoid", size=[0.05,0.10,0.20], position=[-0.12,0,0.1], color=[0.2,0.6,1], is_static=True)["status"])
print(s.add_object("compiled", shape="sphere", size=[0.05], position=[0.12,0,0.025], color=[1,0.3,0.2], is_static=True)["status"])
r=s.add_object("ball", shape="sphere", size=[0.05,0.10,0.20], position=[0,0.3,0.2]); print(r)
m=s._world._model; d=s._world._data; mj.mj_forward(m,d)
ren=mj.Renderer(m,480,640); cam=mj.MjvCamera(); cam.lookat[:]=[0,0,0.08]; cam.distance=0.7; cam.azimuth=90; cam.elevation=-15
ren.update_scene(d,cam); img=Image.fromarray(ren.render()); dr=ImageDraw.Draw(img)
dr.text((150,40),"asked: size=[0.05, 0.10, 0.20]",fill=(255,255,255)); dr.text((380,40),"used to compile: 5 cm ball",fill=(255,255,255))
dr.text((20,450),"now: sphere with size=[0.05, 0.10, 0.20] -> status=error, points at shape='ellipsoid'",fill=(255,255,0))
img.save("/tmp/add_object_size_drop.png"); print(img.size)
