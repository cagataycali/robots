import os; os.environ.setdefault("MUJOCO_GL","egl")
import mujoco, numpy as np
from PIL import Image, ImageDraw
from strands_robots import Robot
from strands_robots.drivers.go2 import GO2_JOINT_INDEX
# targetPos_3 of example/b2/low_level/b2_stand_example.py, by LegID slot (FR, FL, RR, RL)
SLOTS=(-0.5,1.36,-2.65, 0.5,1.36,-2.65, -0.5,1.36,-2.65, 0.5,1.36,-2.65)
r=Robot("b2")
names=r.robot_joint_names("b2")
m=r.mj_model; d=r.mj_data
frames=[]
for label,pose in (("keyed by name (Go2Driver)",{n:SLOTS[GO2_JOINT_INDEX[n]] for n in names}),
                   ("zipped in description order",dict(zip(names,SLOTS)))):
    for n,v in pose.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"b2/"+n)]]=v
    d.qpos[2]=0.8
    mujoco.mj_forward(m,d)
    ren=mujoco.Renderer(m,480,640)
    cam=mujoco.MjvCamera(); cam.lookat[:]=d.qpos[:3]; cam.distance=2.6; cam.azimuth=180; cam.elevation=-12
    ren.update_scene(d,camera=cam); img=Image.fromarray(ren.render())
    ImageDraw.Draw(img).text((12,12),label,fill=(255,255,255))
    frames.append(img); ren.close()
    print(label, {n:round(float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"b2/"+n)]]),2) for n in names if 'hip' in n})
out=Image.new("RGB",(1280,480)); out.paste(frames[0],(0,0)); out.paste(frames[1],(640,0)); out.save("/tmp/b2vis/b2_by_name_vs_zipped.png")
r.cleanup()
