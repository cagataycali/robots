import os, sys, types
os.environ["MUJOCO_GL"]="egl"
import mujoco, numpy as np
from PIL import Image, ImageDraw
from robot_descriptions import h1_mj_description as d
# stub sdk to build a real frame through the driver
class M:
    def __init__(s): s.mode=0;s.q=0.0;s.dq=0.0;s.tau=0.0;s.kp=0.0;s.kd=0.0
class L:
    def __init__(s): s.head=[0,0];s.level_flag=0;s.gpio=0;s.motor_cmd=[M() for _ in range(20)];s.crc=0
class C:
    def Crc(s,c): return 1
for n in ["unitree_sdk2py","unitree_sdk2py.idl","unitree_sdk2py.idl.default","unitree_sdk2py.utils","unitree_sdk2py.utils.crc"]:
    sys.modules[n]=types.ModuleType(n)
sys.modules["unitree_sdk2py.idl.default"].unitree_go_msg_dds__LowCmd_=L
sys.modules["unitree_sdk2py.utils.crc"].CRC=C
from strands_robots.drivers.go2 import WIRE_PROFILES, H1_JOINT_INDEX, build_lowcmd_from_action
prof=WIRE_PROFILES["unitree_h1"]
action={"left_shoulder_pitch":-1.4,"left_shoulder_roll":0.4,"left_elbow":0.2,"left_hip_pitch":-0.6,"left_knee":1.0}
cmd,err=build_lowcmd_from_action(action,prof); assert err is None
m=mujoco.MjModel.from_xml_path(d.MJCF_PATH.replace("h1.xml","scene.xml")); data=mujoco.MjData(m)
names=[m.joint(i).name for i in range(1,m.njnt)]  # description order, skip free joint
def pose(mapping):
    mujoco.mj_resetData(m,data)
    for n,q in mapping.items():
        data.qpos[m.jnt_qposadr[m.joint(n).id]]=q
    mujoco.mj_forward(m,data)
# a) by name through H1_JOINT_INDEX
by_name={n:cmd.motor_cmd[s].q for n,s in H1_JOINT_INDEX.items()}
# b) zip wire slots onto description order (slot 9 skipped)
slots=[s for s in range(20) if s!=9]
zipped={names[i]:cmd.motor_cmd[s].q for i,s in enumerate(slots)}
r=mujoco.Renderer(m,480,400)
cam=mujoco.MjvCamera(); cam.lookat[:]=[0,0,1.0]; cam.distance=3.2; cam.azimuth=160; cam.elevation=-5
imgs=[]
for title,mp in [("by name (H1_JOINT_INDEX)",by_name),("zipped onto model order",zipped)]:
    pose(mp); r.update_scene(data,cam); im=Image.fromarray(r.render()); ImageDraw.Draw(im).text((10,10),title,fill=(255,255,255)); imgs.append(im)
    print(title, {k:round(v,2) for k,v in mp.items() if abs(v)>1e-9})
out=Image.new("RGB",(960,400)); out.paste(imgs[0],(0,0)); out.paste(imgs[1],(480,0)); out.save("/tmp/h1_wire_order.png")
print("saved")
