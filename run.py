import math, sys, time, types
import mujoco, numpy as np
from PIL import Image
M='/home/cagatay/.cache/robot_descriptions/mujoco_menagerie/ufactory_xarm7/scene.xml'
m=mujoco.MjModel.from_xml_path(M); d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0)
frames=[]; qs=[]; r=mujoco.Renderer(m,480,640)
cam=mujoco.MjvCamera(); cam.lookat[:]=[0.25,0,0.35]; cam.distance=1.6; cam.azimuth=135; cam.elevation=-20
class SimArm:
    """XArmAPI-shaped double whose servo writes drive the MuJoCo xArm 7."""
    def __init__(self, port, is_radian=False):
        self.connected=True; self.error_code=0; self.warn_code=0; self.state=2; self.mode=0
        self.joint_speed_limit=[0.0001,3.14]; self.writes=0
    def motion_enable(self, enable=True): return 0
    def set_mode(self, mode=0): self.mode=mode; return 0
    def set_state(self, state=0): self.state=4 if state==4 else 1; return 0
    def get_servo_angle(self, is_radian=None): return 0,[float(d.qpos[i]) for i in range(7)]
    def get_joint_states(self, is_radian=None): return 0,[[float(d.qpos[i]) for i in range(7)],[float(d.qvel[i]) for i in range(7)],[float(d.qfrc_actuator[i]) for i in range(7)]]
    def get_position(self, is_radian=None): return 0,[0.0]*6
    def set_servo_angle_j(self, angles, is_radian=None):
        d.ctrl[:7]=angles; self.writes+=1
        for _ in range(5): mujoco.mj_step(m,d)   # 100 Hz control on a 500 Hz model
        if self.writes%4==0:
            qs.append(d.qpos.copy())
        return 0
    def disconnect(self): self.connected=False
mod=types.ModuleType('xarm.wrapper'); mod.XArmAPI=SimArm; sys.modules['xarm.wrapper']=mod
from strands_robots.drivers.xarm import XArmDriver
drv=XArmDriver('xarm7',port='192.168.1.185'); print('connect', drv.connect_eagerly())
home=[float(d.qpos[i]) for i in range(7)]; t0=[0]
def policy(obs):
    t0[0]+=1; k=t0[0]/100.0
    return {'joint1': home[0]+0.9*math.sin(k*1.2), 'joint2': home[1]+0.25*math.sin(k*2.4), 'joint4': home[3]+0.3*math.sin(k*1.8)}
print(drv.run_policy(policy, n_steps=400, duration=60)['status'])
while drv.get_task_status()['content'][0]['json'].get('running'): time.sleep(0.05)
print('status', drv.get_task_status()['content'][0]['json'])
print('jump', drv.send_action({'joint1': 2.0})['content'][0]['text'][:160])
print('stop', drv.stop_task()['status'])
for q in qs:
    d.qpos[:]=q; mujoco.mj_forward(m,d); r.update_scene(d,cam); frames.append(Image.fromarray(r.render()))
frames[0].save('/tmp/xviz/xarm7_driver.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
frames[len(frames)//2].save('/tmp/xviz/xarm7_driver.png')
print('frames', len(frames))
