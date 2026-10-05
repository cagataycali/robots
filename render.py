"""Drive the MuJoCo Gen3 through KinovaDriver over a kinematic Kortex base double."""
import math, sys, threading, time, types
from types import SimpleNamespace
import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco, numpy as np
from PIL import Image, ImageDraw
from strands_robots.drivers.kinova import KinovaDriver, JOINT_NAMES

xml = os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie/kinova_gen3/scene.xml")
model = mujoco.MjModel.from_xml_path(xml); data = mujoco.MjData(model)
qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]
home = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
for a, v in zip(qadr, home): data.qpos[a] = v
mujoco.mj_forward(model, data)
lock = threading.Lock(); speeds = [0.0]*7; last = [time.monotonic()]; log = []
def integrate():
    now = time.monotonic(); dt = now - last[0]; last[0] = now
    for i, a in enumerate(qadr): data.qpos[a] += math.radians(speeds[i]) * dt
    log.append((now, [float(data.qpos[a]) for a in qadr], list(speeds)))
class T:
    def connect(self, ip, port): pass
    def disconnect(self): pass
class R:
    def __init__(self, t, e=None): pass
class S:
    def __init__(self, r): pass
    def CreateSession(self, info): pass
    def CloseSession(self): pass
class B:
    def __init__(self, r): pass
    def GetActuatorCount(self): return SimpleNamespace(count=7)
    def GetArmState(self): return SimpleNamespace(active_state=7)
    def SetServoingMode(self, m): pass
    def SendJointSpeedsCommand(self, js):
        with lock: integrate(); speeds[:] = [s.value for s in js.joint_speeds]
    def Stop(self):
        with lock: integrate(); speeds[:] = [0.0]*7; log.append((time.monotonic(), None, "STOP"))
class C:
    def __init__(self, r): pass
    def RefreshFeedback(self):
        with lock:
            integrate()
            acts = [SimpleNamespace(position=math.degrees(data.qpos[a]) % 360.0, velocity=s, torque=0.0) for a, s in zip(qadr, speeds)]
        return SimpleNamespace(base=SimpleNamespace(active_state=7, **{f"tool_pose_{k}": 0.0 for k in ("x","y","z","theta_x","theta_y","theta_z")}), actuators=acts)
pb2 = dict(ServoingModeInformation=lambda **k: SimpleNamespace(**k), JointSpeeds=lambda **k: SimpleNamespace(**k),
           JointSpeed=lambda **k: SimpleNamespace(**k), ArmState=SimpleNamespace(Name=lambda v: {7: "ARMSTATE_SERVOING_READY"}.get(v, str(v))),
           SINGLE_LEVEL_SERVOING=2, ARMSTATE_SERVOING_READY=7, ARMSTATE_IN_FAULT=4)
for name, members in {"kortex_api.TCPTransport": {"TCPTransport": T}, "kortex_api.RouterClient": {"RouterClient": R},
        "kortex_api.SessionManager": {"SessionManager": S}, "kortex_api.autogen.client_stubs.BaseClientRpc": {"BaseClient": B},
        "kortex_api.autogen.client_stubs.BaseCyclicClientRpc": {"BaseCyclicClient": C}, "kortex_api.autogen.messages.Base_pb2": pb2,
        "kortex_api.autogen.messages.Session_pb2": {"CreateSessionInfo": lambda **k: SimpleNamespace(**k)},
        "kortex_api.Exceptions.KException": {"KException": RuntimeError}}.items():
    m = types.ModuleType(name); m.__dict__.update(members); sys.modules[name] = m

d = KinovaDriver(port="192.168.1.10", control_frequency=40.0)
assert d.connect_eagerly() is None
start = d.get_observation(); t0 = [None]
def policy(obs):
    if t0[0] is None: t0[0] = time.monotonic()
    t = time.monotonic() - t0[0]
    return {"joint_1": start["joint_1"] + 0.6*math.sin(0.6*t), "joint_2": start["joint_2"] + 0.4*(1-math.cos(0.8*t)),
            "joint_4": start["joint_4"] + 0.5*(1-math.cos(0.8*t))}
res = d.run_policy(policy, n_steps=200)
while d.get_task_status()["content"][0]["json"].get("running"): time.sleep(0.05)
st = d.get_task_status()["content"][0]["json"]; print("rollout", {k: st.get(k) for k in ("steps", "exit_reason", "running")})
# refused jump
print("jump:", d.send_action({"joint_1": d.get_observation()["joint_1"] + 0.5})["content"][0])
# deadman: one write, then silence
d.send_action({"joint_2": d.get_observation()["joint_2"] + 0.02}); w = time.monotonic(); time.sleep(0.3)
stops = [e[0] for e in log if e[2] == "STOP" and e[0] > w]; print("deadman stop after %.0f ms" % ((stops[0]-w)*1000))
d.cleanup()
traj = [e for e in log if e[1] is not None]
print("samples", len(traj), "max |j1-j1_0| rad %.3f" % max(abs(e[1][0]-home[0]) for e in traj))
r = mujoco.Renderer(model, 480, 640); cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = 135, -20, 2.0, (0, 0, 0.5)
idx = np.linspace(0, len(traj)-1, 40).astype(int); frames = []
for k in idx:
    for a, v in zip(qadr, traj[k][1]): data.qpos[a] = v
    mujoco.mj_forward(model, data); r.update_scene(data, cam); im = Image.fromarray(r.render())
    ImageDraw.Draw(im).text((10, 10), "KinovaDriver.run_policy -> SendJointSpeedsCommand  t=%.2fs" % (traj[k][0]-traj[0][0]), fill=(255,255,255))
    frames.append(im)
frames[0].save("/tmp/kin/gen3_stream.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
strip = Image.new("RGB", (640*4, 480)); [strip.paste(frames[i].copy(), (640*j, 0)) for j, i in enumerate([0, 13, 26, 39])]
strip.resize((1280, 240)).save("/tmp/kin/gen3_strip.png"); print("ok")
