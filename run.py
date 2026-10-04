import math, sys, types
import numpy as np, mujoco, imageio
sys.path.insert(0, "/home/cagatay/actions-runner-harness-5/_work/_temp/robots-src/tests/drivers")
from strands_robots import Robot
from strands_robots.drivers.rby1 import JOINT_NAMES, WHEEL_JOINTS, RBY1Driver
import rby1_sdk as real  # real builders; only the robot link is a double

sim = Robot("rby1", position=[0.0, 0.0, 0.0026])
m, d = sim.mj_model, sim.mj_data
jid = {m.joint(i).name.split("/")[-1]: i for i in range(m.njnt)}
act_for = {}
for a in range(m.nu):
    j = m.actuator_trnid[a, 0]
    act_for[m.joint(j).name.split("/")[-1]] = a
MODEL = [*WHEEL_JOINTS, *JOINT_NAMES]

def qpos(name): return float(d.qpos[m.jnt_qposadr[jid[name]]])

class Robot_:
    def __init__(s): s.sent = 0
    def connect(s, max_retries=5, timeout_ms=1000): return True
    def is_connected(s): return True
    def power_on(s, n): return True
    def servo_on(s, n): return True
    def enable_control_manager(s, unlimited_mode_enabled=False): return True
    def disable_control_manager(s): return True
    def get_control_manager_state(s): return types.SimpleNamespace(state=types.SimpleNamespace(name="Enabled"))
    def get_state(s):
        return types.SimpleNamespace(position=[qpos(n) for n in MODEL], velocity=[0.0]*24, torque=[0.0]*24,
            emo_states=[types.SimpleNamespace(state=types.SimpleNamespace(name="Released"))],
            battery_state=types.SimpleNamespace(level_percent=100.0))
    @staticmethod
    def model(): return real.Model_A()
    def get_dynamics(s, urdf_model=""):
        import robot_descriptions, os, glob
        urdf = glob.glob(os.path.expanduser("~/.cache/robot_descriptions/rby1_description/models/rby1a/urdf/model.urdf"))[0]
        return real.dynamics.Robot(real.dynamics.load_robot_from_urdf(urdf, "base"))
    def create_command_stream(s, priority=1): return Stream()
    def cancel_control(s): return True
    def disconnect(s): pass

class Stream:
    def is_done(self): return False
    def cancel(self): pass
    def send_command(self, builder, timeout_ms=1000):
        for n, v in zip(JOINT_NAMES, LAST): d.ctrl[act_for[n]] = v
        return types.SimpleNamespace(status=types.SimpleNamespace(name="Running"), finish_code=types.SimpleNamespace(name="Unknown"))

LAST = []
real.create_robot_a = lambda addr: Robot_()
drv = RBY1Driver("rby1", port="sim:50051")
orig = drv._command
def spy(targets):
    LAST[:] = targets
    return orig(targets)
drv._command = spy
print("connect:", drv.connect_eagerly())
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.0, 0.0, 1.0]; cam.distance = 3.0; cam.azimuth = 150; cam.elevation = -12
start = {n: qpos(n) for n in JOINT_NAMES}
frames, refused, ok = [], 0, 0
steps_per = int(round(0.02 / m.opt.timestep))
for k in range(200):
    t = k * 0.02
    w = 0.5 * (1 - math.cos(min(t, 1.5) / 1.5 * math.pi))  # ease in
    act = {
        "right_arm_1": start["right_arm_1"] - 1.2 * w,
        "right_arm_3": start["right_arm_3"] - 1.4 * w + 0.35 * w * math.sin(2 * math.pi * 0.8 * t),
        "left_arm_3": start["left_arm_3"] - 0.6 * w,
        "head_0": start["head_0"] + 0.35 * w * math.sin(2 * math.pi * 0.4 * t),
        "torso_5": start["torso_5"] + 0.15 * w * math.sin(2 * math.pi * 0.3 * t),
    }
    env = drv.send_action(act)
    if env["status"] == "success": ok += 1
    else:
        refused += 1
        if refused < 3: print(env)
    for _ in range(steps_per): mujoco.mj_step(m, d)
    if k % 2 == 0:
        r.update_scene(d, cam); frames.append(r.render().copy())
print("ok", ok, "refused", refused, "right_arm_1", round(qpos("right_arm_1"),3), "head_0", round(qpos("head_0"),3))
bad = drv.send_action({"right_arm_1": qpos("right_arm_1") - 1.0})
print("jump:", bad["content"][0])
imageio.mimsave("/tmp/rbviz/rby1_driver.gif", frames, duration=0.04, loop=0)
imageio.imwrite("/tmp/rbviz/rby1_driver.png", frames[-1])
