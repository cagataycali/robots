"""Drive the cyclonedds poll loop with a real MuJoCo arm behind it.

The bridge's ``_on_command`` calls ``robot.send_action(action)``, so a shim that
forwards to a live sim makes the loop really move the arm. One batch is fed:
a valid pose, a malformed sample, then the pose the operator meant to end at.
"""
import json, pathlib, sys, types
import numpy as np
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
from strands_robots import Simulation
from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TAG = pathlib.Path(TREE).name
facts = {"tree": TREE}
def save(): (OUT / f"facts-{TAG}.json").write_text(json.dumps(facts, indent=2))

class _JS:
    def __init__(self, name, position): self.name, self.position = name, position

class _Reader:
    def __init__(self, batch): self._q = list(batch)
    def take(self, N=10):
        out, self._q = self._q[:N], self._q[N:]
        return out

class _Stop:
    def __init__(self, n): self.n = n
    def is_set(self):
        if self.n <= 0: return True
        self.n -= 1; return False
    def wait(self, t): return False

class _SimRobot:
    """Duck-typed Robot: the bridge's send_action drives the live sim."""
    def __init__(self, sim, name):
        self.sim, self._name = sim, name
        self.tool_name_str = name
        self.robot = types.SimpleNamespace(name=name)
        self.applied = []
        self.results = []
    def send_action(self, action):
        self.applied.append(dict(action))
        out = self.sim.send_action(action, robot_name=self._name, n_substeps=10)
        self.results.append(out)
        return out

def png(sim, cam):
    r = sim.render(camera_name=cam, width=760, height=680)
    assert r.get("status") == "success", r
    return next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)

def build():
    sim = Simulation(backend="mujoco", mesh=False)
    assert sim.create_world()["status"] == "success"
    assert sim.add_robot(name="so101")["status"] == "success"
    assert sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0, 0, 0.16], fov=42)["status"] == "success"
    return sim

keys = None
def poses(sim):
    """Half-way and final commanded poses, both inside every ctrlrange."""
    global keys
    keys = sim.robot_action_keys("so101")
    m = sim._world._model
    import mujoco as mj
    half, full = {}, {}
    for k in keys:
        # so101 declares no ctrlrange (all zeros), so the reachable span comes
        # from the JOINT each actuator drives.
        aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"so101/{k}")
        jid = int(m.actuator_trnid[aid, 0])
        lo, hi = (float(x) for x in m.jnt_range[jid])
        assert bool(m.jnt_limited[jid]), f"joint for {k} is unlimited"
        span = hi - lo
        full[k] = lo + 0.75 * span
        half[k] = lo + 0.35 * span
    return half, full

def run(label, batch, iterations=1):
    sim = build()
    half, full = poses(sim)
    robot = _SimRobot(sim, "so101")
    bridge = HardwareRtpsBridge.__new__(HardwareRtpsBridge)
    bridge._robot = robot
    bridge._command_reader = _Reader(batch(half, full))
    bridge._poll_period = 0.001
    bridge._joint_limits = None
    bridge._stop = _Stop(iterations)
    bridge._poll_loop()
    for _ in range(60):
        sim.step(10)
    for r in robot.results:
        assert r.get("status") == "success", r
    img = png(sim, "look")
    obs = sim.get_observation(robot_name="so101")
    joints = {k: round(float(obs[k]), 4) for k in keys if k in obs and not hasattr(obs[k], "shape")}
    sim.cleanup()
    return img, {"applied": len(robot.applied), "commands": robot.applied, "joints": joints}

# The operator's intent: end at `full`. A malformed sample sits between.
def batch_with_malformed(half, full):
    return [
        _JS(list(half), [half[k] for k in half]),
        _JS(["1"], ["not-a-number"]),              # a malformed command
        _JS(list(full), [full[k] for k in full]),  # the pose meant to be final
    ]
def batch_clean(half, full):
    return [
        _JS(list(half), [half[k] for k in half]),
        _JS(list(full), [full[k] for k in full]),
    ]

img_ref, ref = run("intended", batch_clean)
(OUT / f"intended-{TAG}.png").write_bytes(img_ref)
facts["intended"] = ref; save()
print("intended :", ref["applied"], "commands ->", ref["joints"])

img_mal, mal = run("with_malformed", batch_with_malformed)
(OUT / f"malformed-{TAG}.png").write_bytes(img_mal)
facts["with_malformed"] = mal; save()
print("malformed:", mal["applied"], "commands ->", mal["joints"])

# The intended run must actually move the arm, or neither panel means anything.
moved = max(abs(v) for v in ref["joints"].values())
facts["intended_max_joint_rad"] = moved; save()
assert moved > 0.5, f"the intended pose barely moved the arm ({moved})"

a = np.frombuffer(img_ref, np.uint8); b = np.frombuffer(img_mal, np.uint8)
facts["png_identical"] = bool(len(a) == len(b) and (a == b).all()); save()
# The joint-space verdict is what the picture illustrates.
facts["joints_match_intended"] = ref["joints"] == mal["joints"]; save()
print("intended max joint (rad):", round(moved, 4))
print("with-malformed reaches the intended pose:", facts["joints_match_intended"])
print("intended vs with-malformed PNG identical:", facts["png_identical"])
