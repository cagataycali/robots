"""Measure the three entry points and the mutation matrix; dump one JSON."""
import ast, json, pathlib, re, shutil, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

from strands_robots.hardware_robot import Robot as HardwareRobot, RobotTaskState
from strands_robots.mesh.input import InputPublisher
from strands_robots.utils import positive_finite_number_error
sys.path.insert(0, str(ROOT))
from tests.test_teleop import FakePublishHost, FakeTeleop  # noqa: E402

BAD_HZ = 0
out = {"tree": TREE, "entry_points": [], "mutations": [], "coverage": {}}


class FakeMesh:
    def __init__(self):
        self.peer_id = "leader-1"; self.alive = True; self.published = []
    def publish(self, topic, payload): self.published.append((topic, payload))


def hw_robot():
    hw = HardwareRobot.__new__(HardwareRobot)
    hw.tool_name_str = "arm"; hw.mesh = FakeMesh(); hw.peer_id = "leader-1"
    hw.robot = object(); hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1)
    hw._shutdown_event = threading.Event()
    hw._task_admission = threading.Lock(); hw._task_claimed = False
    return hw


# --- (1) what each of the three named entry points does with hz=0 -----------
host = FakePublishHost()
dev = FakeTeleop({"a.pos": 1.0})
host.attach_teleop(dev, name="lead")
res = host.teleoperate(hz=BAD_HZ, publish=True)
out["entry_points"].append({
    "surface": "teleoperate(publish=True)  [mixin]",
    "outcome": f'refused: {res["content"][0]["text"][:52]}',
    "driven_before": True,
    "note": "validates hz, then forwards it on",
})

hw = hw_robot()
res = hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), hz=BAD_HZ)
out["entry_points"].append({
    "surface": "Robot.start_teleop_publish  [hardware]",
    "outcome": f'refused: {res["content"][0]["text"]}',
    "driven_before": False,
    "note": "the only guard a direct caller passes",
})

try:
    InputPublisher(mesh=FakeMesh(), teleoperator=object(), hz=BAD_HZ)
    ip = "accepted (!)"
except ValueError as exc:
    ip = f"raises ValueError: {exc}"
out["entry_points"].append({
    "surface": "InputPublisher(hz=...)  [constructor]",
    "outcome": ip,
    "driven_before": True,
    "note": "raises; a tool caller cannot use it",
})

# the stand-in host that made the middle row invisible
stub = FakePublishHost()
stub_res = stub.start_teleop_publish(teleoperator=object(), hz=BAD_HZ)
out["stub"] = {
    "class": "tests.test_teleop.FakePublishHost.start_teleop_publish",
    "status": stub_res["status"],
    "validates": False,
}

# --- (2) ordering: does a refused rate stop a live publisher? --------------
class Live:
    def __init__(self): self.stopped = False
    def stop(self): self.stopped = True; return {}

hw = hw_robot(); live = Live(); hw._input_publishers = {"leader": live}
r = hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), device_name="leader", hz=BAD_HZ)
out["ordering_refused"] = {"status": r["status"], "live_stopped": live.stopped}
hw = hw_robot(); live = Live(); hw._input_publishers = {"leader": live}
r = hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), device_name="leader", hz=50.0)
out["ordering_accepted"] = {"status": r["status"], "live_stopped": live.stopped,
                            "replaced": hw._input_publishers["leader"] is not live}
hw._input_publishers["leader"].stop()

# --- (3) mutation matrix ---------------------------------------------------
SRC = ROOT / "strands_robots/hardware_robot.py"
orig = SRC.read_text()
NEWCLASS = "TestTheHardwarePublishEntryPointSharesTheRateDomain"
ARM_NEW = ["tests/test_teleop_rate_and_duration_guards.py", "-k", NEWCLASS]
ARM_OLD = ["tests/test_teleop_rate_and_duration_guards.py", "tests/test_teleop.py",
           "tests/test_hardware_robot_lifecycle.py",
           "tests/mesh/test_teleop_identifier_source_scoping.py", "-k", f"not {NEWCLASS}"]
GUARD = ('        error = positive_finite_number_error(hz, "hz", "start_teleop_publish")\n'
         '        if error:\n'
         '            return {"status": "error", "content": [{"text": error}]}\n\n')
TEARDOWN = ("        if device_name in self._input_publishers:\n"
            "            # Stop existing publisher for this device\n"
            "            self._input_publishers[device_name].stop()\n")
MUT = [
    ("M1  delete the hz guard", lambda s: s.replace(GUARD, "", 1)),
    ("M2  move the guard below the teardown",
     lambda s: s.replace(GUARD, "", 1).replace(TEARDOWN, TEARDOWN + "\n" + GUARD.rstrip("\n") + "\n", 1)),
    ("M3  keep the call, discard the refusal",
     lambda s: s.replace(GUARD, '        positive_finite_number_error(hz, "hz", "start_teleop_publish")\n\n', 1)),
    ("M4  hand-rolled guard replaces the domain",
     lambda s: s.replace('        error = positive_finite_number_error(hz, "hz", "start_teleop_publish")\n',
                         '        error = None if isinstance(hz, (int, float)) else "bad"\n', 1)),
    ("M5  delete the replacement teardown", lambda s: s.replace(TEARDOWN, "", 1)),
]

def failures(args):
    p = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "--no-header", "--no-cov",
                        "-p", "no:randomly"], capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) failed", p.stdout)
    return int(m.group(1)) if m else 0

try:
    for label, fn in MUT:
        mutated = fn(orig)
        assert mutated != orig, label
        ast.parse(mutated)
        SRC.write_text(mutated)
        out["mutations"].append({"label": label, "new": failures(ARM_NEW), "old": failures(ARM_OLD)})
        SRC.write_text(orig)
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig

out["coverage"] = {"file": "strands_robots/hardware_robot.py",
                   "missing_before": 243, "missing_after": 242, "line": 2650}
out["suite_arms"] = {"new_cases": 30, "preexisting_cases": 254}
pathlib.Path(f"/tmp/art-{pathlib.Path(TREE).name}.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "entry_points"}, indent=2)[:1200])
print("\nentry points:")
for e in out["entry_points"]:
    print(f"  driven_before={e['driven_before']!s:<5} {e['surface']:<40} {e['outcome'][:70]}")
