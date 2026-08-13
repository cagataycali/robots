"""Measure the incident on the real Robot.start_task path, per tree.

Reads whichever executor the tree's own fixture module imports, builds the same
Robot shape, wedges bring-up so the rollout never finishes, gives up on the
future, and then lets the interpreter try to exit.
"""
from __future__ import annotations
import json, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1])
KILL = 45.0

CHILD = r'''
import json, sys, threading, time
import tests.test_hardware_policy_port_domain as m
from strands_robots.hardware_robot import Robot as HwRobot, RobotTaskState

Exec = getattr(m, "DaemonThreadExecutor", None) or m.ThreadPoolExecutor
facts = {"executor": Exec.__name__}

wedge = threading.Event()

hw = HwRobot.__new__(HwRobot)
hw.tool_name_str = "test_arm"
hw.action_horizon = 1
hw.data_config = None
hw.control_frequency = 50.0
hw.action_sleep_time = 1.0 / 50.0
hw._task_state = RobotTaskState()
hw._executor = Exec(max_workers=1, thread_name_prefix="test_arm_executor")
hw._shutdown_event = threading.Event()
hw._stop_requested = threading.Event()
hw._task_admission = threading.Lock()
hw._task_claimed = False
hw.mesh = None
hw.peer_id = None
hw.robot = m._FakeArm()

async def _connected():
    wedge.wait()          # the rollout never gets past bring-up
    return (True, "")
hw._connect_robot = _connected
hw._initialize_policy = lambda policy: policy
hw._publish_ros_telemetry = lambda observation, *, skip_images=False: None

started = hw.start_task("pick", policy_port=5555, duration=0.5)
facts["start_task_status"] = started["status"]

t0 = time.monotonic()
try:
    hw._task_state.task_future.result(timeout=2.0)
    facts["verdict"] = "the wait returned"
except TimeoutError:
    facts["verdict"] = "TimeoutError -- the test fails here"
facts["verdict_after_s"] = round(time.monotonic() - t0, 2)

hw._executor.shutdown(wait=False)
facts["non_daemon_left"] = sorted(
    t.name for t in threading.enumerate()
    if not t.daemon and t is not threading.main_thread()
)
print("FACTS " + json.dumps(facts), flush=True)
'''

def arm(tree: pathlib.Path) -> dict:
    (tree / "_child.py").write_text(CHILD)
    t0 = time.monotonic()
    exited, code, out = True, None, ""
    try:
        done = subprocess.run(
            [sys.executable, "_child.py"], cwd=tree,
            capture_output=True, text=True, timeout=KILL, check=False,
        )
        code, out = done.returncode, done.stdout
    except subprocess.TimeoutExpired as e:
        exited = False
        out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
    wall = round(time.monotonic() - t0, 2)
    (tree / "_child.py").unlink()
    facts = {}
    for line in out.splitlines():
        if line.startswith("FACTS "):
            facts = json.loads(line[6:])
    facts.update({"tree": str(tree), "exited": exited, "exit_code": code, "wall_s": wall,
                  "killed_at_s": None if exited else KILL})
    return facts


base = pathlib.Path(sys.argv[2])
data = {"main": arm(base), "branch": arm(ROOT)}
assert data["main"]["tree"] != data["branch"]["tree"]
OUT.write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2))
