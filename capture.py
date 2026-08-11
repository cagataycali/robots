"""Measure the developer-visible change on whichever tree runs this file."""
from __future__ import annotations
import ast, importlib, json, os, pathlib, subprocess, sys, tempfile

import strands_robots
TREE = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", TREE)
ROOT = pathlib.Path(strands_robots.__file__).parent
OUT = pathlib.Path(sys.argv[1])

# ---- 1. the verbatim pytest render of a refused constructor -------------------
PROBE = '''
from strands_robots.mesh.ros_bridge import RosBridgedRobot

def test_a_refused_bridge_is_diagnosable():
    try:
        RosBridgedRobot(node_name="bad name!", cmd_vel_topic="/cmd_vel", odom_topic="/odom")
    except ValueError as exc:
        tb, found = exc.__traceback__, None
        while tb is not None:
            s = tb.tb_frame.f_locals.get("self")
            if type(s).__name__ == "RosBridgedRobot":
                found = s
            tb = tb.tb_next
        assert found is None, "a half-built bridge is still reachable"
'''
d = pathlib.Path(tempfile.mkdtemp())
(d / "test_probe.py").write_text(PROBE)
r = subprocess.run([sys.executable, "-m", "pytest", str(d / "test_probe.py"), "-q", "--no-header",
                    "--no-cov", "-p", "no:randomly", "-p", "no:cacheprovider"],
                   capture_output=True, text=True, cwd=str(TREE))
render = next((ln.strip() for ln in r.stdout.splitlines()
               if ln.strip().startswith("E ") and " is None" in ln), "")
render = render.removeprefix("E ").strip()

# the real refusal message, for the panel
try:
    from strands_robots.mesh.ros_bridge import RosBridgedRobot
    RosBridgedRobot(node_name="bad name!", cmd_vel_topic="/cmd_vel", odom_topic="/odom")
    refusal = ""
except ValueError as exc:
    refusal = str(exc)

# ---- 2. the 11-class survey ---------------------------------------------------
survey: dict[str, str] = {}
for path in sorted(ROOT.rglob("*.py")):
    src = path.read_text(encoding="utf-8")
    if "def __repr__" not in src:
        continue
    rel = path.relative_to(ROOT).with_suffix("")
    dotted = ("strands_robots." + str(rel).replace("/", ".")).removesuffix(".__init__")
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(isinstance(f, ast.FunctionDef) and f.name == "__repr__" for f in node.body):
            continue
        cls = getattr(importlib.import_module(dotted), node.name)
        factory = cls
        obj = factory.__new__(factory)
        try:
            survey[node.name] = "ok: " + repr(obj)
        except BaseException as exc:
            survey[node.name] = f"raises: {type(exc).__name__}: {exc}"

# ---- 3. fully-built reprs must be byte-identical across trees ----------------
class _Mesh:
    peer_id, alive = "arm", True
    def subscribe(self, *a, **k): return None
    def unsubscribe(self, *a, **k): return None
class _Robot: tool_name_str = "arm"
class _DS:
    repo_id, root = "user/dataset", pathlib.Path("/tmp")

from strands_robots.dataset_recorder import DatasetRecorder
from strands_robots.mesh.core import Mesh
from strands_robots.mesh.input import InputPublisher, InputReceiver
from strands_robots.mesh.ros_bridge import RosBridgedRobot as RBR
from strands_robots.mesh.rosbridge_robot import RosbridgeRobot
from strands_robots.mesh.rtps_robot import RtpsRobot
from strands_robots.mesh.session import PeerInfo
from strands_robots.policies.lerobot_local.processor import ProcessorBridge
built = {
    "RosBridgedRobot": repr(RBR(node_name="/turtle1", cmd_vel_topic="/cmd_vel", odom_topic="/odom")),
    "RosbridgeRobot": repr(RosbridgeRobot(node_name="/turtle1", cmd_vel_topic="/cmd_vel",
                                          odom_topic="/odom", host="127.0.0.1", port=9090)),
    "RtpsRobot": repr(RtpsRobot(node_name="/arm", cmd_vel_topic="/cmd_vel")),
    "Mesh": repr(Mesh(_Robot(), peer_id="arm", peer_type="robot")),
    "PeerInfo": repr(PeerInfo(peer_id="arm", peer_type="robot", last_seen=0.0)).split(", age=")[0],
    "DatasetRecorder": repr(DatasetRecorder(dataset=_DS())),
    "InputPublisher": repr(InputPublisher(_Mesh(), object(), device_name="leader", hz=50.0)),
    "InputReceiver": repr(InputReceiver(_Mesh(), object(), source_peer_id="peer", device_name="leader")),
    "ProcessorBridge": repr(ProcessorBridge(None, None)),
}

# ---- 4. nothing outside diagnostic rendering moved ---------------------------
class _DropReprAndItsImport(ast.NodeTransformer):
    """Drop every ``__repr__`` and the utils import that serves it.

    What remains is every executable line the change does not touch, so an
    identical digest is mechanical proof that nothing outside diagnostic
    rendering moved.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        return None if node.name == "__repr__" else self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):  # noqa: N802
        if (node.module or "").endswith("utils"):
            return None
        return node

import hashlib
TOUCHED = ["dataset_recorder.py", "hardware_rtps_bridge.py", "mesh/core.py", "mesh/input.py",
           "mesh/ros_bridge.py", "mesh/rosbridge_robot.py", "mesh/rtps_robot.py",
           "mesh/session.py", "policies/lerobot_local/processor.py",
           "simulation/isaac/simulation.py"]
digests = {}
for rel in TOUCHED:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    tree = _DropReprAndItsImport().visit(tree)
    ast.fix_missing_locations(tree)
    digests[rel] = hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]

# ---- 5. a real MuJoCo headless render, to ground the no-regression claim -----
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
from strands_robots import Simulation  # noqa: E402
sim = Simulation(backend="mujoco", mesh=False)
sim.create_world()
sim.add_robot(name="so100")
sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)
res = sim.render(camera_name="look", width=560, height=480)
png = next(c["image"]["source"]["bytes"] for c in res["content"] if "image" in c)
frame_path = OUT.with_name(OUT.stem + "_frame.png")
frame_path.write_bytes(png)
sim_ok = res["status"]
sim.cleanup()

OUT.write_text(json.dumps({
    "tree": str(TREE),
    "pytest_render": render,
    "refusal": refusal,
    "survey": survey,
    "built": built,
    "digests": digests,
    "frame": str(frame_path),
    "sim_status": sim_ok,
}, indent=2))
print("wrote", OUT)
