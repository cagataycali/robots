"""Measure both reported claims + the residual defect, in whichever tree runs it."""
import importlib.metadata
import json, os, pathlib, sys

import strands_robots
TREE = pathlib.Path(strands_robots.__file__).parents[1]
OUT = pathlib.Path(f"/tmp/art-{os.environ['GITHUB_RUN_ID']}-{TREE.name}.json")
facts = {"tree": str(TREE)}


def save():
    OUT.write_text(json.dumps(facts, indent=2))


print("TREE:", TREE, flush=True)
save()


class Absent:
    """Report a package as genuinely not installed, exactly as import does."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in self.names:
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        return None


# --- provider census: who defers + translates, who imports at module level ---
import ast
HEAVY = {"torch", "transformers", "diffusers", "zmq", "onnxruntime", "lerobot", "msgpack"}
census = {"defers_and_translates": [], "module_level_heavy": []}
for p in sorted((pathlib.Path(strands_robots.__file__).parent / "policies").rglob("*.py")):
    if "__pycache__" in str(p):
        continue
    t = ast.parse(p.read_text(encoding="utf-8"))
    ml = [(a.name.split(".")[0], n.lineno) for n in t.body if isinstance(n, ast.Import)
          for a in n.names if a.name.split(".")[0] in HEAVY]
    hints = {c.func.id for c in ast.walk(t)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id in {"require_optional", "require_optionals"}}
    rel = str(p.relative_to(pathlib.Path(strands_robots.__file__).parent))
    if hints:
        census["defers_and_translates"].append(rel)
    if ml and not hints:
        census["module_level_heavy"].append({"module": rel, "imports": ml})
facts["census"] = census
save()

# --- CLAIM 2 (recording): what does start_recording do with lerobot absent? ---
row2 = {}
try:
    from strands_robots import Simulation
    import strands_robots.dataset_recorder as dr

    sim = Simulation(backend="mujoco", mesh=False, tool_name="art")
    sim.create_world()
    arm = pathlib.Path(f"/tmp/arm-{os.environ['GITHUB_RUN_ID']}.xml")
    arm.write_text(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="link" pos="0 0 0.1"><joint name="j" type="hinge" axis="0 1 0" damping="1"/>'
        '<geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.02"/></body>'
        "</worldbody><actuator><position name=\"a\" joint=\"j\" kp=\"12\"/></actuator></mujoco>"
    )
    sim.add_robot(name="arm", urdf_path=str(arm))
    # Make the LeRobot dataset stack unavailable the way a minimal install is.
    real_probe = dr.lerobot_dataset_import_error
    dr.lerobot_dataset_import_error = lambda: "No module named 'lerobot.datasets.lerobot_dataset'"
    res = sim.start_recording(repo_id="local/art_probe", task="probe", fps=30,
                              root=str(pathlib.Path(f"/tmp/artds-{os.environ['GITHUB_RUN_ID']}")))
    dr.lerobot_dataset_import_error = real_probe
    text = ""
    for c in res.get("content", []):
        if "text" in c:
            text = c["text"]
            break
    row2 = {"status": res.get("status"), "text": text,
            "degraded_to_mp4": "start_cameras_recording" in text and res.get("status") == "success",
            "names_lerobot": "lerobot" in text}
    sim.cleanup()
except BaseException as e:  # noqa: BLE001 - recording the outcome
    row2 = {"status": "probe_error", "text": f"{type(e).__name__}: {e}"}
facts["claim2_recording"] = row2
save()
print("CLAIM2:", json.dumps(row2)[:400], flush=True)

# --- CLAIM 1 + the residual: what error does a caller actually receive? ------
sys.meta_path.insert(0, Absent({"torch"}))
_realv = importlib.metadata.version
importlib.metadata.version = lambda n: (_ for _ in ()).throw(
    importlib.metadata.PackageNotFoundError(n)) if n.split(".")[0] == "torch" else _realv(n)
for m in [k for k in sys.modules if k.split(".")[0] == "torch"]:
    del sys.modules[m]

from strands_robots.policies import create_policy
import strands_robots.registry.policies as reg

scenarios = {}
# registered provider whose module imports torch at module level
try:
    obj = create_policy("lerobot_local", pretrained_name_or_path="allenai/MolmoAct2-SO100_101")
    scenarios["registered"] = {"outcome": "CONSTRUCTED", "class": type(obj).__name__,
                               "substituted": "Mock" in type(obj).__name__}
except BaseException as e:  # noqa: BLE001
    msg = str(e)
    scenarios["registered"] = {
        "outcome": "RAISED", "exc": type(e).__name__, "message": msg, "substituted": False,
        "names_provider": "lerobot_local" in msg, "names_module": "torch" in msg,
        "names_install": "strands-robots[" in msg}
# a provider whose module exists but cannot import its dependency
_real_import = reg.importlib.import_module


def _fake(name, *a, **k):
    if name == "strands_robots.policies.some_provider":
        raise ModuleNotFoundError("No module named 'some_dep'", name="some_dep")
    return _real_import(name, *a, **k)


reg.importlib.import_module = _fake
try:
    reg.import_policy_class("some_provider")
    scenarios["autodiscovered"] = {"outcome": "CONSTRUCTED"}
except BaseException as e:  # noqa: BLE001
    msg = str(e)
    scenarios["autodiscovered"] = {
        "outcome": "RAISED", "exc": type(e).__name__, "message": msg,
        "blames_the_name": "Unknown policy provider" in msg,
        "names_module": "some_dep" in msg}
reg.importlib.import_module = _real_import
facts["claim1_scenarios"] = scenarios
save()
print("SCENARIOS:", json.dumps(scenarios, indent=2)[:900], flush=True)
print("WROTE", OUT)
