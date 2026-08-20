"""Q56 generalised: does the dashboard ever call a collaborator with a kwarg the REAL class lacks?

Only fakes are exercised in dashboard tests, so a wrong kwarg is invisible until production.
"""
import ast, inspect, pathlib, sys, collections

DASH = pathlib.Path("strands_robots/dashboard")

# real classes the dashboard drives through injected collaborators
reals = {}
def load():
    from strands_robots.dataset_recorder import DatasetRecorder
    reals["DatasetRecorder"] = DatasetRecorder
    try:
        from strands_robots.hardware_robot import HardwareRobot
        reals["HardwareRobot"] = HardwareRobot
    except Exception as e: print("skip HardwareRobot", e)
    try:
        from strands_robots.mesh.core import Mesh
        reals["Mesh"] = Mesh
    except Exception as e: print("skip Mesh", e)
    try:
        from strands_robots.robot import Robot
        reals["Robot"] = Robot
    except Exception as e: print("skip Robot", e)
load()

# method name -> [(cls, signature)]
by_name = collections.defaultdict(list)
for cname, cls in reals.items():
    for mname, m in inspect.getmembers(cls, predicate=inspect.isfunction):
        if mname.startswith("__"): continue
        try: by_name[mname].append((cname, inspect.signature(m)))
        except Exception: pass

problems = []
for path in sorted(DASH.rglob("*.py")):
    try: tree = ast.parse(path.read_text())
    except Exception: continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute): continue
        kws = [k.arg for k in node.keywords if k.arg]
        if not kws: continue
        name = node.func.attr
        cands = by_name.get(name)
        if not cands: continue
        for cname, sig in cands:
            params = sig.parameters
            has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            if has_kwargs: continue
            bad = [k for k in kws if k not in params]
            if bad:
                problems.append((str(path), node.lineno, f"{cname}.{name}{sig}", bad))

for p in problems:
    print("MISMATCH", p[0], "line", p[1], "->", p[2], "unexpected:", p[3])
print(f"\n{len(problems)} candidate mismatch(es); real classes: {list(reals)}")
