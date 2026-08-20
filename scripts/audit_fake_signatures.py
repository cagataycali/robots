"""The camouflage direction: a test fake whose signature is a FALSE CLAIM about the real class."""
import ast, inspect, pathlib, collections

reals = {}
# Same honesty rule as audit_collaborator_kwargs: a narrowed run must not print like a complete one.
SKIPPED: list[str] = []
from strands_robots.dataset_recorder import DatasetRecorder
reals["DatasetRecorder"] = DatasetRecorder
for mod, name in [("strands_robots.mesh.core","Mesh"),("strands_robots.robot","Robot"),
                  ("strands_robots.dashboard.device_manager","DeviceManager"),
                  ("strands_robots.dashboard.record_worker","RecordWorker")]:
    try:
        m = __import__(mod, fromlist=[name]); reals[name] = getattr(m, name)
    except Exception as e:
        SKIPPED.append(name)
        print("skip", name, e)

by_name = collections.defaultdict(list)
for cname, cls in reals.items():
    for mname, f in inspect.getmembers(cls, predicate=inspect.isfunction):
        if mname.startswith("_"): continue
        by_name[mname].append((cname, inspect.signature(f)))

FAKEISH = ("fake","stub","dummy","spy","recording","boom")
out = []
for path in sorted(pathlib.Path("tests").rglob("*.py")):
    try: tree = ast.parse(path.read_text())
    except Exception: continue
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if not any(t in cls.name.lower() for t in FAKEISH): continue
        for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if fn.name.startswith("_"): continue
            cands = by_name.get(fn.name)
            if not cands: continue
            args = [a.arg for a in fn.args.args if a.arg != "self"] + [a.arg for a in fn.args.kwonlyargs]
            if fn.args.kwarg: continue  # **kwargs fake claims nothing
            for cname, sig in cands:
                params = [p for n,p in sig.parameters.items() if n != "self"]
                if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
                    continue
                real_names = {p.name for p in params}
                invented = [a for a in args if a not in real_names]
                if invented:
                    out.append((str(path), fn.lineno, cls.name, f"{cname}.{fn.name}{sig}", invented))
seen=set()
for p in out:
    k=(p[0],p[2],p[3])
    if k in seen: continue
    seen.add(k)
    print("FAKE CLAIMS", p[0], "line", p[1], p[2], "vs", p[3], "invented:", p[4])
print(f"\n{len(seen)} fake/real signature divergence(s)"
      + (f"; {len(SKIPPED)} real class(es) NOT loaded ({', '.join(SKIPPED)}) — a skip is not a pass"
         if SKIPPED else ""))
# Read the hits before believing them: this pairs a fake with a real class BY METHOD NAME, so an
# honest stand-in for an MQTT/websocket/ROS object gets compared to Mesh purely because both have a
# `publish`. Measured 2026-08-20: 24 hits, 0 real (the two genuine ones are pinned in
# tests/test_dashboard_record_fake_fidelity.py). Deliberately NOT a gate for that reason.
