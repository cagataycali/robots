"""Re-measure every number the figure shows, into one JSON."""
import ast, dataclasses, json, os, pathlib, re, subprocess, sys, tempfile

import strands_robots
TREE = pathlib.Path(strands_robots.__file__).parents[1]
RUN = os.environ["GITHUB_RUN_ID"]
SRC = pathlib.Path("strands_robots/training/lerobot.py")
NEWF = "tests/training/test_lerobot_policy_type_discovery.py"
BASEF = "tests/training/test_zz_base_probe.py"
out = {"tree": str(TREE)}

# (1) sibling registry-probe matrix, from the PRISTINE full-suite coverage run
cov = json.load(open(f"/tmp/cov-{RUN}.json"))["files"][str(SRC)]
miss, ex = set(cov["missing_lines"]), set(cov["executed_lines"])
src = SRC.read_text().splitlines()
tree = ast.parse("\n".join(src))
probes = []
for fn in sorted((f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)), key=lambda f: f.lineno):
    body = "\n".join(src[fn.lineno - 1:fn.end_lineno])
    if "_policy_registry()" not in body and "_reward_registry()" not in body:
        continue
    b0 = fn.body[0]
    start = b0.end_lineno + 1 if isinstance(b0, ast.Expr) and isinstance(b0.value, ast.Constant) else fn.lineno
    rng = set(range(start, fn.end_lineno + 1))
    probes.append({"name": fn.name, "missing": sorted(rng & miss),
                   "n": len([l for l in rng if l in ex | miss])})
out["probes"] = probes

# (2) the target lines, before vs after the new tests
out["lines"] = {}
for arm in ("before", "after"):
    d = json.load(open(f"/tmp/d-{arm}-{RUN}.json"))["files"][str(SRC)]
    out["lines"][arm] = sorted({235, 236, 238} & set(d["missing_lines"]))

# (3) what a caller is told, per registry state (the consequence)
from strands_robots.training import TrainSpec
from strands_robots.training import lerobot as L

def _boom(): raise RuntimeError("config default_factory is broken")

@dataclasses.dataclass
class Unreadable:
    normalization_mapping: dict = dataclasses.field(default_factory=_boom)

@dataclasses.dataclass
class NoField:
    chunk_size: int = 8

def consequence(cfg_cls, ptype):
    d = tempfile.mkdtemp()
    meta = pathlib.Path(d) / "meta"; meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    feat = {"mean": [0.0], "std": [1.0], "min": [-1.0], "max": [1.0]}
    (meta / "stats.json").write_text(json.dumps({"observation.state": dict(feat), "action": dict(feat)}))
    spec = TrainSpec(dataset_root=d, base_model="", output_dir=d + "/out",
                     steps=10, extra={"policy_type": ptype})
    orig = L._policy_registry
    if cfg_cls is not None:
        L._policy_registry = lambda: {"molmoact2": cfg_cls, "pi05": cfg_cls, "act": cfg_cls}
    try:
        probs = L.LerobotTrainer(device="cpu").validate(spec)
        answer = L._policy_uses_quantile_norm(ptype)
    finally:
        L._policy_registry = orig
    return {"probe": bool(answer),
            "warned": any("augment_dataset_quantile_stats" in p for p in probs)}

out["consequence"] = [
    {"state": "live registry (control)", "ptype": "molmoact2", **consequence(None, "molmoact2")},
    {"state": "unreadable default_factory", "ptype": "molmoact2", **consequence(Unreadable, "molmoact2")},
    {"state": "unreadable default_factory", "ptype": "act", **consequence(Unreadable, "act")},
    {"state": "no normalization_mapping", "ptype": "molmoact2", **consequence(NoField, "molmoact2")},
]

# (4) mutation table, both arms
original = SRC.read_text()
pathlib.Path(BASEF).write_text(
    subprocess.run(["git", "show", f"upstream/main:{NEWF}"], capture_output=True, text=True, check=True).stdout)
MUT = [
    ("M1", "drop the try/except",
     "            try:\n                mapping = f.default_factory()\n"
     "            except Exception:  # noqa: BLE001 - a broken default falls back to the static set\n"
     "                return ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK\n",
     "            mapping = f.default_factory()\n"),
    ("M2", "unknown collapses to False",
     "                return ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK\n", "                return False\n"),
    ("M3", "missing field collapses to the set",
     "    return False\n", "    return ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK\n"),
    ("M4", "narrow the handler to TypeError",
     "            except Exception:  # noqa: BLE001 - a broken default falls back to the static set\n",
     "            except TypeError:  # noqa: BLE001 - a broken default falls back to the static set\n"),
]
def run(target):
    o = subprocess.run([sys.executable, "-m", "pytest", target, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"], capture_output=True, text=True).stdout
    f, p = re.search(r"(\d+) failed", o), re.search(r"(\d+) passed", o)
    return {"failed": int(f.group(1)) if f else 0, "passed": int(p.group(1)) if p else 0}
fn = next(n for n in ast.walk(ast.parse(original)) if isinstance(n, ast.FunctionDef)
          and n.name == "_policy_uses_quantile_norm")
region = "\n".join(original.splitlines()[fn.lineno - 1:fn.end_lineno]) + "\n"
rows = []
try:
    for tag, label, old, new in MUT:
        assert region.count(old) == 1, tag
        SRC.write_text(original.replace(region, region.replace(old, new, 1), 1))
        try:
            rows.append({"tag": tag, "label": label, "in_file": original.count(old),
                         "new": run(NEWF), "base": run(BASEF)})
        finally:
            SRC.write_text(original)
        assert SRC.read_text() == original
    rows.append({"tag": "--", "label": "unmutated control", "in_file": 0,
                 "new": run(NEWF), "base": run(BASEF)})
finally:
    pathlib.Path(BASEF).unlink(missing_ok=True)
    SRC.write_text(original)
assert SRC.read_text() == original, "source not restored"
out["mutations"] = rows

pathlib.Path(f"/tmp/art-{RUN}.json").write_text(json.dumps(out, indent=2))
print("TREE:", TREE)
print(json.dumps({k: v for k, v in out.items() if k != "probes"}, indent=2)[:1400])
