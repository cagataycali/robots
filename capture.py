"""Measure the max_steps refusal: per-surface verdicts, mutation table, coverage delta."""

from __future__ import annotations

import ast, json, logging, os, pathlib, subprocess, sys, tempfile

import strands_robots

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

from strands_robots.benchmarks.libero import LiberoAdapter, load_libero_suite, parse_bddl
from strands_robots.utils import positive_count_error

BDDL = "(define (problem pick_cube) (:goal (grasped cube_1)))"
INERT = {"auto_generate_scene": False, "install_cameras": False}
VALUES = [("0", 0), ("-5", -5), ("True", True), ("2.7", 2.7), ("3.0", 3.0),
          ("nan", float("nan")), ("inf", float("inf")), ("'10'", "10"), ("[4]", [4])]
NEW = "tests/benchmarks/libero/test_libero_max_steps_domain.py"

out: dict = {"tree": TREE, "surfaces": {}, "mutations": [], "coverage": {}}
tmp = pathlib.Path(tempfile.mkdtemp())
bp = tmp / "task.bddl"; bp.write_text(BDDL)

def verdict(fn):
    try:
        fn(); return "accepted"
    except ValueError as e:
        return "refused" if "max_steps" in str(e) else "other"
    except BaseException as e:
        return f"raised {type(e).__name__}"

for label, fn in (
    ("LiberoAdapter(...)", lambda v: LiberoAdapter(parse_bddl(BDDL), max_steps=v, **INERT)),
    ("from_text(...)", lambda v: LiberoAdapter.from_text(BDDL, max_steps=v, **INERT)),
    ("from_file(...)", lambda v: LiberoAdapter.from_file(bp, max_steps=v, **INERT)),
):
    out["surfaces"][label] = {"channel": "raises ValueError",
                              "verdicts": {n: verdict(lambda v=v: fn(v)) for n, v in VALUES}}

# The suite loader: a different channel.
sdir = tmp / "libero_spatial"; sdir.mkdir()
for t in ("task_a", "task_b"):
    (sdir / f"{t}.bddl").write_text(BDDL)
recs: list[str] = []
class H(logging.Handler):
    def emit(self, r): recs.append(r.getMessage())
lg = logging.getLogger("strands_robots.benchmarks.libero.suite"); lg.addHandler(H()); lg.propagate = False
loader = {}
for n, v in VALUES:
    recs.clear()
    reg = load_libero_suite("libero_spatial", bddl_dir=sdir, max_steps=v,
                            load_init_states=False, adapter_kwargs=dict(INERT))
    named = sum(1 for m in recs if (positive_count_error(v, "max_steps", "LiberoAdapter") or "") in m)
    loader[n] = f"0 registered, {named}/2 reported"
recs.clear()
reg = load_libero_suite("libero_spatial", bddl_dir=sdir, max_steps=250,
                        load_init_states=False, adapter_kwargs=dict(INERT))
out["surfaces"]["load_libero_suite(...)"] = {
    "channel": "skips each task, returns", "verdicts": loader,
    "control": f"max_steps=250 -> {len(reg)} registered",
}

# Mutation table: both arms, AST-scoped anchors.
SRCP = pathlib.Path("strands_robots/benchmarks/libero/adapter.py")
orig = SRCP.read_text()
GUARD = ('            if error := positive_count_error(max_steps, "max_steps", type(self).__name__):\n'
         "                raise ValueError(error)\n")
RAISE = "                raise ValueError(error)\n"
CALL = "positive_count_error(max_steps,"
IMP = "from strands_robots.utils import get_base_dir, positive_count_error, require_optional"
tree_ = ast.parse(orig); lo = hi = None
for node in ast.walk(tree_):
    if isinstance(node, ast.ClassDef) and node.name == "LiberoAdapter":
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                lo, hi = sub.lineno, sub.end_lineno
region = "".join(orig.splitlines(keepends=True)[lo - 1 : hi])
for a in (GUARD, RAISE, CALL):
    assert region.count(a) == 1 and orig.count(a) == 1, "anchor not unique inside __init__"

def counts(args):
    r = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "-p", "no:randomly", "--no-cov"],
                       capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    line = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l][-1]
    f = int(line.split(" failed")[0].split("=")[-1].strip()) if " failed" in line else 0
    p = int(line.split(" passed")[0].rstrip().split()[-1])
    return f, p

for label, fn in (
    ("drop the guard entirely", lambda s: s.replace(GUARD, "")),
    ("keep the call, discard the raise", lambda s: s.replace(RAISE, "                pass\n")),
    ("widen to the whole-number domain",
     lambda s: s.replace(IMP, IMP + "\nfrom strands_robots.utils import positive_whole_number_error")
                .replace(CALL, "positive_whole_number_error(max_steps,")),
):
    mutated = fn(orig); assert mutated != orig; ast.parse(mutated)
    SRCP.write_text(mutated)
    try:
        nf, np_ = counts([NEW])
        ef, ep = counts(["tests/benchmarks/libero", "--deselect", NEW])
    finally:
        SRCP.write_text(orig)
    assert SRCP.read_text() == orig, "restore failed"
    out["mutations"].append({"label": label, "new_failed": nf, "new_passed": np_,
                             "existing_failed": ef, "existing_passed": ep})

for arm in ("before", "after"):
    d = json.load(open(f"/tmp/d-{arm}-{os.environ['GITHUB_RUN_ID']}.json"))["files"]
    a = d["strands_robots/benchmarks/libero/adapter.py"]
    out["coverage"][arm] = {"missing": len(a["missing_lines"]), "pct": a["summary"]["percent_covered_display"],
                            "line_541": "missing" if 541 in a["missing_lines"] else "covered"}

pathlib.Path(f"/tmp/art-{os.environ['GITHUB_RUN_ID']}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["mutations"], indent=1)); print(json.dumps(out["coverage"], indent=1))
