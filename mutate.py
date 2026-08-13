"""Mutation table: 6 plausible regressions x (new module | pre-existing robot_mesh suite).

Each anchor is scoped to ``robot_mesh``'s own AST line range and the in-function
vs in-file counts are printed, so the scoping is checkable. The source is
restored byte-identically in a ``finally``.
"""
import ast, glob, json, os, pathlib, re, subprocess, sys

SRC = pathlib.Path("strands_robots/tools/robot_mesh.py")
NEW = ["tests/mesh/test_rate_limit_reserved_atomically_on_both_gate_paths.py"]
OLD = sorted(glob.glob("tests/mesh/test_robot_mesh*.py")) + ["tests/mesh/test_sim_robot_mesh_attach.py"]

base = subprocess.run(["git", "merge-base", "HEAD", "upstream/main"], capture_output=True, text=True,
                      check=True).stdout.strip()
blob = subprocess.run(["git", "show", f"{base}:strands_robots/tools/robot_mesh.py"], capture_output=True,
                      text=True, check=True).stdout
bl = blob.splitlines(keepends=True)
rec = next(n for n in ast.walk(ast.parse(blob)) if isinstance(n, ast.FunctionDef) and n.name == "_rate_limit_record")
RECORD_ONLY = "\n\n" + "".join(bl[rec.lineno - 1:rec.end_lineno])

UNGATED = """        rl_race_err = _rate_limit_check_and_record(action)
        if rl_race_err is not None:
            _audit_tool_action(action, target, False, f"rate_limit_race: {rl_race_err}")
            return _err(rl_race_err)
"""
AUDIT = ('            _audit_tool_action(action, target, False, f"rate_limit_race: {rl_race_err}")\n'
         "            return _err(rl_race_err)\n"
         '        _audit_tool_action(action, target, True, f"operator approved: {response!r}")\n')

MUTATIONS = {
    "M1 ungated path appends a slot unconditionally (the defect)":
        [(UNGATED, "        _rate_limit_record(action)\n"), ("\n\n__all__ =", RECORD_ONLY + "\n\n__all__ =")],
    "M2 ungated path reserves but discards the verdict":
        [(UNGATED, "        _rate_limit_check_and_record(action)\n")],
    "M3 ungated path reserves with the read-only check":
        [(UNGATED, UNGATED.replace("_rate_limit_check_and_record(action)", "_rate_limit_check(action)"))],
    "M4 the raced call is not audited":
        [(AUDIT, AUDIT.replace('            _audit_tool_action(action, target, False, f"rate_limit_race: {rl_race_err}")\n', "", 1))],
    "M5 race message names an operator approval again":
        [('f"and record (a concurrent call raced past): max {max_calls} "',
          'f"and record (concurrent approval raced past): max {max_calls} "')],
    "M6 the pre-gate check consumes a slot":
        [("    rl_err = _rate_limit_check(action)", "    rl_err = _rate_limit_check_and_record(action)")],
}


def run(paths):
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-header", "-p", "no:randomly",
                        "--no-cov", "--tb=no"], capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    f = re.search(r"(\d+) failed", r.stdout); p = re.search(r"(\d+) passed", r.stdout)
    return (int(f.group(1)) if f else 0, int(p.group(1)) if p else 0)


orig = SRC.read_text()
fn = next(n for n in ast.walk(ast.parse(orig))
          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "robot_mesh")
region = "".join(orig.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
rows = []
try:
    for label, edits in MUTATIONS.items():
        for old, _n in edits:
            if "__all__" not in old:
                print(f"  [{label[:3]}] in_fn={region.count(old)} in_file={orig.count(old)}")
                assert orig.count(old) >= 1
        mutated = orig
        for old, new in edits:
            mutated = mutated.replace(old, new, 1)
        assert mutated != orig, label
        ast.parse(mutated)
        SRC.write_text(mutated)
        nf, np_ = run(NEW); of, op = run(OLD)
        rows.append({"label": label, "new_failed": nf, "new_passed": np_, "old_failed": of, "old_passed": op})
        print(f"{label:<58} new {nf}/{np_}   pre-existing {of}/{op}")
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig
print("restored byte-identically")
nf, np_ = run(NEW); of, op = run(OLD)
ctl = {"label": "unmutated control", "new_failed": nf, "new_passed": np_, "old_failed": of, "old_passed": op}
print(f"{ctl['label']:<58} new {nf}/{np_}   pre-existing {of}/{op}")
pathlib.Path(sys.argv[1]).write_text(json.dumps({"rows": rows, "control": ctl, "old_files": len(OLD)}, indent=2))
print(f"\ncaught by new: {sum(1 for r in rows if r['new_failed'])}/{len(rows)}  "
      f"invisible to pre-existing: {sum(1 for r in rows if not r['old_failed'])}/{len(rows)}")
