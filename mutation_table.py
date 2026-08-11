"""Mutation table: 5 plausible regressions x 2 arms (new file vs pre-existing robot_mesh tests)."""
import ast, pathlib, subprocess, sys
SRC = pathlib.Path("strands_robots/tools/robot_mesh.py")
NEW = "tests/mesh/test_robot_mesh_validate_before_hitl_contract.py"
OLD = [str(p) for p in sorted(pathlib.Path("tests/mesh").glob("test_robot_mesh_*.py")) if NEW not in str(p)]
ORIG = SRC.read_text()

SEND_GUARD = '''        if validated_send_cmd is None:
            raise RuntimeError(
                "send reached its handler without pre-validation -- validate-before-HITL contract broken"
            )
'''
BCAST_GUARD = '''        if validated_broadcast_cmd is None:
            raise RuntimeError(
                "broadcast reached its handler without pre-validation -- validate-before-HITL contract broken"
            )
'''
MUTATIONS = [
    ("M1 delete the send guard", SEND_GUARD, ""),
    ("M2 delete the broadcast guard", BCAST_GUARD, ""),
    ("M3 send guard becomes an assert (-O strippable)", SEND_GUARD,
     '        assert validated_send_cmd is not None, "validate-before-HITL contract broken"\n'),
    ("M4 broadcast guard reads the wrong sentinel", BCAST_GUARD,
     BCAST_GUARD.replace("validated_broadcast_cmd is None", "validated_send_cmd is None")),
    ("M5 broadcast guard test inverted", BCAST_GUARD,
     BCAST_GUARD.replace("is None:", "is not None:")),
]

# AST-scope: both guards live inside robot_mesh; print in_fn vs in_file per anchor.
tree = ast.parse(ORIG)
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "robot_mesh")
fn_src = "\n".join(ORIG.splitlines()[fn.lineno - 1 : fn.end_lineno])
for label, old, _ in MUTATIONS:
    print(f"anchor {label[:34]:36s} in_fn={fn_src.count(old)} in_file={ORIG.count(old)}")
assert all(ORIG.count(o) == 1 and fn_src.count(o) == 1 for _, o, _ in MUTATIONS), "anchors not unique inside robot_mesh"

def run(paths):
    p = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
                       capture_output=True, text=True)
    tail = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l or "error" in l.lower()]
    return tail[-1].strip() if tail else "??"

print("\n%-46s | %-28s | %s" % ("mutation", "new file", "pre-existing robot_mesh tests"))
print("-" * 112)
try:
    for label, old, new in MUTATIONS:
        SRC.write_text(ORIG.replace(old, new, 1))
        ast.parse(SRC.read_text())
        print("%-46s | %-28s | %s" % (label, run([NEW]), run(OLD)))
        SRC.write_text(ORIG)
    print("\n%-46s | %-28s | %s" % ("(unmutated control)", run([NEW]), run(OLD)))
finally:
    SRC.write_text(ORIG)
    assert SRC.read_text() == ORIG, "RESTORE FAILED"
    print("\nsource restored byte-identical:", SRC.read_text() == ORIG)
    print("pre-existing files in arm B:", len(OLD))
