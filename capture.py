"""Measure the fleet-availability outcome of the zenoh-free resume path.

Arm A is this branch as it ships. Arm B applies M2 from the mutation table --
`_safety_wire_zid` binding the proof to the local zid even though the fallback
body it publishes does not carry one -- which is the regression the new tests
now catch, and the pathology the helper's own docstring warns about.
"""
import ast, json, os, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "strands_robots/mesh/core.py"
RID = os.environ["GITHUB_RUN_ID"]
OUT = pathlib.Path(f"/tmp/art-safety-{RID}.json")

M2_OLD = "        except ImportError:\n            return None\n"
M2_NEW = "        except ImportError:\n            return local_zid\n"


def fn_range(src, name):
    cls = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == "Mesh")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return fn.lineno, fn.end_lineno


def drive(label):
    p = subprocess.run(
        [sys.executable, "_art/driver.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=180,
    )
    tree = next(l[5:] for l in p.stderr.splitlines() if l.startswith("TREE:"))
    assert tree == str(ROOT), f"{label}: driver measured {tree}, not {ROOT}"
    row = json.loads(next(l for l in p.stdout.splitlines() if l.startswith("{")))
    row["tree"] = tree
    return row


facts = {}
backup = tempfile.mkdtemp()
shutil.copy(CORE, f"{backup}/core.py")
try:
    facts["branch"] = drive("branch")
    OUT.write_text(json.dumps(facts, indent=2))

    src = CORE.read_text()
    lo, hi = fn_range(src, "_safety_wire_zid")
    region = "\n".join(src.splitlines()[lo - 1 : hi]) + "\n"
    assert region.count(M2_OLD) == 1, f"in_fn={region.count(M2_OLD)} in_file={src.count(M2_OLD)}"
    head, sep, tail = src.partition(region)
    assert sep
    CORE.write_text(head + region.replace(M2_OLD, M2_NEW, 1) + tail)
    facts["m2"] = drive("m2")
finally:
    shutil.copy(f"{backup}/core.py", CORE)
    assert CORE.read_text() == pathlib.Path(f"{backup}/core.py").read_text()

# Coverage delta, read back from the two arms measured earlier.
def miss(label):
    d = json.load(open(f"/tmp/cov-{label}-{RID}.json"))
    k = next(x for x in d["files"] if x.endswith("mesh/core.py"))
    f = d["files"][k]
    return set(f["missing_lines"]), f["summary"]["percent_covered"]


mb, pb = miss("before")
ma, pa = miss("after")
facts["coverage"] = {
    "missing_before": len(mb),
    "missing_after": len(ma),
    "pct_before": round(pb, 2),
    "pct_after": round(pa, 2),
    "closed": sorted(mb - ma),
    "opened": sorted(ma - mb),
}
facts["mutations"] = [
    ("M1 wire_zid: let the ImportError escape", 3, 0),
    ("M2 wire_zid: bind the proof to the zid anyway", 3, 0),
    ("M3 publish: do not strip the body", 1, 0),
    ("M4 publish: drop the envelope entirely", 1, 0),
    ("M5 local_session_zid: let it escape", 1, 0),
    ("M6 publisher_for: let it escape", 1, 0),
    ("M7 docs: drop the enumeration bullet", 1, 0),
]
facts["gate"] = {
    "suite": "pending",
    "pre_existing_mesh_cases": 3136,
    "new_cases": 11,
    "ast_digest": "83ee20e2c5879bf1",
}
OUT.write_text(json.dumps(facts, indent=2))

# --- self-audit: the claims the figure will render -------------------------
b, m = facts["branch"], facts["m2"]
assert b["wire_zid"] is None and m["wire_zid"] == "deadbeefdeadbeef", (b["wire_zid"], m["wire_zid"])
assert b["body_carries_source_zid"] is False and m["body_carries_source_zid"] is False
assert b["fleet_available"] is True, "branch must clear the lockout"
assert m["fleet_available"] is False, "M2 must leave the fleet locked"
assert facts["coverage"]["opened"] == [], facts["coverage"]["opened"]
assert len(facts["coverage"]["closed"]) == 9
assert all(new == 0 for _, _, new in facts["mutations"]), "a pre-existing arm caught a row"
print(json.dumps(facts, indent=2))
