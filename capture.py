"""Measure both the dispatch-consequence ledger and the mutation matrix; dump JSON."""
import ast, contextlib, importlib, json, pathlib, subprocess, sys
from unittest.mock import MagicMock, patch
import strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
rm = importlib.import_module("strands_robots.tools.robot_mesh")
SRC = pathlib.Path("strands_robots/tools/robot_mesh.py"); ORIG = SRC.read_text()
NEW = "tests/mesh/test_robot_mesh_validate_before_hitl_contract.py"
OLD = [str(p) for p in sorted(pathlib.Path("tests/mesh").glob("test_robot_mesh_*.py")) if NEW not in str(p)]

GUARD = {
    "send": '''        if validated_send_cmd is None:
            raise RuntimeError(
                "send reached its handler without pre-validation -- validate-before-HITL contract broken"
            )
''',
    "broadcast": '''        if validated_broadcast_cmd is None:
            raise RuntimeError(
                "broadcast reached its handler without pre-validation -- validate-before-HITL contract broken"
            )
''',
}
KW = {"send": {"target": "peer-b", "command": '{"action": "status"}'},
      "broadcast": {"command": '{"action": "status"}'}}

def ctx():
    c = MagicMock(); c.interrupt.return_value = "y"; return c

def drive(action):
    """Drive one gated action with a validator that returns None; report outcome + wire."""
    mod = importlib.reload(rm)
    mesh = MagicMock(); mesh.peer_id = "local-a"; mesh.peer_type = "sim"; mesh.inbox = {}
    mesh.send.return_value = {"ok": True}; mesh.broadcast.return_value = [{"ok": True}]
    audits = []
    mod._reset_rate_limits()
    with contextlib.ExitStack() as st:
        st.enter_context(patch("strands_robots.mesh.get_local_robots", return_value={"local-a": mesh}))
        st.enter_context(patch("strands_robots.mesh.session.get_peers", return_value=[]))
        st.enter_context(patch.object(mod._security, "validate_command", lambda cmd: None))
        st.enter_context(patch.object(mod, "_audit_tool_action", lambda *a: audits.append(list(a))))
        fn = getattr(mod.robot_mesh, "original", mod.robot_mesh)
        try:
            out = fn(tool_context=ctx(), action=action, **KW[action])
            outcome = f"returned status={out['status']}"
        except BaseException as exc:
            outcome = f"{type(exc).__name__}: {str(exc)[:70]}"
    call = getattr(mesh, action).call_args
    return {"outcome": outcome,
            "wire": (f"{action}(" + ", ".join(repr(a) for a in call.args) + ")") if call else "-- not called --",
            "dispatched": call is not None,
            "audited_success": [a for a in audits if a[2] is True and "approved" not in str(a[3])]}

facts = {"tree": str(ROOT), "with_guard": {}, "without_guard": {}}
for action in ("send", "broadcast"):
    facts["with_guard"][action] = drive(action)
for action in ("send", "broadcast"):
    assert ORIG.count(GUARD[action]) == 1
    SRC.write_text(ORIG.replace(GUARD[action], "", 1))
    try:
        facts["without_guard"][action] = drive(action)
    finally:
        SRC.write_text(ORIG); assert SRC.read_text() == ORIG
importlib.reload(rm)

# ---- mutation matrix (re-derived, nothing hand-typed)
MUT = [
    ("delete the send guard", GUARD["send"], ""),
    ("delete the broadcast guard", GUARD["broadcast"], ""),
    ("send guard -> assert (-O strippable)", GUARD["send"],
     '        assert validated_send_cmd is not None, "validate-before-HITL contract broken"\n'),
    ("broadcast guard reads wrong sentinel", GUARD["broadcast"],
     GUARD["broadcast"].replace("validated_broadcast_cmd is None", "validated_send_cmd is None")),
    ("broadcast guard test inverted", GUARD["broadcast"], GUARD["broadcast"].replace("is None:", "is not None:")),
]
def run(paths):
    p = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
                       capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if " passed" in l or " failed" in l][-1]
    import re
    f = re.search(r"(\d+) failed", line); ps = re.search(r"(\d+) passed", line)
    return {"failed": int(f.group(1)) if f else 0, "passed": int(ps.group(1)) if ps else 0}
rows = []
try:
    for label, old, new in MUT:
        SRC.write_text(ORIG.replace(old, new, 1)); ast.parse(SRC.read_text())
        rows.append({"label": label, "new": run([NEW]), "old": run(OLD)})
        SRC.write_text(ORIG)
    facts["control"] = {"new": run([NEW]), "old": run(OLD)}
finally:
    SRC.write_text(ORIG); assert SRC.read_text() == ORIG
facts["mutations"] = rows
facts["n_old_files"] = len(OLD)
pathlib.Path("/tmp/art-facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2)[:1400])
