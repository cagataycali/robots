"""Measure what use_rosbridge does at each refusal, and what reaches the bridge.

Writes one JSON dump; compose.py asserts every number it draws against it.
"""
from __future__ import annotations
import ast, json, pathlib, re, subprocess, sys, types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import strands_robots
import strands_robots.tools.use_rosbridge as rb
from tests.tools.test_use_rosbridge import _FakeRos, _FakeService, _FakeTopic

OUT = pathlib.Path(f"/tmp/art-rosbridge-{sys.argv[1]}.json")
F: dict = {"tree": str(pathlib.Path(strands_robots.__file__).parents[1])}
def save() -> None:
    OUT.write_text(json.dumps(F, indent=2))
save()
print("TREE:", F["tree"])

SRC = ROOT / "strands_robots/tools/use_rosbridge.py"
FILES = ["tests/tools/test_use_rosbridge.py",
         "tests/tools/test_rosbridge_port_type_identity.py",
         "tests/tools/test_rosbridge_transport_port_limit.py"]
NEW = ("test_invalid_service_rejected or test_topic_and_service_are_held_to_one_name_rule "
       "or test_an_invalid_service_name_is_refused_before_the_bridge_is_dialed "
       "or test_publish_requires_topic_and_type or test_an_incomplete_publish_advertises_no_publisher")

def install() -> types.ModuleType:
    _FakeRos.instances = []
    _FakeRos.fail_next_connect = False
    _FakeRos.ready_without_connection = False
    _FakeRos.scripted_responses = {}
    _FakeRos.scripted_messages = {}
    m = types.ModuleType("roslibpy")
    m.Ros, m.Topic, m.Service = _FakeRos, _FakeTopic, _FakeService
    m.Message = m.ServiceRequest = dict
    sys.modules["roslibpy"] = m
    rb._backend._connections = {}
    rb._backend._available = None
    return m

def txt(r: dict) -> str:
    return "\n".join(i.get("text", "") for i in r.get("content", []))

# --- 1. the refusal inventory, with which lines the suite reaches -------------
def refusals() -> list[dict]:
    src = SRC.read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "use_rosbridge")
    rows = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Return):
            seg = " ".join((ast.get_source_segment(src, n) or "").split())
            if "_err(" in seg:
                rows.append({"line": n.lineno, "text": seg})
    return sorted(rows, key=lambda r: r["line"])
F["refusals"] = refusals()
save()

def cov_arm(k: str | None) -> dict:
    cmd = [sys.executable, "-m", "pytest", *FILES, "-q", "-p", "no:randomly",
           "--cov=strands_robots", "--cov-report=term-missing", "--cov-fail-under=0"]
    if k:
        cmd += ["-k", k]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    line = next((l for l in out.splitlines() if "tools/use_rosbridge.py" in l), "")
    m = re.search(r"(\d+)\s+(\d+)\s+(\d+)%\s*(.*)$", line)
    missing = [int(x) for x in re.findall(r"\b(\d+)\b", m.group(4))] if (m and m.group(4)) else []
    p = re.search(r"(\d+) passed", out)
    return {"stmts": int(m.group(1)), "miss": int(m.group(2)), "pct": int(m.group(3)),
            "missing": missing, "passed": int(p.group(1)) if p else 0}

F["cov_before"] = cov_arm(f"not ({NEW})")
save()
F["cov_after"] = cov_arm(None)
save()
print("cov:", F["cov_before"], "->", F["cov_after"])

# --- 2. the wire trace: what reached the bridge for each call -----------------
def trace(label: str, calls: list[dict]) -> dict:
    fake = install()
    steps = []
    for kw in calls:
        r = rb.use_rosbridge(**kw)
        ros = fake.Ros.instances[0] if fake.Ros.instances else None
        steps.append({
            "call": {k: v for k, v in kw.items()},
            "status": r["status"],
            "text": txt(r),
            "clients_dialed": len(fake.Ros.instances),
            "advertised": [[t.name, bool(t.advertised), bool(t.unadvertised), len(t.published)]
                           for t in (ros.topics if ros else [])],
        })
    return {"label": label, "steps": steps}

F["wire"] = [
    trace("a mistyped service name",
          [{"action": "service_call", "service": "/bad name", "type": "std_srvs/Empty"}]),
    trace("a publish missing its interface type",
          [{"action": "publish", "topic": "/cmd_vel"}]),
    trace("the same publish, complete",
          [{"action": "publish", "topic": "/cmd_vel", "type": "geometry_msgs/Twist", "count": 1}]),
]
save()

# the parity table: one rule for topic and service
par = []
for name in ["/cmd vel", "/x|y", "../etc", "/a$(x)", "/gazebo/reset_world", "~private"]:
    install(); t = rb.use_rosbridge(action="echo", topic=name)
    install(); s = rb.use_rosbridge(action="service_call", service=name, type="std_srvs/Empty")
    par.append({"name": name,
                "topic_refused": "invalid topic name" in txt(t),
                "service_refused": "invalid service name" in txt(s)})
F["parity"] = par
save()

# --- 3. the mutation table ----------------------------------------------------
GUARD_S = ('    if service is not None and not _NAME_RE.match(service):\n'
           '        return _err(f"invalid service name: {service!r}")\n')
GUARD_P = ('                if not topic or not type:\n'
           '                    return _err("publish requires topic and type")\n')
MUTS = [
    ("M1 delete the service name guard", GUARD_S, ""),
    ("M2 service check evaluated, refusal discarded", GUARD_S,
     '    if service is not None and not _NAME_RE.match(service):\n        pass\n'),
    ("M3 service checked with the interface-type rule", GUARD_S,
     '    if service is not None and not _TYPE_RE.match(service):\n'
     '        return _err(f"invalid service name: {service!r}")\n'),
    ("M4 delete the publish required-argument guard", GUARD_P, ""),
    ("M5 publish refused only when BOTH are missing", GUARD_P,
     '                if not topic and not type:\n'
     '                    return _err("publish requires topic and type")\n'),
    ("M6 publish refusal reworded locally", GUARD_P,
     '                if not topic or not type:\n'
     '                    return _err("bad publish request")\n'),
]
def run(k: str) -> tuple[int, int]:
    out = subprocess.run([sys.executable, "-m", "pytest", *FILES, "-q", "--no-cov",
                          "-p", "no:randomly", "-k", k],
                         capture_output=True, text=True, cwd=ROOT).stdout
    f = re.search(r"(\d+) failed", out); p = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0), (int(p.group(1)) if p else 0)

src = SRC.read_text()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "use_rosbridge")
region = "".join(src.splitlines(keepends=True)[fn.body[0].lineno - 1: fn.end_lineno])
rows = []
try:
    for label, old, new in MUTS:
        in_fn, in_file = region.count(old), src.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn}"
        SRC.write_text(src.replace(old, new, 1))
        ast.parse(SRC.read_text())
        na_f, na_p = run(NEW)
        ol_f, ol_p = run(f"not ({NEW})")
        rows.append({"label": label, "new_failed": na_f, "new_passed": na_p,
                     "old_failed": ol_f, "old_passed": ol_p,
                     "in_fn": in_fn, "in_file": in_file})
        print(f"  {label}: new {na_f}F/{na_p}P | pre-existing {ol_f}F/{ol_p}P")
        SRC.write_text(src)
finally:
    SRC.write_text(src)
    assert SRC.read_text() == src
na_f, na_p = run(NEW); ol_f, ol_p = run(f"not ({NEW})")
F["mutations"] = rows
F["control"] = {"new_failed": na_f, "new_passed": na_p, "old_failed": ol_f, "old_passed": ol_p}
F["restored_identically"] = SRC.read_text() == src
save()
print("\nwrote", OUT)
