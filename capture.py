"""Measure the three degradation branches: coverage state, mutation matrix, behaviour."""
import ast, asyncio, json, pathlib, socket, subprocess, sys
from unittest.mock import AsyncMock, MagicMock, patch

import strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
REL = "strands_robots/device_connect/reachy_transport.py"
SRC = ROOT / REL
TESTFILE = "tests/test_reachy_transport_links.py"
NEW = ("test_malformed_imu_frame_is_dropped_and_subscription_survives or "
       "test_a_raising_consumer_does_not_kill_either_subscription or "
       "test_modern_connect_receives_the_additional_headers_keyword or "
       "test_authorization_survives_when_the_connect_signature_is_unreadable or "
       "test_a_resolvable_name_is_translated_to_its_address or "
       "test_an_unresolvable_name_is_returned_verbatim")

facts = {"tree": str(ROOT)}

# ---------- coverage, from the two authoritative full-suite runs ----------
def cov_of(path):
    d = json.load(open(path))["files"][REL]
    return {"stmts": d["summary"]["num_statements"], "missing": d["summary"]["missing_lines"],
            "pct": round(d["summary"]["percent_covered"], 1), "missing_lines": d["missing_lines"]}
facts["cov_before"] = cov_of("/tmp/cov-31440126023.json")
facts["cov_after"] = cov_of("/tmp/cov-after-31440126023.json")

# ---------- behaviour of each branch (measured against this tree) ----------
from strands_robots.device_connect.reachy_transport import WebSocketLink, ZenohLink

def _imu_behaviour():
    tr = MagicMock(); tr.subscribe = AsyncMock(); seen = []
    asyncio.run(ZenohLink(tr, prefix="p").start(on_joints=lambda d: None, on_imu=seen.append))
    cb = tr.subscribe.call_args_list[1].args[1]
    asyncio.run(cb(b"not-json{")); after_bad = list(seen)
    asyncio.run(cb(json.dumps({"accel": [0, 0, 9.8]}).encode()))
    return {"after_malformed": after_bad, "after_good": list(seen)}
facts["imu"] = _imu_behaviour()

class _FakeWS:
    def __aiter__(self):
        async def g():
            if False:
                yield None
        return g()
    async def close(self): pass

def _ws_headers(readable_signature: bool):
    import inspect, os
    os.environ["REACHY_DAEMON_TOKEN"] = "secret-token"
    seen = {}
    async def _connect(url, *, additional_headers=None, extra_headers=None, ssl=None):
        seen["additional_headers"] = additional_headers
        seen["extra_headers"] = extra_headers
        return _FakeWS()
    fw = MagicMock(); fw.connect = _connect
    real = inspect.signature
    def bad(obj, *a, **k):
        if obj is _connect:
            raise ValueError("no signature found for builtin <built-in function connect>")
        return real(obj, *a, **k)
    async def go():
        cms = [patch.dict(sys.modules, {"websockets": fw})]
        if not readable_signature:
            cms.append(patch("inspect.signature", bad))
        for cm in cms: cm.__enter__()
        try:
            link = WebSocketLink("h", 1)
            await link.start(on_joints=lambda m: None, on_imu=lambda m: None)
            await link.stop()
        finally:
            for cm in reversed(cms): cm.__exit__(None, None, None)
    asyncio.run(go())
    os.environ.pop("REACHY_DAEMON_TOKEN", None)
    return seen
facts["ws_readable"] = _ws_headers(True)
facts["ws_unreadable"] = _ws_headers(False)

def _host_behaviour():
    out = {}
    with patch("socket.gethostbyname", return_value="10.1.2.3"):
        out["resolvable"] = strands_robots.device_connect.reachy_transport.resolve_host("reachy-mini.local")
    with patch("socket.gethostbyname", side_effect=socket.gaierror(-2, "Name or service not known")):
        out["unresolvable"] = strands_robots.device_connect.reachy_transport.resolve_host("reachy-mini.local")
    return out
import strands_robots.device_connect.reachy_transport  # noqa: E402
facts["host"] = _host_behaviour()

# ---------- mutation matrix ----------
IMU_OLD = '''        async def _on_imu(data: bytes, _reply=None):
            try:
                on_imu(json.loads(data.decode()))
            except Exception:
                pass  # drop malformed/partial frame; keep the subscription alive
'''
RH_OLD = '''    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return host
'''
FB_OLD = '''            except (ValueError, TypeError):
                _connect_kwargs["extra_headers"] = _extra_headers
'''
MUTATIONS = [
    ("M1", "drop the IMU malformed-frame tolerance", "start", IMU_OLD,
     '        async def _on_imu(data: bytes, _reply=None):\n            on_imu(json.loads(data.decode()))\n'),
    ("M2", "IMU wrapper forwards to the joints consumer", "start", IMU_OLD,
     IMU_OLD.replace("on_imu(json.loads", "on_joints(json.loads")),
    ("M3", "drop the resolve_host lookup-failure fallback", "resolve_host", RH_OLD,
     '    return socket.gethostbyname(host)\n'),
    ("M4", "header fallback drops the headers (unauthenticated)", "start", FB_OLD,
     '            except (ValueError, TypeError):\n                pass\n'),
    ("M5", "header fallback uses the modern keyword on legacy", "start", FB_OLD,
     '            except (ValueError, TypeError):\n                _connect_kwargs["additional_headers"] = _extra_headers\n'),
]

def run(kexpr):
    r = subprocess.run([sys.executable, "-m", "pytest", TESTFILE, "-q", "--no-cov",
                        "-p", "no:randomly", "-k", kexpr], cwd=ROOT, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l]
    txt = line[-1] if line else ""
    return {"failed": " failed" in txt, "n_failed": int(txt.split(" failed")[0].split()[-1]) if " failed" in txt else 0}

original = SRC.read_text()
rows = []
try:
    for mid, label, fnname, old, new in MUTATIONS:
        lines = original.splitlines(keepends=True)
        fns = [n for n in ast.walk(ast.parse(original))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fnname
               and old in "".join(lines[n.lineno - 1:n.end_lineno])]
        assert len(fns) == 1, (mid, len(fns))
        fn = fns[0]
        region = "".join(lines[fn.lineno - 1:fn.end_lineno])
        assert region.count(old) == 1
        mutated = "".join(lines[:fn.lineno - 1]) + region.replace(old, new, 1) + "".join(lines[fn.end_lineno:])
        ast.parse(mutated)
        SRC.write_text(mutated)
        rows.append({"id": mid, "label": label, "new": run(NEW), "old": run(f"not ({NEW})")})
finally:
    SRC.write_text(original)
    assert SRC.read_text() == original, "RESTORE FAILED"
facts["mutations"] = rows

pathlib.Path("/tmp/art_facts.json").write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
