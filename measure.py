"""Measure the silent auto-accept swallow + the stop/status refusal asymmetry."""
import json, logging, pathlib, sys, threading, tempfile, os
import strands_robots
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])

import strands_robots.tools.lerobot_teleoperate as tele
from strands_robots.tools.lerobot_teleoperate import lerobot_teleoperate, SessionManager

# isolate the session dir so nothing touches the real one
tmp = pathlib.Path(tempfile.mkdtemp())
tele.SESSION_DIR = tmp

def texts(r):
    return " ".join(c.get("text", "") for c in r.get("content", []) if "text" in c)

class _SyncThread:
    """Run the target inline so auto_respond is observable."""
    def __init__(self, target=None, daemon=None, **kw): self._t = target
    def start(self): self._t()

class _Stdin:
    def __init__(self, fail): self.fail = fail; self.writes = []
    def write(self, s):
        if self.fail: raise OSError("Broken pipe")
        self.writes.append(s)
    def flush(self): pass
    def close(self): pass

class _Proc:
    def __init__(self, pid, fail): self.pid = pid; self.stdin = _Stdin(fail); self.returncode = None
    def poll(self): return None

records = []
class _H(logging.Handler):
    def emit(self, rec): records.append((rec.levelname, rec.getMessage()))

out = {"tree": str(pathlib.Path(strands_robots.__file__).parents[1])}

for label, fail in (("auto-accept succeeds", False), ("auto-accept write FAILS", True)):
    records.clear()
    lg = logging.getLogger("strands_robots.tools.lerobot_teleoperate")
    h = _H(); lg.addHandler(h); lg.setLevel(logging.DEBUG)
    orig_thread, orig_popen, orig_sleep = threading.Thread, tele.subprocess.Popen, tele.time.sleep
    proc = _Proc(pid=os.getpid(), fail=fail)
    threading.Thread = _SyncThread
    tele.subprocess.Popen = lambda *a, **k: proc
    tele.time.sleep = lambda s: None
    try:
        SessionManager().remove_session("cal")
        r = lerobot_teleoperate(action="start", session_name="cal",
                                robot_type="so101_follower", teleop_type="so101_leader",
                                auto_accept_calibration=True, background=True)
        sess = SessionManager().get_session("cal")
        out[label] = {
            "status": r.get("status"),
            "text_head": texts(r)[:70],
            "enters_delivered": len(proc.stdin.writes),
            "records": [f"{lv}: {m[:60]}" for lv, m in records],
            "session_reports_running": sess is not None,
            "session_pid": (sess or {}).get("pid"),
        }
    finally:
        threading.Thread, tele.subprocess.Popen, tele.time.sleep = orig_thread, orig_popen, orig_sleep
        lg.removeHandler(h)
        SessionManager().remove_session("cal")

# the stop/status refusal asymmetry on a genuinely absent session
for act in ("stop", "status"):
    r = lerobot_teleoperate(action=act, session_name="gone-forever")
    out[f"{act} absent session"] = {"status": r.get("status"), "text": texts(r)}

# is L1109 reachable? plant a record with NO pid and see whether get_session returns it
SessionManager().add_session("nopid", {"start_time": 0.0, "action": "record"})
out["record with no pid survives _load_sessions"] = SessionManager().get_session("nopid") is not None
raw = json.loads((tmp / "active_sessions.json").read_text()) if (tmp / "active_sessions.json").exists() else {}
out["nopid on disk after load"] = list(raw)

print(json.dumps(out, indent=2, default=str))
