"""Measure both auto-accept outcomes on whichever tree this runs in."""
import json, logging, os, pathlib, sys, threading

import strands_robots.tools.lerobot_teleoperate as tele
from strands_robots.tools.lerobot_teleoperate import SessionManager, lerobot_teleoperate

TREE = str(pathlib.Path(tele.__file__).resolve().parents[2])
print("TREE:", TREE)
OUT = pathlib.Path(sys.argv[1])
LOGGER = "strands_robots.tools.lerobot_teleoperate"


class _Stdin:
    def __init__(self, fail): self.fail, self.writes, self.closed = fail, [], False
    def write(self, d):
        if self.fail: raise OSError("Broken pipe")
        self.writes.append(d)
    def flush(self): pass
    def close(self): self.closed = True


class _Proc:
    def __init__(self, fail): self.pid, self.returncode, self.stdin = os.getpid(), None, _Stdin(fail)
    def poll(self): return None


class _Sync:
    def __init__(self, target=None, daemon=None, **kw): self._t = target
    def start(self):
        if self._t: self._t()


class _Cap(logging.Handler):
    def __init__(self): super().__init__(logging.DEBUG); self.records = []
    def emit(self, r): self.records.append((r.levelname, r.getMessage()))


def scenario(fail, tmp):
    session_dir = tmp / f"s-{fail}"
    session_dir.mkdir(parents=True, exist_ok=True)
    tele.SESSION_DIR = session_dir
    proc = _Proc(fail)
    real_thread, real_popen, real_sleep = threading.Thread, tele.subprocess.Popen, tele.time.sleep
    threading.Thread = _Sync
    tele.subprocess.Popen = lambda *a, **k: proc
    tele.time.sleep = lambda s: None
    log = logging.getLogger(LOGGER)
    cap = _Cap(); log.addHandler(cap); prev = log.level; log.setLevel(logging.DEBUG)
    try:
        res = lerobot_teleoperate(action="start", session_name="cal", robot_type="so101_follower",
                                  teleop_type="so101_leader", auto_accept_calibration=True, background=True)
    finally:
        threading.Thread, tele.subprocess.Popen, tele.time.sleep = real_thread, real_popen, real_sleep
        log.removeHandler(cap); log.setLevel(prev)
    text = "\n".join(i.get("text", "") for i in res.get("content", []) if "text" in i)
    recs = [(lv, m) for lv, m in cap.records if "auto-accept" in m]
    return {
        "status": res.get("status"),
        "says_started": "Session Started" in text,
        "store_live": SessionManager().get_session("cal") is not None,
        "newlines_written": len(proc.stdin.writes),
        "record": recs[0][1] if recs else None,
        "level": recs[0][0] if recs else None,
    }


tmp = pathlib.Path(f"/tmp/art-tele-{os.environ['GITHUB_RUN_ID']}-{abs(hash(TREE)) % 10000}")
facts = {"tree": TREE, "fail": scenario(True, tmp), "ok": scenario(False, tmp)}
OUT.write_text(json.dumps(facts, indent=2))
print(json.dumps(facts, indent=2))
