"""Measure the verdict of the target test file per tree x host load, plus the
fresh-socket connect cost the 2 ms budget had to cover."""
import json, os, subprocess, sys, time

NODE = "tests/test_zmq_timeout_ms_domain.py"
SPIN = ("import time,sys\n"
        "end=time.time()+%d\n"
        "sys.settrace(lambda *a: None)\n"
        "x=0\n"
        "while time.time()<end: x=(x*31+7)%%1000003\n")

def tree_of(cwd):
    r = subprocess.run([sys.executable, "-c",
                        "import strands_robots.utils as u, pathlib; print(pathlib.Path(u.__file__).parents[1])"],
                       cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()

def run_file(cwd):
    env = dict(os.environ, MUJOCO_GL="egl")
    r = subprocess.run([sys.executable, "-m", "pytest", NODE, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"],
                       cwd=cwd, capture_output=True, text=True, env=env)
    tail = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l]
    line = tail[-1] if tail else "<no summary>"
    failed = 0
    for tok, nxt in zip(line.split(), line.split()[1:]):
        if nxt.startswith("failed"):
            failed = int(tok)
    return {"failed": failed, "summary": line.strip("= ")}

def spinners(seconds, n=16):
    return [subprocess.Popen([sys.executable, "-c", SPIN % seconds],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]

def latency(n=60):
    import zmq, msgpack, threading, statistics
    ctx = zmq.Context(); rep = ctx.socket(zmq.REP)
    port = rep.bind_to_random_port("tcp://127.0.0.1"); stop = threading.Event()
    def serve():
        p = zmq.Poller(); p.register(rep, zmq.POLLIN)
        while not stop.is_set():
            if p.poll(50):
                rep.recv(); rep.send(msgpack.packb({"ok": True}, use_bin_type=True))
    threading.Thread(target=serve, daemon=True).start()
    out = []
    for _ in range(n):
        s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 5000); s.setsockopt(zmq.SNDTIMEO, 5000)
        s.connect(f"tcp://127.0.0.1:{port}")
        t0 = time.perf_counter(); s.send(msgpack.packb({"ping": 1}, use_bin_type=True)); s.recv()
        out.append((time.perf_counter() - t0) * 1000.0); s.close()
    stop.set(); rep.close(); ctx.term()
    return out

cwd, tag = sys.argv[1], sys.argv[2]
tree = tree_of(cwd)
print("TREE:", tree)
res = {"tag": tag, "tree": tree, "idle": [], "loaded": [], "lat_idle": [], "lat_loaded": []}
res["lat_idle"] = latency()
for _ in range(5):
    res["idle"].append(run_file(cwd))
procs = spinners(150)
time.sleep(3)
res["lat_loaded"] = latency()
for _ in range(5):
    res["loaded"].append(run_file(cwd))
for p in procs:
    p.kill()
json.dump(res, open(f"/tmp/art_{tag}.json", "w"), indent=1)
print(tag, "idle_failed:", [r["failed"] for r in res["idle"]], "loaded_failed:", [r["failed"] for r in res["loaded"]])
