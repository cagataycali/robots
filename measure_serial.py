"""Measure serial_tool against a real pty: what reaches the wire, and what it reports."""
import json, os, sys, threading, time
from pathlib import Path
import strands_robots.tools.serial_tool as mod

TREE = str(Path(mod.__file__).parents[2])
print("TREE:", TREE)


def newpty():
    m, s = os.openpty()
    return m, os.ttyname(s)


def feed(master, delay=0.15, payload=b"\xaa\xbb\xcc\xdd"):
    def _w():
        time.sleep(delay)
        try:
            os.write(master, payload)
        except OSError:
            pass
    threading.Thread(target=_w, daemon=True).start()


def run(label, want_feed=False, **kw):
    m, dev = newpty()
    if want_feed:
        feed(m)
    t0 = time.time()
    res = mod.serial_tool(port=dev, **kw)
    dt = time.time() - t0
    try:
        os.set_blocking(m, False)
        wire = os.read(m, 64)
    except Exception:
        wire = b""
    os.close(m)
    text = " ".join(b.get("text", "") for b in res["content"]).strip().split("\n")[0]
    return {"label": label, "status": res["status"], "text": text,
            "bytes": wire.hex().upper(), "secs": round(dt, 3)}


out = {"tree": TREE, "positions": [], "options": []}
for value in [0, 2048, 4095, 5000, 70000, -1, 65536]:
    r = run(str(value), action="feetech_position", motor_id=1, position=value)
    pkt = bytes.fromhex(r["bytes"]) if r["bytes"] else b""
    r["requested"] = value
    r["wire"] = (pkt[6] | (pkt[7] << 8)) if len(pkt) >= 8 else None
    out["positions"].append(r)

for label, feed_it, kw in [
    ("motor_id=255", False, dict(action="feetech_position", motor_id=255, position=100)),
    ("motor_id=True", False, dict(action="feetech_position", motor_id=True, position=100)),
    ("velocity=70000", False, dict(action="feetech_velocity", motor_id=1, velocity=70000)),
    ("read_bytes=4", True, dict(action="read", timeout=1.0, read_bytes=4)),
    ("read_bytes=0", True, dict(action="read", timeout=1.0, read_bytes=0)),
    ("read_bytes=2.7", True, dict(action="read", timeout=1.0, read_bytes=2.7)),
    ("timeout=nan", True, dict(action="read", timeout=float("nan"), read_bytes=4)),
    ("timeout=inf", True, dict(action="read", timeout=float("inf"), read_bytes=4)),
    ("baudrate=2.7", True, dict(action="read", baudrate=2.7, timeout=1.0, read_bytes=4)),
]:
    out["options"].append(run(label, want_feed=feed_it, **kw))

Path(sys.argv[1]).write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote", sys.argv[1])
