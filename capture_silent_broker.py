"""Same knob, but against a broker that never reports CONNACK (genuinely unreachable)."""
import json, pathlib, threading, time
import strands_robots.mesh.transport.iot_transport as iot
TREE = str(pathlib.Path(iot.__file__).parents[3])
CERTS = pathlib.Path("/tmp/iot-art-certs")
# Usable budgets are kept under the 2.5 s join below so a bounded wait cannot be
# mistaken for a hang; 15.0 (the default) is bounded too, just longer than the join.
CASES = [("0.5", 0.5), ("1.2", 1.2), ("0", 0), ("nan", float("nan")), ("True", True), ("None", None)]

def make(value):
    import awsiot.mqtt5_client_builder as builder
    built = []
    class C:
        def __init__(self, **kw): self.stopped = False
        def start(self): pass
        def stop(self): self.stopped = True
    builder.mtls_from_path = lambda **kw: built.append(C(**kw)) or built[-1]
    t = iot.IotMqttTransport(thing_name="thor-arm", endpoint="e", cert_dir=str(CERTS), connect_timeout=value)
    return t, built

rows = []
for label, value in CASES:
    row = {"label": label}
    try:
        t, built = make(value)
    except ValueError:
        row.update(verdict="refused at construction", waited_ms=None, lock_held=False)
        rows.append(row); continue
    res = {}
    def run():
        t0 = time.perf_counter()
        try: res["ok"] = t.connect()
        except BaseException as e: res["exc"] = type(e).__name__
        res["ms"] = (time.perf_counter() - t0) * 1000
    w = threading.Thread(target=run, daemon=True); w.start(); w.join(timeout=2.5)
    if w.is_alive():
        row.update(verdict="never returns", waited_ms=None,
                   lock_held=not t._lock.acquire(blocking=False))
        t._connected.set(); w.join(timeout=2.0)
    elif "exc" in res:
        row.update(verdict=f"raised {res['exc']}", waited_ms=round(res["ms"], 1), lock_held=False)
    else:
        row.update(verdict=f"returned {res['ok']}", waited_ms=round(res["ms"], 1), lock_held=False)
    rows.append(row)
print(json.dumps({"tree": TREE, "rows": rows}))
