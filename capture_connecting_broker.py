"""Measure IotMqttTransport(connect_timeout=...) against a CONNECTING broker."""
import json, pathlib, sys, threading, time

import strands_robots.mesh.transport.iot_transport as iot
TREE = str(pathlib.Path(iot.__file__).parents[3])

CONNACK_DELAY = 0.05
CERTS = pathlib.Path("/tmp/iot-art-certs"); CERTS.mkdir(exist_ok=True)
for n in ("thor-arm.cert.pem", "thor-arm.private.key", "AmazonRootCA1.pem"):
    (CERTS / n).write_text("x")

CASES = [
    ("15.0", 15.0), ("0.5", 0.5), ("0", 0), ("-1", -1.0), ("nan", float("nan")),
    ("inf", float("inf")), ("True", True), ("'15'", "15"), ("None", None),
]

def build(value):
    import awsiot.mqtt5_client_builder as builder
    built = []
    class C:
        def __init__(self, **kw): self.kw = kw; self.stopped = False
        def start(self):
            cb = self.kw["on_lifecycle_connection_success"]
            threading.Timer(CONNACK_DELAY, lambda: cb(object())).start()
        def stop(self): self.stopped = True
    def fake(**kw):
        c = C(**kw); built.append(c); return c
    builder.mtls_from_path = fake
    t = iot.IotMqttTransport(thing_name="thor-arm", endpoint="x-ats.iot.us-west-2.amazonaws.com",
                             cert_dir=str(CERTS), connect_timeout=value)
    return t, built

def silent(value):
    import awsiot.mqtt5_client_builder as builder
    built = []
    class C:
        def __init__(self, **kw): self.kw = kw; self.stopped = False
        def start(self): pass
        def stop(self): self.stopped = True
    def fake(**kw):
        c = C(**kw); built.append(c); return c
    builder.mtls_from_path = fake
    t = iot.IotMqttTransport(thing_name="thor-arm", endpoint="e", cert_dir=str(CERTS), connect_timeout=value)
    return t, built

rows = []
for label, value in CASES:
    row = {"label": label}
    try:
        t, built = build(value)
    except ValueError as exc:
        row.update(verdict="refused at construction", detail=str(exc).split(": ", 1)[1],
                   clients_started=0, leaked=False)
        rows.append(row); continue
    res = {}
    def run():
        t0 = time.perf_counter()
        try:
            res["ok"] = t.connect()
        except BaseException as exc:  # noqa: BLE001 - an escape IS the measurement
            res["exc"] = type(exc).__name__
        res["elapsed"] = time.perf_counter() - t0
    w = threading.Thread(target=run, daemon=True); w.start(); w.join(timeout=2.0)
    if w.is_alive():
        row.update(verdict="connect() never returned", detail="holds the instance lock; close() blocks",
                   clients_started=len(built), leaked=t._client is not None)
        t._connected.set(); w.join(timeout=2.0)
    elif "exc" in res:
        row.update(verdict=f"raised {res['exc']}", detail="out of a method documented to return bool",
                   clients_started=len(built), leaked=t._client is not None and not built[0].stopped)
    elif res["ok"]:
        row.update(verdict="connected", detail=f"CONNACK seen in {res['elapsed']*1000:.0f} ms",
                   clients_started=len(built), leaked=False)
    else:
        row.update(verdict="reported unreachable", detail=f"'timed out' after {res['elapsed']*1000:.1f} ms",
                   clients_started=len(built), leaked=False)
    rows.append(row)

# No-regression ledger: the two paths a usable budget must keep.
ledger = {}
t, built = build(0.5)
ledger["connecting_broker"] = {"connect": bool(t.connect()), "client_stopped": built[0].stopped}
t2, built2 = silent(0.05)
t0 = time.perf_counter()
ledger["silent_broker"] = {"connect": bool(t2.connect())}
ledger["silent_broker"]["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
ledger["silent_broker"]["client_stopped"] = built2[0].stopped
ledger["silent_broker"]["client_cleared"] = t2._client is None

print(json.dumps({"tree": TREE, "rows": rows, "ledger": ledger}))
