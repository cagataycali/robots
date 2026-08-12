"""Drive all four IotMqttTransport client-teardown paths; dump measured facts."""
from __future__ import annotations
import json, logging, pathlib, sys, tempfile

import strands_robots.mesh.transport.iot_transport as itmod
from strands_robots.mesh.transport.iot_transport import IotMqttTransport
import awsiot.mqtt5_client_builder as builder

TREE = str(pathlib.Path(itmod.__file__).parents[3])
print("TREE:", TREE)
LOG = "strands_robots.mesh.transport.iot_transport"
BOOM = "io thread already gone"


class Rec(logging.Handler):
    def __init__(self): super().__init__(); self.recs = []
    def emit(self, r): self.recs.append((r.levelname, r.getMessage()))


class Client:
    def __init__(self, connack=True, stop_raises=False, start_raises=False, **kw):
        self.kw, self.connack, self.stop_raises, self.start_raises = kw, connack, stop_raises, start_raises
        self.started = False; self.stop_attempts = 0
    def start(self):
        if self.start_raises: raise RuntimeError("io thread failed to launch")
        self.started = True
        if self.connack: self.kw["on_lifecycle_connection_success"](object())
    def stop(self):
        self.stop_attempts += 1
        if self.stop_raises: raise RuntimeError(BOOM)


def certs(tmp, thing="thor-arm"):
    d = pathlib.Path(tmp) / "iot"; d.mkdir(parents=True, exist_ok=True)
    for n in (f"{thing}.cert.pem", f"{thing}.private.key", "AmazonRootCA1.pem"):
        (d / n).write_text("x")
    return str(d)


def scenario(name, *, cfg, action):
    """Return the measured outcome of one teardown path."""
    h = Rec(); lg = logging.getLogger(LOG)
    lg.addHandler(h); lvl, prop = lg.level, lg.propagate
    lg.setLevel(logging.DEBUG); lg.propagate = False
    real = builder.mtls_from_path
    built = []
    def f(**kw):
        c = Client(**cfg, **kw); built.append(c); return c
    builder.mtls_from_path = f
    tmp = tempfile.mkdtemp()
    t = IotMqttTransport(thing_name="thor-arm", endpoint="x-ats.iot.us-west-2.amazonaws.com",
                         cert_dir=certs(tmp), connect_timeout=0.05)
    escaped = None
    try:
        action(t, h)
    except BaseException as e:
        escaped = f"{type(e).__name__}: {e}"
    finally:
        builder.mtls_from_path = real
        lg.removeHandler(h); lg.setLevel(lvl); lg.propagate = prop
    recorded = [f"{lv}: {m}" for lv, m in h.recs if BOOM in m]
    return {
        "site": name,
        "escaped": escaped,
        "client_left_set": t._client is not None,
        "stop_attempted": bool(built and built[0].stop_attempts),
        "recorded_level": recorded[0].split(":")[0] if recorded else None,
        "recorded": recorded[0] if recorded else None,
        "all_logs": [f"{lv}: {m}" for lv, m in h.recs],
    }


def a_reconnect(t, h):
    assert t.connect() is True
    t._connected.clear(); h.recs.clear()
    assert t.connect() is True

def a_construct(t, h):
    t.connect()

def a_timeout(t, h):
    t.connect()

def a_close(t, h):
    assert t.connect() is True
    h.recs.clear(); t.close()

rows = [
    scenario("connect(): reconnect stale client", cfg=dict(stop_raises=True), action=a_reconnect),
    scenario("connect(): construction failure", cfg=dict(stop_raises=True, start_raises=True), action=a_construct),
    scenario("connect(): CONNACK timeout", cfg=dict(stop_raises=True, connack=False), action=a_timeout),
    scenario("close(): public teardown", cfg=dict(stop_raises=True), action=a_close),
]
controls = [
    scenario("CONTROL connect(): clean timeout", cfg=dict(connack=False), action=a_timeout),
    scenario("CONTROL close(): clean teardown", cfg={}, action=a_close),
]
out = {"tree": TREE, "rows": rows, "controls": controls}
json.dump(out, open(f"/tmp/art-{pathlib.Path(TREE).name}.json", "w"), indent=2)
print(json.dumps([{k: r[k] for k in ("site","escaped","client_left_set","recorded_level")} for r in rows], indent=2))
print("CONTROLS:", json.dumps([{k: r[k] for k in ("site","escaped","recorded_level")} for r in controls]))
