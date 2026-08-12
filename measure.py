"""Measure the four client.stop() call sites in IotMqttTransport."""
from __future__ import annotations
import logging, pathlib, sys
from typing import Any

import strands_robots.mesh.transport.iot_transport as itmod
print("TREE:", pathlib.Path(itmod.__file__).parents[3])
from strands_robots.mesh.transport.iot_transport import IotMqttTransport
import awsiot.mqtt5_client_builder as builder


class _Rec(logging.Handler):
    def __init__(self): super().__init__(); self.msgs=[]
    def emit(self, r): self.msgs.append(f"{r.levelname}:{r.getMessage()}")


def certs(tmp: pathlib.Path, thing="thor-arm"):
    d = tmp / "iot"; d.mkdir(parents=True, exist_ok=True)
    (d/f"{thing}.cert.pem").write_text("cert")
    (d/f"{thing}.private.key").write_text("key")
    (d/"AmazonRootCA1.pem").write_text("ca")
    return str(d)


def transport(tmp, timeout=0.05):
    return IotMqttTransport(thing_name="thor-arm",
        endpoint="x-ats.iot.us-west-2.amazonaws.com",
        cert_dir=certs(tmp), connect_timeout=timeout)


class Client:
    """Fake mqtt5 client. connack=False -> never fires CONNACK (timeout)."""
    def __init__(self, connack=True, stop_raises=False, **kw):
        self.kw=kw; self.connack=connack; self.stop_raises=stop_raises
        self.started=False; self.stop_calls=0
    def start(self):
        self.started=True
        if self.connack: self.kw["on_lifecycle_connection_success"](object())
    def stop(self):
        self.stop_calls += 1
        if self.stop_raises: raise RuntimeError("io thread already gone")


def install(monkey_target, **cfg):
    built=[]
    def f(**kw):
        c=Client(**cfg, **kw); built.append(c); return c
    builder.mtls_from_path = f
    return built


import tempfile, json
out={}
real = builder.mtls_from_path

def run(name, fn):
    h=_Rec(); lg=logging.getLogger("strands_robots.mesh.transport.iot_transport")
    lg.addHandler(h); old=lg.level; lg.setLevel(logging.DEBUG); prop=lg.propagate; lg.propagate=False
    try: res=fn(h)
    except BaseException as e: res={"raised": f"{type(e).__name__}: {e}"}
    finally:
        lg.removeHandler(h); lg.setLevel(old); lg.propagate=prop; builder.mtls_from_path=real
    res.setdefault("logs", h.msgs)
    out[name]=res
    print(f"\n### {name}"); print(json.dumps(res, indent=2, default=str))


# --- 1. connect() TIMEOUT, stop() raises  (L454, UNWRAPPED) ---
def c1(h):
    tmp=pathlib.Path(tempfile.mkdtemp())
    built=install(builder, connack=False, stop_raises=True)
    t=transport(tmp)
    try:
        r=t.connect(); esc=None
    except BaseException as e:
        r="<did not return>"; esc=f"{type(e).__name__}: {e}"
    return {"connect_returned": r, "escaped": esc,
            "client_left_set": t._client is not None,
            "client_stop_calls": built[0].stop_calls if built else None,
            "client_started_still": built[0].started if built else None,
            "is_alive": t.is_alive()}
run("A_connect_timeout_stop_raises", c1)

# --- 2. connect() TIMEOUT, stop() OK (control) ---
def c2(h):
    tmp=pathlib.Path(tempfile.mkdtemp())
    built=install(builder, connack=False, stop_raises=False)
    t=transport(tmp)
    r=t.connect()
    return {"connect_returned": r, "client_left_set": t._client is not None,
            "client_stop_calls": built[0].stop_calls}
run("B_connect_timeout_stop_ok_CONTROL", c2)

# --- 3. close(), stop() raises  (L472-473, SILENT swallow) ---
def c3(h):
    tmp=pathlib.Path(tempfile.mkdtemp())
    built=install(builder, connack=True, stop_raises=True)
    t=transport(tmp); assert t.connect() is True
    h.msgs.clear()
    try: t.close(); esc=None
    except BaseException as e: esc=f"{type(e).__name__}: {e}"
    return {"escaped": esc, "client_cleared": t._client is None,
            "handlers_cleared": len(t._handlers)==0, "is_alive": t.is_alive()}
run("C_close_stop_raises", c3)

# --- 4. reconnect stale stop() raises (L379-380, LOGS) ---
def c4(h):
    tmp=pathlib.Path(tempfile.mkdtemp())
    built=install(builder, connack=True, stop_raises=True)
    t=transport(tmp); assert t.connect() is True
    t._connected.clear()          # broker drop
    h.msgs.clear()
    r=t.connect()
    return {"connect_returned": r, "clients_built": len(built)}
run("D_reconnect_stale_stop_raises_SIBLING", c4)

# --- 5. construction failure stop() raises (L438/444, LOGS) ---
def c5(h):
    tmp=pathlib.Path(tempfile.mkdtemp())
    class Boom(Client):
        def start(self): raise RuntimeError("io thread failed to launch")
    built=[]
    def f(**kw):
        c=Boom(stop_raises=True, **kw); built.append(c); return c
    builder.mtls_from_path=f
    t=transport(tmp)
    h.msgs.clear()
    r=t.connect()
    return {"connect_returned": r, "client_cleared": t._client is None}
run("E_construct_fail_stop_raises_SIBLING", c5)

json.dump(out, open(f"/tmp/measure-{sys.argv[1]}.json","w"), indent=2, default=str)
print("\n=== VERDICT TABLE ===")
for k,v in out.items():
    esc = v.get("escaped") or v.get("raised")
    print(f"  {k:42s} escaped={str(esc)[:44]:46s} logs={len(v.get('logs',[]))}")
