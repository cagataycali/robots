"""Measure, on whichever tree runs this, what an api_port reaches."""
import asyncio, json, pathlib, sys, urllib.error, urllib.request

import strands_robots.device_connect.reachy_transport as rt
import strands_robots.device_connect.reachy_mini_driver as rmd

TREE = str(pathlib.Path(rmd.__file__).parents[2])
HOST = "reachy-mini.local"
PATH = "/api/daemon/status"
CANDS = [("8000  (default)", 8000), ("9001", 9001), ("0", 0), ("-1", -1),
         ("99999", 99999), ("True", True), ("2.7", 2.7), ("None", None)]

def measure(port):
    row = {}
    try:
        drv = rmd.ReachyMiniDriver(host=HOST, api_port=port)
    except ValueError as e:
        return {"construct": "REFUSED", "reason": str(e), "rest": None, "ws": None, "variant": None}
    row["construct"] = "accepted"
    row["reason"] = None
    urls, links = [], []
    def spy(req, body=None, timeout=None, **kw):
        urls.append(req.full_url); raise urllib.error.URLError("[Errno 111] Connection refused")
    real = urllib.request.urlopen
    urllib.request.urlopen = spy
    z, w = rmd.ZenohLink, rmd.WebSocketLink
    class FZ:
        def __init__(s,*a,**k): links.append("Zenoh (Wireless)")
        async def start(s,**k): pass
    class FW:
        def __init__(s,h,p): links.append("WebSocket (Lite)")
        async def start(s,**k): pass
    rmd.ZenohLink, rmd.WebSocketLink = FZ, FW
    try:
        asyncio.run(drv.connect())
    finally:
        urllib.request.urlopen = real
        rmd.ZenohLink, rmd.WebSocketLink = z, w
    row["rest"] = urls[0] if urls else None
    row["variant"] = links[0] if links else None
    row["ws"] = f"{rt._ws_scheme()}://{HOST}:{port}/ws/sdk"
    row["api_result"] = rt.api(HOST, port, PATH) if False else None
    return row

out = {"tree": TREE, "rows": {}}
for label, v in CANDS:
    out["rows"][label] = measure(v)

# the indistinguishability claim, measured on this tree
real = urllib.request.urlopen
def refuse(req, body=None, timeout=None, **kw):
    raise urllib.error.URLError("[Errno 111] Connection refused")
urllib.request.urlopen = refuse
out["api_valid_down"] = rt.api(HOST, 8000, PATH)
urllib.request.urlopen = real
out["api_out_of_range"] = rt.api("127.0.0.1", 99999, PATH)

pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("TREE:", TREE)
