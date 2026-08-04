"""Measure robot_mesh's four numeric options. Run in each tree; prints its tree."""
from __future__ import annotations
import json, sys, threading, time
from pathlib import Path
from unittest.mock import MagicMock, patch

import strands_robots.tools.robot_mesh as rm
TREE = str(Path(rm.__file__).parents[2])
from strands_robots.tools.robot_mesh import robot_mesh

# One set, unusable for every one of the four options. ``0`` and ``None`` are
# excluded because they are legitimate for duration (documented [0, MAX_DURATION_S])
# and policy_port (None = omitted), so they would not make the columns comparable.
PROBES = [("-1", -1.0), ("nan", float("nan")), ("inf", float("inf")),
          ("True", True), ("'30'", "30")]
LIMIT_PROBES = PROBES


class Wire:
    """Records the budget handed to the transport; never waits."""
    def __init__(self):
        self.peer_id, self.peer_type, self.inbox, self.got = "local-a", "sim", {}, []
    def send(self, target, cmd, timeout=30.0):
        self.got.append(timeout); return {"status": "ok"}
    def broadcast(self, cmd, timeout=5.0):
        self.got.append(timeout); return []
    def tell(self, target, instruction, **kw):
        self.got.append(kw); return {"status": "ok"}


def call(mesh, **kw):
    c = MagicMock(); c.interrupt.return_value = "y"
    fn = getattr(robot_mesh, "original", None) or robot_mesh
    with patch("strands_robots.mesh.get_local_robots", return_value={"local-a": mesh}), \
         patch("strands_robots.mesh.session.get_peers", return_value=[]):
        return fn(tool_context=c, **kw)


def verdict(mesh, **kw):
    """bounded  = the call is refused naming the option
       UNBOUNDED = accepted, or escaped with a raw internal"""
    try:
        out = call(mesh, **kw)
    except BaseException as e:  # noqa: BLE001 - an escape past the envelope IS the finding
        return "raised", f"{type(e).__name__}"
    txt = out["content"][0]["text"]
    if out["status"] == "error" and ("timeout must be" in txt or "limit must be" in txt
                                     or "out of bounds" in txt or "must be finite" in txt
                                     or "must be a number" in txt or "must be an integer" in txt):
        return "bounded", txt[:60]
    if out["status"] == "error":
        return "leaked", txt[:60]
    return "accepted", txt[:60]


facts = {"tree": TREE, "matrix": {}, "stop_wire": {}, "inbox_cap": {}}

for label, v in PROBES:
    facts["matrix"].setdefault(label, {})
    facts["matrix"][label]["duration"] = verdict(Wire(), action="tell", target="p", instruction="go", duration=v)
    facts["matrix"][label]["policy_port"] = verdict(Wire(), action="tell", target="p", instruction="go", policy_port=v)
    facts["matrix"][label]["timeout"] = verdict(Wire(), action="send", target="p", command='{"action":"status"}', timeout=v)

for label, v in LIMIT_PROBES:
    m = Wire(); m.inbox = {"s": [("strands/peer/stream", {"i": i}) for i in range(120)]}
    facts["matrix"].setdefault(label, {})
    facts["matrix"][label]["limit"] = verdict(m, action="inbox", name="s", limit=v)

# stop: what budget reached the transport (the min(timeout, 5.0) cap)
for label, v in [("30.0", 30.0), ("2.5", 2.5), ("nan", float("nan")), ("True", True), ("-1", -1.0)]:
    m = Wire()
    out = None
    try:
        out = call(m, action="stop", target="p", timeout=v)
        st = out["status"]
    except BaseException as e:  # noqa: BLE001
        st = f"raised {type(e).__name__}"
    facts["stop_wire"][label] = {"status": st, "on_wire": [repr(x) for x in m.got]}

# inbox: how many of 120 messages came back
for label, v in [("50", 50), ("5", 5), ("0", 0), ("-5", -5), ("nan", float("nan"))]:
    m = Wire(); m.inbox = {"s": [("strands/peer/stream", {"i": i}) for i in range(120)]}
    try:
        out = call(m, action="inbox", name="s", limit=v)
        txt = out["content"][0]["text"]
        n = txt.count("strands/peer/stream") if out["status"] == "success" else -1
        facts["inbox_cap"][label] = {"status": out["status"], "returned": n, "text": txt.splitlines()[0][:70]}
    except BaseException as e:  # noqa: BLE001
        facts["inbox_cap"][label] = {"status": "raised", "returned": -1, "text": f"{type(e).__name__}: {e}"[:70]}

Path(sys.argv[1]).write_text(json.dumps(facts, indent=1))
print("TREE:", TREE, "-> wrote", sys.argv[1])
