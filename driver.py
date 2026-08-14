"""Run the zenoh-free resume sequence end to end and report the fleet state.

Printed as one JSON line so the capture script can read it back per tree arm.
"""
import json, pathlib, sys, types
from unittest.mock import MagicMock

import strands_robots.mesh.core as core

print("TREE:" + str(pathlib.Path(core.__file__).parents[2]), file=sys.stderr)

RESUME_KEY = "strands/safety/resume"


class _Pub:
    def __init__(self):
        self.puts = []

    def put(self, payload, **kw):
        self.puts.append((payload, kw))


def _hide_zenoh():
    sys.modules["zenoh"] = None  # `import zenoh` then raises ImportError


def _sample(envelope):
    return types.SimpleNamespace(
        payload=types.SimpleNamespace(to_bytes=lambda: json.dumps(envelope).encode()),
        source_info=None,
    )


import os

os.environ["STRANDS_MESH_OVERRIDE_CODE"] = "operator-secret"
_hide_zenoh()

issuer = core.Mesh(robot=object(), peer_id="issuer")
issuer.publish_safety_event = MagicMock()
# A session IS open and a publisher IS declarable: the import failure is the
# only thing that rules the native SourceInfo path out.
issuer._local_session_zid = lambda: "deadbeefdeadbeef"
issuer._safety_publisher_for = lambda key: _Pub()

published = {}
core.put = lambda key, payload: published.update(key=key, payload=payload)

issuer._estop_lockout.set()
issuer._last_estop_ts = core.time.time()
resume_status = issuer._resume_lockout("operator-secret")

env = published.get("payload", {})
receiver = core.Mesh(robot=object(), peer_id="receiver")
receiver.publish_safety_event = MagicMock()
receiver._estop_lockout.set()
locked_before = receiver._estop_lockout.is_set()
try:
    receiver._on_safety_resume(_sample(env))
    verify_error = None
except Exception as exc:  # noqa: BLE001 - the arm's outcome is the measurement
    verify_error = f"{type(exc).__name__}: {exc}"
locked_after = receiver._estop_lockout.is_set()

print(
    json.dumps(
        {
            "wire_zid": issuer._safety_wire_zid(RESUME_KEY),
            "published_key": published.get("key"),
            "body_carries_source_zid": "source_zid" in env,
            "body_carries_proof": "override_proof" in env,
            "resume_status": resume_status,
            "receiver_locked_before": locked_before,
            "receiver_locked_after": locked_after,
            "fleet_available": locked_after is False,
            "verify_error": verify_error,
        }
    )
)
