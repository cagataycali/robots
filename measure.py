"""Drive begin_registration across the first-enrollment scenarios and report the verdict.

Run once per tree (PYTHONPATH selects which strands_robots is imported).
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TREE = sys.argv[1]
OUT = sys.argv[2]

_ENVS = (
    "STRANDS_DASH_AUTH_ENABLED",
    "STRANDS_DASH_AUTH_RP_ID",
    "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN",
    "STRANDS_DASH_AUTH_ENROLL_TOKEN_FILE",
)
for k in _ENVS:
    os.environ.pop(k, None)

from fastapi import HTTPException  # noqa: E402

from strands_robots.dashboard import auth  # noqa: E402

print(f"[{TREE}] strands_robots.dashboard.auth from {auth.__file__}")

STRANGER = "203.0.113.9"


class FakeRequest:
    def __init__(self, headers=None, client_host="127.0.0.1"):
        self.headers = {"host": "localhost:8090", **(headers or {})}
        self.url = SimpleNamespace(scheme="http")
        self.client = None if client_host is None else type("C", (), {"host": client_host})()


# id, label, peer, headers, token_kind, env_token
SCENARIOS = [
    ("A", "remote attacker via socat TCP-LISTEN:8443,fork TCP:127.0.0.1:8090", "127.0.0.1", {}, "none", None),
    ("B", "same, forwarder bound on ::1", "::1", {}, "none", None),
    ("C", "same, peer reported as 'localhost'", "localhost", {}, "none", None),
    ("D", "L7 proxy emitting only x-client-ip (not on the roster)", "127.0.0.1", {"x-client-ip": STRANGER}, "none", None),
    ("E", "L7 proxy emitting only via (not on the roster)", "127.0.0.1", {"via": "1.1 squid"}, "none", None),
    ("F", "L7 proxy emitting x-forwarded-for (on the roster)", "127.0.0.1", {"x-forwarded-for": STRANGER}, "none", None),
    ("G", "direct connection from another machine", STRANGER, {}, "none", None),
    ("H", "connection with no socket peer at all", None, {}, "none", None),
    ("I", "operator at the machine, presenting the minted token", "127.0.0.1", {}, "file", None),
    ("J", "operator over ssh, presenting the token they read on the host", STRANGER, {}, "file", None),
    ("K", "guess from loopback", "127.0.0.1", {}, "wrong", None),
    ("L", "BOOTSTRAP_TOKEN configured, correct value from a remote peer", STRANGER, {}, "env", "let-me-in"),
    ("M", "BOOTSTRAP_TOKEN configured, nothing presented, from loopback", "127.0.0.1", {}, "none", "let-me-in"),
]

rows = []
for sid, label, peer, headers, kind, env_token in SCENARIOS:
    workdir = Path(tempfile.mkdtemp(prefix=f"enroll-{sid}-"))
    os.environ["STRANDS_DASH_AUTH_STORE"] = str(workdir / "auth.json")
    for k in _ENVS:
        os.environ.pop(k, None)
    if env_token:
        os.environ["STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"] = env_token
    auth._cache = {}
    auth._corrupt = None

    if kind == "none":
        bootstrap = ""
    elif kind == "env":
        bootstrap = env_token or ""
    elif kind == "wrong":
        bootstrap = "hunter2"
    else:  # "file" - the token this build mints, when it mints one at all
        mint = getattr(auth, "_local_enroll_token", None)
        bootstrap = mint() if mint else secrets.token_urlsafe(32)

    try:
        opts = auth.begin_registration(FakeRequest(headers, peer), label="probe", bootstrap=bootstrap)
        verdict = "ADMITTED" if opts.get("challenge_id") else "ADMITTED?"
        detail = f"challenge_id issued, user_id {auth._load().get('user_id', '')[:12]}..."
    except HTTPException as e:
        verdict = f"{e.status_code}"
        detail = str(e.detail)
    rows.append(
        {
            "id": sid,
            "label": label,
            "peer": peer,
            "headers": headers,
            "token": kind,
            "env_token": bool(env_token),
            "verdict": verdict,
            "detail": detail,
            "token_file_exists": (Path(os.environ["STRANDS_DASH_AUTH_STORE"]).with_name("enroll_token")).exists(),
        }
    )
    print(f"[{TREE}] {sid} {verdict:9s} {label}")
    shutil.rmtree(workdir, ignore_errors=True)

# Does this build read the bootstrap value at all when no env token is set?
workdir = Path(tempfile.mkdtemp(prefix="ignored-"))
os.environ["STRANDS_DASH_AUTH_STORE"] = str(workdir / "auth.json")
for k in _ENVS:
    os.environ.pop(k, None)
auth._cache = {}
auth._corrupt = None
try:
    auth.begin_registration(FakeRequest(), label="probe", bootstrap="a-value-nobody-configured")
    ignored = "ADMITTED"
except HTTPException as e:
    ignored = str(e.status_code)
shutil.rmtree(workdir, ignore_errors=True)

Path(OUT).write_text(
    json.dumps(
        {"tree": TREE, "auth_file": auth.__file__, "rows": rows, "unconfigured_value_verdict": ignored}, indent=2
    )
)
print(f"[{TREE}] arbitrary value with no env token configured -> {ignored}")
print(f"[{TREE}] wrote {OUT}")
