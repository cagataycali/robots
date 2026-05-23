"""ACL config builder for the strands-robots mesh.

Reads a JSON5 ACL file at ``STRANDS_MESH_ACL_FILE`` and returns the
serialised ``access_control`` block ready for
``zenoh.Config.insert_json5``. When the env var is unset, returns the
permissive :func:`default_acl` skeleton.

Zenoh 1.x quirks (each verified against a live session in
``tests/mesh/test_redteam_zenoh.py``):

* ``enabled: true`` is required — without it the entire block is a
  no-op even if rules and subjects are populated.
* ``cert_common_names`` matches LITERAL CNs only; globs and regexes
  match nothing. Operators tighten the default by enumerating each
  peer's exact cert CN in ``STRANDS_MESH_ACL_FILE``.
* Subject ``interfaces`` must be a non-empty list — leaving it unset
  causes the subject to match nothing.
* ``key_exprs`` match the user-side key (the namespace prefix is
  stripped from the matcher's view), so ``**/cmd`` is the robust
  glob; ``"<namespace>/*/cmd"`` never matches.
* ``declare_subscriber`` rules live in the ``egress`` flow (the
  declare goes from subscriber to publisher); ``put`` rules live in
  ``ingress`` (the publisher's cert CN is known to the receiver).

JSON5 is the on-disk format (line + block comments, trailing commas,
unquoted keys). The loader uses a stdlib JSON5-lite preprocessor —
no third-party dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from strands_robots.mesh._zenoh_config import _local_interfaces

logger = logging.getLogger(__name__)

#: Maximum bytes of an ACL file we will load. Anything larger is almost
#: certainly an attacker probing for an OOM.
ACL_FILE_MAX_BYTES: int = 256 * 1024


# ─── JSON5-lite preprocessor ───────────────────────────────────────────


def _strip_json5_comments(raw: str) -> str:
    """Strip ``//`` line comments and ``/* */`` block comments.

    Preserves string contents (an inline ``//`` inside a JSON string is
    left verbatim).
    """
    out: list[str] = []
    i = 0
    n = len(raw)
    in_string = False
    string_quote = ""
    while i < n:
        ch = raw[i]
        nxt = raw[i + 1] if i + 1 < n else ""
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if ch == string_quote:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and raw[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i < n - 1 and not (raw[i] == "*" and raw[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_trailing_commas(raw: str) -> str:
    """Remove trailing commas before ``}`` / ``]``."""
    out: list[str] = []
    i = 0
    n = len(raw)
    in_string = False
    string_quote = ""
    while i < n:
        ch = raw[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == string_quote:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and raw[j] in " \t\r\n":
                j += 1
            if j < n and raw[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:")


def _quote_unquoted_keys(raw: str) -> str:
    """Wrap unquoted object keys in double quotes."""
    out: list[str] = []
    i = 0
    n = len(raw)
    in_string = False
    string_quote = ""
    while i < n:
        ch = raw[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == string_quote:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue
        m = _KEY_RE.match(raw, i)
        if m and (not out or out[-1] in "{,\n \t"):
            out.append('"' + m.group(1) + '"' + ":")
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _json5_to_json(raw: str) -> str:
    """Apply our JSON5-lite preprocessor to *raw*."""
    return _strip_trailing_commas(_quote_unquoted_keys(_strip_json5_comments(raw)))


# ─── ACL file loader ───────────────────────────────────────────────────


def _load_acl_file(path: Path) -> dict[str, Any]:
    """Load and validate an ACL file.

    Refuses any file that omits ``enabled: true`` — Zenoh silently
    no-ops the block in that case, and the loader fails closed rather
    than ship a quietly-disabled gate.
    """
    if not path.is_file():
        raise FileNotFoundError(f"ACL file not found: {path}")
    size = path.stat().st_size
    if size > ACL_FILE_MAX_BYTES:
        raise ValueError(f"ACL file {path} is {size} bytes; refusing to load >{ACL_FILE_MAX_BYTES}")
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(_json5_to_json(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"ACL file {path} is not valid JSON5: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"ACL file {path} root must be an object")
    for required in ("default_permission", "rules", "subjects", "policies"):
        if required not in data:
            raise ValueError(f"ACL file {path} missing required field: {required!r}")
    if not data.get("enabled", False):
        raise ValueError(
            f"ACL file {path} must set ``enabled: true`` — without it Zenoh silently disables the access_control block."
        )
    if data["default_permission"] not in ("allow", "deny"):
        raise ValueError(f"ACL file {path} default_permission={data['default_permission']!r} must be 'allow' or 'deny'")
    if data["default_permission"] == "allow":
        logger.warning(
            "[acl] %s uses default_permission='allow' — this is a blacklist "
            "policy and any rule gap exposes the mesh. Prefer 'deny'.",
            path,
        )
    return data


# ─── Default ACL ───────────────────────────────────────────────────────


def default_acl(namespace: str) -> dict[str, Any]:
    """Return a permissive default ACL skeleton.

    The default allows any peer with a valid CA-signed cert (verified
    at the mTLS handshake) to publish and subscribe on any key.
    Operators who want per-role enforcement supply their own ACL via
    ``STRANDS_MESH_ACL_FILE`` enumerating each peer's exact cert CN
    (Zenoh 1.x cert_common_names does not support globs — see
    ``examples/mesh_acl_example.json5`` for the canonical template).

    Why permissive default rather than default-deny: a default-deny
    skeleton with no enumerated subjects rejects every legitimate
    message — silent total outage on first run. The mTLS handshake at
    the link layer already gates fleet membership; the application-
    layer ``validate_command`` gates payload semantics. ACL is the
    third line of defence and operators opt in explicitly.
    """
    _ = namespace  # `namespace` config does the routing isolation; ACL key_exprs do not need it
    return {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            {
                "id": "any_subscribe",
                "messages": ["declare_subscriber"],
                "flows": ["egress"],
                "permission": "allow",
                "key_exprs": ["**"],
            },
            {
                "id": "any_publish",
                "messages": ["put"],
                "flows": ["ingress", "egress"],
                "permission": "allow",
                "key_exprs": ["**"],
            },
        ],
        "subjects": [
            {
                "id": "any_authenticated_peer",
                "interfaces": _local_interfaces(),
            },
        ],
        "policies": [
            {
                "rules": ["any_subscribe", "any_publish"],
                "subjects": ["any_authenticated_peer"],
            },
        ],
    }


def resolve_acl(namespace: str) -> dict[str, Any]:
    """Return the ACL dict for the current configuration.

    Resolution order: ``STRANDS_MESH_ACL_FILE`` → :func:`default_acl`.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if path_env:
        return _load_acl_file(Path(path_env))
    return default_acl(namespace)


def acl_block(namespace: str) -> tuple[str, str]:
    """Return ``("access_control", <json5>)`` for the current config."""
    return ("access_control", json.dumps(resolve_acl(namespace)))
