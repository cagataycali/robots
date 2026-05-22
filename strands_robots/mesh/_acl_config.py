"""ACL config builder for the strands-robots mesh.

Reads a JSON5 ACL file at ``STRANDS_MESH_ACL_FILE`` and returns the
serialised ``access_control`` block ready for
``zenoh.Config.insert_json5``. When the env var is unset, returns the
:func:`default_acl` shape — default-deny with two roles:

* ``robot_peer`` (cert CN matches ``robot-*``): may publish telemetry on
  ``{ns}/<peer>/{presence,state,health,response,...}`` and subscribe to
  ``{ns}/<peer>/cmd`` + ``{ns}/broadcast``.
* ``operator_peer`` (cert CN matches ``op-*``): may publish on every
  cmd / broadcast topic and subscribe to ``{ns}/**`` for observation.

Every other peer (whose cert CN matches no rule) is silently dropped
at the transport. There is no application-layer reject path; ACL
denials show up in Zenoh's own logs and are emitted to our audit log
via the ``acl_drop`` hook in :mod:`strands_robots.mesh.core`.

JSON5 is the on-disk format because (a) the Zenoh config itself takes
JSON5, (b) it allows comments which operators want for ACL reasoning,
(c) it tolerates trailing commas which makes the file diff-friendly
when adding rules.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Maximum bytes of an ACL file we will load. Anything larger is almost
#: certainly an attacker probing for an OOM.
ACL_FILE_MAX_BYTES: int = 256 * 1024


def _strip_json5_comments(raw: str) -> str:
    """Strip ``//`` line comments and ``/* */`` block comments.

    A small stdlib-only JSON5 preprocessor that handles the subset
    operators actually use (line + block comments, trailing commas).
    String contents are left untouched: we walk the source char by char
    and toggle an ``in_string`` flag on unescaped quotes so a ``//``
    inside a JSON string is preserved verbatim.
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
                # Keep escape pair verbatim.
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
            # Line comment: skip to end of line.
            i += 2
            while i < n and raw[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            # Block comment: skip to closing */.
            i += 2
            while i < n - 1 and not (raw[i] == "*" and raw[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_trailing_commas(raw: str) -> str:
    """Remove trailing commas before ``}`` / ``]`` (JSON5 allows them).

    Walks the source preserving string contents (an unbalanced ``,]``
    inside a string must NOT be touched).
    """
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
            # Look ahead past whitespace; if next non-ws is } or ], drop the comma.
            j = i + 1
            while j < n and raw[j] in " \t\r\n":
                j += 1
            if j < n and raw[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _quote_unquoted_keys(raw: str) -> str:
    """Wrap unquoted object keys in double quotes (JSON5 allows them).

    The replacement is intentionally narrow: it only matches an
    identifier (``[A-Za-z_][A-Za-z0-9_]*``) immediately followed by
    optional whitespace and a colon, in object-key position. The
    string-walker preserves string contents.
    """
    import re

    out: list[str] = []
    i = 0
    n = len(raw)
    in_string = False
    string_quote = ""
    key_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:")
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
        # Try to match an unquoted key starting at i.
        m = key_re.match(raw, i)
        if m and (not out or out[-1] in "{,\n \t"):
            out.append('"' + m.group(1) + '"' + ":")
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _json5_to_json(raw: str) -> str:
    """Apply our JSON5-lite preprocessor to *raw*.

    Three passes (in order): strip comments, quote unquoted keys, drop
    trailing commas. The output is plain JSON suitable for
    :func:`json.loads`.
    """
    return _strip_trailing_commas(_quote_unquoted_keys(_strip_json5_comments(raw)))


def _load_acl_file(path: Path) -> dict[str, Any]:
    """Load and validate an ACL file.

    The on-disk format is JSON5-lite: line and block comments, trailing
    commas, and unquoted object keys are all accepted. Strict JSON
    files also work. Zenoh's ``access_control`` parser does the deep
    schema validation when we hand the dict to ``insert_json5``.
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
    if data["default_permission"] not in ("allow", "deny"):
        raise ValueError(f"ACL file {path} default_permission={data['default_permission']!r} must be 'allow' or 'deny'")
    if data["default_permission"] == "allow":
        logger.warning(
            "[acl] %s uses default_permission='allow' — this is a blacklist "
            "policy and any rule gap exposes the mesh. Prefer 'deny'.",
            path,
        )
    return data


def default_acl(namespace: str) -> dict[str, Any]:
    """Return the built-in default-deny ACL for *namespace*.

    Two roles, mapped from cert CN globs:

    * ``robot-*`` certs: publish own telemetry, subscribe to own cmd
      topic plus broadcast.
    * ``op-*`` certs: publish to any cmd / broadcast topic, subscribe
      to everything for monitoring.

    Anything else is denied. This is what every fresh install gets
    when ``STRANDS_MESH_ACL_FILE`` is unset.
    """
    return {
        "default_permission": "deny",
        "rules": [
            {
                "id": "robot_publish_telemetry",
                "messages": ["put"],
                "flows": ["egress"],
                "permission": "allow",
                "key_exprs": [
                    f"{namespace}/*/presence",
                    f"{namespace}/*/state/**",
                    f"{namespace}/*/health",
                    f"{namespace}/*/pose",
                    f"{namespace}/*/imu",
                    f"{namespace}/*/odom",
                    f"{namespace}/*/lidar/**",
                    f"{namespace}/*/hand/**",
                    f"{namespace}/*/camera/**",
                    f"{namespace}/*/response/**",
                    f"{namespace}/*/safety/**",
                ],
            },
            {
                "id": "robot_subscribe_cmds",
                "messages": ["declare_subscriber", "put"],
                "flows": ["ingress"],
                "permission": "allow",
                "key_exprs": [
                    f"{namespace}/*/cmd",
                    f"{namespace}/broadcast",
                    f"{namespace}/*/safety/**",
                ],
            },
            {
                "id": "operator_publish_cmds",
                "messages": ["put"],
                "flows": ["egress"],
                "permission": "allow",
                "key_exprs": [
                    f"{namespace}/*/cmd",
                    f"{namespace}/broadcast",
                    f"{namespace}/*/safety/**",
                ],
            },
            {
                "id": "operator_observe",
                "messages": ["declare_subscriber"],
                "flows": ["ingress"],
                "permission": "allow",
                "key_exprs": [f"{namespace}/**"],
            },
        ],
        "subjects": [
            {"id": "robot_peer", "cert_common_names": ["robot-*"]},
            {"id": "operator_peer", "cert_common_names": ["op-*"]},
        ],
        "policies": [
            {
                "rules": ["robot_publish_telemetry", "robot_subscribe_cmds"],
                "subjects": ["robot_peer"],
            },
            {
                "rules": ["operator_publish_cmds", "operator_observe"],
                "subjects": ["operator_peer"],
            },
        ],
    }


def resolve_acl(namespace: str) -> dict[str, Any]:
    """Return the ACL dict for the current configuration.

    Resolution order:

    1. ``STRANDS_MESH_ACL_FILE`` set → load and validate that file.
    2. Otherwise → :func:`default_acl(namespace)`.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if path_env:
        return _load_acl_file(Path(path_env))
    return default_acl(namespace)


def acl_block(namespace: str) -> tuple[str, str]:
    """Return ``("access_control", <json5>)`` for the current config."""
    return ("access_control", json.dumps(resolve_acl(namespace)))
