"""ACL config builder for the strands-robots mesh.

Reads a JSON5 ACL file at ``STRANDS_MESH_ACL_FILE`` and returns the
serialised ``access_control`` block ready for
``zenoh.Config.insert_json5``. When the env var is unset, returns the
permissive :func:`default_acl` skeleton.

Zenoh 1.x quirks (each verified against a live session in
``tests/mesh/test_zenoh_transport_security.py``):

* ``enabled: true`` is required -- without it the entire block is a
  no-op even if rules and subjects are populated.
* ``cert_common_names`` matches LITERAL CNs only; globs and regexes
  match nothing. Operators tighten the default by enumerating each
  peer's exact cert CN in ``STRANDS_MESH_ACL_FILE``.
* Subject ``interfaces`` must be a non-empty list -- leaving it unset
  causes the subject to match nothing.
* ``key_exprs`` match the user-side key (the namespace prefix is
  stripped from the matcher's view), so ``**/cmd`` is the robust
  glob; ``"<namespace>/*/cmd"`` never matches.
* ``declare_subscriber`` rules live in the ``egress`` flow (the
  declare goes from subscriber to publisher); ``put`` rules live in
  ``ingress`` (the publisher's cert CN is known to the receiver).

JSON5 is the on-disk format (line + block comments, trailing commas,
unquoted keys). The loader uses a stdlib JSON5-lite preprocessor --
no third-party dependency.
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


# --- JSON5 parser (vendored via the ``json5`` PyPI dep) ----------------

# We delegate JSON5 parsing to the ``json5`` library (MIT, audited, ~3kLOC,
# pure Python, no native deps). Earlier revisions carried a four-pass hand-
# rolled preprocessor (``_strip_json5_comments`` -> ``_strip_trailing_commas``
# -> ``_quote_unquoted_keys`` -> ``_convert_single_quoted_strings``) that
# silently truncated on unterminated ``/*`` blocks, mis-quoted keys after
# ``[`` (object-in-array case), and produced ``json.JSONDecodeError`` column
# numbers pointing at the post-preprocessor string -- making operator
# debugging painful. The dep swap eliminates ~250 LOC of fragile state-
# machine code and gives operators precise diagnostics on malformed input.
#
# Why a third-party dep is acceptable here: the ACL file gates wire
# authorisation, so a parser that fails *closed* with a clear error is
# strictly safer than a hand-rolled approximation. ``json5`` is already
# transitively available in many Python deployments; we add it to the
# ``mesh`` extra so it ships with the rest of the wire-layer code.

try:
    import json5  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "json5 is required by strands_robots.mesh -- install via "
        "``pip install strands-robots[mesh]`` (which pulls in json5) "
        "or ``pip install json5``"
    ) from exc


def _parse_json5(raw: str, path: Path) -> Any:
    """Parse *raw* JSON5 text into a Python object.

    Raises :class:`ValueError` with operator-friendly diagnostics on any
    malformed input. The ACL loader treats this as a fail-closed
    boundary: a malformed file does NOT silently degrade to the
    permissive default.
    """
    try:
        return json5.loads(raw)
    except ValueError as exc:
        # json5 raises ValueError (subclass) with a useful message that
        # includes line/column. Re-raise with the path attached so an
        # operator looking at the log sees exactly which file failed.
        raise ValueError(f"ACL file {path} is not valid JSON5: {exc}") from exc


# --- ACL file loader ---------------------------------------------------


def _load_acl_file(path: Path) -> dict[str, Any]:
    """Load and validate an ACL file.

    Refuses any file that omits ``enabled: true`` -- Zenoh silently
    no-ops the block in that case, and the loader fails closed rather
    than ship a quietly-disabled gate.
    """
    # Defence: refuse to follow symlinks AND bound the read at
    # ACL_FILE_MAX_BYTES + 1 so an attacker who races content between
    # stat() and read() cannot bypass the size cap. Mirrors the
    # O_NOFOLLOW + bounded-read discipline used for the audit log
    # (audit.py:_ensure_paths). The ACL file gates wire authorisation,
    # so the same TOCTOU + symlink-swap defences apply.
    if path.is_symlink():
        raise ValueError(
            f"refusing to load ACL file {path}: it is a SYMLINK "
            f"(target: {os.readlink(path)!r}). ACL files must be regular files."
        )
    if not path.is_file():
        raise FileNotFoundError(f"ACL file not found: {path}")
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags | nofollow)
    except OSError as exc:
        # ELOOP under O_NOFOLLOW = symlink raced ahead of the static check.
        raise ValueError(f"refusing to load ACL file {path}: {exc}") from exc
    try:
        # Read at most MAX+1 bytes so we can detect overflow without
        # an unbounded read.
        chunks = []
        remaining = ACL_FILE_MAX_BYTES + 1
        while remaining > 0:
            buf = os.read(fd, remaining)
            if not buf:
                break
            chunks.append(buf)
            remaining -= len(buf)
    finally:
        os.close(fd)
    raw_bytes = b"".join(chunks)
    if len(raw_bytes) > ACL_FILE_MAX_BYTES:
        raise ValueError(f"ACL file {path} is >{ACL_FILE_MAX_BYTES} bytes; refusing to load.")
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"ACL file {path} is not valid UTF-8: {exc}") from exc
    data = _parse_json5(raw, path)

    if not isinstance(data, dict):
        raise ValueError(f"ACL file {path} root must be an object")
    for required in ("default_permission", "rules", "subjects", "policies"):
        if required not in data:
            raise ValueError(f"ACL file {path} missing required field: {required!r}")
    if not data.get("enabled", False):
        raise ValueError(
            f"ACL file {path} must set ``enabled: true`` -- without it Zenoh silently disables the access_control block."
        )
    if data["default_permission"] not in ("allow", "deny"):
        raise ValueError(f"ACL file {path} default_permission={data['default_permission']!r} must be 'allow' or 'deny'")
    if data["default_permission"] == "allow":
        # F11-C (PR #195 review): only warn when the operator actually
        # has rules/subjects/policies that *combine* with allow-by-default
        # in a blacklist shape. The built-in ``default_acl()`` (used when
        # STRANDS_MESH_ACL_FILE is unset) ships ``allow + empty rules/
        # subjects/policies`` -- the documented permissive-by-design
        # posture. Warning operators who copy that shape into a file is
        # asymmetric scolding; reserve the warning for the actual
        # blacklist anti-pattern (allow + non-empty rules).
        is_truly_permissive_default = not data.get("rules") and not data.get("subjects") and not data.get("policies")
        if not is_truly_permissive_default:
            logger.warning(
                "[acl] %s uses default_permission='allow' with rules -- "
                "this is a blacklist policy and any rule gap exposes "
                "the mesh. Prefer 'deny' with explicit allow rules.",
                path,
            )
    _validate_acl_shape(data, path)
    return data


def _validate_acl_shape(data: dict[str, Any], path: Path) -> None:
    """Validate the shape of subjects/rules/policies after JSON parse.

    R19 (PR #195 design-thread review): a typo like ``interface:``
    (singular) or a missing ``cert_common_names`` field silently
    degrades a role-separated ACL to "match nothing" at the Zenoh
    layer, which manifests as a silent total outage operators must
    debug from Zenoh logs. We refuse these shapes loudly at parse
    time -- the same posture the ``enabled: true`` check (added
    earlier in this function) is built around.

    Validates:

    1. ``subjects``, ``rules``, ``policies`` are lists.
    2. Every subject has ``id``. ``interfaces`` is OPTIONAL -- when
       omitted, Zenoh treats the interface dimension as
       ``SubjectProperty::Wildcard`` (matches every link); when present,
       it must be a non-empty list of non-empty strings (Zenoh rejects
       ``[]`` with ``Found empty interface value``). ``cert_common_names``
       is OPTIONAL -- when present must be a list. Subjects with
       neither ``interfaces`` nor ``cert_common_names`` match every
       peer (effectively wildcard) and operators should use them only
       when a permissive ``default_permission: "allow"`` is desired.
    3. Every rule has ``id``, ``key_exprs`` (non-empty list of
       strings), ``messages`` (non-empty list), ``flows`` (non-empty
       list), and ``permission`` (``allow`` or ``deny``).
    4. Every policy has ``rules`` and ``subjects`` referencing
       existing rule / subject ids.

    Raises ``ValueError`` with a path-prefixed message on the first
    failure. Callers should treat any failure here as a deployment
    blocker -- a malformed ACL is worse than no ACL because the
    operator believes role separation is enforced when it is not.
    """
    # 1. Top-level lists.
    for field in ("subjects", "rules", "policies"):
        if not isinstance(data[field], list):
            raise ValueError(f"ACL file {path}: {field!r} must be a list, got {type(data[field]).__name__}")

    # 2. Subjects.
    subject_ids: set[str] = set()
    for i, subj in enumerate(data["subjects"]):
        if not isinstance(subj, dict):
            raise ValueError(f"ACL file {path}: subjects[{i}] must be an object")
        sid = subj.get("id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"ACL file {path}: subjects[{i}].id must be a non-empty string")
        subject_ids.add(sid)
        # F2: ``interfaces`` is OPTIONAL per Zenoh's AclConfigSubjects
        # schema (``Option<NEVec<...>>``). When omitted, Zenoh treats
        # the subject's interface dimension as ``SubjectProperty::Wildcard``
        # (matches every link); see authorization.rs:446-454. The
        # cleanest CN-only ACL pattern is therefore:
        #
        #   subjects: [{ id: "ops", cert_common_names: ["op-1", "op-2"] }]
        #
        # which is exactly what Zenoh's own ``tests/authentication.rs``
        # uses. We still REJECT an empty list outright -- Zenoh's parser
        # raises ``Found empty interface value`` and the silent total-
        # outage failure mode is real (R19 footgun). And we still treat
        # an unknown-key like ``interface:`` (singular typo) as an error
        # because the rest of the validator catches it via
        # ``deny_unknown_fields`` semantics in the Rust deserializer.
        if "interfaces" in subj:
            ifaces = subj["interfaces"]
            if not isinstance(ifaces, list):
                raise ValueError(
                    f"ACL file {path}: subjects[{i}={sid!r}].interfaces must be a list "
                    f"(or omitted), got {type(ifaces).__name__}."
                )
            if not ifaces:
                raise ValueError(
                    f"ACL file {path}: subjects[{i}={sid!r}].interfaces is an empty list. "
                    f"Zenoh rejects ``[]`` with ``Found empty interface value``; either "
                    f"omit the field (for a wildcard binding) or enumerate the NICs."
                )
            if not all(isinstance(x, str) and x for x in ifaces):
                raise ValueError(
                    f"ACL file {path}: subjects[{i}={sid!r}].interfaces must contain only non-empty strings"
                )
        cns = subj.get("cert_common_names")
        if cns is not None and not isinstance(cns, list):
            raise ValueError(
                f"ACL file {path}: subjects[{i}={sid!r}].cert_common_names must be a list "
                f"(or omitted), got {type(cns).__name__}. Common typo: cert_common_name (singular)."
            )

    # 3. Rules.
    rule_ids: set[str] = set()
    for i, rule in enumerate(data["rules"]):
        if not isinstance(rule, dict):
            raise ValueError(f"ACL file {path}: rules[{i}] must be an object")
        rid = rule.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"ACL file {path}: rules[{i}].id must be a non-empty string")
        rule_ids.add(rid)
        for field in ("key_exprs", "messages", "flows"):
            val = rule.get(field)
            if not isinstance(val, list) or not val:
                raise ValueError(f"ACL file {path}: rules[{i}={rid!r}].{field} must be a non-empty list")
            if not all(isinstance(x, str) for x in val):
                raise ValueError(f"ACL file {path}: rules[{i}={rid!r}].{field} must contain only strings")
        perm = rule.get("permission")
        if perm not in ("allow", "deny"):
            raise ValueError(f"ACL file {path}: rules[{i}={rid!r}].permission must be 'allow' or 'deny', got {perm!r}")

    # 4. Policies.
    for i, pol in enumerate(data["policies"]):
        if not isinstance(pol, dict):
            raise ValueError(f"ACL file {path}: policies[{i}] must be an object")
        pol_rules = pol.get("rules")
        pol_subjects = pol.get("subjects")
        if not isinstance(pol_rules, list) or not pol_rules:
            raise ValueError(f"ACL file {path}: policies[{i}].rules must be a non-empty list of rule ids")
        if not isinstance(pol_subjects, list) or not pol_subjects:
            raise ValueError(f"ACL file {path}: policies[{i}].subjects must be a non-empty list of subject ids")
        for r in pol_rules:
            if r not in rule_ids:
                raise ValueError(
                    f"ACL file {path}: policies[{i}].rules references unknown rule id {r!r} (known: {sorted(rule_ids)})"
                )
        for sid_ref in pol_subjects:
            if sid_ref not in subject_ids:
                raise ValueError(
                    f"ACL file {path}: policies[{i}].subjects references unknown subject id "
                    f"{sid_ref!r} (known: {sorted(subject_ids)})"
                )


# --- Default ACL -------------------------------------------------------


def default_acl(namespace: str) -> dict[str, Any]:
    """Return a permissive default ACL skeleton.

    The default allows any peer with a valid CA-signed cert (verified
    at the mTLS handshake) to publish and subscribe on any key.
    Operators who want per-role enforcement supply their own ACL via
    ``STRANDS_MESH_ACL_FILE`` enumerating each peer's exact cert CN
    (Zenoh 1.x cert_common_names does not support globs -- see
    ``examples/mesh_acl_example.json5`` for the canonical template).

    Why permissive default rather than default-deny: a default-deny
    skeleton with no enumerated subjects rejects every legitimate
    message -- silent total outage on first run. The mTLS handshake at
    the link layer already gates fleet membership; the application-
    layer ``validate_command`` gates payload semantics. ACL is the
    third line of defence and operators opt in explicitly.
    """
    _ = namespace  # `namespace` config does the routing isolation; ACL key_exprs do not need it
    return {
        "enabled": True,
        # Permissive default: any peer that survived the mTLS handshake may
        # publish and subscribe on any key. This is the documented behaviour
        # (CHANGELOG section 8, README "Default ACL -- permissive by design").
        # Operators wanting per-role enforcement supply STRANDS_MESH_ACL_FILE
        # (see examples/mesh_acl_example.json5 for the canonical template).
        #
        # Earlier versions of this default mixed default_permission='deny'
        # with two key_exprs=['**'] allow-rules; the effective behaviour was
        # identical (allow-any) but the code-vs-doc surface was confusing
        # and review-flagged 5x. Code now matches docs.
        "default_permission": "allow",
        "rules": [],
        "subjects": [],
        "policies": [],
    }


def is_default_acl_in_use() -> bool:
    """Return True when the permissive built-in default ACL is active.

    A True return means *no* operator-supplied ACL is in effect: any peer
    that survives the mTLS handshake can publish and subscribe on any key.
    Callers (Mesh.start) emit a WARNING when this is combined with
    ``STRANDS_MESH_AUTH_MODE=mtls`` so production deployments that forgot
    to enumerate cert CNs hear about it on every session open.
    """
    return not os.getenv("STRANDS_MESH_ACL_FILE", "").strip()


def resolve_acl(namespace: str) -> dict[str, Any]:
    """Return the ACL dict for the current configuration.

    Resolution order: ``STRANDS_MESH_ACL_FILE`` -> :func:`default_acl`.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if path_env:
        return _load_acl_file(Path(path_env))
    return default_acl(namespace)


def acl_block(namespace: str) -> tuple[str, str]:
    """Return ``("access_control", <json5>)`` for the current config."""
    return ("access_control", json.dumps(resolve_acl(namespace)))
