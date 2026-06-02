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
import threading
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
# imported lazily inside ``_parse_json5`` -- only paid by
# operators who actually load an ACL file.

# json5 is imported lazily inside
# ``_parse_json5`` rather than at module top-level. Importing it eagerly every
# import of ``strands_robots.mesh`` (including ``session.py`` for
# ``auth_mode=none`` dev paths) triggered the json5 import even when
# no ACL file is loaded. Operators running with no ACL file (the
# permissive default) and no ``mesh`` extra installed got an
# ``ImportError`` at import time when they didn't need the dep.
# The loader is the only consumer; lazy-import there.


def _parse_json5(raw: str, path: Path) -> Any:
    """Parse *raw* JSON5 text into a Python object.

    Raises :class:`ValueError` with operator-friendly diagnostics on any
    malformed input. The ACL loader treats this as a fail-closed
    boundary: a malformed file does NOT silently degrade to the
    permissive default.
    """
    try:
        import json5  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "json5 is required to parse STRANDS_MESH_ACL_FILE -- install "
            "via ``pip install strands-robots[mesh]`` (which pulls in "
            "json5) or ``pip install json5``"
        ) from exc
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
    # Require literal boolean ``True`` (identity check) rather than truthy
    # non-bool. ``enabled: 1`` (JSON5 int), ``enabled: "true"`` (string
    # typo), ``enabled: [false]`` (non-empty list) all pass ``not False``
    # but the downstream Zenoh deserializer expects a strict ``bool`` and
    # fails with an opaque "expected boolean" several frames deeper. The
    # whole point of this gate is to fail closed before Zenoh sees the
    # config, so accept only the literal type.
    if data.get("enabled") is not True:
        raise ValueError(
            f"ACL file {path} must set ``enabled: true`` (literal boolean) -- "
            f"got {data.get('enabled')!r}. Without it Zenoh silently "
            f"disables the access_control block."
        )
    if data["default_permission"] not in ("allow", "deny"):
        raise ValueError(f"ACL file {path} default_permission={data['default_permission']!r} must be 'allow' or 'deny'")
    if data["default_permission"] == "allow":
        # only warn when the operator actually
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

    a typo like ``interface:``
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
        # ``interfaces`` is OPTIONAL per Zenoh's AclConfigSubjects
        # schema (``Option<NEVec<...>>``). When omitted, Zenoh treats
        # the subject's interface dimension as ``SubjectProperty::Wildcard``
        # (matches every link); see authorization.rs:446-454. The
        # cleanest CN-only ACL pattern is therefore:
        #
        #  subjects: [{ id: "ops", cert_common_names: ["op-1", "op-2"] }]
        #
        # which is exactly what Zenoh's own ``tests/authentication.rs``
        # uses. We still REJECT an empty list outright -- Zenoh's parser
        # raises ``Found empty interface value`` and the silent total-
        # outage failure mode is real (prior footgun). And we still treat
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
    # ``namespace`` parameter is kept for API symmetry with the public
    # functions in this module (``acl_block``, ``resolve_acl``,
    # ``snapshot_acl``, ``is_default_acl_in_use``) -- they all take a
    # namespace string so callers can pass it positionally without
    # special-casing ``default_acl``. The built-in default ACL itself is
    # namespace-independent (Zenoh's namespace config does the routing
    # isolation; ACL key_exprs are RELATIVE to the active namespace and
    # do not need a namespace prefix). Review thread PR#224 _acl_config.py:343.
    _ = namespace  # noqa: F841 -- kept for API symmetry
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


# TOCTOU defence. In an earlier revision ``Mesh.start``
# called ``is_default_acl_in_use()`` (which now reads the file) and then
# ``resolve_acl()`` (which reads it again) -- a small TOCTOU window where
# an attacker who can rewrite the ACL file between the two reads sees
# the gate observe the SAFE shape and the wire load the UNSAFE shape.
# We close it with a single-load cache keyed on the file's identity
# tuple ``(path, dev, ino, size, mtime_ns)``. Both functions take the
# same snapshot; if the file changes mid-flight the next call refreshes,
# but a single ``Mesh.start`` call sees one snapshot.
_ACL_CACHE_LOCK = threading.Lock()
_ACL_CACHE: dict[tuple, dict[str, Any]] = {}


def _file_identity(path: Path) -> tuple | None:
    """Return ``(path_str, dev, ino, size, mtime_ns)`` or None on stat err."""
    try:
        st = os.stat(str(path), follow_symlinks=False)
    except OSError:
        return None
    return (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _load_acl_cached(path: Path) -> dict[str, Any]:
    """Load + cache an ACL file, keyed on its identity tuple.

    Two callers in the same ``Mesh.start`` flow (the gate check and the
    config builder) get the same dict object instead of two independent
    reads -- closing the prior TOCTOU surface. If the file changes a
    later call computes a fresh identity tuple and re-loads.
    """
    identity = _file_identity(path)
    if identity is None:
        # Stat failed -- fall through to the loader so it raises with
        # the canonical error path (FileNotFoundError, etc.).
        return _load_acl_file(path)
    with _ACL_CACHE_LOCK:
        cached = _ACL_CACHE.get(identity)
        if cached is not None:
            return cached
    loaded = _load_acl_file(path)
    with _ACL_CACHE_LOCK:
        # Cap the cache at 4 entries -- ACL files are tiny and the
        # operator usually has one. Bound prevents an attacker who can
        # touch the file repeatedly from inflating memory.
        if len(_ACL_CACHE) >= 4:
            _ACL_CACHE.pop(next(iter(_ACL_CACHE)))
        _ACL_CACHE[identity] = loaded
    return loaded


def _clear_acl_cache_for_test() -> None:
    """Test-only escape hatch -- pytest fixtures that mutate ACL files
    in tmp_path between assertions need to invalidate the cache."""
    with _ACL_CACHE_LOCK:
        _ACL_CACHE.clear()


def _is_permissive_acl_shape(data: dict[str, Any]) -> bool:
    """inspect the resolved ACL *shape* for
    the permissive pattern, regardless of where the dict came from
    (built-in default or operator-supplied file).

    The dangerous shape is ``default_permission == "allow"`` AND every
    explicit rule/subject/policy collection empty. This collapses the
    Original behaviour ("operator file with permissive shape silently
    bypasses the gate" surface into one truth source -- the gate now
    triggers on the wire-effective posture, not on the env-var
    presence.
    """
    if not isinstance(data, dict):
        return False
    if data.get("default_permission") != "allow":
        return False
    return not data.get("rules") and not data.get("subjects") and not data.get("policies")


def is_default_acl_in_use(namespace: str = "strands") -> bool:
    """Return True when the wire-effective ACL is permissive-by-shape.

    A True return means *the resolved ACL* (whether built-in default
    or operator-supplied) grants any CA-signed peer publish/subscribe
    on any key. The check is shape-based (see
    :func:`_is_permissive_acl_shape`) so an operator file with the
    same permissive pattern as :func:`default_acl` triggers the same
    refuse-to-start gate at :class:`Mesh` start.

    Callers (Mesh.start) emit ERROR + refuse-to-start when this is
    combined with ``STRANDS_MESH_AUTH_MODE=mtls`` and the operator has
    not opted in via ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1``.

    Earlier revisions returned ``not env_var_set``
    -- an operator who supplied a permissive file silenced the gate
    while running with the same posture the gate was supposed to
    refuse. Now we resolve the file (or fall back to default) and
    inspect its shape.

    Failure mode: if the operator-supplied file fails to load
    (parse error, bad shape, IO error), the gate fails CLOSED -- we
    return ``True`` so :class:`Mesh.start` refuses to bring up the
    wire. A broken ACL file is a configuration emergency, not a
    "fall back to permissive" situation.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if not path_env:
        # Built-in default: known to be permissive-by-shape.
        return True
    try:
        resolved = _load_acl_cached(Path(path_env))
    except (OSError, ValueError) as exc:
        # fail closed. A broken ACL file is treated as the
        # most-dangerous-known posture so the operator hears about it
        # at start-up rather than silently degrading to permissive.
        logger.warning(
            "[mesh] ACL file %s could not be loaded for shape check (%s); "
            "treating as permissive-by-default for the start-time gate",
            path_env,
            exc,
        )
        return True
    return _is_permissive_acl_shape(resolved)


def resolve_acl(namespace: str) -> dict[str, Any]:
    """Return the ACL dict for the current configuration.

    Resolution order: ``STRANDS_MESH_ACL_FILE`` -> :func:`default_acl`.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if path_env:
        # shared cache so the prior shape gate and the wire
        # config builder see the SAME snapshot of the ACL file.
        return _load_acl_cached(Path(path_env))
    return default_acl(namespace)


def snapshot_acl(namespace: str = "strands") -> tuple[bool, dict[str, Any]]:
    """Atomically resolve the ACL and report its permissive-by-shape state.

    Issue #218: closes the TOCTOU window between ``is_default_acl_in_use``
    and ``resolve_acl``. The previous two-call pattern computed a fresh
    identity tuple per call, missing the cache when an attacker rewrote
    the ACL file between calls. ``snapshot_acl`` performs a single
    ``_load_acl_cached`` call and derives both signals from it.

    Returns:
        (is_permissive_by_shape, resolved_acl_dict)

    Mesh.start should call this once at the top and thread the
    resolved dict through both the refuse-to-start gate and the
    ``acl_block`` insertion.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if not path_env:
        # Built-in default: known permissive-by-shape.
        return True, default_acl(namespace)
    try:
        resolved = _load_acl_cached(Path(path_env))
    except (OSError, ValueError) as exc:
        # fail closed: unloadable file is treated as permissive so the
        # gate at Mesh.start refuses to bring up the wire.
        logger.warning(
            "[mesh] ACL file %s could not be loaded for snapshot (%s); "
            "treating as permissive-by-default for the start-time gate",
            path_env,
            exc,
        )
        return True, default_acl(namespace)
    return _is_permissive_acl_shape(resolved), resolved


def acl_block_from(resolved: dict[str, Any]) -> tuple[str, str]:
    """Return ``("access_control", <json5>)`` from a pre-resolved dict.

    Companion to :func:`snapshot_acl` -- pass the dict returned by the
    snapshot to bypass a second file read. Use this in Mesh.start so
    the refuse-to-start gate and the wire config builder share exactly
    one snapshot of the ACL file.
    """
    return ("access_control", json.dumps(resolved))


def acl_block(namespace: str) -> tuple[str, str]:
    """Return ``("access_control", <json5>)`` for the current config.

    .. note::
        Prefer :func:`snapshot_acl` + :func:`acl_block_from` in new code
        to avoid the TOCTOU window between gate-shape-check and
        wire-config-build (issue #218).
    """
    return ("access_control", json.dumps(resolve_acl(namespace)))
