"""Zenoh config builders for the strands-robots mesh.

This module owns every ``insert_json5`` call that hardens the Zenoh
session: namespace isolation, scouting policy, transport DoS bounds,
per-key-expression rate / size caps, mTLS, and access control.

Public functions return ``(path, json5_value)`` pairs ready to feed to
``zenoh.Config.insert_json5``. They never touch a ``zenoh.Config``
object directly so the builders can be unit-tested without the wheel
installed.

The mesh enables the Zenoh built-in security primitives by default;
operators do not get to disable them. There is no permissive fallback,
no PSK, no application-layer envelope. Identity is bound at the TLS
handshake (``cert_common_names``); authorisation is bound at the ACL
(``key_exprs`` + ``messages`` + ``flows``); rate / size caps are
enforced at the transport before bytes hit the deserialiser.

Configuration env vars
----------------------
``STRANDS_MESH_NAMESPACE``
    Fleet prefix prepended to every key-expression. Default
    ``strands``. Two fleets with different namespaces cannot
    collide on the same network.

``STRANDS_MESH_MULTICAST``
    ``true`` to enable multicast scouting. Default ``false`` -- gossip
    is the only discovery channel and ``connect/endpoints`` must be set
    explicitly. This closes the LAN-attacker-enrollment surface.

``STRANDS_MESH_MAX_SESSIONS``
    Hard cap on simultaneous unicast sessions. Default ``256``.

``STRANDS_MESH_MAX_CMD_BYTES``
    Per-message byte cap on ``cmd`` / ``broadcast`` topics enforced via
    ``low_pass_filter``. Default ``16384`` (mesh commands are small
    JSON; anything larger is jumbo-frame DoS).

``STRANDS_MESH_MAX_CAMERA_BYTES``
    Per-message byte cap on camera topics. Default ``1048576`` (1 MiB).

``STRANDS_MESH_CMD_RATE_HZ``
    Per-key-expression frequency cap for ``cmd`` topics enforced via
    ``downsampling``. Default ``20.0`` Hz.

``STRANDS_MESH_SAFETY_RATE_HZ``
    Per-key-expression frequency cap on ``safety/**`` topics. Default
    ``2.0`` Hz. Caps novel-``t`` estop/resume floods that bypass the
    receiver-side replay cache. Operators with a legitimate need for
    higher safety throughput (sensor-driven safety event streams) can
    raise this; the floor is the receiver-side HMAC + freshness cost.

``STRANDS_MESH_MAX_SAFETY_BYTES``
    Per-message byte cap on ``safety/**`` topics. Default ``4096``.
    Safety envelopes are small JSON dicts; jumbo-frame envelopes on
    this topic are DoS targeting the receiver HMAC + freshness math.

``STRANDS_MESH_AUTH_MODE``
    ``mtls`` (default) or ``none``. ``none`` is a development-only mode
    that skips the TLS terminator and ACL -- never run it on a network
    you do not fully trust. The mesh still emits namespace, scouting,
    and DoS-cap config in ``none`` mode; only the auth + ACL blocks are
    omitted.

``STRANDS_MESH_TLS_CA``
    Filesystem path to the CA bundle used to validate peer certificates.
    Required when ``STRANDS_MESH_AUTH_MODE=mtls``.

``STRANDS_MESH_TLS_CERT``
    Filesystem path to this peer's certificate (PEM).

``STRANDS_MESH_TLS_KEY``
    Filesystem path to this peer's private key (PEM, mode 0o600 on POSIX).
    On non-POSIX hosts (Windows) ``_resolve_tls_paths`` does not enforce
    the file mode -- the loader skips the ``stat().st_mode`` check because
    POSIX modes do not map cleanly onto NTFS ACLs. Operators on Windows
    must rely on filesystem ACLs (e.g. restrict the key file to a single
    Windows account) rather than the loader's mode gate.

``STRANDS_MESH_ACL_FILE``
    Filesystem path to a JSON5 ACL file. When unset, the built-in
    permissive ACL from :func:`~strands_robots.mesh._acl_config.default_acl`
    is used: any CA-signed peer may publish/subscribe on any key. Operators
    who require role separation between robots and operators must supply
    a custom ACL file (template at ``examples/mesh_acl_example.json5``).
    See CHANGELOG.md Section 8 for the rationale (Zenoh 1.x ACL CN-glob
    quirks made a true default-deny silently total-deny on first run).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


#: Fleet namespace fallback when ``STRANDS_MESH_NAMESPACE`` is unset.
#:
#: This must match the literal topic prefix every mesh component emits
#: (`mesh/core.py`, `mesh/sensors.py`, `mesh/input.py`, the IoT path).
#: The `namespace` Zenoh config field provides routing isolation --
#: two fleets with different namespaces cannot exchange messages even
#: when their key-expressions collide. The default below tracks the
#: hardcoded `strands/...` topic prefix so the built-in ACL key_exprs
#: match the wire keys exactly.
DEFAULT_NAMESPACE: str = "strands"

#: Hard cap on simultaneous Zenoh unicast sessions.
DEFAULT_MAX_SESSIONS: int = 256

#: Per-message byte cap on cmd / broadcast topics.
DEFAULT_MAX_CMD_BYTES: int = 16 * 1024

#: Per-message byte cap on camera frames.
DEFAULT_MAX_CAMERA_BYTES: int = 1 * 1024 * 1024

#: Per-key-expression frequency cap on cmd topics (Hz).
DEFAULT_CMD_RATE_HZ: float = 20.0

#: Per-key-expression frequency cap on safety topics (Hz).
#:
#: R21: legitimate operator estop / resume traffic is far below
#: 1 Hz steady-state. A peer publishing on ``safety/**`` faster
#: than this rate is throttled at the transport, capping the
#: novel-`t` flood surface that bypasses the receiver-side R9
#: replay cache.
DEFAULT_SAFETY_RATE_HZ: float = 2.0

#: Per-message byte cap on safety topics. Safety envelopes are
#: small JSON dicts; a 100 KiB envelope on this topic is jumbo-
#: frame DoS targeting the receiver-side HMAC + freshness math.
DEFAULT_MAX_SAFETY_BYTES: int = 4 * 1024


def resolve_namespace() -> str:
    """Return the configured fleet namespace.

    Reads ``STRANDS_MESH_NAMESPACE`` and falls back to
    :data:`DEFAULT_NAMESPACE`. Empty / whitespace values fall through
    to the default so an operator setting ``STRANDS_MESH_NAMESPACE=""``
    does not accidentally produce keys like ``"//presence"``.
    """
    raw = os.getenv("STRANDS_MESH_NAMESPACE", "").strip()
    return raw or DEFAULT_NAMESPACE


def resolve_auth_mode() -> str:
    """Return the configured auth mode.

    One of ``"mtls"`` (default) or ``"none"``. Any other value is
    rejected with a ``ValueError`` so a typo does not silently disable
    auth.

    ``"none"`` disables both the mTLS terminator and the ACL block --
    a single env var that turns the entire wire-layer security model
    off. To prevent a typo / forgotten env-var / leaked CI fixture
    from silently disabling wire auth in production, ``"none"`` is
    additionally gated on ``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1``
    (case-insensitive: ``1``, ``true``, ``yes``). Without that explicit
    second factor, ``"none"`` raises ``ValueError`` at config-build
    time -- the burden of proof lives with the operator who is turning
    auth off.
    """
    raw = os.getenv("STRANDS_MESH_AUTH_MODE", "mtls").strip().lower()
    if raw not in ("mtls", "none"):
        raise ValueError(f"STRANDS_MESH_AUTH_MODE={raw!r} not supported (expected 'mtls' or 'none')")
    if raw == "none":
        ack = os.getenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "").strip().lower()
        if ack not in ("1", "true", "yes"):
            raise ValueError(
                "STRANDS_MESH_AUTH_MODE=none disables BOTH the mTLS "
                "terminator AND the ACL block -- the entire wire-layer "
                "security model. Refusing without an explicit second "
                "factor: set STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1 to "
                "confirm. This guard prevents a typo / forgotten env-var "
                "/ leaked CI fixture from silently disabling wire auth "
                "in production."
            )
    return raw


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var with a strict truthy/falsy mapping."""
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("true", "1", "yes", "on"):
        return True
    if raw in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (use true/false)")


def _int_env(name: str, default: int, *, lo: int = 1, hi: int = 1 << 30) -> int:
    """Parse an integer env var clamped to ``[lo, hi]``."""
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} out of bounds [{lo}, {hi}]")
    return value


def _float_env(name: str, default: float, *, lo: float = 0.0, hi: float = 1e6) -> float:
    """Parse a float env var clamped to ``[lo, hi]``."""
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a float") from exc
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} out of bounds [{lo}, {hi}]")
    return value


def namespace_block() -> tuple[str, str]:
    """Return ``("namespace", <json5>)`` for the configured fleet namespace."""
    return ("namespace", json.dumps(resolve_namespace()))


def scouting_block() -> list[tuple[str, str]]:
    """Return scouting config: multicast off (default) + gossip on.

    Multicast on a hostile LAN is a discovery attack surface -- any host
    that joins the multicast group (224.0.0.224:7446) sees every peer's
    presence broadcast. Gossip-only with explicit ``connect/endpoints``
    is the production posture.

    Operators on a controlled LAN can opt back into multicast with
    ``STRANDS_MESH_MULTICAST=true``. We do NOT recommend it.
    """
    multicast = _bool_env("STRANDS_MESH_MULTICAST", default=False)
    return [
        ("scouting/multicast/enabled", "true" if multicast else "false"),
        ("scouting/gossip/enabled", "true"),
    ]


def transport_caps_block() -> list[tuple[str, str]]:
    """Return transport-level DoS bounds.

    Currently emits ``transport/unicast/max_sessions``. Future caps
    (timeouts, queue sizes) land here.
    """
    max_sessions = _int_env(
        "STRANDS_MESH_MAX_SESSIONS",
        DEFAULT_MAX_SESSIONS,
        lo=1,
        hi=65535,
    )
    return [("transport/unicast/max_sessions", str(max_sessions))]


def downsampling_block() -> tuple[str, str]:
    """Return ``("downsampling", <json5>)`` capping the cmd-publish rate.

    A peer publishing to ``{namespace}/*/cmd`` faster than the
    configured frequency is throttled at the transport layer -- the
    extra messages are dropped before they reach the JSON parser, so
    flood attacks cost the receiver almost nothing.
    """
    freq = _float_env(
        "STRANDS_MESH_CMD_RATE_HZ",
        DEFAULT_CMD_RATE_HZ,
        lo=0.001,
        hi=10000.0,
    )
    # R21: safety topics need their own (lower) rate cap. Without it
    # a peer with any CA-signed cert can flood ``safety/estop`` at
    # line rate with novel ``t`` on each envelope -- bypassing the
    # receiver-side replay cache (key=(issuer_id, t)) and consuming
    # CPU on freshness arithmetic + per-receiver replay-cache pressure.
    safety_freq = _float_env(
        "STRANDS_MESH_SAFETY_RATE_HZ",
        DEFAULT_SAFETY_RATE_HZ,
        lo=0.001,
        hi=1000.0,
    )
    # See ``low_pass_filter_block`` for the namespace-vs-key_expr note:
    # ``**/cmd`` matches any prefix including the namespace one;
    # ``f"{namespace}/*/cmd"`` would not.
    rules = [
        {"key_expr": "**/cmd", "freq": freq},
        {"key_expr": "**/broadcast", "freq": freq},
        {"key_expr": "**/safety/**", "freq": safety_freq},
    ]
    return (
        "downsampling",
        json.dumps(
            [
                {
                    "id": "strands_cmd_rate_cap",
                    "messages": ["put"],
                    "flows": ["ingress"],
                    "rules": rules,
                }
            ]
        ),
    )


def _filter_interfaces() -> list[str] | None:
    """Return the operator-supplied interface allowlist, or ``None``.

    Zenoh's ``low_pass_filter`` block treats an absent ``interfaces``
    field as ``SubjectProperty::Wildcard`` (matches every link). See
    ``zenoh/src/net/routing/interceptor/low_pass.rs`` (1.x):

        let interfaces = lpf_config.interfaces
            .map(...)
            .unwrap_or(vec![SubjectProperty::Wildcard]);

    A wildcard binding is the correct posture for a fleet-wide cap:
    the cap applies regardless of which NIC a peer's link rides on,
    and there is no NIC enumeration that needs to stay in sync with
    the actual deployment topology.

    Operators with a *specific* need to bind the cap to a subset of
    NICs (e.g. excluding a high-volume telemetry NIC from the cmd
    cap) can set ``STRANDS_MESH_FILTER_INTERFACES`` (comma-separated)
    and we honour it literally. Empty / unset returns ``None`` so the
    builder omits the field entirely -- not the empty list, which
    Zenoh's ``Option<NEVec<String>>`` parser rejects with
    ``Found empty interface value`` (deny_unknown_fields + non-empty
    vec).
    """
    raw = os.getenv("STRANDS_MESH_FILTER_INTERFACES", "").strip()
    if not raw:
        return None
    parts = [iface.strip() for iface in raw.split(",") if iface.strip()]
    return parts or None


def low_pass_filter_block() -> tuple[str, str]:
    """Return ``("low_pass_filter", <json5>)`` capping per-message bytes.

    Three filters:

    * cmd / broadcast topics: 16 KiB default cap, both flows.
    * camera topics: 1 MiB default cap, ingress-only.
    * safety topics: 4 KiB default cap, both flows.

    Anything over the cap is dropped at the transport before the
    JSON parser runs.

    Interface binding: ``interfaces`` is OMITTED so Zenoh applies the
    cap to every link (``SubjectProperty::Wildcard``). Operators with a
    specific need to scope the cap to a subset of NICs supply
    ``STRANDS_MESH_FILTER_INTERFACES`` (comma-separated); see
    :func:`_filter_interfaces`. Earlier revisions enumerated every
    local NIC via psutil with a hardcoded fallback; that pattern
    silently bypassed the cap on hosts with non-canonical interface
    names (``enp0s3``, ``wlp2s0``, ``cni0``, ``wg0``, ...) when psutil
    was absent. Wildcard-by-default removes that footgun.
    """
    cmd_bytes = _int_env(
        "STRANDS_MESH_MAX_CMD_BYTES",
        DEFAULT_MAX_CMD_BYTES,
        lo=128,
        hi=16 * 1024 * 1024,
    )
    cam_bytes = _int_env(
        "STRANDS_MESH_MAX_CAMERA_BYTES",
        DEFAULT_MAX_CAMERA_BYTES,
        lo=1024,
        hi=128 * 1024 * 1024,
    )
    safety_bytes = _int_env(
        "STRANDS_MESH_MAX_SAFETY_BYTES",
        DEFAULT_MAX_SAFETY_BYTES,
        lo=128,
        hi=1 * 1024 * 1024,
    )
    interfaces = _filter_interfaces()

    def _rule(rule: dict) -> dict:
        # Only attach `interfaces` when the operator explicitly opted into a
        # subset; otherwise leave it unset so Zenoh treats the rule as
        # SubjectProperty::Wildcard (applies to every link).
        if interfaces is not None:
            rule["interfaces"] = interfaces
        return rule

    return (
        "low_pass_filter",
        json.dumps(
            [
                # NOTE on key_expr globs: the Zenoh ``namespace`` config
                # field prefixes keys on the wire (see
                # zenoh/src/net/routing/namespace.rs). The interceptor
                # matches against the wire key (post-prefix), but ``**``
                # matches any prefix including the namespace one, so
                # ``**/cmd`` is robust regardless of the configured
                # namespace.
                _rule(
                    {
                        "id": "strands_cmd_size_cap",
                        "messages": ["put"],
                        "flows": ["ingress", "egress"],
                        "key_exprs": ["**/cmd", "**/broadcast"],
                        "size_limit": cmd_bytes,
                    }
                ),
                _rule(
                    {
                        "id": "strands_camera_size_cap",
                        "messages": ["put"],
                        "flows": ["ingress"],  # R22-C: ingress-only, publisher trusts own frames
                        "key_exprs": ["**/camera/**"],
                        "size_limit": cam_bytes,
                    }
                ),
                _rule(
                    {
                        # R21: safety topics need their own (smaller)
                        # byte cap. A 100 KiB safety envelope is jumbo-
                        # frame DoS targeting the receiver-side HMAC and
                        # freshness math; legitimate safety envelopes are
                        # well under 1 KiB.
                        "id": "strands_safety_size_cap",
                        "messages": ["put"],
                        "flows": ["ingress", "egress"],
                        "key_exprs": ["**/safety/**"],
                        "size_limit": safety_bytes,
                    }
                ),
            ]
        ),
    )


# --- mTLS ---------------------------------------------------------------


def _resolve_tls_paths() -> tuple[Path, Path, Path]:
    """Return ``(ca, cert, key)`` paths from env vars.

    Raises :class:`FileNotFoundError` on a missing path so
    misconfiguration fails loud at session-open time rather than
    silently downgrading to plain TCP.
    """
    ca = os.getenv("STRANDS_MESH_TLS_CA", "").strip()
    cert = os.getenv("STRANDS_MESH_TLS_CERT", "").strip()
    key = os.getenv("STRANDS_MESH_TLS_KEY", "").strip()
    if not ca or not cert or not key:
        raise ValueError(
            "STRANDS_MESH_AUTH_MODE=mtls requires "
            "STRANDS_MESH_TLS_CA, STRANDS_MESH_TLS_CERT, "
            "and STRANDS_MESH_TLS_KEY to be set"
        )
    paths = (Path(ca), Path(cert), Path(key))
    # F7-A (PR #195 review): the existence + symlink check must come
    # before any other inspection. ``is_file`` follows symlinks; we
    # do an explicit ``is_symlink`` reject first so the path used for
    # mode + load is always the real file, never an attacker-redirected
    # link target.
    for label, p in zip(("CA", "cert", "key"), paths, strict=True):
        if p.is_symlink():
            raise ValueError(
                f"mTLS {label} file {p} is a SYMLINK "
                f"(target: {os.readlink(p)!r}). Refusing -- mTLS files "
                "must be real regular files at the operator-supplied path."
            )
        if not p.is_file():
            raise FileNotFoundError(f"mTLS {label} file does not exist: {p}")
    # R24-C: enforce the mode 0o600 contract that the docstring (line 73)
    # and README env-var matrix promise for the private key. A 0o644 key
    # file on a shared host is a real exfiltration surface; the operator
    # who set STRANDS_MESH_TLS_KEY thinks they get the documented protection.
    # Skipped on non-POSIX (Windows file modes do not map cleanly).
    #
    # F7-A (PR #195 review): use ``lstat()`` + ``is_symlink()`` reject
    # so a symlink to an attacker-writable file does not silently pass
    # the mode check. Without this, ``STRANDS_MESH_TLS_KEY=/safe/key.pem``
    # pointing at a co-tenant-controlled ``/tmp/evil.pem`` (which the
    # attacker has chmod'd 0o600) would pass while the actual TLS load
    # later opens the symlink target. Symmetric with the
    # ``O_NOFOLLOW`` + lstat-reject discipline applied across
    # ``audit.py:_ensure_paths``, ``_load_seq_counters``, and
    # ``_acl_config.py:_load_acl_file``.
    # See PR #195 threads PRRT_kwDORUMiZs6EUu8N + post-F3 follow-up.
    if os.name == "posix":
        key_path = paths[2]
        if key_path.is_symlink():
            raise ValueError(
                f"mTLS private key {key_path} is a SYMLINK "
                f"(target: {os.readlink(key_path)!r}). Refusing -- "
                "the mode check would otherwise stat() the target and "
                "an attacker-writable target with a 0o600 mode would "
                "silently pass while the TLS load follows the symlink. "
                "Set STRANDS_MESH_TLS_KEY to the real key file path."
            )
        # ``lstat`` returns the link's own metadata; since we just
        # rejected symlinks, this is equivalent to ``stat`` here, but
        # using lstat keeps the semantic explicit and matches the
        # rest of the codebase.
        key_mode = key_path.lstat().st_mode & 0o777
        if key_mode & 0o077:
            raise ValueError(
                f"mTLS private key {key_path} has mode 0o{key_mode:03o}; "
                "refusing world/group readable key. "
                "Run: chmod 600 " + str(key_path)
            )
    return paths


def tls_block() -> tuple[str, str]:
    """Return ``("transport/link/tls", <json5>)`` for mTLS terminator.

    Both listen-side and connect-side present the same cert (a peer can
    be either initiator or responder depending on who reaches whom
    first). Mutual TLS is mandatory; ``verify_name_on_connect`` is on so
    a peer that swaps its cert at the network layer cannot bypass CN
    matching on the ACL side.
    """
    ca, cert, key = _resolve_tls_paths()
    return (
        "transport/link/tls",
        json.dumps(
            {
                "root_ca_certificate": str(ca),
                "listen_private_key": str(key),
                "listen_certificate": str(cert),
                "connect_private_key": str(key),
                "connect_certificate": str(cert),
                "enable_mtls": True,
                "verify_name_on_connect": True,
                "close_link_on_expiration": True,
            }
        ),
    )


def link_protocols_block() -> tuple[str, str]:
    """Restrict the transport to TLS only when mTLS is on.

    Without this, an attacker could downgrade to plain TCP by being
    the first to bind a TCP listener -- Zenoh would happily accept the
    cleartext peer.
    """
    return ("transport/link/protocols", json.dumps(["tls"]))


# --- adminspace ---------------------------------------------------------


def adminspace_block() -> tuple[str, str]:
    """Lock down the admin space.

    Default in upstream Zenoh is already disabled but we set it
    explicitly so an operator who later toggles it on at the env-var
    layer can find the override centralised here.
    """
    return (
        "adminspace",
        json.dumps({"enabled": False, "permissions": {"read": False, "write": False}}),
    )
