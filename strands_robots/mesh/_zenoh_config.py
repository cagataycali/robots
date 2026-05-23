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
    ``strands_robots``. Two fleets with different namespaces cannot
    collide on the same network.

``STRANDS_MESH_MULTICAST``
    ``true`` to enable multicast scouting. Default ``false`` — gossip
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

``STRANDS_MESH_AUTH_MODE``
    ``mtls`` (default) or ``none``. ``none`` is a development-only mode
    that skips the TLS terminator and ACL — never run it on a network
    you do not fully trust. The mesh still emits namespace, scouting,
    and DoS-cap config in ``none`` mode; only the auth + ACL blocks are
    omitted.

``STRANDS_MESH_TLS_CA``
    Filesystem path to the CA bundle used to validate peer certificates.
    Required when ``STRANDS_MESH_AUTH_MODE=mtls``.

``STRANDS_MESH_TLS_CERT``
    Filesystem path to this peer's certificate (PEM).

``STRANDS_MESH_TLS_KEY``
    Filesystem path to this peer's private key (PEM, mode 0o600).

``STRANDS_MESH_ACL_FILE``
    Filesystem path to a JSON5 ACL file. When unset, the built-in
    default-deny ACL from :func:`default_acl_block` is used (robots
    publish telemetry + receive cmds; operators publish cmds + observe).
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
#: The `namespace` Zenoh config field provides routing isolation —
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
    """
    raw = os.getenv("STRANDS_MESH_AUTH_MODE", "mtls").strip().lower()
    if raw not in ("mtls", "none"):
        raise ValueError(f"STRANDS_MESH_AUTH_MODE={raw!r} not supported (expected 'mtls' or 'none')")
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

    Multicast on a hostile LAN is a discovery attack surface — any host
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


def downsampling_block(namespace: str) -> tuple[str, str]:
    """Return ``("downsampling", <json5>)`` capping the cmd-publish rate.

    A peer publishing to ``{namespace}/*/cmd`` faster than the
    configured frequency is throttled at the transport layer — the
    extra messages are dropped before they reach the JSON parser, so
    flood attacks cost the receiver almost nothing.
    """
    freq = _float_env(
        "STRANDS_MESH_CMD_RATE_HZ",
        DEFAULT_CMD_RATE_HZ,
        lo=0.001,
        hi=10000.0,
    )
    # See ``low_pass_filter_block`` for the namespace-vs-key_expr note:
    # ``**/cmd`` matches any prefix including the namespace one;
    # ``f"{namespace}/*/cmd"`` would not.
    rules = [
        {"key_expr": "**/cmd", "freq": freq},
        {"key_expr": "**/broadcast", "freq": freq},
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


def _local_interfaces() -> list[str]:
    """Enumerate every local network interface name.

    Zenoh's ``low_pass_filter`` and ``downsampling`` blocks require an
    explicit ``interfaces`` allowlist; an empty list silently disables
    the filter, and ``["*"]`` does NOT mean "all interfaces" (it is
    matched literally and almost always misses). We enumerate every
    interface visible to the OS and pass them all so the cap applies
    fleet-wide regardless of which NIC a peer connects through.

    Operators can override the list with
    ``STRANDS_MESH_FILTER_INTERFACES`` (comma-separated). This is
    useful when an environment has dozens of virtual interfaces and
    the operator wants to bind the filter to a specific subset.
    """
    raw = os.getenv("STRANDS_MESH_FILTER_INTERFACES", "").strip()
    if raw:
        return [iface.strip() for iface in raw.split(",") if iface.strip()]
    try:
        import psutil  # type: ignore[import-not-found]

        return sorted(psutil.net_if_addrs().keys())
    except ImportError:
        # psutil is a transitive dep of strands-agents; this branch is
        # for dev environments where it is genuinely missing. Fall back
        # to the canonical list of interfaces a container or laptop is
        # likely to expose. The filter still functions if any of these
        # match the actual link the traffic rides; missing entries just
        # mean traffic on those interfaces bypasses the cap.
        return ["lo", "lo0", "eth0", "en0", "en1", "wlan0"]


def low_pass_filter_block(namespace: str) -> tuple[str, str]:
    """Return ``("low_pass_filter", <json5>)`` capping per-message bytes.

    Two filters: cmd / broadcast topics get a 16 KiB cap; camera
    topics get a 1 MiB cap. Anything larger is dropped at the
    transport.

    The ``interfaces`` field is REQUIRED — without it Zenoh treats the
    block as a no-op. We enumerate every local interface (or use the
    operator-supplied ``STRANDS_MESH_FILTER_INTERFACES`` allowlist) so
    the cap applies regardless of which NIC a peer's link rides on.
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
    interfaces = _local_interfaces()
    return (
        "low_pass_filter",
        json.dumps(
            [
                # NOTE on key_expr globs: the Zenoh ``namespace`` config
                # prefixes keys on the wire but ``low_pass_filter`` matches
                # against the user-side (un-prefixed) key, so a filter
                # written as ``f"{namespace}/*/cmd"`` never fires. ``**/cmd``
                # matches any prefix (including the empty / namespace one)
                # and is the robust choice. The namespace is therefore not
                # used in these key_exprs — fleet isolation is provided by
                # the namespace itself, not by these globs.
                {
                    "id": "strands_cmd_size_cap",
                    "interfaces": interfaces,
                    "messages": ["put"],
                    "flows": ["ingress", "egress"],
                    "key_exprs": ["**/cmd", "**/broadcast"],
                    "size_limit": cmd_bytes,
                },
                {
                    "id": "strands_camera_size_cap",
                    "interfaces": interfaces,
                    "messages": ["put"],
                    "flows": ["ingress", "egress"],
                    "key_exprs": ["**/camera/**"],
                    "size_limit": cam_bytes,
                },
            ]
        ),
    )


# ─── mTLS ───────────────────────────────────────────────────────────────


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
    for label, p in zip(("CA", "cert", "key"), paths, strict=True):
        if not p.is_file():
            raise FileNotFoundError(f"mTLS {label} file does not exist: {p}")
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
    the first to bind a TCP listener — Zenoh would happily accept the
    cleartext peer.
    """
    return ("transport/link/protocols", json.dumps(["tls"]))


# ─── adminspace ─────────────────────────────────────────────────────────


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
