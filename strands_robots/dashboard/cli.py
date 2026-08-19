"""``strands-robots dashboard`` CLI entry point."""

from __future__ import annotations

import argparse
import errno
import logging
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

#: Where an ``lsof`` binary is looked for. ``PATH`` first, then the two absolute
#: locations it actually ships in, because ``/usr/sbin`` is missing from the
#: ``PATH`` of a login-less shell (launchd jobs, CI runners, an agent's
#: subprocess) on exactly the platform where ``lsof`` is the only owner lookup
#: available to an unprivileged process.
_LSOF_CANDIDATES: tuple[str, ...] = ("/usr/sbin/lsof", "/usr/bin/lsof")

#: Longest command line reproduced in the refusal. The point is recognition, not
#: a full argv - a dashboard command line with mesh endpoints on it runs long
#: enough to bury the pid that precedes it.
_COMMAND_CHARS = 120


def _lsof_path() -> str | None:
    """Absolute path of an ``lsof`` binary, or ``None`` if none is installed."""
    found = shutil.which("lsof")
    if found:
        return found
    for candidate in _LSOF_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _listening_pid(port: int) -> int | None:
    """Pid of the process *listening* on ``port``, or ``None`` if not found.

    Two lookups, in order of how much they can be trusted:

    * ``psutil``, when it is installed *and* the platform lets an unprivileged
      process enumerate another's sockets (macOS raises ``AccessDenied``
      instead, which is why this is not the only route);
    * ``lsof``, restricted to ``-sTCP:LISTEN``. The restriction is
      load-bearing: a bare ``lsof -ti tcp:8090`` also returns every *client*
      with an established connection to the port, so the first pid it printed on
      this machine was a browser tab, not the server it was talking to.

    Owner discovery is best-effort by construction - the refusal it decorates is
    decided by the bind probe, never by whether a pid could be named.

    Args:
        port: TCP port to look up.

    Returns:
        The listening pid, or ``None`` when neither lookup can name one.
    """
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            for conn in psutil.net_connections(kind="tcp"):
                laddr = getattr(conn, "laddr", None)
                if getattr(laddr, "port", None) == port and conn.status == psutil.CONN_LISTEN:
                    if conn.pid:
                        return int(conn.pid)
        except Exception:  # AccessDenied on macOS, NotImplementedError elsewhere
            pass

    lsof = _lsof_path()
    if lsof is None:
        return None
    try:
        out = subprocess.run(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.split():
        try:
            return int(line)
        except ValueError:
            continue
    return None


def _process_command(pid: int) -> str | None:
    """Command line of ``pid``, truncated for a one-line message.

    Args:
        pid: Process id to describe.

    Returns:
        The command line, or ``None`` if the process is gone or unreadable.
    """
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            cmdline = " ".join(psutil.Process(pid).cmdline()).strip()
            if cmdline:
                return cmdline[:_COMMAND_CHARS]
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.splitlines()[0][:_COMMAND_CHARS] if out else None


def _port_in_use(port: int, host: str = "0.0.0.0") -> str | None:
    """Describe whoever holds ``port``, or ``None`` when it is free to bind.

    A second dashboard on a taken port does not fail cleanly: it initialises a
    mesh peer and a zenoh session *before* uvicorn asks the kernel for the
    socket, so the duplicate hub exists - and has already partitioned the mesh
    it joined - by the time the bind error is printed. The guard therefore
    probes the port itself, up front, rather than letting the server surface it.

    The probe binds and closes a socket with ``SO_REUSEADDR`` *set* - the same
    option uvicorn's own listener gets from asyncio on POSIX - on ``host`` plus
    both wildcard and loopback: a listener on ``0.0.0.0`` and a listener on
    ``127.0.0.1`` each conflict with the other, so one address alone reports a
    free port that the server then cannot bind. ``SO_REUSEADDR`` matters
    because without it the probe is *stricter than the server it fronts*: a
    leftover ``CLOSE_WAIT``/``TIME_WAIT`` socket from the previous instance
    (no listener at all) fails the bare bind with ``EADDRINUSE`` and the guard
    refuses a restart that uvicorn would have completed happily. With the
    option set, only a live LISTENer on the same address still collides -
    which is exactly the pileup this guard exists to catch.
    Only ``EADDRINUSE`` counts as occupied - ``EACCES`` on a privileged port and
    ``EADDRNOTAVAIL`` on an address this host does not own are different
    failures, and reporting them as a pileup would name an owner that does not
    exist.

    Args:
        port: TCP port the dashboard was asked to serve on.
        host: Address the dashboard was asked to bind, probed alongside the
            wildcard and loopback addresses.

    Returns:
        A human-readable description of the holder (``"pid 28346 (python -m
        strands_robots dashboard)"``, or ``"an unidentified process"`` when the
        owner cannot be looked up), or ``None`` if the port is free.
    """
    candidates = [host, "0.0.0.0", "127.0.0.1"]
    seen: set[str] = set()
    occupied = False
    for address in candidates:
        if address in seen:
            continue
        seen.add(address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((address, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                occupied = True
                break
        finally:
            sock.close()
    if not occupied:
        return None

    pid = _listening_pid(port)
    if pid is None:
        return "an unidentified process"
    command = _process_command(pid)
    return f"pid {pid} ({command})" if command else f"pid {pid}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="strands-robots dashboard",
        description="Fleet cockpit for the strands-robots mesh (PWA + API).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--peer-id", default=None, help="Mesh peer id for the dashboard (auto if omitted)")
    parser.add_argument("--local-dev", action="store_true", help="Set STRANDS_MESH_LOCAL_DEV=1 (no TLS, single machine)")
    parser.add_argument("--log-level", default="info")

    mesh = parser.add_argument_group(
        "remote mesh",
        "The dashboard is a mesh peer, not a hub - it can join a mesh running "
        "anywhere. These are persisted to the settings file and editable live "
        "from the UI (Settings -> Mesh).",
    )
    mesh.add_argument(
        "--zenoh-connect", default=None, metavar="EP[,EP...]",
        help="Endpoints to dial, e.g. tls/robot.lan:7447 (sets ZENOH_CONNECT)",
    )
    mesh.add_argument(
        "--zenoh-listen", default=None, metavar="EP[,EP...]",
        help="Endpoints to listen on, e.g. tls/0.0.0.0:7447 (sets ZENOH_LISTEN)",
    )
    mesh.add_argument("--mesh-port", type=int, default=None, help="Mesh port (default 7447)")
    mesh.add_argument(
        "--mesh-backend", default=None, choices=("zenoh", "iot", "bridge"),
        help="Transport: zenoh (LAN/VPN), iot (AWS IoT Core over the internet), bridge",
    )
    mesh.add_argument("--camera-hz", type=float, default=None, help="Camera publish rate cap")

    sec = parser.add_argument_group("security")
    sec.add_argument(
        "--auth-token", default=None, metavar="TOKEN",
        help="Require this bearer token on every /api and /ws request. Without "
             "it the dashboard is open to anyone who can reach the port - and "
             "it moves real motors. NOTE: a token on the command line is "
             "readable by every local user via ps; prefer --auth-token-file.",
    )
    sec.add_argument(
        "--auth-token-file", default=None, metavar="PATH",
        help="Read the bearer token from this file (first line, whitespace "
             "stripped) instead of the command line, so it never appears in "
             "ps output or shell history.",
    )
    sec.add_argument(
        "--cors-origin", action="append", default=None, metavar="ORIGIN",
        help="Allowed CORS origin (repeatable). Default '*'.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Start even when --port is already bound. Without it a second "
             "dashboard on a taken port is refused, because it joins the mesh "
             "as a duplicate hub before the bind fails.",
    )
    args = parser.parse_args()

    # Before anything joins the mesh: a second dashboard on a bound port creates
    # a duplicate zenoh hub and partitions the fleet, and it does so during
    # MeshBridge construction - well before uvicorn's bind would have failed.
    if not args.force:
        owner = _port_in_use(args.port, args.host)
        if owner is not None:
            print(
                f"dashboard: port {args.port} is already in use by {owner} "
                f"- refusing to start a second instance (--force to override, "
                f"--port N for another port)",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.local_dev:
        os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import uvicorn

    from strands_robots.dashboard import settings
    from strands_robots.dashboard.mesh_bridge import MeshBridge
    from strands_robots.dashboard.server import create_app

    # CLI flags are persisted, so they behave the same as the same fields set in
    # the UI - one resolution order (settings -> env -> default), not two.
    patch: dict[str, dict[str, object]] = {"mesh": {}, "security": {}}
    if args.zenoh_connect is not None:
        patch["mesh"]["connect"] = args.zenoh_connect
    if args.zenoh_listen is not None:
        patch["mesh"]["listen"] = args.zenoh_listen
    if args.mesh_port is not None:
        patch["mesh"]["port"] = args.mesh_port
    if args.mesh_backend is not None:
        patch["mesh"]["backend"] = args.mesh_backend
    if args.camera_hz is not None:
        patch["mesh"]["camera_hz"] = args.camera_hz
    if args.auth_token is not None:
        patch["security"]["auth_token"] = args.auth_token
    if args.auth_token_file is not None:
        # JOURNEYS #15: a token in argv is readable by every local user via ps
        # (this machine's audit literally lifted it that way). The file form
        # keeps it out of process listings and shell history. Refusing on a
        # missing/empty file beats silently starting open: the operator asked
        # for auth and would not get it.
        try:
            file_token = Path(args.auth_token_file).read_text().strip().splitlines()[0].strip()
        except (OSError, IndexError):
            file_token = ""
        if not file_token:
            parser.error(f"--auth-token-file {args.auth_token_file!r} is missing or empty")
        patch["security"]["auth_token"] = file_token
    if args.cors_origin is not None:
        patch["security"]["cors_origins"] = args.cors_origin
    changed = settings.update({k: v for k, v in patch.items() if v})

    resolved = settings.load(refresh=True)
    mesh_cfg = resolved["mesh"]
    local_dev = os.getenv("STRANDS_MESH_LOCAL_DEV", "") not in ("", "0", "false")

    print(f"🤖 strands-robots dashboard → http://{args.host}:{args.port}")
    print(f"   mesh: backend={mesh_cfg.get('backend') or 'zenoh'} "
          f"port={mesh_cfg.get('port') or 7447} "
          f"connect={mesh_cfg.get('connect') or ['<multicast/local>']} "
          f"listen={mesh_cfg.get('listen') or ['<default>']}")
    if local_dev:
        print("   ⚠️  STRANDS_MESH_LOCAL_DEV=1 - WIRE SECURITY DISABLED (no TLS, no auth)")
    if resolved["security"].get("auth_token"):
        print("   auth: bearer token required on /api and /ws")
    else:
        print("   ⚠️  no auth token - anyone who can reach this port can move motors "
              "(--auth-token to require one)")
    if changed:
        print(f"   saved to {settings.SETTINGS_FILE}: {', '.join(changed)}")
    # Piped stdout is block-buffered, so without this the security warnings land
    # *after* uvicorn's log lines - or only at exit - in a redirected logfile.
    sys.stdout.flush()

    app = create_app(MeshBridge(peer_id=args.peer_id))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
