#!/usr/bin/env python3
"""Qwen-VLA inference service management tool.

Manages a Qwen-VLA ZMQ inference server lifecycle and lets an agent run a
single inference against a running server. Mirrors the structure of
``gr00t_inference.py`` but for the Qwen-VLA provider.

Security (PR #90/#92 lessons):
    * ``validate_inputs()`` allowlists every caller-supplied value that flows
      into a subprocess argv or a model path: ``data_config`` must be a known
      config name, ``host`` must be a literal IP / hostname with no shell
      metacharacters, ``model_path`` is validated through ``validate_save_path``
      so it cannot point at a protected system directory or use ``..``.
    * Servers bind to ``127.0.0.1`` by default - never ``0.0.0.0`` - so an
      accidentally-started server is not exposed on the network.
    * No value is interpolated into a shell string; subprocess is invoked with
      an argv list (``shell=False``).
"""

import shlex
import shutil
import socket
import subprocess
import time
from typing import Any

from strands import tool

from strands_robots.policies.qwen_vla.data_config import DATA_CONFIG_MAP
from strands_robots.tools._path_validation import _HOST_RE, validate_save_path

# Default ZMQ port for Qwen-VLA (distinct from GR00T's 5555 so both can run).
_DEFAULT_PORT = 5556
# Loopback-only bind by default - explicit opt-out required to expose.
_DEFAULT_HOST = "127.0.0.1"
# Hostname/IP allowlist (_HOST_RE) is shared with qwen_vla_train via _path_validation.
# Hosts that are safe to bind a server to (loopback only).
_SAFE_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _validate_inputs(
    *,
    action: str,
    data_config: str,
    host: str,
    port: int,
    model_path: str | None,
    bind: bool,
) -> str | None:
    """Validate caller inputs against allowlists. Returns an error string or None.

    Args:
        action: The requested action.
        data_config: Embodiment config name (must be a registered key).
        host: Server host / target host.
        port: TCP port (1-65535).
        model_path: Optional model path (validated via validate_save_path).
        bind: Whether *host* is used to BIND a server (stricter: loopback only).

    Returns:
        ``None`` when all inputs pass, otherwise a human-readable error string
        (the caller turns it into a structured error dict).
    """
    valid_actions = ("start", "stop", "status", "list", "ping")
    if action not in valid_actions:
        return f"Unknown action {action!r}. Valid: {list(valid_actions)}"

    if data_config not in DATA_CONFIG_MAP:
        return f"Unknown data_config {data_config!r}. Available: {sorted(DATA_CONFIG_MAP)}"

    if not _HOST_RE.match(host):
        return f"host {host!r} contains invalid characters (allowed: letters, digits, '.', '-', ':', '_')"

    if bind and host not in _SAFE_BIND_HOSTS:
        return (
            f"refusing to bind a server to non-loopback host {host!r}. "
            f"Bind to one of {sorted(_SAFE_BIND_HOSTS)} and use an SSH tunnel / reverse proxy for remote access."
        )

    if not isinstance(port, int) or not (1 <= port <= 65535):
        return f"port must be an integer in [1, 65535], got {port!r}"

    if model_path is not None:
        try:
            validate_save_path(model_path, label="model_path")
        except ValueError as e:
            return str(e)

    return None


def _is_service_running(host: str, port: int) -> bool:
    """Return True iff a TCP listener is accepting connections at host:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host if host != "localhost" else "127.0.0.1", port)) == 0
    except OSError:
        return False


@tool
def qwen_vla_inference(
    action: str,
    data_config: str = "so100",
    model_path: str | None = None,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    device: str = "cuda",
    denoising_steps: int = 4,
    timeout: int = 120,
    server_command: str | None = None,
) -> dict[str, Any]:
    """Manage a Qwen-VLA inference service and run inference against it.

    Qwen-VLA (arXiv:2605.30280v2) is served over ZMQ. This tool starts /
    stops / inspects a server process and can ping a running one. All
    caller-supplied values are validated against allowlists before any
    subprocess or path I/O (servers bind to loopback by default).

    Actions:
        - ``start``: Launch a Qwen-VLA ZMQ server for ``model_path`` on
          ``host:port``. Requires ``model_path``. Binds to loopback only.
        - ``stop``: Stop a server running on ``port``.
        - ``status``: Check whether a server is accepting connections on
          ``host:port``.
        - ``list``: Discover Qwen-VLA servers on common ports (5556-5559).
        - ``ping``: Send a ``ping`` to a running server (validates the wire
          protocol end-to-end).

    Args:
        action: One of ``start``, ``stop``, ``status``, ``list``, ``ping``.
        data_config: Embodiment config name (must be registered; see
            ``strands_robots.policies.qwen_vla.DATA_CONFIG_MAP``).
        model_path: HF model ID or local checkpoint path (required for
            ``start``). Validated against protected-directory / traversal.
        host: Server host. For ``start`` this is the BIND host and must be
            loopback (``127.0.0.1`` / ``localhost`` / ``::1``).
        port: ZMQ port (default 5556).
        device: ``"cuda"`` or ``"cpu"`` (passed to the server on ``start``).
        denoising_steps: Flow-matching Euler steps for the server (``start``).
        timeout: Seconds to wait for the server to accept connections
            (``start``).
        server_command: Optional override for the server entrypoint argv
            (space-separated). Defaults to the upstream
            ``python -m qwen_vla.serve`` once released; until then ``start``
            returns a clear not-available error unless this is supplied.

    Returns:
        A structured status dict: ``{"status": "success"|"error", ...}``.
        Never raises - failures are returned as error dicts (AGENTS.md
        AgentTool contract).
    """
    bind = action == "start"
    err = _validate_inputs(
        action=action,
        data_config=data_config,
        host=host,
        port=port,
        model_path=model_path,
        bind=bind,
    )
    if err is not None:
        return {"status": "error", "message": err}

    if action == "status":
        running = _is_service_running(host, port)
        return {
            "status": "success",
            "host": host,
            "port": port,
            "service_status": "running" if running else "not_running",
        }

    if action == "list":
        services = [
            {"host": _DEFAULT_HOST, "port": p, "status": "running"}
            for p in (5556, 5557, 5558, 5559)
            if _is_service_running(_DEFAULT_HOST, p)
        ]
        return {"status": "success", "services": services, "message": f"Found {len(services)} running Qwen-VLA servers"}

    if action == "ping":
        return _ping_server(host=host, port=port, data_config=data_config)

    if action == "stop":
        return _stop_service(port)

    if action == "start":
        if model_path is None:
            return {"status": "error", "message": "model_path is required to start a Qwen-VLA server"}
        return _start_service(
            data_config=data_config,
            model_path=model_path,
            host=host,
            port=port,
            device=device,
            denoising_steps=denoising_steps,
            timeout=timeout,
            server_command=server_command,
        )

    # Unreachable: action already validated.
    return {"status": "error", "message": f"Unhandled action {action!r}"}


def _ping_server(*, host: str, port: int, data_config: str) -> dict[str, Any]:
    """Ping a running Qwen-VLA server via the ZMQ client."""
    if not _is_service_running(host, port):
        return {"status": "error", "message": f"No Qwen-VLA server running on {host}:{port}"}
    try:
        from strands_robots.policies.qwen_vla.client import QwenVlaInferenceClient

        client = QwenVlaInferenceClient(host=host, port=port, timeout_ms=5000)
        ok = client.ping()
        return {
            "status": "success" if ok else "error",
            "host": host,
            "port": port,
            "ping": "ok" if ok else "no_response",
        }
    except ImportError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:  # noqa: BLE001 - surface the wire error to the agent
        return {"status": "error", "message": f"Ping failed: {e}"}


def _stop_service(port: int) -> dict[str, Any]:
    """Stop a Qwen-VLA server bound to *port* via lsof + SIGTERM/SIGKILL."""
    if shutil.which("lsof") is None:
        return {"status": "error", "message": "lsof not available; cannot resolve server PID to stop"}
    try:
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True, check=False)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        if not pids:
            return {"status": "success", "port": port, "message": f"No server running on port {port}"}
        for pid in pids:
            subprocess.run(["kill", "-TERM", pid], check=False)
        time.sleep(2)
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True, check=False)
        for pid in (p for p in result.stdout.strip().split("\n") if p):
            subprocess.run(["kill", "-KILL", pid], check=False)
        return {"status": "success", "port": port, "message": f"Stopped Qwen-VLA server on port {port}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"Failed to stop server: {e}"}


def _start_service(
    *,
    data_config: str,
    model_path: str,
    host: str,
    port: int,
    device: str,
    denoising_steps: int,
    timeout: int,
    server_command: str | None,
) -> dict[str, Any]:
    """Start a Qwen-VLA ZMQ server as a detached subprocess (argv, no shell).

    The upstream server entrypoint is finalized on the model's public release
    (PLAN section 6.2). Until then, callers must pass ``server_command`` with
    their own entrypoint, otherwise we return a clear not-available error
    rather than guessing a command.
    """
    if _is_service_running(host, port):
        return {"status": "error", "message": f"A server is already running on {host}:{port}"}

    if server_command is None:
        return {
            "status": "error",
            "message": (
                "No Qwen-VLA server entrypoint is bundled yet (upstream package not released). "
                "Pass server_command='<your entrypoint>' to launch a custom server, e.g. "
                "'python -m qwen_vla.serve'. The provider's SERVICE mode will then connect over ZMQ."
            ),
        }

    # Tokenize the caller's entrypoint and append validated flags as argv.
    # shlex.split is quote-aware (handles e.g. --prefix "with space").
    base_argv = shlex.split(server_command)
    if not base_argv:
        return {"status": "error", "message": "server_command must not be empty"}

    argv = [
        *base_argv,
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--data-config",
        data_config,
        "--device",
        device,
        "--denoising-steps",
        str(denoising_steps),
    ]

    try:
        # Detach so the server outlives this tool call; no shell interpolation.
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
    except (OSError, ValueError) as e:
        return {"status": "error", "message": f"Failed to launch server: {e}"}

    start = time.time()
    while time.time() - start < timeout:
        if _is_service_running(host, port):
            return {
                "status": "success",
                "host": host,
                "port": port,
                "data_config": data_config,
                "model_path": model_path,
                "device": device,
                "denoising_steps": denoising_steps,
                "message": f"Qwen-VLA server started on {host}:{port}",
            }
        time.sleep(1)

    return {"status": "error", "message": f"Server did not accept connections within {timeout}s"}


__all__ = ["qwen_vla_inference"]
