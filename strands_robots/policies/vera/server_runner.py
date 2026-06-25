"""Managed VERA policy-server subprocess.

Launches ``python -m vera.server.start_vera_server`` with **list args** (never a
shell string — see PR #621 feedback), health-checks the websocket before
returning, streams the server's stdout/stderr to the logger, and shuts the
process down cleanly on :meth:`stop`.

The server holds the GPU and the two-stage model; this provider talks to it over
the websocket (see :mod:`client`). Auto-launch is optional — point the provider
at an already-running server by setting ``auto_launch_server=False``.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import VeraConfig

logger = logging.getLogger(__name__)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if a TCP connection to ``host:port`` succeeds (server is listening)."""
    # 0.0.0.0 is a bind address, not connectable — probe loopback instead.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return True
    except OSError:
        return False


class VeraServerRunner:
    """Launch and supervise a ``vera.server.start_vera_server`` subprocess.

    Args:
        config: The :class:`~strands_robots.policies.vera.config.VeraConfig`
            driving embodiment, ports, checkpoints and planner knobs.
    """

    def __init__(self, config: VeraConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen[str] | None = None
        self._log_thread: threading.Thread | None = None

    # -- command construction ----------------------------------------------

    def _build_command(self) -> list[str]:
        """Assemble the server launch argv as a list (no shell string)."""
        cfg = self.config
        python = cfg.python_executable or sys.executable
        cmd: list[str] = [
            python,
            "-m",
            "vera.server.start_vera_server",
            "--embodiment",
            str(cfg.embodiment),
            "--host",
            str(cfg.host),
            "--port",
            str(cfg.server_port),
        ]
        if cfg.vis_port:
            cmd += ["--vis-port", str(cfg.vis_port)]
        if cfg.algo_config is not None:
            cmd += ["--algo-config", str(cfg.algo_config)]
        if cfg.dynamics_run_id:
            cmd += ["--dynamics-run-id", str(cfg.dynamics_run_id)]
        if cfg.text_prompt:
            cmd += ["--text", str(cfg.text_prompt)]
        if cfg.sample_steps is not None:
            cmd += ["--sample-steps", str(cfg.sample_steps)]
        if not cfg.teacache:
            cmd += ["--no-teacache"]
        else:
            cmd += ["--teacache-thresh", str(cfg.teacache_thresh)]
        return cmd

    # -- lifecycle ----------------------------------------------------------

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Launch the server (idempotent) and block until its websocket is up."""
        cfg = self.config

        # Already serving (ours or someone else's) — reuse it.
        if _port_open(cfg.host, cfg.server_port):
            logger.info("VERA server already listening on %s:%s; reusing", cfg.host, cfg.server_port)
            return

        cmd = self._build_command()
        env = {**os.environ, **cfg.server_env()}
        logger.info("launching VERA server: %s", " ".join(cmd))

        self._proc = subprocess.Popen(  # noqa: S603 - list args, no shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._start_log_pump()
        self._wait_until_ready()

    def _start_log_pump(self) -> None:
        """Stream server stdout/stderr to the logger on a daemon thread."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        def _pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                logger.info("[vera.server] %s", line.rstrip())

        self._log_thread = threading.Thread(target=_pump, name="vera-server-log", daemon=True)
        self._log_thread.start()

    def _wait_until_ready(self) -> None:
        """Poll the websocket port until ready, or raise on timeout / early exit."""
        cfg = self.config
        deadline = time.monotonic() + cfg.server_ready_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                code = self._proc.returncode
                raise RuntimeError(
                    f"VERA server exited early (code {code}) before becoming ready. "
                    f"Check the [vera.server] log lines above; common causes are "
                    f"missing checkpoints (set VERA_CKPT_ROOT / ckpt_root) or CUDA OOM."
                )
            if _port_open(cfg.host, cfg.server_port):
                logger.info("VERA server ready on %s:%s", cfg.host, cfg.server_port)
                return
            time.sleep(1.0)
        self.stop()
        raise TimeoutError(
            f"VERA server did not become ready on {cfg.host}:{cfg.server_port} "
            f"within {cfg.server_ready_timeout:.0f}s (WAN model load can be slow — "
            f"raise server_ready_timeout / VERA_SERVER_READY_TIMEOUT if needed)."
        )

    def stop(self) -> None:
        """Terminate the server subprocess cleanly (SIGTERM, then SIGKILL)."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("VERA server did not stop on SIGTERM; killing")
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("VERA server unresponsive to SIGKILL")
        self._proc = None
        logger.info("VERA server stopped")
